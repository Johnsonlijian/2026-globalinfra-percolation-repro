"""R205-S1: measured projected crossing exposure for the 71 standardized city windows.

Purpose (escape-upgrade U1): the manuscript's non-crossing mechanism is manipulated
(null ladder, synthetic dial) but never MEASURED on the real cities. This script
measures, per city window, how far the real embedded road graph violates the
non-crossing idealization in projection:

  chi_poly  = (# unordered pairs of node-disjoint edges whose true OSM polyline
               geometries properly cross in projection) / n_edges
  chi_chord = same count with every edge represented by its endpoint chord
              (the representation actually used by the strict geometry null,
               main text section 2.5), / n_edges

Both are reported so the measured exposure and the null's chord-space rule can be
compared directly. Bridge/tunnel OSM tags are recorded per crossing pair as a
grade-separation plausibility check (a projected crossing on a real road network
should usually involve a tagged grade-separated edge).

Inputs : cache/R61_osmnx_city_windows/*.pkl  (OSMnx MultiDiGraph, epsg:4326)
         public repo source_data/tables/R61_central_window_screen_results.csv
         public repo source_data/tables/R81_full71_high_intensity_geometry_null_city_summary.csv (join if present)
Outputs: outputs/R205_S1_crossing_density_71cities.csv
         outputs/R205_S1_dose_response_summary.json
No manuscript files are modified. Derived results only.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import LineString
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache" / "R61_osmnx_city_windows"
TABLES = ROOT / "source_data" / "tables"
OUTDIR = ROOT / "data" / "R205_measured_geometry_dose_response"

N_WORKERS = 16


def _tag_truthy(v) -> bool:
    if v is None:
        return False
    if isinstance(v, (list, tuple, set)):
        return any(_tag_truthy(x) for x in v)
    s = str(v).strip().lower()
    return s not in ("", "no", "false", "none", "0", "nan")


def _simple_edges(G):
    """Simple undirected edge list with polyline geometry, chord geometry, tag flag."""
    nodes_xy = {nid: (d["x"], d["y"]) for nid, d in G.nodes(data=True)}
    seen = {}
    for u, v, data in G.edges(data=True):
        if u == v:
            continue
        key = (u, v) if u <= v else (v, u)
        if key in seen:
            continue
        geom = data.get("geometry")
        if geom is None:
            geom = LineString([nodes_xy[u], nodes_xy[v]])
        chord = LineString([nodes_xy[u], nodes_xy[v]])
        tagged = _tag_truthy(data.get("bridge")) or _tag_truthy(data.get("tunnel"))
        seen[key] = (geom, chord, tagged)
    keys = list(seen.keys())
    geoms = [seen[k][0] for k in keys]
    chords = [seen[k][1] for k in keys]
    tags = np.array([seen[k][2] for k in keys], dtype=bool)
    return keys, geoms, chords, tags


def _crossing_pairs(keys, geoms):
    """Unordered node-disjoint properly-crossing pairs via STRtree bulk query."""
    tree = STRtree(geoms)
    qi, ti = tree.query(geoms, predicate="crosses")
    pairs = set()
    for a, b in zip(qi.tolist(), ti.tolist()):
        if a >= b:
            continue
        ka, kb = keys[a], keys[b]
        if ka[0] in kb or ka[1] in kb:
            continue  # share a node: adjacent, not a grade-separation crossing
        pairs.add((a, b))
    return pairs


def one_city(pkl_path_str: str) -> dict:
    t0 = time.time()
    p = Path(pkl_path_str)
    row = {"graph_cache_file": p.name, "status": "fail"}
    try:
        with open(p, "rb") as f:
            G = pickle.load(f)
        keys, geoms, chords, tags = _simple_edges(G)
        m = len(keys)
        cross_poly = _crossing_pairs(keys, geoms)
        cross_chord = _crossing_pairs(keys, chords)
        tag_hits = sum(1 for a, b in cross_poly if tags[a] or tags[b])
        row.update(
            n_nodes=G.number_of_nodes(),
            n_edges_simple=m,
            n_cross_pairs_poly=len(cross_poly),
            n_cross_pairs_chord=len(cross_chord),
            chi_poly=len(cross_poly) / m,
            chi_chord=len(cross_chord) / m,
            bridge_tunnel_tagged_edge_share=float(tags.mean()),
            cross_pairs_with_tagged_edge_share=(tag_hits / len(cross_poly)) if cross_poly else np.nan,
            seconds=round(time.time() - t0, 1),
            status="ok",
        )
    except Exception:
        row["error"] = traceback.format_exc(limit=3)
    return row


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pkls = sorted(CACHE.glob("*.pkl"))
    print(f"[S1] {len(pkls)} cached city windows", flush=True)
    rows = []
    with Pool(N_WORKERS, maxtasksperchild=4) as pool:
        for i, row in enumerate(pool.imap_unordered(one_city, [str(p) for p in pkls]), 1):
            rows.append(row)
            print(f"[S1] {i}/{len(pkls)} {row.get('graph_cache_file')} "
                  f"chi_poly={row.get('chi_poly', float('nan')):.5f} "
                  f"chi_chord={row.get('chi_chord', float('nan')):.5f} ({row['status']})", flush=True)
            if i % 8 == 0:
                pd.DataFrame(rows).to_csv(OUTDIR / "R205_S1_crossing_density_71cities.csv", index=False)
    df = pd.DataFrame(rows)

    # Join the R61 screen (thresholds) on cache file name.
    r61 = pd.read_csv(TABLES / "R61_central_window_screen_results.csv")
    r61["graph_cache_file"] = r61["graph_cache_file"].map(lambda s: Path(str(s).replace("\\", "/")).name)
    df = df.merge(
        r61[["city", "country", "macro_region", "graph_cache_file", "mean_degree",
             "pc_cebh", "pc_rank", "road_gap"]],
        on="graph_cache_file", how="left")
    df["anchor_pc"] = 2.0 / df["mean_degree"]
    df["anchor_deviation"] = df["pc_rank"] * df["mean_degree"] / 2.0 - 1.0
    # Post-geometry-null residual (71-city high-intensity ensemble), joined by city.
    try:
        r81 = pd.read_csv(TABLES / "R81_full71_high_intensity_geometry_null_city_summary.csv")
        cand = [c for c in r81.columns if "resid" in c.lower()]
        keep = ["city"] + cand
        df = df.merge(r81[keep].rename(columns={c: f"r81_{c}" for c in cand}), on="city", how="left")
    except Exception as e:
        print(f"[S1] R81 join skipped: {e}", flush=True)

    df.to_csv(OUTDIR / "R205_S1_crossing_density_71cities.csv", index=False)

    # Dose-response statistics with city bootstrap.
    from scipy.stats import pearsonr, spearmanr
    ok = df[df["status"] == "ok"].dropna(subset=["road_gap"])
    summary = {"n_cities_ok": int(len(ok))}
    rng = np.random.default_rng(20260702)
    for xcol in ("chi_poly", "chi_chord", "bridge_tunnel_tagged_edge_share"):
        for ycol in ("road_gap", "anchor_deviation"):
            x, y = ok[xcol].to_numpy(), ok[ycol].to_numpy()
            rs, ps = spearmanr(x, y)
            rp, pp = pearsonr(x, y)
            boots = []
            for _ in range(5000):
                idx = rng.integers(0, len(x), len(x))
                if len(set(x[idx])) > 2:
                    boots.append(spearmanr(x[idx], y[idx])[0])
            lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
            summary[f"{xcol}__vs__{ycol}"] = {
                "spearman": round(float(rs), 4), "spearman_p": round(float(ps), 6),
                "spearman_ci95": [round(float(lo), 4), round(float(hi), 4)],
                "pearson": round(float(rp), 4), "pearson_p": round(float(pp), 6),
            }
    for col in ("chi_poly", "chi_chord"):
        summary[f"{col}_distribution"] = {
            "min": float(ok[col].min()), "median": float(ok[col].median()),
            "mean": float(ok[col].mean()), "max": float(ok[col].max()),
        }
    summary["cross_pairs_with_tagged_edge_share_median"] = float(
        ok["cross_pairs_with_tagged_edge_share"].median())
    with open(OUTDIR / "R205_S1_dose_response_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[S1] DONE", json.dumps(summary, indent=2)[:2000], flush=True)


if __name__ == "__main__":
    main()
