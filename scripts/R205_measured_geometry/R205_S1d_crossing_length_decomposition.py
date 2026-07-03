"""R205-S1d: decompose measured crossings by edge length - shortcut vs local flyover.

S2b showed that locality-bounded few-crossing swaps (flyover-faithful) leave pc
essentially unchanged across the whole real-city chi range, while unconstrained
long-chord swaps (S2) collapse pc immediately. Refined mechanism: what moves the
threshold is LONG-RANGE SHORTCUT crossings, not crossing count per se.

Real-data test: split each city's measured crossing pairs by the longer member's
length relative to the city's own edge-length distribution:

  chi_long  = crossings whose longer edge > city P90 edge length, / n_edges
  chi_short = remaining crossings / n_edges

Prediction: the chi -> road_gap association is carried by chi_long; chi_short
contributes little. Outputs:
  outputs/R205_S1d_crossing_length_decomposition.csv
  outputs/R205_S1d_summary.json
"""

from __future__ import annotations

import json
import pickle
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


def one_city(pkl_path_str: str) -> dict:
    t0 = time.time()
    p = Path(pkl_path_str)
    row = {"graph_cache_file": p.name, "status": "fail"}
    try:
        with open(p, "rb") as f:
            G = pickle.load(f)
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
            # metric length from OSMnx attribute when present (metres)
            try:
                mlen = float(data.get("length"))
            except (TypeError, ValueError):
                mlen = float(geom.length)
            seen[key] = (geom, mlen)
        keys = list(seen.keys())
        geoms = [seen[k][0] for k in keys]
        lens = np.array([seen[k][1] for k in keys])
        m = len(keys)
        p90 = float(np.quantile(lens, 0.90))
        tree = STRtree(geoms)
        qi, ti = tree.query(geoms, predicate="crosses")
        long_cnt = short_cnt = 0
        pair_longer = []
        for a, b in zip(qi.tolist(), ti.tolist()):
            if a >= b:
                continue
            ka, kb = keys[a], keys[b]
            if ka[0] in kb or ka[1] in kb:
                continue
            longer = max(lens[a], lens[b])
            pair_longer.append(longer)
            if longer > p90:
                long_cnt += 1
            else:
                short_cnt += 1
        row.update(
            n_edges_simple=m,
            edge_len_p90_m=p90,
            edge_len_median_m=float(np.median(lens)),
            chi_long=long_cnt / m,
            chi_short=short_cnt / m,
            crossing_longer_median_m=(float(np.median(pair_longer)) if pair_longer else np.nan),
            crossing_len_ratio=(float(np.median(pair_longer) / np.median(lens)) if pair_longer else np.nan),
            seconds=round(time.time() - t0, 1),
            status="ok",
        )
    except Exception:
        row["error"] = traceback.format_exc(limit=3)
    return row


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pkls = sorted(CACHE.glob("*.pkl"))
    rows = []
    with Pool(16, maxtasksperchild=4) as pool:
        for i, row in enumerate(pool.imap_unordered(one_city, [str(p) for p in pkls]), 1):
            rows.append(row)
            if i % 12 == 0:
                print(f"[S1d] {i}/{len(pkls)}", flush=True)
    df = pd.DataFrame(rows)
    r61 = pd.read_csv(TABLES / "R61_central_window_screen_results.csv")
    r61["graph_cache_file"] = r61["graph_cache_file"].map(lambda s: Path(str(s).replace("\\", "/")).name)
    df = df.merge(r61[["city", "graph_cache_file", "mean_degree", "pc_cebh", "pc_rank", "road_gap"]],
                  on="graph_cache_file", how="left")
    # spatial-null residual join (crossings-allowed null error)
    r81 = pd.read_csv(TABLES / "R81_full71_high_intensity_geometry_null_city_summary.csv")
    sp = r81[r81.swap_fraction == 0.01][["city", "road_minus_spatial"]]
    df = df.merge(sp, on="city", how="left")
    df.to_csv(OUTDIR / "R205_S1d_crossing_length_decomposition.csv", index=False)

    from scipy.stats import spearmanr
    ok = df[df.status == "ok"].dropna(subset=["road_gap"])
    summary = {"n": int(len(ok)),
               "chi_long_median": float(ok.chi_long.median()),
               "chi_short_median": float(ok.chi_short.median()),
               "crossing_len_ratio_median": float(ok.crossing_len_ratio.median())}
    for x in ("chi_long", "chi_short", "crossing_len_ratio"):
        for y in ("road_gap", "road_minus_spatial"):
            sub = ok.dropna(subset=[x, y])
            rs, ps = spearmanr(sub[x], sub[y])
            summary[f"{x}__vs__{y}"] = {"n": int(len(sub)), "spearman": round(float(rs), 4),
                                        "p": round(float(ps), 6)}
    # partial: chi_long controlling chi_short and vice versa (rank-residual)
    from scipy.stats import rankdata
    def partial(xcol, ycol, ctrl):
        sub = ok.dropna(subset=[xcol, ycol, ctrl])
        rx, ry, rc = rankdata(sub[xcol]), rankdata(sub[ycol]), rankdata(sub[ctrl])
        A = np.column_stack([np.ones(len(rc)), rc])
        ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
        ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
        return float(np.corrcoef(ex, ey)[0, 1]), int(len(sub))
    for x, c in (("chi_long", "chi_short"), ("chi_short", "chi_long")):
        r, n = partial(x, "road_gap", c)
        summary[f"partial_{x}__road_gap__ctrl_{c}"] = {"n": n, "partial_spearman": round(r, 4)}
    with open(OUTDIR / "R205_S1d_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
