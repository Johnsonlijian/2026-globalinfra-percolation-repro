"""R206-A2: rail chi_geo artifact validation - crossing angles + LCC restriction.

OSM rail is track-level topology: parallel tracks in stations/yards can produce
near-parallel projected "crossings" that are digitization artifacts, not grade
separations. For each rail network this script:
  1. recomputes crossing pairs from polyline geometry,
  2. estimates the crossing ANGLE at each intersection (tangent directions of the
     two polylines at the crossing point, folded to (0, 90] degrees),
  3. reports chi under filters: all / angle >= 15 deg / LCC-only / LCC + angle,
  4. reports the angle distribution quartiles.
Decision rule: if angle-filtering collapses chi, rail crossings are dominated by
near-parallel artifacts and only the filtered value (or chi_topo) may be used.

Output: outputs/R206_A2_rail_artifact_validation.csv (+ .json summary)
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "data" / "R206_R208_ress_reframe"

RAIL_CITIES = ["Berlin, Germany", "Hamburg, Germany", "Vienna, Austria",
               "Madrid, Spain", "Milan, Italy", "Osaka, Japan", "Chicago, Illinois, USA"]
RAIL_FILTER = '["railway"~"rail|light_rail|subway|narrow_gauge"]'
ANGLE_MIN_DEG = 15.0


def tangent_at(geom, d, eps):
    a = geom.interpolate(max(d - eps, 0.0))
    b = geom.interpolate(min(d + eps, geom.length))
    v = np.array([b.x - a.x, b.y - a.y])
    nv = np.linalg.norm(v)
    return v / nv if nv > 0 else np.array([1.0, 0.0])


def crossing_angle(g1, g2, pt):
    d1 = g1.project(pt)
    d2 = g2.project(pt)
    t1 = tangent_at(g1, d1, max(g1.length * 1e-4, 1e-9))
    t2 = tangent_at(g2, d2, max(g2.length * 1e-4, 1e-9))
    cosang = abs(float(np.clip(np.dot(t1, t2), -1, 1)))
    return float(np.degrees(np.arccos(cosang)))  # in [0, 90]


def one_city(place):
    import osmnx as ox
    from shapely.geometry import LineString
    from shapely.strtree import STRtree
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(ROOT / "cache" / "osmnx")
    ox.settings.log_console = False
    ox.settings.requests_timeout = 300
    G = ox.graph_from_place(place, custom_filter=RAIL_FILTER, retain_all=True, simplify=True)
    nodes_xy = {nid: (d["x"], d["y"]) for nid, d in G.nodes(data=True)}
    seen = {}
    for u, v, data in G.edges(data=True):
        if u == v:
            continue
        key = (u, v) if u <= v else (v, u)
        if key in seen:
            continue
        geom = data.get("geometry") or LineString([nodes_xy[u], nodes_xy[v]])
        seen[key] = geom
    keys = list(seen.keys())
    geoms = [seen[k] for k in keys]
    m = len(keys)
    # LCC membership on the simple undirected graph
    S = nx.Graph()
    S.add_edges_from(keys)
    lcc = max(nx.connected_components(S), key=len)
    in_lcc = np.array([(k[0] in lcc and k[1] in lcc) for k in keys])
    m_lcc = int(in_lcc.sum())
    tree = STRtree(geoms)
    qi, ti = tree.query(geoms, predicate="crosses")
    angles, pair_lcc = [], []
    n_pairs = 0
    for a, b in zip(qi.tolist(), ti.tolist()):
        if a >= b:
            continue
        ka, kb = keys[a], keys[b]
        if ka[0] in kb or ka[1] in kb:
            continue
        n_pairs += 1
        inter = geoms[a].intersection(geoms[b])
        pt = inter if inter.geom_type == "Point" else (
            list(inter.geoms)[0] if hasattr(inter, "geoms") and len(list(inter.geoms)) else inter.centroid)
        try:
            ang = crossing_angle(geoms[a], geoms[b], pt)
        except Exception:
            ang = np.nan
        angles.append(ang)
        pair_lcc.append(bool(in_lcc[a] and in_lcc[b]))
    angles = np.array(angles, float)
    pair_lcc = np.array(pair_lcc, bool)
    ok_ang = angles >= ANGLE_MIN_DEG
    row = {
        "city": place.split(",")[0], "n_edges": m, "n_edges_lcc": m_lcc,
        "n_cross_pairs": n_pairs,
        "chi_all": n_pairs / m,
        "chi_angle15": int(np.nansum(ok_ang)) / m,
        "chi_lcc": int(pair_lcc.sum()) / max(m_lcc, 1),
        "chi_lcc_angle15": int(np.nansum(ok_ang & pair_lcc)) / max(m_lcc, 1),
        "angle_q25": float(np.nanpercentile(angles, 25)) if n_pairs else np.nan,
        "angle_median": float(np.nanpercentile(angles, 50)) if n_pairs else np.nan,
        "angle_q75": float(np.nanpercentile(angles, 75)) if n_pairs else np.nan,
        "share_below_15deg": float(np.nanmean(angles < ANGLE_MIN_DEG)) if n_pairs else np.nan,
        "status": "ok",
    }
    return row


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for place in RAIL_CITIES:
        try:
            row = one_city(place)
        except Exception:
            row = {"city": place, "status": "fail", "error": traceback.format_exc(limit=3)}
        rows.append(row)
        print(f"[A2] {row.get('city')}: chi_all={row.get('chi_all')} "
              f"chi_lcc_angle15={row.get('chi_lcc_angle15')} "
              f"share<15deg={row.get('share_below_15deg')} ({row['status']})", flush=True)
        pd.DataFrame(rows).to_csv(OUTDIR / "R206_A2_rail_artifact_validation.csv", index=False)
    ok = pd.DataFrame([r for r in rows if r["status"] == "ok"])
    if len(ok):
        summary = {
            "n_cities": int(len(ok)),
            "chi_all_median": float(ok.chi_all.median()),
            "chi_angle15_median": float(ok.chi_angle15.median()),
            "chi_lcc_angle15_median": float(ok.chi_lcc_angle15.median()),
            "share_below_15deg_median": float(ok.share_below_15deg.median()),
            "verdict_hint": "artifact-dominated if chi collapses under angle filter",
        }
        with open(OUTDIR / "R206_A2_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
