"""R205-S3: put all four infrastructure domains on ONE measured geometry axis.

The manuscript currently treats water/rail/power as threshold-scale transfer
checks and explicitly disclaims a shared mechanism, because the non-crossing
mechanism was only ever manipulated on roads. This script measures, per real
network, how far each domain actually violates the non-crossing idealization,
using two complementary measures:

  chi_geo  : projected crossing density (node-disjoint properly-crossing pairs
             per edge) - polyline geometry where available (rail, roads),
             endpoint chords otherwise (water; coordinates may be schematic and
             are flagged as such)
  chi_topo : topological non-planarity = 1 - |maximal planar subgraph|/m
             (greedy incremental planarity, mean over 3 random orders) -
             computable WITHOUT coordinates, hence also for the pandapower
             grids (PEGASE/RTE/... have no geographic coordinates)

Cross-measure calibration: chi_topo is also computed for the rail networks,
the water networks and the 3 smallest road windows, so the power grids can be
placed on the shared axis with an explicit calibration rather than by fiat.

Dose-response targets come from the ALREADY-PUBLISHED per-network tables
(R106 power / R111 water / R112 rail / R61 roads): cebh_gap and anchor error.
Prediction under the embedded-merging mechanism: deviation from the planar-end
anchor grows with measured non-planarity, and the CEBH gap shrinks.

Outputs: outputs/R205_S3_cross_domain_nonplanarity.csv
         outputs/R205_S3_summary.json
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache" / "R61_osmnx_city_windows"
TABLES = ROOT / "source_data" / "tables"
OUTDIR = ROOT / "data" / "R205_measured_geometry_dose_response"

POWER_GRIDS = [
    ("case1354pegase", "PEGASE"), ("case2869pegase", "PEGASE"), ("case9241pegase", "PEGASE"),
    ("case1888rte", "RTE"), ("case2848rte", "RTE"), ("case6470rte", "RTE"),
    ("case3120sp", "Polish"), ("GBnetwork", "GB"), ("case_illinois200", "Illinois"), ("case300", "IEEE300"),
]
RAIL_CITIES = ["Berlin, Germany", "Hamburg, Germany", "Vienna, Austria",
               "Madrid, Spain", "Milan, Italy", "Osaka, Japan", "Chicago, Illinois, USA"]
RAIL_FILTER = '["railway"~"rail|light_rail|subway|narrow_gauge"]'


# ---------------- shared helpers ----------------

def chi_geo_from_geoms(keys, geoms):
    from shapely.strtree import STRtree
    tree = STRtree(geoms)
    qi, ti = tree.query(geoms, predicate="crosses")
    cnt = 0
    for a, b in zip(qi.tolist(), ti.tolist()):
        if a >= b:
            continue
        ka, kb = keys[a], keys[b]
        if ka[0] in kb or ka[1] in kb:
            continue
        cnt += 1
    return cnt / max(len(keys), 1)


def chi_topo(edges, n_orders=3, seed=11, time_budget_s=2400):
    """1 - |greedy maximal planar subgraph| / m, mean over random orders."""
    m = len(edges)
    fracs = []
    t0 = time.time()
    for k in range(n_orders):
        rng = np.random.default_rng(seed + k)
        order = rng.permutation(m)
        H = nx.Graph()
        kept = 0
        for ei in order:
            u, v = int(edges[ei][0]), int(edges[ei][1])
            H.add_edge(u, v)
            ok, _ = nx.check_planarity(H, counterexample=False)
            if ok:
                kept += 1
            else:
                H.remove_edge(u, v)
                if H.degree(u) == 0:
                    H.remove_node(u)
                if H.degree(v) == 0:
                    H.remove_node(v)
        fracs.append(1.0 - kept / m)
        if time.time() - t0 > time_budget_s:
            break
    return float(np.mean(fracs)), float(np.std(fracs)), len(fracs)


def euler_slack(n, m):
    """m / (3n - 6): >1 is impossible for a simple planar graph."""
    return m / max(3 * n - 6, 1)


# ---------------- domain loaders ----------------

def power_task(args):
    name, family = args
    try:
        import pandapower.networks as ppn
        net = getattr(ppn, name)()
        g = nx.Graph()
        g.add_nodes_from(net.bus.index)
        for _, r in net.line.iterrows():
            g.add_edge(int(r.from_bus), int(r.to_bus))
        for _, r in net.trafo.iterrows():
            g.add_edge(int(r.hv_bus), int(r.lv_bus))
        g.remove_nodes_from(list(nx.isolates(g)))
        g = nx.convert_node_labels_to_integers(g)
        edges = np.array(g.edges(), dtype=np.int64)
        n, m = g.number_of_nodes(), len(edges)
        ct, ct_sd, n_ord = chi_topo(edges)
        return {"domain": "power", "network": name, "family": family, "n_nodes": n,
                "n_edges": m, "chi_topo": ct, "chi_topo_sd": ct_sd, "chi_topo_orders": n_ord,
                "euler_slack": euler_slack(n, m), "chi_geo": np.nan,
                "geometry_kind": "none", "status": "ok"}
    except Exception:
        return {"domain": "power", "network": name, "status": "fail",
                "error": traceback.format_exc(limit=3)}


def rail_task(place):
    try:
        import osmnx as ox
        from shapely.geometry import LineString
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
        # LCC restriction to match the R112 percolation graphs' spirit is NOT
        # applied; chi is a whole-network geometric property.
        cg = chi_geo_from_geoms(keys, geoms)
        idx = {nid: i for i, nid in enumerate({x for k in keys for x in k})}
        edges = np.array([(idx[a], idx[b]) for a, b in keys], dtype=np.int64)
        ct, ct_sd, n_ord = chi_topo(edges)
        city = place.split(",")[0]
        return {"domain": "rail", "network": city, "family": "osm_rail",
                "n_nodes": len(idx), "n_edges": len(keys), "chi_geo": cg,
                "chi_topo": ct, "chi_topo_sd": ct_sd, "chi_topo_orders": n_ord,
                "euler_slack": euler_slack(len(idx), len(keys)),
                "geometry_kind": "osm_polyline", "status": "ok"}
    except Exception:
        return {"domain": "rail", "network": place, "status": "fail",
                "error": traceback.format_exc(limit=3)}


def water_networks():
    import wntr
    base = os.path.dirname(wntr.__file__)
    hits = {}
    for root, _dirs, files in os.walk(base):
        for f in files:
            if f.lower().endswith(".inp"):
                hits[f[:-4]] = os.path.join(root, f)
    wanted = pd.read_csv(TABLES / "R111_water_percolation.csv")["network"].tolist()
    return [(w, hits[w]) for w in wanted if w in hits]


def water_task(args):
    name, inp = args
    try:
        import wntr
        from shapely.geometry import LineString
        wn = wntr.network.WaterNetworkModel(inp)
        coords = {n: wn.get_node(n).coordinates for n in wn.node_name_list}
        keys, geoms = [], []
        seen = set()
        for lname in wn.link_name_list:
            link = wn.get_link(lname)
            a, b = link.start_node_name, link.end_node_name
            if a == b:
                continue
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            keys.append(key)
            geoms.append(LineString([coords[a], coords[b]]))
        cg = chi_geo_from_geoms(keys, geoms)
        idx = {nid: i for i, nid in enumerate({x for k in keys for x in k})}
        edges = np.array([(idx[a], idx[b]) for a, b in keys], dtype=np.int64)
        ct, ct_sd, n_ord = chi_topo(edges)
        return {"domain": "water", "network": name, "family": "epanet_wntr",
                "n_nodes": len(idx), "n_edges": len(keys), "chi_geo": cg,
                "chi_topo": ct, "chi_topo_sd": ct_sd, "chi_topo_orders": n_ord,
                "euler_slack": euler_slack(len(idx), len(keys)),
                "geometry_kind": "inp_chord_possibly_schematic", "status": "ok"}
    except Exception:
        return {"domain": "water", "network": name, "status": "fail",
                "error": traceback.format_exc(limit=3)}


def road_task(pkl_name):
    try:
        p = CACHE / pkl_name
        with open(p, "rb") as f:
            G = pickle.load(f)
        seen = set()
        for u, v in G.edges():
            if u != v:
                seen.add((u, v) if u <= v else (v, u))
        keys = list(seen)
        idx = {nid: i for i, nid in enumerate({x for k in keys for x in k})}
        edges = np.array([(idx[a], idx[b]) for a, b in keys], dtype=np.int64)
        ct, ct_sd, n_ord = chi_topo(edges, n_orders=2)
        return {"domain": "road", "network": pkl_name.split("__")[0], "family": "osmnx_road",
                "n_nodes": len(idx), "n_edges": len(keys),
                "chi_topo": ct, "chi_topo_sd": ct_sd, "chi_topo_orders": n_ord,
                "euler_slack": euler_slack(len(idx), len(keys)),
                "chi_geo": np.nan, "geometry_kind": "see_S1", "status": "ok"}
    except Exception:
        return {"domain": "road", "network": pkl_name, "status": "fail",
                "error": traceback.format_exc(limit=3)}


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    # 3 smallest road windows by file size for chi_topo calibration.
    road_pkls = sorted(CACHE.glob("*.pkl"), key=lambda p: p.stat().st_size)[:3]
    tasks = ([("power", g) for g in POWER_GRIDS]
             + [("rail", c) for c in RAIL_CITIES]
             + [("water", w) for w in water_networks()]
             + [("road", p.name) for p in road_pkls])
    print(f"[S3] {len(tasks)} networks", flush=True)

    def dispatch(t):
        kind, arg = t
        return {"power": power_task, "rail": rail_task,
                "water": water_task, "road": road_task}[kind](arg)

    rows = []
    with Pool(10, maxtasksperchild=1) as pool:
        for row in pool.imap_unordered(_worker, tasks):
            rows.append(row)
            print(f"[S3] {row.get('domain')}/{row.get('network')}: "
                  f"chi_topo={row.get('chi_topo', float('nan'))} "
                  f"chi_geo={row.get('chi_geo', float('nan'))} ({row['status']})", flush=True)
            pd.DataFrame(rows).to_csv(OUTDIR / "R205_S3_cross_domain_nonplanarity.csv", index=False)

    df = pd.DataFrame(rows)

    # Join published per-network outcome tables.
    p106 = pd.read_csv(TABLES / "R106_power_grid_percolation.csv")
    p106 = p106.rename(columns={"grid": "network"})[
        ["network", "mean_degree", "cebh_pc", "observed_pc", "cebh_gap", "anchor_pc", "anchor_abs_err", "cebh_abs_err"]]
    p111 = pd.read_csv(TABLES / "R111_water_percolation.csv")[
        ["network", "mean_degree", "cebh_pc", "observed_pc", "cebh_gap", "anchor_pc", "anchor_abs_err", "cebh_abs_err"]]
    p112 = pd.read_csv(TABLES / "R112_rail_percolation.csv").rename(columns={"city": "network"})
    keep112 = [c for c in ["network", "mean_degree", "cebh_pc", "observed_pc", "cebh_gap",
                           "derived_pc", "derived_abs_err", "cebh_abs_err"] if c in p112.columns]
    p112 = p112[keep112]
    out = pd.concat([
        df[df.domain == "power"].merge(p106, on="network", how="left"),
        df[df.domain == "water"].merge(p111, on="network", how="left"),
        df[df.domain == "rail"].merge(p112, on="network", how="left"),
        df[df.domain == "road"],
    ], ignore_index=True)
    out["anchor_rel_dev"] = np.where(
        out["observed_pc"].notna() & out["mean_degree"].notna(),
        np.abs(out["observed_pc"] * out["mean_degree"] / 2.0 - 1.0), np.nan)
    out.to_csv(OUTDIR / "R205_S3_cross_domain_nonplanarity.csv", index=False)

    from scipy.stats import spearmanr
    summary = {}
    ok = out[(out.status == "ok")]
    summary["chi_topo_by_domain_median"] = ok.groupby("domain")["chi_topo"].median().round(5).to_dict()
    summary["euler_slack_by_domain_median"] = ok.groupby("domain")["euler_slack"].median().round(4).to_dict()
    pw = ok[(ok.domain == "power") & ok.anchor_abs_err.notna()]
    if len(pw) >= 6:
        for y in ("anchor_abs_err", "cebh_abs_err", "cebh_gap", "anchor_rel_dev"):
            rs, ps = spearmanr(pw["chi_topo"], pw[y])
            summary[f"power_chi_topo__vs__{y}"] = {"n": int(len(pw)), "spearman": round(float(rs), 4),
                                                   "p": round(float(ps), 5)}
    pooled = ok[ok.anchor_rel_dev.notna() & ok.chi_topo.notna()]
    if len(pooled) >= 10:
        rs, ps = spearmanr(pooled["chi_topo"], pooled["anchor_rel_dev"])
        summary["pooled_chi_topo__vs__anchor_rel_dev"] = {
            "n": int(len(pooled)), "spearman": round(float(rs), 4), "p": round(float(ps), 6),
            "domains": sorted(pooled.domain.unique().tolist())}
    rl = ok[(ok.domain == "rail") & ok.chi_geo.notna() & ok.chi_topo.notna()]
    if len(rl) >= 4:
        rs, ps = spearmanr(rl["chi_geo"], rl["chi_topo"])
        summary["rail_chi_geo__vs__chi_topo"] = {"n": int(len(rl)), "spearman": round(float(rs), 4),
                                                 "p": round(float(ps), 5)}
    with open(OUTDIR / "R205_S3_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[S3] DONE", json.dumps(summary, indent=2), flush=True)


def _worker(t):
    kind, arg = t
    fn = {"power": power_task, "rail": rail_task, "water": water_task, "road": road_task}[kind]
    return fn(arg)


if __name__ == "__main__":
    main()
