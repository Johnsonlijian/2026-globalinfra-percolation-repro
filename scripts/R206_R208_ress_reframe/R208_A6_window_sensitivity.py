"""R208-A6: window-radius sensitivity (reviewer P1-8). For 12 cities spanning chi
and size, induce the 5-km inner disc of the cached 10-km window (offline), take
the LCC, recompute pc (R61 settings: 6 orders, 0.025 grid), CEBH and the gap, and
compare with the 10-km values. Output: outputs/R208_A6_window_sensitivity.csv
"""

from __future__ import annotations

import pickle
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "R205_measured_geometry"))
from R205_S2_matched_dial_and_planar_families import pc_estimate  # noqa: E402

CACHE = ROOT / "cache" / "R61_osmnx_city_windows"
TABLES = ROOT / "source_data" / "tables"
OUTDIR = ROOT / "data" / "R206_R208_ress_reframe"

CITIES = ["kampala_uganda", "addis_ababa_ethiopia", "montevideo_uruguay",
          "algiers_algeria", "auckland_new_zealand", "amsterdam_netherlands",
          "barcelona_spain", "chicago_united_states", "boston_united_states",
          "kuala_lumpur_malaysia", "taipei_taiwan", "hong_kong_china"]
RADIUS_M = 5000.0


def one(args):
    stem, lat0, lon0 = args
    try:
        with open(CACHE / f"{stem}__drive_10000m.pkl", "rb") as f:
            G = pickle.load(f)
        m_per_deg_lat = 111_132.0
        m_per_deg_lon = 111_320.0 * np.cos(np.radians(lat0))
        keep = [n for n, d in G.nodes(data=True)
                if ((d["x"] - lon0) * m_per_deg_lon) ** 2 + ((d["y"] - lat0) * m_per_deg_lat) ** 2
                <= RADIUS_M ** 2]
        H = nx.Graph(G.subgraph(keep))
        H.remove_edges_from(nx.selfloop_edges(H))
        if H.number_of_nodes() == 0:
            raise RuntimeError("empty 5km disc")
        lcc = max(nx.connected_components(H), key=len)
        H = H.subgraph(lcc).copy()
        H = nx.convert_node_labels_to_integers(H)
        n, m = H.number_of_nodes(), H.number_of_edges()
        edges = np.array(H.edges(), dtype=np.int64)
        deg = np.array([d for _, d in H.degree()])
        k1, k2 = deg.mean(), (deg ** 2).mean()
        cebh = k1 / (k2 - k1)
        pc = pc_estimate(n, edges, n_perm=6, seed=17)
        return {"stem": stem, "n_nodes_5km": n, "n_edges_5km": m,
                "mean_degree_5km": 2 * m / n, "pc_5km": pc, "pc_cebh_5km": cebh,
                "gap_5km": pc - cebh, "dim_pc_5km": 2 / (2 * m / n), "status": "ok"}
    except Exception:
        return {"stem": stem, "status": "fail", "error": traceback.format_exc(limit=2)}


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    r61 = pd.read_csv(TABLES / "R61_central_window_screen_results.csv")
    r61["stem"] = r61["graph_cache_file"].map(
        lambda s: Path(str(s).replace("\\", "/")).name.replace("__drive_10000m.pkl", ""))
    sel = r61[r61.stem.isin(CITIES)]
    tasks = [(row.stem, row.latitude, row.longitude) for row in sel.itertuples()]
    rows = []
    with Pool(min(12, len(tasks))) as pool:
        for r in pool.imap_unordered(one, tasks):
            rows.append(r)
            print(f"[A6] {r['stem']}: {r.get('gap_5km', 'fail')}", flush=True)
    df = pd.DataFrame(rows).merge(
        sel[["stem", "city", "mean_degree", "pc_cebh", "pc_rank", "road_gap"]], on="stem")
    df["gap_10km"] = df["road_gap"]
    df["gap_diff"] = df["gap_5km"] - df["gap_10km"]
    df.to_csv(OUTDIR / "R208_A6_window_sensitivity.csv", index=False)
    ok = df[df.status == "ok"]
    print("[A6] n=%d  gap 10km mean=%.4f  5km mean=%.4f  mean|diff|=%.4f  max|diff|=%.4f  sign-preserved=%s"
          % (len(ok), ok.gap_10km.mean(), ok.gap_5km.mean(),
             ok.gap_diff.abs().mean(), ok.gap_diff.abs().max(), bool((ok.gap_5km > 0).all())), flush=True)


if __name__ == "__main__":
    main()
