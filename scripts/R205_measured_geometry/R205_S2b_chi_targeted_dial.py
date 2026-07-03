"""R205-S2b: chi-targeted, locally-bounded crossing dial (the flyover-faithful dial).

S2's unconstrained swap dial showed that a random replacement chord crosses tens
of edges at once, so chi overshoots the empirically observed range
(real 71-city chi_chord spans ~0.009-0.19) after a handful of swaps. Real
grade separations cross one or two carriageways, not fifty.

This dial therefore accepts a degree-preserving double-edge swap only when EACH
replacement chord produces between 1 and 3 new proper crossings with non-adjacent
existing chords. The chain is driven until the measured chord-space crossing
density chi reaches preset stage targets inside and slightly beyond the real-city
range; at each stage chi is recounted exactly (STRtree) and pc re-estimated with
the standard rank-order estimator.

Output: outputs/R205_S2b_chi_targeted_dial.csv
        -> dpc/dchi in the real regime; overlay curve for the S1 cross-city
           scatter; converts "crossings shortcut the plane" into a quantitative,
           degree- and edge-count-controlled prediction.
"""

from __future__ import annotations

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "R205_measured_geometry"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "scripts"))

from R205_S2_matched_dial_and_planar_families import (  # noqa: E402
    ChordGrid, _delaunay_edges, _proper_cross, count_chi_chord, pc_estimate,
)
from build_R105_finite_size_scaling import load_road_simple  # noqa: E402

CACHE = ROOT / "cache" / "R61_osmnx_city_windows"
OUTDIR = ROOT / "data" / "R205_measured_geometry_dose_response"

DIAL_CITIES = [
    "bengaluru_india__drive_10000m.pkl",
    "osaka_japan__drive_10000m.pkl",
    "tehran_iran__drive_10000m.pkl",
]
CHI_INCREMENTS = [0.01, 0.02, 0.05, 0.10, 0.20]  # added ON TOP of each graph's baseline chi
MAX_CROSS_PER_CHORD = 3
MAX_ATTEMPTS_PER_STAGE = 4_000_000


def chord_cross_count(grid: ChordGrid, u, v, skip=(), cap=64):
    """Number of non-adjacent existing chords properly crossed by chord u-v."""
    pu, pv = grid.pos[u], grid.pos[v]
    seen = set(skip)
    cnt = 0
    for c in grid._cover(u, v):
        for eid in grid.cells.get(c, ()):
            if eid in seen:
                continue
            seen.add(eid)
            a, b = grid.edges[eid]
            if a in (u, v) or b in (u, v):
                continue
            if _proper_cross(pu, pv, grid.pos[a], grid.pos[b]):
                cnt += 1
                if cnt >= cap:
                    return cnt
    return cnt


def run_chain(task):
    label, n, edges, pos, seed = task
    rng = np.random.default_rng(seed)
    edges = edges.copy()
    grid = ChordGrid(pos, edges)
    edge_set = {tuple(sorted(e)) for e in edges.tolist()}
    m = len(edges)
    chi_est = count_chi_chord(pos, edges)
    chi0 = chi_est
    rows = [{
        "graph": label, "stage_target_chi": chi0, "chi_chord": chi_est,
        "chi_added": 0.0, "accepted_swaps": 0, "swaps_per_edge": 0.0,
        "pc": pc_estimate(n, edges, seed=seed + 7),
        "mean_degree": 2.0 * m / n, "seconds": 0.0,
    }]
    print(f"[S2b] {label} baseline chi={chi_est:.5f} pc={rows[0]['pc']:.3f}", flush=True)
    accepted = 0
    t0 = time.time()
    def neighbour_edge(i):
        """Pick a swap partner from the spatial neighbourhood of edge i
        (locality-faithful: real grade separations connect nearby corridors).
        Every 8th call falls back to a uniform partner for chain mobility."""
        u, v = grid.edges[i]
        cand = set()
        for (cx, cy) in grid._cover(u, v):
            for dx in (-2, -1, 0, 1, 2):
                for dy in (-2, -1, 0, 1, 2):
                    cand.update(grid.cells.get((cx + dx, cy + dy), ()))
        cand.discard(i)
        if not cand:
            return int(rng.integers(0, m))
        cand = list(cand)
        return int(cand[rng.integers(0, len(cand))])

    for inc in CHI_INCREMENTS:
        stage = chi0 + inc
        attempts = 0
        while chi_est < stage and attempts < MAX_ATTEMPTS_PER_STAGE:
            attempts += 1
            i = int(rng.integers(0, m))
            j = int(rng.integers(0, m)) if attempts % 8 == 0 else neighbour_edge(i)
            if i == j:
                continue
            a, b = int(edges[i, 0]), int(edges[i, 1])
            c, d = int(edges[j, 0]), int(edges[j, 1])
            if len({a, b, c, d}) < 4:
                continue
            if rng.random() < 0.5:
                new1, new2 = (a, c), (b, d)
            else:
                new1, new2 = (a, d), (b, c)
            k1, k2 = tuple(sorted(new1)), tuple(sorted(new2))
            if k1 == k2 or k1 in edge_set or k2 in edge_set:
                continue
            c1 = chord_cross_count(grid, *new1, skip=(i, j), cap=MAX_CROSS_PER_CHORD + 1)
            if not (1 <= c1 <= MAX_CROSS_PER_CHORD):
                continue
            c2 = chord_cross_count(grid, *new2, skip=(i, j), cap=MAX_CROSS_PER_CHORD + 1)
            if not (1 <= c2 <= MAX_CROSS_PER_CHORD):
                continue
            old1 = chord_cross_count(grid, a, b, skip=(i, j))
            old2 = chord_cross_count(grid, c, d, skip=(i, j))
            edge_set.discard(tuple(sorted((a, b))))
            edge_set.discard(tuple(sorted((c, d))))
            edge_set.add(k1)
            edge_set.add(k2)
            grid.remove(i)
            grid.remove(j)
            edges[i] = new1
            edges[j] = new2
            grid.add(i, *new1)
            grid.add(j, *new2)
            accepted += 1
            chi_est += (c1 + c2 - old1 - old2) / m
        chi_true = count_chi_chord(pos, edges)
        chi_est = chi_true
        pc = pc_estimate(n, edges, seed=seed + 17)
        rows.append({
            "graph": label, "stage_target_chi": stage, "chi_chord": chi_true,
            "chi_added": chi_true - chi0, "accepted_swaps": accepted,
            "swaps_per_edge": accepted / m,
            "pc": pc, "mean_degree": 2.0 * m / n,
            "seconds": round(time.time() - t0, 1),
        })
        print(f"[S2b] {label} target={stage:.5f} chi={chi_true:.5f} "
              f"swaps/edge={accepted / m:.4f} pc={pc:.3f}", flush=True)
        if attempts >= MAX_ATTEMPTS_PER_STAGE:
            print(f"[S2b] {label} attempts exhausted at stage {stage}", flush=True)
            break
    return rows


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tasks = []
    rng = np.random.default_rng(4242)
    pts = rng.random((20000, 2))
    tasks.append(("delaunay20k", len(pts), _delaunay_edges(pts), pts, 90210))
    for i, fn in enumerate(DIAL_CITIES):
        n, edges, pos = load_road_simple(CACHE / fn)
        tasks.append((fn.split("__")[0], n, np.asarray(edges, dtype=np.int64),
                      np.asarray(pos, float), 90300 + i))
    all_rows = []
    with Pool(len(tasks), maxtasksperchild=1) as pool:
        for rows in pool.imap_unordered(run_chain, tasks):
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_csv(OUTDIR / "R205_S2b_chi_targeted_dial.csv", index=False)
    print("[S2b] DONE", flush=True)


if __name__ == "__main__":
    main()
