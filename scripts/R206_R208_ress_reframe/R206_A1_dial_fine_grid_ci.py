"""R206-A1: publication-grade dial - fine local-grid pc with bootstrap CIs, 3 seeds.

Upgrades the S2b chi-targeted dial from single-chain 0.025-grid pc read-outs to:
  - 3 independent swap-chain seeds per graph,
  - stage increments {+0.02, +0.05, +0.10, +0.20} over each graph's baseline chi,
  - at each stage a two-pass pc estimate: coarse 0.025 rank-order pass locates the
    transition, then a local +-0.06 window at 0.005 spacing with 24 edge-order
    permutations; a 200-draw bootstrap over permutation curves gives a 95% CI.

Output: outputs/R206_A1_dial_fine_grid.csv
"""

from __future__ import annotations

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
R205S = ROOT / "scripts" / "R205_measured_geometry"
sys.path.insert(0, str(R205S))
sys.path.insert(0, str(ROOT / "scripts"))

from R205_S2_matched_dial_and_planar_families import (  # noqa: E402
    ChordGrid, _delaunay_edges, count_chi_chord, pc_estimate,
)
from R205_S2b_chi_targeted_dial import chord_cross_count  # noqa: E402
from build_R105_finite_size_scaling import _find, load_road_simple  # noqa: E402

CACHE = ROOT / "cache" / "R61_osmnx_city_windows"
OUTDIR = ROOT / "data" / "R206_R208_ress_reframe"

DIAL_CITIES = ["bengaluru_india__drive_10000m.pkl", "osaka_japan__drive_10000m.pkl",
               "tehran_iran__drive_10000m.pkl"]
CHI_INCREMENTS = [0.02, 0.05, 0.10, 0.20]
MAX_CROSS_PER_CHORD = 3
MAX_ATTEMPTS_PER_STAGE = 4_000_000
SEEDS = [11, 22, 33]


def _rank_curves(n, edges, grid, n_perm, rng):
    """Per-permutation LCC(p) curves at the given occupation grid."""
    m = len(edges)
    tk = np.clip(np.rint(grid * m).astype(int), 0, m)
    curves = np.zeros((n_perm, len(grid)))
    for r in range(n_perm):
        parent = np.arange(n)
        size = np.ones(n, int)
        largest = 1
        t = 0
        order = rng.permutation(m)
        for step, ei in enumerate(order, 1):
            u, v = int(edges[ei, 0]), int(edges[ei, 1])
            ru, rv = _find(parent, u), _find(parent, v)
            if ru != rv:
                if size[ru] < size[rv]:
                    ru, rv = rv, ru
                parent[rv] = ru
                size[ru] += size[rv]
                largest = max(largest, int(size[ru]))
            while t < len(grid) and step >= tk[t]:
                curves[r, t] = largest / n
                t += 1
        while t < len(grid):
            curves[r, t] = largest / n
            t += 1
    return curves


def pc_fine(n, edges, seed, n_perm=24, boot=200):
    rng = np.random.default_rng(seed)
    pc0 = pc_estimate(n, edges, n_perm=8, seed=seed)
    lo, hi = max(0.03, pc0 - 0.06), min(0.97, pc0 + 0.06)
    grid = np.round(np.arange(lo, hi + 1e-9, 0.005), 4)
    curves = _rank_curves(n, edges, grid, n_perm, rng)
    mean = curves.mean(0)
    pc_hat = float(grid[int(np.argmax(np.gradient(mean, grid)))])
    draws = []
    for _ in range(boot):
        idx = rng.integers(0, n_perm, n_perm)
        mb = curves[idx].mean(0)
        draws.append(float(grid[int(np.argmax(np.gradient(mb, grid)))]))
    lo_ci, hi_ci = np.percentile(draws, [2.5, 97.5])
    return pc_hat, float(lo_ci), float(hi_ci)


def run_chain(task):
    label, n, edges, pos, seed = task
    rng = np.random.default_rng(seed)
    edges = edges.copy()
    grid = ChordGrid(pos, edges)
    edge_set = {tuple(sorted(e)) for e in edges.tolist()}
    m = len(edges)
    chi_est = count_chi_chord(pos, edges)
    chi0 = chi_est
    pc_hat, lo, hi = pc_fine(n, edges, seed + 7)
    rows = [{"graph": label, "seed": seed, "chi_chord": chi_est, "chi_added": 0.0,
             "accepted_swaps": 0, "pc": pc_hat, "pc_ci_lo": lo, "pc_ci_hi": hi,
             "mean_degree": 2.0 * m / n}]
    print(f"[A1] {label} s{seed} base chi={chi_est:.5f} pc={pc_hat:.3f} [{lo:.3f},{hi:.3f}]", flush=True)

    def neighbour_edge(i):
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

    accepted = 0
    t0 = time.time()
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
        pc_hat, lo, hi = pc_fine(n, edges, seed + 17)
        rows.append({"graph": label, "seed": seed, "chi_chord": chi_true,
                     "chi_added": chi_true - chi0, "accepted_swaps": accepted,
                     "pc": pc_hat, "pc_ci_lo": lo, "pc_ci_hi": hi,
                     "mean_degree": 2.0 * m / n})
        print(f"[A1] {label} s{seed} chi={chi_true:.5f} pc={pc_hat:.3f} "
              f"[{lo:.3f},{hi:.3f}] ({time.time() - t0:.0f}s)", flush=True)
    return rows


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tasks = []
    for seed in SEEDS:
        rng = np.random.default_rng(4242)
        pts = rng.random((20000, 2))
        tasks.append(("delaunay20k", len(pts), _delaunay_edges(pts), pts, seed))
    for i, fn in enumerate(DIAL_CITIES):
        n, edges, pos = load_road_simple(CACHE / fn)
        for seed in SEEDS:
            tasks.append((fn.split("__")[0], n, np.asarray(edges, dtype=np.int64),
                          np.asarray(pos, float), seed + 100 * (i + 1)))
    all_rows = []
    with Pool(min(12, len(tasks)), maxtasksperchild=1) as pool:
        for rows in pool.imap_unordered(run_chain, tasks):
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_csv(OUTDIR / "R206_A1_dial_fine_grid.csv", index=False)
    print("[A1] DONE", flush=True)


if __name__ == "__main__":
    main()
