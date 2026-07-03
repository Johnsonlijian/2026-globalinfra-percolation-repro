"""R208-A8: middle-regime dial arm (methods reviewer P1-6).

The observational chi->gap association is carried by LONG-edge crossings
(longer member ~11x median edge length) of low multiplicity. Neither existing
arm matches that regime (local arm: short neighbour chords; unconstrained arm:
extreme chords crossing tens-hundreds). This arm injects degree- and
edge-count-preserving swaps whose replacement chords are 5-15x the median edge
length AND cross 1-3 non-adjacent chords - the motorway-segment regime.
Three cities, chains to chi increments +0.02/+0.05/+0.10, fine-grid pc + CI.
Output: outputs/R208_A8_midrange_dial.csv
"""

from __future__ import annotations

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "R206_R208_ress_reframe"))
sys.path.insert(0, str(ROOT / "scripts" / "R205_measured_geometry"))
sys.path.insert(0, str(ROOT / "scripts"))

from R205_S2_matched_dial_and_planar_families import ChordGrid, count_chi_chord  # noqa: E402
from R205_S2b_chi_targeted_dial import chord_cross_count  # noqa: E402
from R206_A1_dial_fine_grid_ci import pc_fine  # noqa: E402
from build_R105_finite_size_scaling import load_road_simple  # noqa: E402

CACHE = ROOT / "cache" / "R61_osmnx_city_windows"
OUTDIR = ROOT / "data" / "R206_R208_ress_reframe"

CITIES = ["bengaluru_india__drive_10000m.pkl", "osaka_japan__drive_10000m.pkl",
          "tehran_iran__drive_10000m.pkl"]
CHI_INCREMENTS = [0.02, 0.05, 0.10]
LEN_LO, LEN_HI = 5.0, 15.0
MAX_CROSS = 3
MAX_ATTEMPTS_PER_STAGE = 6_000_000


def run_chain(task):
    label, n, edges, pos, seed = task
    rng = np.random.default_rng(seed)
    edges = edges.copy()
    grid = ChordGrid(pos, edges)
    edge_set = {tuple(sorted(e)) for e in edges.tolist()}
    m = len(edges)
    med_len = float(np.median(np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]], axis=1)))
    chi_est = count_chi_chord(pos, edges)
    chi0 = chi_est
    pc_hat, lo, hi = pc_fine(n, edges, seed + 7)
    rows = [{"graph": label, "chi_chord": chi_est, "chi_added": 0.0, "accepted_swaps": 0,
             "pc": pc_hat, "pc_ci_lo": lo, "pc_ci_hi": hi}]
    print(f"[A8] {label} base chi={chi_est:.5f} pc={pc_hat:.3f} [{lo:.3f},{hi:.3f}]", flush=True)

    def chord_len_ok(u, v):
        L = float(np.linalg.norm(pos[u] - pos[v]))
        return LEN_LO * med_len <= L <= LEN_HI * med_len

    accepted = 0
    t0 = time.time()
    for inc in CHI_INCREMENTS:
        stage = chi0 + inc
        attempts = 0
        while chi_est < stage and attempts < MAX_ATTEMPTS_PER_STAGE:
            attempts += 1
            i, j = int(rng.integers(0, m)), int(rng.integers(0, m))
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
            if not (chord_len_ok(*new1) and chord_len_ok(*new2)):
                continue
            c1 = chord_cross_count(grid, *new1, skip=(i, j), cap=MAX_CROSS + 1)
            if not (1 <= c1 <= MAX_CROSS):
                continue
            c2 = chord_cross_count(grid, *new2, skip=(i, j), cap=MAX_CROSS + 1)
            if not (1 <= c2 <= MAX_CROSS):
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
        rows.append({"graph": label, "chi_chord": chi_true, "chi_added": chi_true - chi0,
                     "accepted_swaps": accepted, "pc": pc_hat, "pc_ci_lo": lo, "pc_ci_hi": hi})
        print(f"[A8] {label} chi={chi_true:.5f} (+{chi_true - chi0:.3f}) "
              f"pc={pc_hat:.3f} [{lo:.3f},{hi:.3f}] swaps={accepted} ({time.time() - t0:.0f}s)", flush=True)
    return rows


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tasks = []
    for i, fn in enumerate(CITIES):
        n, edges, pos = load_road_simple(CACHE / fn)
        tasks.append((fn.split("__")[0], n, np.asarray(edges, dtype=np.int64),
                      np.asarray(pos, float), 8800 + i))
    rows = []
    with Pool(len(tasks), maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(run_chain, tasks):
            rows.extend(r)
            pd.DataFrame(rows).to_csv(OUTDIR / "R208_A8_midrange_dial.csv", index=False)
    print("[A8] DONE", flush=True)


if __name__ == "__main__":
    main()
