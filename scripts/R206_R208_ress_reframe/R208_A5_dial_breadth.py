"""R208-A5: broaden the local-crossing dial to 8 additional cities stratified by
baseline chi (reviewer P1-4). One chain per city, increments +0.05/+0.10, fine-grid
pc with bootstrap CI at baseline and each stage.
Output: outputs/R208_A5_dial_breadth.csv
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "R206_R208_ress_reframe"))
sys.path.insert(0, str(ROOT / "scripts" / "R205_measured_geometry"))
sys.path.insert(0, str(ROOT / "scripts"))

import R206_A1_dial_fine_grid_ci as A1  # noqa: E402
from build_R105_finite_size_scaling import load_road_simple  # noqa: E402

# stratified by S1 chi_poly: low / mid / high exposure
CITIES = [
    "kampala_uganda__drive_10000m.pkl",
    "addis_ababa_ethiopia__drive_10000m.pkl",
    "algiers_algeria__drive_10000m.pkl",
    "auckland_new_zealand__drive_10000m.pkl",
    "barcelona_spain__drive_10000m.pkl",
    "chicago_united_states__drive_10000m.pkl",
    "kuala_lumpur_malaysia__drive_10000m.pkl",
    "taipei_taiwan__drive_10000m.pkl",
]

A1.CHI_INCREMENTS = [0.05, 0.10]
OUTDIR = ROOT / "data" / "R206_R208_ress_reframe"


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    tasks = []
    for i, fn in enumerate(CITIES):
        try:
            n, edges, pos = load_road_simple(A1.CACHE / fn)
        except Exception as e:
            print(f"[A5] skip {fn}: {e}", flush=True)
            continue
        tasks.append((fn.split("__")[0], n, np.asarray(edges, dtype=np.int64),
                      np.asarray(pos, float), 7000 + i))
    rows = []
    with Pool(len(tasks), maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(A1.run_chain, tasks):
            rows.extend(r)
            pd.DataFrame(rows).to_csv(OUTDIR / "R208_A5_dial_breadth.csv", index=False)
    print("[A5] DONE", flush=True)


if __name__ == "__main__":
    main()
