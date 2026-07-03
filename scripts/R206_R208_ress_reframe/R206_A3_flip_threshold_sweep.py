"""R206-A3: decision-flip robustness - sweep the screening threshold.

The R113 decision table fixed the critical-fraction threshold at 0.40. To show
the 89-100% CEBH flip headline is not threshold-picking, sweep t over
[0.10, 0.60] and recompute, per domain and predictor:
  flip(t) = share of networks with sign(fc_pred - t) != sign(fc_obs - t)
Networks where fc_obs == t are counted as flips=False (measure-zero on the grid).

Output: outputs/R206_A3_flip_threshold_sweep.csv (+ .json summary)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "source_data" / "tables"
OUTDIR = ROOT / "data" / "R206_R208_ress_reframe"

d = pd.read_csv(TABLES / "R113_decision_margins.csv")
ts = np.round(np.arange(0.10, 0.601, 0.025), 3)
rows = []
for t in ts:
    for dom, sub in d.groupby("domain"):
        for pred, col in (("cebh", "fc_cebh"), ("derived", "fc_derived")):
            # conservative flip definition: strictly opposite signs only;
            # boundary ties (fc == t, an estimator-grid coincidence) count as
            # non-flips - consistent with the R113 ledger.
            flips = (sub[col] - t) * (sub["fc_obs"] - t) < 0
            rows.append({"threshold": t, "domain": dom, "predictor": pred,
                         "n": len(sub), "flip_rate": float(flips.mean())})
df = pd.DataFrame(rows)
OUTDIR.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTDIR / "R206_A3_flip_threshold_sweep.csv", index=False)

summary = {}
for (dom, pred), sub in df.groupby(["domain", "predictor"]):
    # restrict to the informative band where the screen can actually bind
    band = sub[(sub.threshold >= 0.15) & (sub.threshold <= 0.55)]
    summary[f"{dom}_{pred}"] = {
        "flip_min": float(band.flip_rate.min()),
        "flip_max": float(band.flip_rate.max()),
        "flip_at_0.40": float(sub[sub.threshold == 0.40].flip_rate.iloc[0]),
        "flip_mean_band": round(float(band.flip_rate.mean()), 4),
    }
with open(OUTDIR / "R206_A3_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
