"""R205-S4: decision-referenced consequence layer (recovered and extended).

R109/R113 already computed screening margins and flips but the arc was stripped
from the manuscript during the post-CSF rebuild. This script re-derives the
decision consequences from the ALREADY-PUBLISHED tables, in a form usable either
as a physics-paper "why it matters" paragraph or as the results core of a
reliability-engineering reframe:

 1. Screening flips (R113): a network passes a robustness screen under the CEBH
    threshold but fails under the observed threshold (false-safe), per domain.
 2. Ranking fidelity (R61): if an analyst ranks the 71 cities by predicted
    threshold, how well does the CEBH ranking reproduce the observed ranking
    versus the zero-parameter geometric baseline 2/<k>?
 3. Error decomposition: MAE of CEBH vs anchor per domain (R106/R111/R112/R61).

Outputs: outputs/R205_S4_decision_experiment.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "source_data" / "tables"
OUTDIR = ROOT / "data" / "R205_measured_geometry_dose_response"

summary = {}

# 1) screening flips from R113
d = pd.read_csv(TABLES / "R113_decision_margins.csv")
summary["r113_flip_rate_by_domain"] = d.groupby("domain")["flip"].mean().round(4).to_dict()
summary["r113_n_by_domain"] = d.groupby("domain").size().to_dict()
d["margin_anchor_err"] = (d["margin_cebh"] - d["margin_true"]).abs()
d["margin_derived_err"] = (d["margin_derived"] - d["margin_true"]).abs()
summary["r113_margin_abs_err_cebh_mean"] = round(float(d["margin_anchor_err"].mean()), 4)
summary["r113_margin_abs_err_derived_mean"] = round(float(d["margin_derived_err"].mean()), 4)
d["flip_derived"] = np.sign(d["margin_derived"]) != np.sign(d["margin_true"])
summary["r113_flip_rate_derived_by_domain"] = d.groupby("domain")["flip_derived"].mean().round(4).to_dict()

# 2) ranking fidelity across the 71 windows
r61 = pd.read_csv(TABLES / "R61_central_window_screen_results.csv")
r61["anchor_pc"] = 2.0 / r61["mean_degree"]
for pred in ("pc_cebh", "anchor_pc"):
    rs, ps = spearmanr(r61[pred], r61["pc_rank"])
    kt, pk = kendalltau(r61[pred], r61["pc_rank"])
    summary[f"r61_rank_fidelity_{pred}"] = {
        "spearman": round(float(rs), 4), "spearman_p": round(float(ps), 6),
        "kendall": round(float(kt), 4), "kendall_p": round(float(pk), 6)}
summary["r61_mae_cebh"] = round(float((r61["pc_cebh"] - r61["pc_rank"]).abs().mean()), 4)
summary["r61_mae_anchor"] = round(float((r61["anchor_pc"] - r61["pc_rank"]).abs().mean()), 4)

# top-decile most-fragile identification test (rank by predicted pc, higher pc = more fragile)
k = 7  # top decile of 71
true_top = set(r61.nlargest(k, "pc_rank")["city"])
for pred in ("pc_cebh", "anchor_pc"):
    hit = len(set(r61.nlargest(k, pred)["city"]) & true_top) / k
    summary[f"r61_top_decile_fragile_hit_rate_{pred}"] = round(hit, 4)

# 3) per-domain MAE decomposition
rows = {}
p106 = pd.read_csv(TABLES / "R106_power_grid_percolation.csv")
rows["power"] = {"n": len(p106),
                 "mae_cebh": float(p106["cebh_abs_err"].mean()),
                 "mae_anchor": float(p106["anchor_abs_err"].mean())}
p111 = pd.read_csv(TABLES / "R111_water_percolation.csv")
rows["water"] = {"n": len(p111),
                 "mae_cebh": float(p111["cebh_abs_err"].mean()),
                 "mae_anchor": float(p111["anchor_abs_err"].mean())}
p112 = pd.read_csv(TABLES / "R112_rail_percolation.csv")
mae_anchor_rail = float(p112["derived_abs_err"].mean()) if "derived_abs_err" in p112 else np.nan
rows["rail"] = {"n": len(p112),
                "mae_cebh": float(p112["cebh_abs_err"].mean()) if "cebh_abs_err" in p112
                else float((p112["cebh_pc"] - p112["observed_pc"]).abs().mean()),
                "mae_anchor": mae_anchor_rail}
rows["road71"] = {"n": len(r61), "mae_cebh": summary["r61_mae_cebh"], "mae_anchor": summary["r61_mae_anchor"]}
summary["per_domain_mae"] = {k2: {kk: round(vv, 4) for kk, vv in v.items()} for k2, v in rows.items()}

OUTDIR.mkdir(parents=True, exist_ok=True)
with open(OUTDIR / "R205_S4_decision_experiment.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
