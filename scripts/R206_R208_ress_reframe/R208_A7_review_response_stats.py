"""R208-A7: quick statistics answering reviewer P1s.
1. Bias-corrected CEBH competitor (road-fitted +0.217 offset) on power/water/rail.
2. Fine-grid (0.005, 32-order) road MAE for 2/<k> on the 21-city layer (R78).
3. Bridge/tunnel tagging-validation lower tail + association with chi.
4. Region-cluster robustness: chi vs road_gap partial with macro-region dummies.
5. Per-family power flip statement (PEGASE/RTE nesting).
Output: outputs/R208_A7_review_response_stats.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "source_data" / "tables"
R205O = ROOT / "data" / "R205_measured_geometry_dose_response"
OUTDIR = ROOT / "data" / "R206_R208_ress_reframe"

out = {}
r61 = pd.read_csv(TABLES / "R61_central_window_screen_results.csv")
OFFSET = float((r61.pc_rank - r61.pc_cebh).mean())
out["road_fitted_offset"] = round(OFFSET, 4)

for name, f, obs_col, cebh_col in [
        ("power", "R106_power_grid_percolation.csv", "observed_pc", "cebh_pc"),
        ("water", "R111_water_percolation.csv", "observed_pc", "cebh_pc"),
        ("rail", "R112_rail_percolation.csv", "observed_pc", "cebh_pc")]:
    d = pd.read_csv(TABLES / f)
    mae = float((d[cebh_col] + OFFSET - d[obs_col]).abs().mean())
    out[f"bias_corrected_cebh_mae_{name}"] = round(mae, 4)
# water primary-4 subset
w = pd.read_csv(TABLES / "R111_water_percolation.csv")
w4 = w[w.network.isin(["Net6", "ky4", "ky10", "Net3"])]
out["bias_corrected_cebh_mae_water_primary4"] = round(float((w4.cebh_pc + OFFSET - w4.observed_pc).abs().mean()), 4)

# 2) fine-grid road MAE for 2/<k> (21-city fine observed thresholds)
r78 = pd.read_csv(TABLES / "R78_observed_road_fine_estimator_21city.csv")
print("[A7] R78 columns:", list(r78.columns)[:14], flush=True)
pc_col = next(c for c in r78.columns if "pc" in c.lower() and "fine" in c.lower()) if \
    any("fine" in c.lower() and "pc" in c.lower() for c in r78.columns) else None
if pc_col is None:
    cand = [c for c in r78.columns if c.lower().startswith("pc") or "threshold" in c.lower()]
    pc_col = cand[0]
join = r78.merge(r61[["city", "mean_degree", "pc_cebh"]], on="city", how="inner")
join["dim_pc"] = 2 / join["mean_degree"]
out["fine_grid_21city"] = {
    "pc_column_used": pc_col,
    "n": int(len(join)),
    "mae_dim_baseline": round(float((join["dim_pc"] - join[pc_col]).abs().mean()), 4),
    "mae_cebh": round(float((join["pc_cebh"] - join[pc_col]).abs().mean()), 4),
}

# 3) tagging lower tail
s1 = pd.read_csv(R205O / "R205_S1_crossing_density_71cities.csv")
s1 = s1[s1.status == "ok"]
tag = s1["cross_pairs_with_tagged_edge_share"].dropna()
out["tagging_validation"] = {
    "median": round(float(tag.median()), 3),
    "q25": round(float(tag.quantile(0.25)), 3),
    "min": round(float(tag.min()), 3),
    "n_below_50pct": int((tag < 0.5).sum()),
    "spearman_tagshare_vs_chi": round(float(spearmanr(
        s1.dropna(subset=["cross_pairs_with_tagged_edge_share"])["chi_poly"],
        tag)[0]), 3),
}

# 4) region-cluster robustness (rank partial with region dummies)
sub = s1.dropna(subset=["chi_poly", "road_gap", "macro_region"])
rx, ry = rankdata(sub.chi_poly), rankdata(sub.road_gap)
D = pd.get_dummies(sub.macro_region, drop_first=True).astype(float).to_numpy()
A = np.column_stack([np.ones(len(rx)), D])
ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
r = float(np.corrcoef(ex, ey)[0, 1])
rng = np.random.default_rng(11)
perm = [np.corrcoef(rng.permutation(ex), ey)[0, 1] for _ in range(5000)]
out["chi_vs_gap_region_fixed_effects"] = {
    "partial_spearman": round(r, 4),
    "perm_p": round(float((np.abs(perm) >= abs(r)).mean()), 5),
    "n": int(len(sub)),
}

# 5) power per-family flips at 0.40
p106 = pd.read_csv(TABLES / "R106_power_grid_percolation.csv")
fam_flips = {}
for fam, d in p106.groupby("family"):
    flips = ((1 - d.cebh_pc - 0.4) * (1 - d.observed_pc - 0.4) < 0)
    fam_flips[fam] = f"{int(flips.sum())}/{len(d)}"
out["power_cebh_flips_by_family_at_0.40"] = fam_flips
out["power_n_independent_systems"] = 6

OUTDIR.mkdir(parents=True, exist_ok=True)
with open(OUTDIR / "R208_A7_review_response_stats.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
