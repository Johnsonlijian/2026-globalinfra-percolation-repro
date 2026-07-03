"""R205-S1c: confound controls for the chi dose-response.

Obvious reviewer attack: measured crossing density is a wealth/size/degree proxy,
so the chi -> gap association is spurious. This script computes rank-based
partial correlations of chi_poly with (a) road_gap and (b) the post-geometry-null
residual, controlling for mean degree, window size (log edges) and GDP per
capita (WDI, log). Rank-transform all variables (Spearman-style), residualize on
controls by OLS, correlate residuals; p by permutation (5000).
Output: outputs/R205_S1c_confound_controls.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "source_data" / "tables"
OUTDIR = ROOT / "data" / "R205_measured_geometry_dose_response"

s1 = pd.read_csv(OUTDIR / "R205_S1_crossing_density_71cities.csv")
cov = pd.read_csv(TABLES / "R65_R65_integrated_public_covariates.csv")
gdp_cols = [c for c in cov.columns if "gdp" in c.lower()]
pop_cols = [c for c in cov.columns if "population" in c.lower() and "density" not in c.lower()]
print("[S1c] gdp cols:", gdp_cols, "pop cols:", pop_cols[:3])
keep = ["city"] + gdp_cols[:1] + pop_cols[:1]
df = s1.merge(cov[keep], on="city", how="left")
gdp = gdp_cols[0] if gdp_cols else None

df["log_edges"] = np.log(df["n_edges_simple"])
controls_sets = {
    "none": [],
    "k": ["mean_degree"],
    "k_size": ["mean_degree", "log_edges"],
}
if gdp:
    df["log_gdp"] = np.log(df[gdp].clip(lower=1))
    controls_sets["k_size_gdp"] = ["mean_degree", "log_edges", "log_gdp"]

targets = ["road_gap"] + [c for c in df.columns
                          if c.startswith(("spatial_resid_f", "geom_resid_f", "noncross_marginal_f"))]
rng = np.random.default_rng(7)
out = {}
for tgt in targets:
    for cname, ctrls in controls_sets.items():
        sub = df.dropna(subset=["chi_poly", tgt] + ctrls)
        if len(sub) < 20:
            continue
        R = {v: rankdata(sub[v]) for v in ["chi_poly", tgt] + ctrls}
        def resid(y, X):
            if not X:
                return y - y.mean()
            A = np.column_stack([np.ones(len(y))] + X)
            beta, *_ = np.linalg.lstsq(A, y, rcond=None)
            return y - A @ beta
        Xc = [R[c] for c in ctrls]
        rx = resid(R["chi_poly"], Xc)
        ry = resid(R[tgt], Xc)
        r = float(np.corrcoef(rx, ry)[0, 1])
        null = []
        for _ in range(5000):
            null.append(np.corrcoef(rng.permutation(rx), ry)[0, 1])
        p = float((np.abs(null) >= abs(r)).mean())
        out[f"chi_poly__{tgt}__ctrl_{cname}"] = {"n": int(len(sub)), "partial_spearman": round(r, 4),
                                                 "perm_p": round(p, 5)}
OUTDIR.mkdir(parents=True, exist_ok=True)
with open(OUTDIR / "R205_S1c_confound_controls.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
