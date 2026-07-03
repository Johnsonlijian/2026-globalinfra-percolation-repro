"""R205-S1b: repair the S1 join (R81 has one row per city x swap fraction, which
doubled the merged table) and recompute dose-response statistics on n=71 cities.
Also adds the post-strict-geometry-null residual as a dose-response target,
pivoted per swap fraction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "source_data" / "tables"
OUTDIR = ROOT / "data" / "R205_measured_geometry_dose_response"

df = pd.read_csv(OUTDIR / "R205_S1_crossing_density_71cities.csv")
stale = [c for c in df.columns if c.startswith(
    ("r81_", "geom_resid_f", "spatial_resid_f", "noncross_marginal_f"))]
df = df.drop(columns=stale).drop_duplicates(subset=["graph_cache_file"]).reset_index(drop=True)
assert len(df) == 71, f"expected 71 unique cities, got {len(df)}"

r81 = pd.read_csv(TABLES / "R81_full71_high_intensity_geometry_null_city_summary.csv")
# Explicit column semantics (an earlier substring match silently picked the
# residual-CHANGE column; keep all three quantities under unambiguous names):
#   road_minus_spatial                 -> post-SPATIAL-null residual (crossings allowed)
#   road_minus_geometry_mean           -> post-STRICT-GEOMETRY-null residual
#   geometry_vs_spatial_residual_change-> marginal effect of the non-crossing rule
R81_COLS = {
    "road_minus_spatial": "spatial_resid",
    "road_minus_geometry_mean": "geom_resid",
    "geometry_vs_spatial_residual_change": "noncross_marginal",
}
for src, short in R81_COLS.items():
    piv = r81.pivot_table(index="city", columns="swap_fraction", values=src, aggfunc="mean")
    piv.columns = [f"{short}_f{c}" for c in piv.columns]
    df = df.merge(piv.reset_index(), on="city", how="left")

df.to_csv(OUTDIR / "R205_S1_crossing_density_71cities.csv", index=False)

ok = df[df["status"] == "ok"].dropna(subset=["road_gap"])
summary = {"n_cities_ok": int(len(ok))}
rng = np.random.default_rng(20260702)
ycols = (["road_gap", "anchor_deviation"]
         + [c for c in df.columns
            if c.startswith(("spatial_resid_f", "geom_resid_f", "noncross_marginal_f"))])
for xcol in ("chi_poly", "chi_chord", "bridge_tunnel_tagged_edge_share"):
    for ycol in ycols:
        sub = ok.dropna(subset=[xcol, ycol])
        x, y = sub[xcol].to_numpy(), sub[ycol].to_numpy()
        if len(x) < 10:
            continue
        rs, ps = spearmanr(x, y)
        rp, pp = pearsonr(x, y)
        boots = []
        for _ in range(5000):
            idx = rng.integers(0, len(x), len(x))
            if len(set(x[idx])) > 2:
                boots.append(spearmanr(x[idx], y[idx])[0])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        summary[f"{xcol}__vs__{ycol}"] = {
            "n": int(len(x)),
            "spearman": round(float(rs), 4), "spearman_p": round(float(ps), 6),
            "spearman_ci95": [round(float(lo), 4), round(float(hi), 4)],
            "pearson": round(float(rp), 4), "pearson_p": round(float(pp), 6),
        }
for col in ("chi_poly", "chi_chord"):
    summary[f"{col}_distribution"] = {
        "min": float(ok[col].min()), "median": float(ok[col].median()),
        "mean": float(ok[col].mean()), "max": float(ok[col].max()),
    }
summary["cross_pairs_with_tagged_edge_share_median"] = float(
    ok["cross_pairs_with_tagged_edge_share"].median())
top = ok.nlargest(5, "chi_poly")[["city", "chi_poly", "road_gap"]]
bot = ok.nsmallest(5, "chi_poly")[["city", "chi_poly", "road_gap"]]
summary["chi_poly_top5"] = top.round(4).to_dict("records")
summary["chi_poly_bottom5"] = bot.round(4).to_dict("records")

with open(OUTDIR / "R205_S1_dose_response_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
