"""R205-S5: headline 4-panel figure - the measured crossing-density control parameter.

a. Measured chi_poly across the 71 standardized city windows (ranked strip,
   extremes named, bridge/tunnel validation note).
b. Dose-response: chi_poly vs road CEBH gap (Spearman + partial rho after
   k/size/GDP controls).
c. Mechanism closure: chi_poly vs full-71 strict-geometry-null residual
   (R81, f=0.01) - the null ladder's leftover is predicted by measured chi.
d. Manipulation vs observation on one axis: chi-targeted flyover dial chains
   (S2b; degree- and edge-count-preserving) against the 71-city cross-section.

Output: outputs/Fig_R205_measured_control_parameter.{png,pdf,svg}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "data" / "R205_measured_geometry_dose_response"

try:
    import pub_style
    pub_style.apply()
    W2 = getattr(pub_style, "FIG_WIDTH_2COL", 7.2)
    C = getattr(pub_style, "COLORS", {})
except Exception:
    pub_style = None
    W2, C = 7.2, {}

col_obs = C.get("observed", "#1f5fa8")
col_geo = C.get("geometry_null", "#2a9d8f")
col_cebh = C.get("cebh", "#c1121f")
col_acc = C.get("accent", "#7a5195")

s1 = pd.read_csv(OUT / "R205_S1_crossing_density_71cities.csv")
s1 = s1[s1.status == "ok"].copy()
with open(OUT / "R205_S1_dose_response_summary.json", encoding="utf-8") as f:
    s1sum = json.load(f)
with open(OUT / "R205_S1c_confound_controls.json", encoding="utf-8") as f:
    s1c = json.load(f)
dial = pd.read_csv(OUT / "R205_S2b_chi_targeted_dial.csv")

fig, axes = plt.subplots(1, 4, figsize=(W2 * 1.06, 2.42), constrained_layout=True)

# ---- a: ranked measured chi across 71 cities ----
ax = axes[0]
d = s1.sort_values("chi_poly").reset_index(drop=True)
ax.scatter(np.arange(len(d)), d["chi_poly"], s=7, c=col_obs, alpha=0.75, lw=0)
ax.set_yscale("log")
for name, dy in (("Hong Kong", 0), ("Taipei", 0), ("Dar es Salaam", 0), ("Kampala", 0)):
    row = d[d.city == name]
    if len(row):
        i = int(row.index[0])
        ax.annotate(name, (i, row["chi_poly"].iloc[0]), fontsize=5,
                    xytext=(-2, 3 + dy), textcoords="offset points",
                    ha="right" if i > 35 else "left")
ax.set_xlabel("city rank")
ax.set_ylabel(r"measured crossing density $\chi$")
if pub_style:
    pub_style.panel_title(ax, "a", "71 cities span two decades")
ax.text(0.02, 0.97, "median 100% of crossings\ntouch bridge/tunnel tags",
        transform=ax.transAxes, va="top", fontsize=5, color="#555555")

# ---- b: chi vs road gap ----
ax = axes[1]
ax.scatter(s1["chi_poly"], s1["road_gap"], s=8, c=col_obs, alpha=0.75, lw=0)
ax.set_xscale("log")
ax.set_xlabel(r"$\chi$ (log)")
ax.set_ylabel(r"road gap $p_c^{\rm road}-p_c^{\rm CEBH}$")
r = s1sum["chi_poly__vs__road_gap"]
pr = s1c.get("chi_poly__road_gap__ctrl_k_size_gdp", {})
x = np.log(s1["chi_poly"])
b1, b0 = np.polyfit(x, s1["road_gap"], 1)
xs = np.linspace(x.min(), x.max(), 50)
ax.plot(np.exp(xs), b0 + b1 * xs, "-", color=col_cebh, lw=1)
ax.text(0.03, 0.06,
        f"Spearman {r['spearman']:.2f} (p={r['spearman_p']:.0e})\n"
        f"partial (k,size,GDP) {pr.get('partial_spearman', float('nan')):.2f} "
        f"(p={pr.get('perm_p', float('nan')):.3f})",
        transform=ax.transAxes, fontsize=5, va="bottom")
if pub_style:
    pub_style.panel_title(ax, "b", "More crossings, closer to CEBH")

# ---- c: null-family errors vs chi (spatial structured, strict flat) ----
ax = axes[2]
sp_col, ge_col = "spatial_resid_f0.01", "geom_resid_f0.01"
sub = s1.dropna(subset=[sp_col, ge_col])
ax.scatter(sub["chi_poly"], sub[sp_col], s=8, c=col_cebh, alpha=0.8, lw=0,
           label="spatial null (crossings allowed)")
ax.scatter(sub["chi_poly"], sub[ge_col], s=8, c=col_geo, alpha=0.8, lw=0,
           label="strict non-crossing null")
ax.axhline(0, color="#888888", lw=0.6, ls=":")
ax.set_xscale("log")
ax.set_xlabel(r"$\chi$ (log)")
ax.set_ylabel("road $-$ null threshold")
ax.set_ylim(-0.085, 0.17)
rsp = s1sum[f"chi_poly__vs__{sp_col}"]
psp = s1c.get(f"chi_poly__{sp_col}__ctrl_k_size_gdp", {})
x = np.log(sub["chi_poly"])
b1, b0 = np.polyfit(x, sub[sp_col], 1)
xs = np.linspace(x.min(), x.max(), 50)
ax.plot(np.exp(xs), b0 + b1 * xs, "-", color=col_cebh, lw=1)
ax.text(0.04, 0.02,
        f"spatial $\\rho$={rsp['spearman']:.2f}; partial {psp.get('partial_spearman', float('nan')):.2f}\n"
        "strict flat ($\\rho$=-0.11, p=0.38)",
        transform=ax.transAxes, fontsize=4.2, va="bottom",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.90, pad=0.7))
ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.92,
          fontsize=3.9, loc="upper right", handletextpad=0.35,
          borderpad=0.2, labelspacing=0.15)
if pub_style:
    pub_style.panel_title(ax, "c", "Null error is structured by $\\chi$")

# ---- d: manipulation vs observation ----
ax = axes[3]
ax.scatter(s1["chi_chord"], s1["pc_rank"], s=7, c="#b8b8b8", alpha=0.7, lw=0,
           label="71 cities (observed)")
palette = [col_obs, col_geo, col_acc, col_cebh]
for i, g in enumerate([g for g in dial.graph.unique()]):
    sub = dial[dial.graph == g].sort_values("chi_chord")
    ax.plot(sub["chi_chord"].clip(lower=3e-3), sub["pc"], "o-", ms=2.5, lw=0.9,
            color=palette[i % 4], alpha=0.9, label=g.replace("_", " "))
ax.set_xscale("log")
ax.set_xlabel(r"$\chi$ (chord space, log)")
ax.set_ylabel(r"$p_c$")
ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.94,
          fontsize=3.9, loc="upper center", bbox_to_anchor=(0.5, -0.22),
          ncol=2, handletextpad=0.35, columnspacing=0.8, borderpad=0.2,
          labelspacing=0.15)
if pub_style:
    pub_style.panel_title(ax, "d", "Local crossings leave $p_c$ stable")

for ext in ("png", "pdf", "svg"):
    fig.savefig(OUT / f"Fig_R205_measured_control_parameter.{ext}", dpi=450,
                bbox_inches="tight", pad_inches=0.04)
print("[S5] figure written to", OUT, flush=True)
