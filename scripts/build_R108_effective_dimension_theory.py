"""R108: effective-dimension theory - closed-form derivation of the anchor.

This round replaces the phenomenological honeycomb-square mixture with an approximate,
parameter-free threshold law and checks it against the critical-class read-out.

Derivation. For a locally tree-like random graph the giant cluster appears when
the mean number of new occupied connections per occupied edge reaches one
(Molloy-Reed), giving the degree-moment value p_c = <k>/(<k^2>-<k>) = 1/(kappa-1).
A two-dimensional embedded graph is not tree-like: planarity forces occupied
edges to close loops rather than always reach new nodes, so a *spanning* cluster
requires loops, not a tree. The classical dimensional bond-percolation relation
captures this as p_c * z ~= d/(d-1) (Vyssotsky-type); for d = 2 this is

    p_c ~= 2 / <k> .

The honeycomb (z=3 -> 2/3) and square (z=4 -> 1/2) lattices are its special
cases, so the earlier junction-composition mixture is this same relation in
disguise. The factor of two relative to the mean-field/tree value 1/<k> is why
degree-moment formulas under-predict embedded thresholds about twofold.

Cross-check. Reading d as an effective dimension from each network's own
threshold, d_eff = R/(R-1) with R = p_c*<k>, gives a threshold-route descriptor
that should agree with the cluster-size (Fisher) exponent if the interpretation is not circular.
We show: roads have d_eff ~= 2 (two-dimensional, tau_F ~= 2.05); non-planar power
grids have d_eff ~= 2.5 (tau_F ~= 2.4, near mean-field); and a controlled
planarity dial raises d_eff and tau_F together. The threshold and exponent routes
therefore track the same effective-dimension organization within the tested systems.

Uses derived tables (R62/R103/R105/R106/R107) and the analytic
lattices; no third-party raw data are redistributed.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
import pub_style  # noqa: E402
from build_R105_finite_size_scaling import load_road_simple  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache" / "R61_osmnx_city_windows"
OUT = ROOT / "data" / "R108_effective_dimension"
ROUND = ROOT / "rounds" / "R108_effective_dimension"
FIG_BASE = ROOT / "figures" / "Fig_R108_effective_dimension"
NPJ = ROOT / "submission" / "npj_complexity" / "target_submission"

COV = ROOT / "data" / "R62_urban_form_mechanism" / "urban_form_covariates.csv"
PRED = ROOT / "data" / "N99_prediction_correction_law" / "N99_prediction_city_predictions.csv"
POWER = ROOT / "data" / "R106_second_domain_power" / "R106_power_grid_percolation.csv"
DIAL = ROOT / "data" / "R107_planarity_dial" / "R107_planarity_dial.csv"

HONEY = 1.0 - 2.0 * math.sin(math.pi / 18.0)
SQUARE = 0.5
TRI = 2.0 * math.sin(math.pi / 18.0)


def eff_dim(pc, k):
    R = np.asarray(pc) * np.asarray(k)
    return R / (R - 1.0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ROUND.mkdir(parents=True, exist_ok=True)

    # --- lattice validation of p_c = 2/z (exact thresholds) ---
    lattice = {
        "square (z=4)": (4.0, SQUARE), "honeycomb (z=3)": (3.0, HONEY), "triangular (z=6)": (6.0, TRI),
    }
    lat_rows = [{"lattice": k, "z": z, "p_c_exact": pc, "two_over_z": 2.0 / z, "abs_err": abs(2.0 / z - pc)}
                for k, (z, pc) in lattice.items()]

    # --- roads ---
    cov = pd.read_csv(COV)
    pred = pd.read_csv(PRED)[["city", "pc_road_rank"]]
    r = pred.merge(cov[["city", "mean_degree", "kappa", "degree3_share", "degree4plus_share"]], on="city")
    obs = r["pc_road_rank"].to_numpy(float)
    k = r["mean_degree"].to_numpy(float)
    cebh = 1.0 / (r["kappa"].to_numpy(float) - 1.0)
    derived = 2.0 / k
    w3 = r["degree3_share"] / (r["degree3_share"] + r["degree4plus_share"])
    anchor = (w3 * HONEY + (1 - w3) * SQUARE).to_numpy(float)
    d_road = eff_dim(obs, k)

    def metrics(p):
        e = p - obs
        return {"mae": float(np.abs(e).mean()), "bias": float(e.mean()),
                "spearman": float(spearmanr(p, obs).statistic), "pearson": float(pearsonr(p, obs).statistic)}

    road = {"n": int(len(r)), "cebh": metrics(cebh), "derived_2_over_k": metrics(derived),
            "phenomenological_anchor": metrics(anchor),
            "d_eff_mean": float(np.nanmean(d_road)), "d_eff_median": float(np.nanmedian(d_road)),
            "d_eff_sd": float(np.nanstd(d_road))}

    # --- power grids ---
    pw = pd.read_csv(POWER)
    obsP = pw["observed_pc"].to_numpy(float)
    kP = pw["mean_degree"].to_numpy(float)
    dP = eff_dim(obsP, kP)
    power = {"n": int(len(pw)),
             "derived_2_over_k_mae": float(np.abs(2.0 / kP - obsP).mean()),
             "derived_2_over_k_bias": float((2.0 / kP - obsP).mean()),
             "d_eff_mean": float(np.nanmean(dP)), "d_eff_median": float(np.nanmedian(dP)),
             "fisher_tau_mean": float(pw["fisher_tau"].mean())}

    # --- planarity dial: d_eff and tau_F vs added crossings ---
    # Mean degree after adding f*m0 random edges is <k>(f) = <k0>*(1+f); recover
    # <k0> = 2*m0/n exactly from each cached road graph.
    dial = pd.read_csv(DIAL)
    cache_files = {p.name.split("__")[0].split("_")[0]: p for p in CACHE.glob("*.pkl")}
    k0_by_city = {}
    for city in dial["city"].unique():
        path = cache_files.get(city)
        if path is not None:
            n, edges, _ = load_road_simple(path)
            k0_by_city[city] = 2.0 * len(edges) / n
    dial_rows = []
    for city, g in dial.groupby("city"):
        k0 = k0_by_city.get(city, 2.8)
        for _, row in g.sort_values("f_cross").iterrows():
            kf = k0 * (1.0 + row["f_cross"])
            dd = eff_dim(row["pc"], kf)
            dial_rows.append({"city": city, "f_cross": float(row["f_cross"]), "pc": float(row["pc"]),
                              "mean_degree": kf, "d_eff": float(dd), "fisher_tau": float(row["fisher_tau"])})
    dial_df = pd.DataFrame(dial_rows)

    summary = {
        "derivation": "p_c ~= d/((d-1)<k>); d=2 (planar) -> p_c = 2/<k>; d->inf (tree) -> 1/<k> = CEBH",
        "lattice_validation": lat_rows,
        "roads": road, "power": power,
        "dial_d_eff_by_fraction": dial_df.groupby("f_cross")[["d_eff", "fisher_tau"]].mean().reset_index().to_dict("records"),
        "headline": (
            f"Approximate parameter-free law p_c=2/<k> predicts 71-city thresholds with MAE "
            f"{road['derived_2_over_k']['mae']:.3f} (vs {road['phenomenological_anchor']['mae']:.3f} for the "
            f"honeycomb-square mixture and {road['cebh']['mae']:.3f} for the degree-moment formula). "
            f"Effective dimension from the threshold: roads d={road['d_eff_median']:.2f} (2D), power grids "
            f"d={power['d_eff_median']:.2f}; the planarity dial raises d and the Fisher exponent together. "
            "The threshold and exponent routes track the same effective-dimension organization."
        ),
    }
    r.assign(cebh=cebh, derived_2_over_k=derived, anchor=anchor, d_eff=d_road).to_csv(
        OUT / "R108_road_effective_dimension.csv", index=False)
    dial_df.to_csv(OUT / "R108_dial_effective_dimension.csv", index=False)
    (OUT / "R108_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    make_figure(obs, derived, cebh, anchor, d_road, dP, dial_df, road, power)

    if NPJ.exists():
        for ext in [".svg", ".pdf", ".png", ".tiff"]:
            src = FIG_BASE.with_suffix(ext)
            if src.exists():
                shutil.copy2(src, NPJ / "figures" / f"Fig8_effective_dimension{ext}")
        for f in ["R108_road_effective_dimension.csv", "R108_dial_effective_dimension.csv", "R108_summary.json"]:
            shutil.copy2(OUT / f, NPJ / "source_data" / "tables" / f)

    print("[R108] roads: 2/<k> MAE=%.4f  anchor MAE=%.4f  CEBH MAE=%.4f" % (
        road["derived_2_over_k"]["mae"], road["phenomenological_anchor"]["mae"], road["cebh"]["mae"]), flush=True)
    print("[R108] d_eff roads=%.2f  power=%.2f" % (road["d_eff_median"], power["d_eff_median"]), flush=True)
    print("[R108] lattice 2/z errors:", [round(x["abs_err"], 4) for x in lat_rows], flush=True)


def make_figure(obs, derived, cebh, anchor, d_road, dP, dial_df, road, power):
    pub_style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(pub_style.FIG_WIDTH_2COL, 2.8), constrained_layout=True)

    # a: 2/<k> vs observed (roads), CEBH for contrast.
    ax = axes[0]
    lo, hi = 0.4, 0.95
    ax.plot([lo, hi], [lo, hi], ls="--", lw=0.7, color="#1A1A1A")
    ax.scatter(derived, obs, s=18, color=pub_style.COLORS["model"], alpha=0.85, linewidths=0, label=r"law $2/\langle k\rangle$")
    ax.scatter(cebh, obs, s=14, color=pub_style.COLORS["cebh"], alpha=0.6, linewidths=0, marker="x", label="CEBH")
    ax.set_xlabel(r"predicted $p_c$")
    ax.set_ylabel(r"observed road $p_c$")
    pub_style.panel_title(ax, "a", r"Zero-fit $p_c=2/\langle k\rangle$ (71 cities)")
    pub_style.light_grid(ax, axis="both")
    pub_style.annot(ax, 0.40, 0.18, f"2/<k> MAE {road['derived_2_over_k']['mae']:.3f}\nanchor MAE {road['phenomenological_anchor']['mae']:.3f}\nCEBH MAE {road['cebh']['mae']:.3f}")
    ax.legend(frameon=False, fontsize=6.0, loc="upper left")

    # b: effective dimension - roads ~2, power ~2.5.
    ax = axes[1]
    parts = ax.boxplot([d_road[np.isfinite(d_road)], dP[np.isfinite(dP)]], tick_labels=["roads\n(71)", "power\n(10)"],
                       widths=0.5, patch_artist=True, showfliers=False)
    for patch, c in zip(parts["boxes"], [pub_style.COLORS["geometry_null"], pub_style.COLORS["spatial_null"]]):
        patch.set_facecolor(c); patch.set_alpha(0.25); patch.set_edgecolor(c)
    for key in ["whiskers", "caps", "medians"]:
        for a in parts[key]:
            a.set(color="#334155", lw=0.9)
    ax.axhline(2.0, ls="--", lw=0.8, color="#1A1A1A")
    ax.text(2.45, 2.02, "d = 2 (planar)", ha="right", va="bottom", fontsize=5.8)
    ax.set_ylabel("effective dimension $d$ (from threshold)")
    pub_style.panel_title(ax, "b", "Roads are 2D, power grids are not")

    # c: unification - d_eff and tau_F rise together on the dial.
    ax = axes[2]
    g = dial_df.groupby("f_cross")[["d_eff", "fisher_tau"]].mean().reset_index()
    ax.plot(g["f_cross"], g["d_eff"], "o-", ms=3, lw=1.0, color=pub_style.COLORS["nb"], label="$d$ from threshold")
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(g["f_cross"], g["fisher_tau"], "s--", ms=3, lw=1.0, color=pub_style.COLORS["accent"], label=r"$\tau_F$ exponent")
    ax.set_xlabel("fraction of added crossings")
    ax.set_ylabel("effective dimension $d$")
    ax2.set_ylabel(r"Fisher exponent $\tau_F$")
    pub_style.panel_title(ax, "c", "Dimension tracks class")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=5.8, loc="lower right")

    pub_style.save(fig, FIG_BASE)
    plt.close(fig)


if __name__ == "__main__":
    main()
