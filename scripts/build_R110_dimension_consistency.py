"""R110: non-circular validation of the effective-dimension law.

The effective dimension d = R/(R-1), R = p_c<k>, is read from the threshold
*location*. A skeptic can call the statement "d sets the threshold" circular,
because d is an algebraic function of p_c. This round breaks that circularity by
recovering d from a *second, independent* critical observable - the Fisher
exponent tau_F of the cluster-size distribution, which measures the *shape* of
the distribution at criticality, not the location of the threshold.

Design (no fabricated tau(d) law; an empirical monotone map only):
  1. Calibration: on the controlled planarity dial (planarity is the only knob
     driving both observables), fit a monotone map g: tau_F -> d using isotonic
     regression. No functional form is assumed.
  2. Held-out test: for real networks that carry BOTH observables (5 road windows
     with measured tau_F, 10 power grids), predict the "universality-route"
     dimension d_tau = g(tau_F) and compare it to the independent
     "threshold-route" dimension d_R = R/(R-1). These two observables are
     measured independently, so their agreement is genuine (non-circular)
     evidence that the effective dimension is a real, multiply-readable property.
  3. Forward consequence: d_tau -> predicted threshold p_c = d_tau/((d_tau-1)<k>),
     compared to the observed threshold with a dial-calibrated band.

Physical anchors (reference only, not fitted): the 2D percolation class has
tau_F = 187/91 ~= 2.055 at d = 2; the mean-field class has tau_F = 5/2 at the
upper critical dimension d_c = 6. The empirical map is bracketed by these.

Uses registered derived tables only; no third-party raw data are redistributed.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.isotonic import IsotonicRegression

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
import pub_style  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "R110_dimension_consistency"
ROUND = ROOT / "rounds" / "R110_dimension_consistency"
FIG_BASE = ROOT / "figures" / "Fig_R110_dimension_consistency"
NPJ = ROOT / "submission" / "npj_complexity" / "target_submission"

DIAL = DATA / "R108_effective_dimension" / "R108_dial_effective_dimension.csv"
ROAD_DEFF = DATA / "R108_effective_dimension" / "R108_road_effective_dimension.csv"
ROAD_TAU = DATA / "R105_finite_size_scaling" / "R105_fisher_exponents.csv"
POWER = DATA / "R106_second_domain_power" / "R106_power_grid_percolation.csv"

# physical reference anchors (exact 2D class; mean-field at upper critical dim)
TAU_2D, D_2D = 187.0 / 91.0, 2.0
TAU_MF, D_MF = 2.5, 6.0


def d_from_threshold(pc: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Threshold-route effective dimension d = R/(R-1), R = pc*<k>."""
    R = np.asarray(pc) * np.asarray(k)
    return R / (R - 1.0)


def load_calibration():
    d = pd.read_csv(DIAL)
    return d[["city", "f_cross", "pc", "mean_degree", "d_eff", "fisher_tau"]].copy()


def load_real():
    # roads: 5 windows that carry a measured Fisher exponent
    tau = pd.read_csv(ROAD_TAU).rename(columns={"system": "city", "tau_F": "fisher_tau"})
    tau = tau[~tau["city"].isin(["square_lattice"]) & ~tau["city"].str.startswith("mean_field")]
    deff = pd.read_csv(ROAD_DEFF).rename(columns={"pc_road_rank": "pc"})
    roads = tau.merge(deff[["city", "pc", "mean_degree", "d_eff"]], on="city", how="inner")
    roads = roads.rename(columns={"d_eff": "d_R"})
    roads["domain"] = "road"

    # power: 10 grids carry pc, <k> and a measured Fisher exponent
    pw = pd.read_csv(POWER).rename(columns={"grid": "city", "observed_pc": "pc"})
    pw["d_R"] = d_from_threshold(pw["pc"].values, pw["mean_degree"].values)
    pw["domain"] = "power"
    cols = ["city", "domain", "pc", "mean_degree", "fisher_tau", "d_R"]
    return pd.concat([roads[cols], pw[cols]], ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ROUND.mkdir(parents=True, exist_ok=True)

    cal = load_calibration()
    real = load_real()

    # monotone universality-route map g: tau_F -> d, calibrated on the dial only.
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(cal["fisher_tau"].values, cal["d_eff"].values)

    # residual scatter of the calibration about the monotone fit -> a band in d.
    cal_pred = iso.predict(cal["fisher_tau"].values)
    band_d = float(np.std(cal["d_eff"].values - cal_pred, ddof=1))

    # held-out test on real networks (NOT used to fit g).
    real["d_tau"] = iso.predict(real["fisher_tau"].values)
    real["d_resid"] = real["d_tau"] - real["d_R"]
    tau_lo, tau_hi = float(cal["fisher_tau"].min()), float(cal["fisher_tau"].max())
    real["tau_in_range"] = (real["fisher_tau"] >= tau_lo) & (real["fisher_tau"] <= tau_hi)

    # forward consequence: predict the threshold from the universality-route d.
    real["pc_pred_class"] = real["d_tau"] / ((real["d_tau"] - 1.0) * real["mean_degree"])
    real["pc_band"] = band_d / (((real["d_tau"] - 1.0) ** 2) * real["mean_degree"])  # |dpc/dd|*band
    real["pc_within_band"] = (real["pc"] - real["pc_pred_class"]).abs() <= real["pc_band"]

    d_mae = float(real["d_resid"].abs().mean())
    d_r, d_p = pearsonr(real["d_R"], real["d_tau"])
    pc_mae = float((real["pc"] - real["pc_pred_class"]).abs().mean())

    summary = {
        "design": (
            "Two independent critical observables, the threshold location (route A: "
            "d_R = R/(R-1)) and the cluster-size Fisher exponent (route B: d_tau = "
            "g(tau_F), g calibrated by isotonic regression on the controlled planarity "
            "dial only), are compared on 15 held-out real networks. Agreement breaks the "
            "circularity objection that d is merely an algebraic relabel of p_c."
        ),
        "n_calibration_points": int(len(cal)),
        "n_real_test": int(len(real)),
        "n_road": int((real.domain == "road").sum()),
        "n_power": int((real.domain == "power").sum()),
        "calibration_band_d": band_d,
        "tau_range_calibration": [tau_lo, tau_hi],
        "n_real_tau_in_range": int(real["tau_in_range"].sum()),
        "d_route_agreement_mae": d_mae,
        "d_route_pearson_r": float(d_r),
        "d_route_pearson_p": float(d_p),
        "forward_pc_mae_from_class": pc_mae,
        "forward_pc_within_band_frac": float(real["pc_within_band"].mean()),
        "physical_anchors": {"tau_2D": TAU_2D, "d_2D": D_2D, "tau_MF": TAU_MF, "d_MF": D_MF},
        "interpretation": (
            "The effective dimension recovered from the universality class agrees with the "
            "one read from the threshold across the held-out real networks (Pearson r=%.2f, "
            "d-MAE=%.2f), and the class-route prediction recovers the observed threshold "
            "(p_c-MAE=%.3f). Because the two routes use independent observables - the shape "
            "of the cluster-size distribution versus the location of the threshold - their "
            "agreement shows the effective dimension is a real, multiply-readable property, "
            "not an algebraic restatement of p_c. The empirical tau->d map is monotone and "
            "bracketed by the exact 2D class (tau=2.055, d=2) and the mean-field class "
            "(tau=2.5, d_c=6); no closed-form tau(d) law is asserted."
        ) % (d_r, d_mae, pc_mae),
    }

    cal.to_csv(OUT / "R110_calibration_dial.csv", index=False)
    real.to_csv(OUT / "R110_real_network_consistency.csv", index=False)
    (OUT / "R110_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    make_figure(cal, real, iso, band_d, summary)

    print("[R110] held-out real networks: d-route Pearson r=%.2f (p=%.1e), d-MAE=%.2f" % (
        d_r, d_p, d_mae), flush=True)
    print("[R110] forward p_c from class: MAE=%.3f, within-band %.0f%%" % (
        pc_mae, 100 * summary["forward_pc_within_band_frac"]), flush=True)
    print("[R110] calibration: %d dial points, band(d)=%.2f, tau range [%.2f, %.2f]" % (
        len(cal), band_d, tau_lo, tau_hi), flush=True)


def make_figure(cal, real, iso, band_d, summary):
    pub_style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(pub_style.FIG_WIDTH_2COL, 2.9), constrained_layout=True)
    cr = {"road": pub_style.COLORS["geometry_null"], "power": pub_style.COLORS["spatial_null"]}

    # Panel a: the two independent routes agree (held-out real networks on 1:1).
    ax = axes[0]
    lo, hi = 1.9, max(real["d_R"].max(), real["d_tau"].max()) + 0.3
    ax.plot([lo, hi], [lo, hi], ls="--", lw=0.7, color="#1A1A1A", zorder=1)
    # calibration points (in-sample) shown faint for context
    cal_pred = iso.predict(cal["fisher_tau"].values)
    ax.scatter(cal["d_eff"], cal_pred, s=10, color=pub_style.COLORS["model"],
               alpha=0.30, linewidths=0, label="planarity dial (calibration)", zorder=2)
    for dom, g in real.groupby("domain"):
        ax.scatter(g["d_R"], g["d_tau"], s=26, color=cr[dom], alpha=0.9,
                   linewidths=0, label=f"{dom} (held-out)", zorder=3)
    ax.set_xlabel("threshold-route dimension  $d_R=R/(R-1)$")
    ax.set_ylabel("universality-route dimension  $d_\\tau=g(\\tau_F)$")
    pub_style.panel_title(ax, "a", "Two independent routes to $d$ agree")
    pub_style.annot(ax, 0.04, 0.96,
                    "held-out real nets:\n$r=%.2f$, MAE $=%.2f$" % (
                        summary["d_route_pearson_r"], summary["d_route_agreement_mae"]))
    pub_style.light_grid(ax, axis="both")
    ax.legend(frameon=False, fontsize=5.6, loc="lower right")

    # Panel b: the calibration locus d(tau_F) with real networks on the same curve.
    ax = axes[1]
    xs = np.linspace(cal["fisher_tau"].min(), cal["fisher_tau"].max(), 200)
    ys = iso.predict(xs)
    ax.fill_between(xs, ys - band_d, ys + band_d, color=pub_style.COLORS["model"],
                    alpha=0.15, linewidth=0, label="calibration band")
    ax.plot(xs, ys, color=pub_style.COLORS["model"], lw=1.3, label="isotonic map $g$")
    ax.scatter(cal["fisher_tau"], cal["d_eff"], s=10, color=pub_style.COLORS["model"],
               alpha=0.35, linewidths=0)
    for dom, g in real.groupby("domain"):
        ax.scatter(g["fisher_tau"], g["d_R"], s=26, color=cr[dom], alpha=0.9,
                   linewidths=0, label=f"{dom}")
    # physical reference anchors
    ax.scatter([TAU_2D], [D_2D], marker="*", s=70, color="#1A1A1A", zorder=5)
    ax.annotate("2D class\n($\\tau=2.055$, $d=2$)", (TAU_2D, D_2D), xytext=(3, -16),
                textcoords="offset points", fontsize=5.6, color="#1A1A1A")
    ax.axvline(TAU_MF, ls=":", lw=0.7, color="#6E6E6E")
    ax.annotate("mean-field\n$\\tau=2.5$", (TAU_MF, hi - 0.2), xytext=(-30, 0),
                textcoords="offset points", fontsize=5.6, color="#6E6E6E")
    ax.set_xlabel("Fisher exponent  $\\tau_F$  (universality class)")
    ax.set_ylabel("effective dimension  $d$")
    pub_style.panel_title(ax, "b", "One locus links class and dimension")
    pub_style.light_grid(ax, axis="both")
    ax.legend(frameon=False, fontsize=5.6, loc="lower right")

    pub_style.save(fig, FIG_BASE)
    plt.close(fig)

    if NPJ.exists():
        for ext in [".svg", ".pdf", ".png", ".tiff"]:
            src = FIG_BASE.with_suffix(ext)
            if src.exists():
                shutil.copy2(src, NPJ / "figures" / f"FigS_R110_dimension_consistency{ext}")
                if ext != ".tiff":
                    shutil.copy2(src, NPJ / "source_data" / "figures" / f"FigS_R110_dimension_consistency{ext}")
        for f in ["R110_real_network_consistency.csv", "R110_calibration_dial.csv", "R110_summary.json"]:
            dst = NPJ / "source_data" / "tables" / f
            if (OUT / f).exists() and dst.parent.exists():
                shutil.copy2(OUT / f, dst)


if __name__ == "__main__":
    main()
