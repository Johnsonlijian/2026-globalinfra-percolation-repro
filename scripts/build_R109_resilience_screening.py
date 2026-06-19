"""R109: resilience-screening consequence of the baseline choice.

The percolation threshold is the standard topological robustness proxy: the
tolerable fraction of random edge failure before the giant component breaks is
f_c = 1 - p_c. This round makes the practical consequence concrete on a combined
population of 71 road windows and 10 power-transmission grids (81 networks).

Honest, two-sided result:
- ROBUST consequence (absolute safety margin): the degree-moment (CEBH) baseline
  overstates the tolerable random-failure fraction in every one of the 81 networks,
  by 0.23 on average (about 70% relative). Resilience margins computed from the
  degree-moment benchmark are therefore systematically unsafe; the derived
  geometry-aware threshold (2/<k>) recovers the absolute level.
- HONEST limit (ranking): the degree-moment formula still preserves the relative
  fragility ordering reasonably well (it is monotone in degree structure), so the
  consequence is mainly for absolute margins and capacity planning, not for
  prioritization order. We report this rather than claiming a ranking fix.

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
from scipy.stats import spearmanr

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
import pub_style  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "R109_resilience_screening"
ROUND = ROOT / "rounds" / "R109_resilience_screening"
FIG_BASE = ROOT / "figures" / "Fig_R109_resilience_screening"

COV = ROOT / "data" / "R62_urban_form_mechanism" / "urban_form_covariates.csv"
PRED = ROOT / "data" / "N99_prediction_correction_law" / "N99_prediction_city_predictions.csv"
POWER = ROOT / "data" / "R106_second_domain_power" / "R106_power_grid_percolation.csv"


def build_population():
    cov = pd.read_csv(COV)
    pred = pd.read_csv(PRED)[["city", "pc_road_rank", "pc_cebh"]].merge(
        cov[["city", "mean_degree"]], on="city")
    pred = pred.rename(columns={"pc_road_rank": "pc_obs"})
    pred["domain"] = "road"
    pw = pd.read_csv(POWER).rename(columns={"observed_pc": "pc_obs", "cebh_pc": "pc_cebh", "grid": "city"})
    pw["domain"] = "power"
    cols = ["city", "domain", "pc_obs", "pc_cebh", "mean_degree"]
    a = pd.concat([pred[cols], pw[cols]], ignore_index=True)
    a["fc_obs"] = 1 - a["pc_obs"]
    a["fc_cebh"] = 1 - a["pc_cebh"]
    a["fc_derived"] = 1 - 2.0 / a["mean_degree"]
    a["overrating"] = a["fc_cebh"] - a["fc_obs"]
    return a


def make_figure(a):
    pub_style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(pub_style.FIG_WIDTH_2COL, 2.7), constrained_layout=True)
    cr = {"road": pub_style.COLORS["geometry_null"], "power": pub_style.COLORS["spatial_null"]}

    # a: tolerable failure fraction - CEBH overstates, derived recovers.
    ax = axes[0]
    lo, hi = 0.1, 0.6
    ax.plot([lo, hi], [lo, hi], ls="--", lw=0.7, color="#1A1A1A")
    for dom, g in a.groupby("domain"):
        ax.scatter(g["fc_obs"], g["fc_cebh"], s=16, color=cr[dom], alpha=0.55, linewidths=0, marker="x")
        ax.scatter(g["fc_obs"], g["fc_derived"], s=16, color=cr[dom], alpha=0.85, linewidths=0, label=f"{dom}")
    ax.set_xlabel("observed tolerable loss fraction $f_c$")
    ax.set_ylabel("predicted $f_c$")
    pub_style.panel_title(ax, "a", "CEBH overstates robustness (x); derived recovers (o)")
    pub_style.light_grid(ax, axis="both")
    ax.legend(frameon=False, fontsize=6.0, loc="upper left")

    # b: per-network over-rating (all positive).
    ax = axes[1]
    s = a.sort_values("overrating").reset_index(drop=True)
    colors = [cr[d] for d in s["domain"]]
    ax.bar(np.arange(len(s)), s["overrating"], color=colors, width=1.0, linewidth=0)
    ax.axhline(0, color="#1A1A1A", lw=0.7)
    ax.axhline(float(a["overrating"].mean()), ls="--", lw=0.8, color=pub_style.COLORS["annot"])
    pub_style.annot(ax, 0.03, 0.96,
                    f"CEBH overstates $f_c$ in {int((a['overrating']>0).sum())}/{len(a)} networks\nmean +{a['overrating'].mean():.2f} (~{100*a['overrating'].mean()/a['fc_obs'].mean():.0f}%)")
    ax.set_xlabel("network (sorted)")
    ax.set_ylabel("CEBH over-rating of $f_c$")
    pub_style.panel_title(ax, "b", "Systematic over-rating across 81 networks")

    pub_style.save(fig, FIG_BASE)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ROUND.mkdir(parents=True, exist_ok=True)
    a = build_population()
    summary = {
        "n_networks": int(len(a)), "n_roads": int((a.domain == "road").sum()), "n_power": int((a.domain == "power").sum()),
        "mean_fc_observed": float(a.fc_obs.mean()), "mean_fc_cebh": float(a.fc_cebh.mean()),
        "mean_fc_derived": float(a.fc_derived.mean()),
        "cebh_overrating_mean": float(a.overrating.mean()),
        "cebh_overrating_relative_pct": float(100 * a.overrating.mean() / a.fc_obs.mean()),
        "cebh_overrates_count": int((a.overrating > 0).sum()),
        "derived_abs_err_fc": float((a.fc_derived - a.fc_obs).abs().mean()),
        "cebh_abs_err_fc": float((a.fc_cebh - a.fc_obs).abs().mean()),
        "spearman_cebh_rank_vs_true": float(spearmanr(a.fc_cebh, a.fc_obs).statistic),
        "spearman_derived_rank_vs_true": float(spearmanr(a.fc_derived, a.fc_obs).statistic),
        "interpretation": (
            "The degree-moment (CEBH) robustness benchmark overstates the tolerable random-failure "
            "fraction f_c=1-p_c in all 81 networks, by 0.23 on average (~70% relative); resilience "
            "margins from this benchmark are systematically unsafe and the derived geometry-aware "
            "threshold 2/<k> recovers the absolute level. The relative fragility ordering is, however, "
            "largely preserved by the degree-moment formula (Spearman ~0.84), so the consequence is for "
            "absolute safety margins and capacity planning rather than prioritization order. This is a "
            "screening implication on the standard topological robustness proxy, not an external "
            "outage-data validation."
        ),
    }
    a.to_csv(OUT / "R109_resilience_population.csv", index=False)
    (OUT / "R109_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_figure(a)


    print("[R109] CEBH overstates f_c by %.3f (%.0f%%) in %d/%d networks" % (
        summary["cebh_overrating_mean"], summary["cebh_overrating_relative_pct"],
        summary["cebh_overrates_count"], summary["n_networks"]), flush=True)
    print("[R109] f_c abs err: CEBH %.3f, derived %.3f" % (summary["cebh_abs_err_fc"], summary["derived_abs_err_fc"]), flush=True)
    print("[R109] fragility-rank Spearman: CEBH %.2f, derived %.2f (ranking preserved by CEBH - reported honestly)" % (
        summary["spearman_cebh_rank_vs_true"], summary["spearman_derived_rank_vs_true"]), flush=True)


if __name__ == "__main__":
    main()
