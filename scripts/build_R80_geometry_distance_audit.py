"""R80: geometry-null distance diagnostics.

This round uses already archived replicate-level geometry-null metrics to
answer the reviewer concern that the strict geometry null is only a lightly
perturbed copy of the observed road graph. It does not reconstruct exact edge
overlap, because final rewired edge sets were not archived.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import pub_style  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "R80_geometry_distance_audit"
ROUND = ROOT / "rounds" / "R80_geometry_distance_audit"
FIG_BASE = ROOT / "figures" / "Fig_R80_geometry_null_distance_audit"

R67 = ROOT / "data" / "R67_geometry_null_sensitivity" / "geometry_null_sensitivity_replicates.csv"
R68 = ROOT / "data" / "R68_full71_geometry_null_ensemble" / "full71_geometry_null_replicates.csv"
R72 = ROOT / "data" / "R72_geometry_defense" / "matched_intensity_geometry_vs_spatial_city.csv"


METRIC_LABELS = {
    "delta_connected_components": "Connected components",
    "delta_bridge_edge_share": "Bridge-edge share",
    "delta_articulation_point_share": "Articulation-point share",
    "delta_largest_biconnected_node_fraction": "Largest biconnected fraction",
    "delta_cyclomatic_density": "Cyclomatic density",
    "delta_dual_endpoint_degree_mean": "Dual endpoint degree mean",
    "delta_dual_endpoint_degree_cv": "Dual endpoint degree CV",
    "delta_angular_entropy": "Angular entropy",
    "delta_straight_angle_share_10deg": "Straight-angle share",
    "delta_right_angle_share_10deg": "Right-angle share",
}


def mean_ci(x: pd.Series) -> tuple[float, float, float]:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    if vals.empty:
        return np.nan, np.nan, np.nan
    mean = float(vals.mean())
    if len(vals) < 2:
        return mean, mean, mean
    half = 1.96 * float(vals.std(ddof=1)) / np.sqrt(len(vals))
    return mean, mean - half, mean + half


def build_metric_audit(reps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    passed = reps[reps["status"].eq("pass")].copy()
    for frac, group in passed.groupby("swap_fraction"):
        for col, label in METRIC_LABELS.items():
            if col not in group:
                continue
            m, lo, hi = mean_ci(group[col].abs())
            signed, slo, shi = mean_ci(group[col])
            rows.append(
                {
                    "swap_fraction": float(frac),
                    "n_replicates": int(group[col].notna().sum()),
                    "metric": col,
                    "metric_label": label,
                    "mean_abs_delta": m,
                    "ci95_low_abs_delta": lo,
                    "ci95_high_abs_delta": hi,
                    "mean_signed_delta": signed,
                    "ci95_low_signed_delta": slo,
                    "ci95_high_signed_delta": shi,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "R80_geometry_null_metric_distance_audit.csv", index=False)
    return out


def build_city_audit(reps: pd.DataFrame) -> pd.DataFrame:
    passed = reps[reps["status"].eq("pass")].copy()
    rows = []
    metric_cols = [c for c in METRIC_LABELS if c in passed]
    for (city, country, frac), group in passed.groupby(["city", "country", "swap_fraction"]):
        record = {
            "city": city,
            "country": country,
            "swap_fraction": float(frac),
            "n_replicates": int(len(group)),
            "n_edges": int(group["n_edges"].iloc[0]),
            "road_minus_geometry_mean": float(group["road_minus_geometry"].mean()),
            "pc_geometry_null_std": float(group["pc_geometry_null"].std(ddof=1)) if len(group) > 1 else 0.0,
            "accepted_swaps_per_edge_mean": float(group["accepted_swaps_per_edge"].mean()),
            "acceptance_rate_mean": float(group["acceptance_rate"].mean()),
            "null_mobility_index_mean": float(group["null_mobility_index"].mean()),
            "length_bin_tvd_mean": float(group["length_bin_tvd"].mean()),
            "degree_max_abs_drift_max": float(group["degree_max_abs_drift"].max()),
        }
        for col in metric_cols:
            record[f"mean_abs_{col}"] = float(group[col].abs().mean())
        rows.append(record)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "R80_geometry_null_city_distance_audit.csv", index=False)
    return out


def build_fraction_audit(reps: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    passed = reps[reps["status"].eq("pass")].copy()
    rows = []
    for frac, group in passed.groupby("swap_fraction"):
        rec = {"swap_fraction": float(frac), "n_geometry_replicates": int(len(group))}
        for col in [
            "road_minus_geometry",
            "accepted_swaps_per_edge",
            "acceptance_rate",
            "null_mobility_index",
            "attempted_swaps",
            "accepted_swaps",
            "topology_rejects",
            "length_rejects",
            "crossing_rejects",
            "length_bin_tvd",
            "degree_max_abs_drift",
        ]:
            rec[f"{col}_mean"] = float(group[col].mean())
        rec["attempts_per_accepted_swap"] = float(group["attempted_swaps"].sum() / max(group["accepted_swaps"].sum(), 1))
        match = matched[np.isclose(matched["swap_fraction"], frac)]
        if not match.empty:
            rec["road_minus_spatial_mean"] = float(match["road_minus_spatial"].mean())
            rec["spatial_minus_geometry_mean"] = float(match["spatial_minus_geometry_residual"].mean())
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values("swap_fraction")
    out.to_csv(OUT / "R80_geometry_null_fraction_distance_audit.csv", index=False)
    return out


def build_full71_audit(full: pd.DataFrame) -> pd.DataFrame:
    passed = full[full["status"].eq("pass")].copy()
    rows = []
    for col, label in METRIC_LABELS.items():
        if col in passed:
            m, lo, hi = mean_ci(passed[col].abs())
            rows.append(
                {
                    "scope": "full71_low_intensity",
                    "swap_fraction": 0.005,
                    "n_records": int(passed[col].notna().sum()),
                    "metric": col,
                    "metric_label": label,
                    "mean_abs_delta": m,
                    "ci95_low_abs_delta": lo,
                    "ci95_high_abs_delta": hi,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "R80_full71_low_intensity_distance_audit.csv", index=False)
    return out


def make_figure(metric_audit: pd.DataFrame, fraction: pd.DataFrame, city: pd.DataFrame) -> None:
    pub_style.apply()
    fig, axes = plt.subplots(2, 2, figsize=(pub_style.FIG_WIDTH_2COL, 5.8), constrained_layout=True)
    ax = axes[0, 0]
    x = np.arange(len(fraction))
    width = 0.36
    ax.bar(
        x - width / 2,
        fraction["road_minus_spatial_mean"],
        width,
        label="Spatial-scale residual",
        color=pub_style.COLORS["spatial_null"],
    )
    ax.bar(
        x + width / 2,
        fraction["road_minus_geometry_mean"],
        width,
        label="Strict-geometry residual",
        color=pub_style.COLORS["geometry_null"],
    )
    ax.set_xticks(x, [f"{v:g}" for v in fraction["swap_fraction"]])
    ax.axhline(0, color="#1A1A1A", lw=0.6)
    ax.set_xlabel("Accepted-swap target")
    ax.set_ylabel("Mean road-minus-null residual")
    pub_style.panel_title(ax, "a", "Matched mean residuals")
    pub_style.light_grid(ax, axis="y")
    ax.legend(frameon=False, fontsize=6.2)

    ax = axes[0, 1]
    ax.plot(
        fraction["swap_fraction"],
        fraction["null_mobility_index_mean"],
        marker="o",
        color=pub_style.COLORS["geometry_null"],
        label="Mobility index",
    )
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(
        fraction["swap_fraction"],
        fraction["attempts_per_accepted_swap"],
        marker="s",
        color=pub_style.COLORS["annot"],
        label="Attempts/accepted",
    )
    ax.set_xlabel("Accepted-swap target")
    ax.set_ylabel("Mobility index")
    ax2.set_ylabel("Attempts per accepted swap")
    pub_style.panel_title(ax, "b", "Perturbation and constraint pressure")
    handles = [
        plt.Line2D([], [], color=pub_style.COLORS["geometry_null"], marker="o", label="Mobility index"),
        plt.Line2D([], [], color=pub_style.COLORS["annot"], marker="s", label="Attempts/accepted"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=6.2, loc="center right")

    ax = axes[1, 0]
    top = metric_audit[metric_audit["swap_fraction"].eq(0.01)].copy()
    if top.empty:
        top = metric_audit.copy()
    top = top[~top["metric"].eq("delta_connected_components")].copy()
    top = top.sort_values("mean_abs_delta", ascending=True).tail(7)
    ax.barh(top["metric_label"], top["mean_abs_delta"], color=pub_style.COLORS["model"])
    ax.set_xlabel("Mean absolute shift in share/entropy metrics")
    pub_style.panel_title(ax, "c", "Embedded-structure drift (f = 0.01)")
    pub_style.light_grid(ax, axis="x")

    ax = axes[1, 1]
    frac_colors = {0.005: pub_style.COLORS["geometry_null"], 0.01: pub_style.COLORS["nb"], 0.02: pub_style.COLORS["model"]}
    for frac, group in city.groupby("swap_fraction"):
        ax.scatter(
            group["null_mobility_index_mean"],
            group["road_minus_geometry_mean"],
            s=np.clip(group["n_edges"] / 1400, 8, 40),
            alpha=0.62,
            color=frac_colors.get(round(float(frac), 3), "#7F7F7F"),
            linewidths=0,
            label=f"{frac:g}",
        )
    ax.axhline(0, color="#1A1A1A", lw=0.6)
    ax.set_xlabel("City mean mobility index")
    ax.set_ylabel("Road-minus-geometry residual")
    pub_style.panel_title(ax, "d", "Residual versus null mobility")
    pub_style.light_grid(ax, axis="both")
    ax.legend(title="Accepted-swap target", frameon=False, fontsize=6.2, title_fontsize=6.2)

    pub_style.save(fig, FIG_BASE)
    plt.close(fig)


def write_report(summary: dict) -> None:
    ROUND.mkdir(parents=True, exist_ok=True)
    report = f"""# R80 geometry-null distance audit

Date: 2026-06-08

## Purpose

This round tests whether the strict geometry null could be too close to the original graph. It adds a geometry-null distance audit from archived replicate metrics. It does not claim exact rewired-edge overlap or a full-city intensive ensemble.

## Main diagnostics

- 21-city geometry-null replicate records: {summary['n_r67_replicates']}
- Full 71-city low-intensity records: {summary['n_r68_records']}
- Mean geometry mobility index at 0.005/0.01/0.02: {summary['mobility_by_fraction']}
- Mean road-minus-geometry residual at 0.005/0.01/0.02: {summary['geometry_residual_by_fraction']}
- Mean road-minus-spatial residual at matched targets: {summary['spatial_residual_by_fraction']}

## Claim boundary

The audit supports the statement that accepted geometry-null swaps induce measurable embedded-structure and mobility changes while preserving degree and length-bin fidelity. It does not support exact edge-overlap claims because final rewired edge sets were not archived.
"""
    (ROUND / "gate_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reps = pd.read_csv(R67)
    full = pd.read_csv(R68)
    matched = pd.read_csv(R72)
    metric_audit = build_metric_audit(reps)
    city_audit = build_city_audit(reps)
    fraction = build_fraction_audit(reps, matched)
    build_full71_audit(full)
    make_figure(metric_audit, fraction, city_audit)
    summary = {
        "round": "R80_geometry_distance_audit",
        "n_r67_replicates": int((reps["status"] == "pass").sum()),
        "n_r68_records": int((full["status"] == "pass").sum()),
        "mobility_by_fraction": {str(k): float(v) for k, v in fraction.set_index("swap_fraction")["null_mobility_index_mean"].items()},
        "geometry_residual_by_fraction": {str(k): float(v) for k, v in fraction.set_index("swap_fraction")["road_minus_geometry_mean"].items()},
        "spatial_residual_by_fraction": {str(k): float(v) for k, v in fraction.set_index("swap_fraction")["road_minus_spatial_mean"].items()},
        "figure": str(FIG_BASE.with_suffix(".svg")),
        "claim_boundary": "Distance audit from archived metrics; no exact edge-overlap reconstruction.",
    }
    (OUT / "R80_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
