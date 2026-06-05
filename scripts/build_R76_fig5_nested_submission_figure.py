"""Build submission Fig. 5 with nested validation and public-control checks.

This figure replaces the older public-control-only Fig. 5 in the target
submission packages. It uses existing derived source tables only; no raw OSM,
GHSL or WDI data are redistributed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "R76_fig5_nested_submission"
FIG_ROOT = ROOT / "figures" / "Fig_R76_fig5_nested_submission"

TARGETS = [
    ROOT / "submission" / "communications_physics" / "target_submission",
    ROOT / "submission" / "npj_complexity" / "target_submission",
]
CP_SOURCE = TARGETS[0] / "source_data"

MODEL_ORDER = [
    "M0_size",
    "M1_degree",
    "M2_degree_edge_scale",
    "M3_embedded_form",
    "M4_embedded_form_public_controls",
]
MODEL_LABELS = {
    "M0_size": "M0\nsize",
    "M1_degree": "M1\ndegree",
    "M2_degree_edge_scale": "M2\n+ scale",
    "M3_embedded_form": "M3\nembedded",
    "M4_embedded_form_public_controls": "M4\n+ public",
}
TARGET_LABELS = {
    "road_gap": "Road minus CEBH",
    "road_minus_spatial": "Post-spatial residual",
}
TARGET_COLORS = {
    "road_gap": "#4E79A7",
    "road_minus_spatial": "#59A14F",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
        }
    )


def clean_covariate_label(name: str) -> str:
    return (
        name.replace("_share", "")
        .replace("_fraction", "")
        .replace("_node", "")
        .replace("_edge", "")
        .replace("_10deg", "")
        .replace("_", " ")
        .replace("dual degree mean", "dual-degree mean")
        .replace("largest biconnected", "largest biconnected frac.")
        .replace("articulation point", "articulation points")
        .replace("straight angle", "straight angles")
    )


def make_combined_source_data(
    nested: pd.DataFrame,
    public: pd.DataFrame,
    corr: pd.DataFrame,
    region: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for _, row in nested.iterrows():
        rows.append(
            {
                "panel": "a",
                "record_type": "nested_model_ladder",
                "target": row["target"],
                "model_group": row["model_group"],
                "model_label": row["model_label"],
                "metric": "cv5_r2",
                "value": row["cv5_r2"],
                "n": row["n"],
            }
        )
        rows.append(
            {
                "panel": "a",
                "record_type": "nested_model_ladder",
                "target": row["target"],
                "model_group": row["model_group"],
                "model_label": row["model_label"],
                "metric": "leave_region_out_r2",
                "value": row["leave_region_out_r2"],
                "n": row["n"],
            }
        )

    public_map = {
        "graph_only": "road_form_only",
        "ghsl_only": "public_only",
        "wdi_only": "public_only",
        "graph_plus_ghsl_wdi": "road_form_plus_public",
    }
    public_work = public[public["model_group"].isin(public_map)].copy()
    public_work["display_group"] = public_work["model_group"].map(public_map)
    public_best = (
        public_work.sort_values("cv_r2", ascending=False)
        .groupby(["target", "display_group"], as_index=False)
        .first()
    )
    for _, row in public_best.iterrows():
        rows.append(
            {
                "panel": "b",
                "record_type": "public_control_screen",
                "target": row["target"],
                "model_group": row["display_group"],
                "source_model_group": row["model_group"],
                "metric": "cv_r2",
                "value": row["cv_r2"],
                "n": row["n"],
            }
        )

    corr_work = corr[(corr["target"] == "road_gap") & (corr["n"] >= 60)].copy()
    corr_work = corr_work.sort_values("abs_spearman_rho", ascending=False).head(7)
    for _, row in corr_work.iterrows():
        rows.append(
            {
                "panel": "c",
                "record_type": "high_order_correlation",
                "target": row["target"],
                "covariate": row["covariate"],
                "metric": "spearman_rho",
                "value": row["spearman_rho"],
                "n": row["n"],
            }
        )

    region_work = region[
        (region["target"] == "road_minus_spatial")
        & (region["model_group"] == "M3_embedded_form")
    ].copy()
    for _, row in region_work.iterrows():
        rows.append(
            {
                "panel": "e",
                "record_type": "leave_region_out_region_error",
                "target": row["target"],
                "model_group": row["model_group"],
                "macro_region": row["macro_region"],
                "metric": "leave_region_out_mae_region",
                "value": row["leave_region_out_mae_region"],
                "n": row["n_region"],
            }
        )

    rows.extend(
        [
            {
                "panel": "f",
                "record_type": "claim_boundary",
                "metric": "full_city_window_count",
                "value": 71,
                "note": "Standardized 10-km-radius centre windows, not whole metropolitan systems.",
            },
            {
                "panel": "f",
                "record_type": "claim_boundary",
                "metric": "strict_geometry_full_city_replicates",
                "value": 1,
                "note": "One low-intensity strict geometry-null record per city at accepted-swap fraction 0.005.",
            },
            {
                "panel": "f",
                "record_type": "claim_boundary",
                "metric": "causal_design_rule",
                "value": 0,
                "note": "Nested validation is descriptive prediction, not causal urban-design inference.",
            },
            {
                "panel": "f",
                "record_type": "claim_boundary",
                "metric": "exact_edge_overlap_available",
                "value": 0,
                "note": "Existing null summaries do not store final edge sets; mobility/fidelity proxies are reported instead.",
            },
        ]
    )
    return pd.DataFrame(rows)


def make_figure(
    nested: pd.DataFrame,
    public: pd.DataFrame,
    corr: pd.DataFrame,
    preds: pd.DataFrame,
    region: pd.DataFrame,
) -> None:
    set_style()
    fig = plt.figure(figsize=(7.35, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.0, 0.9])

    ax_a = fig.add_subplot(gs[0, 0])
    x = np.arange(len(MODEL_ORDER))
    width = 0.34
    for i, target in enumerate(TARGET_LABELS):
        sub = nested[nested["target"] == target].set_index("model_group")
        cv = [sub.loc[m, "cv5_r2"] for m in MODEL_ORDER]
        lro = [sub.loc[m, "leave_region_out_r2"] for m in MODEL_ORDER]
        offset = (i - 0.5) * width
        ax_a.bar(
            x + offset,
            cv,
            width,
            color=TARGET_COLORS[target],
            alpha=0.86,
            label=f"{TARGET_LABELS[target]} CV",
        )
        ax_a.plot(x + offset, lro, "o", color="#222222", ms=3.5)
    ax_a.axhline(0, color="#222222", lw=0.75)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax_a.set_ylabel("Cross-validated $R^2$")
    ax_a.set_title("a  Nested feature ladder", loc="left")
    ax_a.grid(axis="y", color="#E6E6E6", lw=0.35)
    ax_a.legend(fontsize=6, loc="upper left")
    ax_a.text(0.02, 0.03, "black dots: leave-region-out", transform=ax_a.transAxes, fontsize=6)

    ax_b = fig.add_subplot(gs[0, 1])
    public_map = {
        "graph_only": "road form",
        "ghsl_only": "public only",
        "wdi_only": "public only",
        "graph_plus_ghsl_wdi": "road + public",
    }
    work = public[public["model_group"].isin(public_map)].copy()
    work["display_group"] = work["model_group"].map(public_map)
    work = work.sort_values("cv_r2", ascending=False).groupby(["target", "display_group"], as_index=False).first()
    order = ["public only", "road form", "road + public"]
    xb = np.arange(len(order))
    for i, target in enumerate(TARGET_LABELS):
        sub = work[work["target"] == target].set_index("display_group")
        vals = [sub.loc[g, "cv_r2"] for g in order]
        ax_b.bar(xb + (i - 0.5) * width, vals, width, color=TARGET_COLORS[target], alpha=0.86)
    ax_b.axhline(0, color="#222222", lw=0.75)
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels(["Public\nonly", "Road\nform", "Road +\npublic"])
    ax_b.set_ylabel("Five-fold CV $R^2$")
    ax_b.set_title("b  Public controls do not replace road form", loc="left")
    ax_b.grid(axis="y", color="#E6E6E6", lw=0.35)

    ax_c = fig.add_subplot(gs[1, 0])
    corr_work = corr[(corr["target"] == "road_gap") & (corr["n"] >= 60)].copy()
    corr_work = corr_work.sort_values("abs_spearman_rho", ascending=True).tail(7)
    yc = np.arange(len(corr_work))
    bar_colors = ["#4E79A7" if v < 0 else "#E15759" for v in corr_work["spearman_rho"]]
    ax_c.barh(yc, corr_work["spearman_rho"], color=bar_colors, alpha=0.9)
    ax_c.axvline(0, color="#222222", lw=0.75)
    ax_c.set_yticks(yc)
    ax_c.set_yticklabels([clean_covariate_label(c) for c in corr_work["covariate"]], fontsize=6)
    ax_c.set_xlabel("Spearman rho with road-minus-CEBH gap")
    ax_c.set_title("c  Embedded graph features carry the signal", loc="left")
    ax_c.grid(axis="x", color="#E6E6E6", lw=0.35)

    ax_d = fig.add_subplot(gs[1, 1])
    for target in TARGET_LABELS:
        sub = preds[
            (preds["target"] == target)
            & (preds["model_group"] == "M3_embedded_form")
        ].copy()
        ax_d.scatter(
            sub["observed"],
            sub["predicted_leave_region_out"],
            s=24,
            color=TARGET_COLORS[target],
            alpha=0.78,
            edgecolor="#222222",
            linewidth=0.25,
            label=TARGET_LABELS[target],
        )
    lo = min(preds["observed"].min(), preds["predicted_leave_region_out"].min())
    hi = max(preds["observed"].max(), preds["predicted_leave_region_out"].max())
    pad = (hi - lo) * 0.05
    ax_d.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls="--", color="#222222", lw=0.75)
    ax_d.set_xlim(lo - pad, hi + pad)
    ax_d.set_ylim(lo - pad, hi + pad)
    ax_d.set_xlabel("Observed")
    ax_d.set_ylabel("Leave-region-out prediction")
    ax_d.set_title("d  M3 predicts held-out regions", loc="left")
    ax_d.grid(color="#E6E6E6", lw=0.35)
    ax_d.legend(fontsize=6, loc="upper left")

    ax_e = fig.add_subplot(gs[2, 0])
    reg = region[
        (region["target"] == "road_minus_spatial")
        & (region["model_group"] == "M3_embedded_form")
    ].copy()
    reg = reg.sort_values("leave_region_out_mae_region", ascending=True)
    ye = np.arange(len(reg))
    ax_e.barh(ye, reg["leave_region_out_mae_region"], color="#7A9E59", alpha=0.9)
    ax_e.set_yticks(ye)
    ax_e.set_yticklabels(reg["macro_region"], fontsize=6)
    ax_e.set_xlabel("Post-spatial residual holdout MAE")
    ax_e.set_title("e  Regional error remains heterogeneous", loc="left")
    ax_e.grid(axis="x", color="#E6E6E6", lw=0.35)

    ax_f = fig.add_subplot(gs[2, 1])
    ax_f.axis("off")
    ax_f.set_title("f  Claim boundary", loc="left", pad=5)
    boundary = [
        ("71", "standardized 10-km-radius centre windows"),
        ("0.005", "low-intensity full-city geometry-null target"),
        ("1/city", "full-city strict-geometry replicate count"),
        ("no", "causal urban-design rule or planar theorem"),
        ("no", "exact edge overlap from archived summaries"),
    ]
    y0 = 0.86
    for i, (num, txt) in enumerate(boundary):
        y = y0 - i * 0.18
        ax_f.text(0.02, y, num, fontsize=12, fontweight="bold", color="#222222", transform=ax_f.transAxes)
        ax_f.text(0.22, y + 0.01, txt, fontsize=7, color="#222222", transform=ax_f.transAxes, va="center")

    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"dpi": 600} if ext in {"png", "tiff"} else {}
        fig.savefig(FIG_ROOT.with_suffix(f".{ext}"), bbox_inches="tight", **kwargs)
    plt.close(fig)


def update_manifest_for_target(target: Path, files: list[Path]) -> None:
    manifest = target / "source_data" / "Supplementary_Data_1_source_data_manifest.csv"
    df = pd.read_csv(manifest)
    for src in files:
        copied = f"figures/{src.name}"
        entry = {
            "round": "R76",
            "source_role": "Fig5 nested validation and public-control source data",
            "source_path": f"data/R76_fig5_nested_submission/{src.name}",
            "copied_file": copied,
            "format": src.suffix.lstrip("."),
            "n_rows": int(pd.read_csv(src).shape[0]) if src.suffix == ".csv" else 1,
            "n_columns_or_keys": int(pd.read_csv(src).shape[1]) if src.suffix == ".csv" else 1,
            "sha256": sha256_file(src),
            "redistribution_boundary": "derived project table; raw OSM/Geofabrik archives excluded",
        }
        df = df[df["copied_file"] != copied]
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    df.to_csv(manifest, index=False)


def copy_outputs() -> None:
    generated = [FIG_ROOT.with_suffix(ext) for ext in [".svg", ".pdf", ".png", ".tiff"]]
    source_files = [OUT / "Fig5_combined_nested_public_source_data.csv"]
    for target in TARGETS:
        fig_dir = target / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        for src in generated:
            shutil.copy2(src, fig_dir / f"Fig5_public_controls_boundary_checks{src.suffix}")
        sd_fig_dir = target / "source_data" / "figures"
        sd_fig_dir.mkdir(parents=True, exist_ok=True)
        for src in source_files:
            shutil.copy2(src, sd_fig_dir / src.name)
        update_manifest_for_target(target, source_files)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tables = CP_SOURCE / "tables"
    nested = pd.read_csv(tables / "R73_nested_model_summary.csv")
    preds = pd.read_csv(tables / "R73_nested_model_predictions.csv")
    region = pd.read_csv(tables / "R73_leave_region_out_region_summary.csv")
    public = pd.read_csv(tables / "R65_R65_model_comparison.csv")
    corr = pd.read_csv(tables / "R63_high_order_external_correlations.csv")

    combined = make_combined_source_data(nested, public, corr, region)
    combined.to_csv(OUT / "Fig5_combined_nested_public_source_data.csv", index=False)
    make_figure(nested, public, corr, preds, region)
    copy_outputs()

    payload = {
        "status": "pass",
        "round": "R76_fig5_nested_submission",
        "figure": str(FIG_ROOT.with_suffix(".svg")),
        "target_figure_name": "Fig5_public_controls_boundary_checks",
        "source_data": str(OUT / "Fig5_combined_nested_public_source_data.csv"),
        "claim_boundary": (
            "Fig5 integrates nested validation, public-control comparison, high-order "
            "embedded-feature correlations and explicit non-causal/exact-overlap boundaries."
        ),
    }
    (OUT / "R76_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
