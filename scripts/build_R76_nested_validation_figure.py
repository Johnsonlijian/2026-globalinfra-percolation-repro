"""Build the nested-validation and public-control release figure.

This figure replaces the older public-control-only figure in the public
source-data bundle. It uses existing derived source tables only; no raw OSM,
GHSL or WDI data are redistributed.

R101 renumbering: in the staged manuscript package this figure became the
urban-form validation figure. The R101 restyle also removed the former
dashboard-style "claim boundary"
text panel; claim boundaries live in the caption and main text instead.
"""

from __future__ import annotations

import hashlib
import csv
import json
import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Keep mapped-drive paths unexpanded on Windows.  Resolving W: to a long UNC
# path can make pandas fail when writing long source-data filenames.
ROOT = Path(__file__).parent.parent
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import pub_style  # noqa: E402

OUT = ROOT / "data" / "R76_nested_validation_figure"
FIG_ROOT = ROOT / "figures" / "Fig_R76_nested_validation"

PUBLIC_SD = ROOT / "source_data"
COMBINED_SOURCE_NAME = "Fig4_combined_nested_public_source_data.csv"

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
    "road_gap": "#0072B2",
    "road_minus_spatial": "#009E73",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_style() -> None:
    pub_style.apply()


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

    # R101: the dashboard-style claim-boundary panel was removed from the
    # figure; claim boundaries are stated in the caption and main text.
    return pd.DataFrame(rows)


def make_figure(
    nested: pd.DataFrame,
    public: pd.DataFrame,
    corr: pd.DataFrame,
    preds: pd.DataFrame,
    region: pd.DataFrame,
) -> None:
    set_style()
    fig = plt.figure(figsize=(pub_style.FIG_WIDTH_2COL, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.0, 0.72])

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
        ax_a.plot(x + offset, lro, "o", color="#1A1A1A", ms=3.2)
    ax_a.axhline(0, color="#1A1A1A", lw=0.7)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax_a.set_ylabel("Cross-validated $R^2$")
    pub_style.panel_title(ax_a, "a", "Nested feature ladder")
    pub_style.light_grid(ax_a, axis="y")
    handles, labels = ax_a.get_legend_handles_labels()
    handles.append(
        mpl.lines.Line2D([], [], color="#1A1A1A", marker="o", ms=3.2, lw=0, label="Leave-region-out")
    )
    labels.append("Leave-region-out")
    ax_a.legend(handles, labels, fontsize=6.0, loc="upper left")

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
    ax_b.axhline(0, color="#1A1A1A", lw=0.7)
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels(["Public\nonly", "Road\nform", "Road +\npublic"])
    ax_b.set_ylabel("Five-fold CV $R^2$")
    pub_style.panel_title(ax_b, "b", "Road form versus public controls")
    pub_style.light_grid(ax_b, axis="y")

    ax_c = fig.add_subplot(gs[1, 0])
    corr_work = corr[(corr["target"] == "road_gap") & (corr["n"] >= 60)].copy()
    corr_work = corr_work.sort_values("abs_spearman_rho", ascending=True).tail(7)
    yc = np.arange(len(corr_work))
    bar_colors = ["#0072B2" if v < 0 else "#D55E00" for v in corr_work["spearman_rho"]]
    ax_c.barh(yc, corr_work["spearman_rho"], color=bar_colors, alpha=0.9)
    ax_c.axvline(0, color="#1A1A1A", lw=0.7)
    ax_c.set_yticks(yc)
    ax_c.set_yticklabels([clean_covariate_label(c) for c in corr_work["covariate"]], fontsize=6)
    ax_c.set_xlabel(r"Spearman $\rho$ with road-minus-CEBH gap")
    pub_style.panel_title(ax_c, "c", "High-order embedded correlations")
    pub_style.light_grid(ax_c, axis="x")

    ax_d = fig.add_subplot(gs[1, 1])
    for target in TARGET_LABELS:
        sub = preds[
            (preds["target"] == target)
            & (preds["model_group"] == "M3_embedded_form")
        ].copy()
        ax_d.scatter(
            sub["observed"],
            sub["predicted_leave_region_out"],
            s=22,
            color=TARGET_COLORS[target],
            alpha=0.78,
            edgecolor="white",
            linewidth=0.25,
            label=TARGET_LABELS[target],
        )
    lo = min(preds["observed"].min(), preds["predicted_leave_region_out"].min())
    hi = max(preds["observed"].max(), preds["predicted_leave_region_out"].max())
    pad = (hi - lo) * 0.05
    ax_d.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls="--", color="#1A1A1A", lw=0.7)
    ax_d.set_xlim(lo - pad, hi + pad)
    ax_d.set_ylim(lo - pad, hi + pad)
    ax_d.set_xlabel("Observed residual or gap")
    ax_d.set_ylabel("Leave-region-out prediction")
    pub_style.panel_title(ax_d, "d", "Leave-region-out predictions (M3)")
    pub_style.light_grid(ax_d, axis="both")
    ax_d.legend(fontsize=6.0, loc="upper left")

    ax_e = fig.add_subplot(gs[2, :])
    reg = region[
        (region["target"] == "road_minus_spatial")
        & (region["model_group"] == "M3_embedded_form")
    ].copy()
    reg = reg.sort_values("leave_region_out_mae_region", ascending=True)
    ye = np.arange(len(reg))
    ax_e.barh(ye, reg["leave_region_out_mae_region"], color="#009E73", alpha=0.9, height=0.62)
    ax_e.set_yticks(ye)
    ax_e.set_yticklabels(reg["macro_region"], fontsize=6.2)
    ax_e.set_xlabel("Post-spatial residual leave-region-out MAE")
    pub_style.panel_title(ax_e, "e", "Regional holdout error")
    pub_style.light_grid(ax_e, axis="x")

    pub_style.save(fig, FIG_ROOT)
    svg_path = FIG_ROOT.with_suffix(".svg")
    if svg_path.exists():
        svg_path.write_text(
            "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
            encoding="utf-8",
        )
    plt.close(fig)


def update_manifest_for_source_data(sd: Path, files: list[tuple[Path, str]]) -> None:
    manifest = sd / "Supplementary_Data_1_source_data_manifest.csv"
    df = pd.read_csv(manifest)
    for src, copied_name in files:
        copied = f"figures/{copied_name}"
        entry = {
            "round": "R76",
            "source_role": "Nested validation and public-control source data",
            "source_path": f"data/R76_nested_validation_figure/{src.name}",
            "copied_file": copied,
            "format": src.suffix.lstrip("."),
            "n_rows": int(pd.read_csv(src).shape[0]) if src.suffix == ".csv" else 1,
            "n_columns_or_keys": int(pd.read_csv(src).shape[1]) if src.suffix == ".csv" else 1,
            "sha256": sha256_file(src),
            "redistribution_boundary": "derived project table; raw OSM/Geofabrik archives excluded",
        }
        mask = df["copied_file"] == copied
        if mask.any():
            for key, value in entry.items():
                df.loc[mask, key] = value
        else:
            df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    df.to_csv(manifest, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL, lineterminator="\n")


def copy_outputs() -> None:
    combined_src = OUT / "Fig4_combined_nested_public_source_data.csv"
    sd_fig_dir = PUBLIC_SD / "figures"
    sd_fig_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(combined_src, sd_fig_dir / COMBINED_SOURCE_NAME)
    update_manifest_for_source_data(PUBLIC_SD, [(combined_src, COMBINED_SOURCE_NAME)])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tables = PUBLIC_SD / "tables"
    nested = pd.read_csv(tables / "R73_nested_model_summary.csv")
    preds = pd.read_csv(tables / "R73_nested_model_predictions.csv")
    region = pd.read_csv(tables / "R73_leave_region_out_region_summary.csv")
    public = pd.read_csv(tables / "R65_R65_model_comparison.csv")
    corr = pd.read_csv(tables / "R63_high_order_external_correlations.csv")

    combined = make_combined_source_data(nested, public, corr, region)
    combined.to_csv(OUT / "Fig4_combined_nested_public_source_data.csv", index=False)
    make_figure(nested, public, corr, preds, region)
    copy_outputs()

    payload = {
        "status": "pass",
        "round": "R76_nested_validation_figure",
        "figure": FIG_ROOT.with_suffix(".svg").relative_to(ROOT).as_posix(),
        "target_figure_name": "urban-form validation figure",
        "source_data": (OUT / "Fig4_combined_nested_public_source_data.csv").relative_to(ROOT).as_posix(),
        "claim_boundary": (
            "The validation figure integrates nested validation, public-control comparison, "
            "high-order embedded-feature correlations and leave-region-out checks; claim "
            "boundaries are stated in the caption and main text, not as a figure panel."
        ),
    }
    (OUT / "R76_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
