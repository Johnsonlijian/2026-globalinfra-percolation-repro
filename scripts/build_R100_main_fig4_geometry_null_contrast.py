"""Build the R100 main geometry-null contrast figure (manuscript Fig. 5).

This figure promotes the low-overlap N99 audit from supplementary-only evidence
into the main geometry-null storyline.  It combines four reviewer-facing checks:
full 71-city coverage, 21-city matched-intensity contrast, R90 fine-grid
regeneration and the 12-city low-overlap strict-geometry surrogate audit.

R101 renumbering: the geometry contrast is manuscript Fig. 5 so that first
in-text figure citations remain sequential (urban-form validation is Fig. 4).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import pub_style  # noqa: E402

OUT_DIR = ROOT / "figures"
SUBMISSION_FIG_DIR = ROOT / "submission" / "npj_complexity" / "target_submission" / "figures"
SOURCE_OUT = ROOT / "data" / "R100_main_figure_hardening"
FIG_STEM = OUT_DIR / "Fig_R100_geometry_null_contrast"
TARGET_STEM = SUBMISSION_FIG_DIR / "Fig5_geometry_null_contrast"

R68 = ROOT / "data" / "R68_full71_geometry_null_ensemble" / "full71_geometry_null_city_summary.csv"
R81 = ROOT / "data" / "R81_full71_high_intensity_geometry_null_ensemble" / "full71_geometry_null_city_summary.csv"
R72 = ROOT / "data" / "R72_geometry_defense" / "matched_intensity_geometry_vs_spatial_city.csv"
R90 = ROOT / "data" / "R90_finegrid_edge_audit" / "R90_finegrid_null_thresholds.csv"
N99 = ROOT / "data" / "N99_low_overlap_geometry_surrogate" / "N99_low_overlap_surrogate_records.csv"


def jitter(center: float, n: int, seed: int, scale: float = 0.045) -> np.ndarray:
    return np.random.default_rng(seed).normal(center, scale, n)


def add_dot_box(ax: plt.Axes, values: list[np.ndarray], labels: list[str], colors: list[str], seed: int) -> None:
    positions = np.arange(1, len(values) + 1)
    bp = ax.boxplot(values, positions=positions, widths=0.52, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set(facecolor=color, alpha=0.18, edgecolor=color, linewidth=1.0)
    for key in ["whiskers", "caps", "medians"]:
        for artist in bp[key]:
            artist.set(color="#334155", linewidth=0.9)
    for idx, vals in enumerate(values, start=1):
        ax.scatter(jitter(idx, len(vals), seed + idx), vals, s=13, color=colors[idx - 1], alpha=0.45, linewidths=0)
        ax.hlines(float(np.nanmean(vals)), idx - 0.25, idx + 0.25, color=colors[idx - 1], lw=2.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.axhline(0, color="#475569", lw=0.8, ls="--")


def build_source_table() -> pd.DataFrame:
    r68 = pd.read_csv(R68)
    r81 = pd.read_csv(R81)
    r72 = pd.read_csv(R72)
    r90 = pd.read_csv(R90)
    n99 = pd.read_csv(N99)
    rows: list[dict[str, object]] = []
    rows.append(
        {
            "panel": "a",
            "source_file": R68.relative_to(ROOT).as_posix(),
            "series": "71-city spatial residual",
            "n": int(r68["road_minus_spatial"].notna().sum()),
            "mean": float(r68["road_minus_spatial"].mean()),
            "median": float(r68["road_minus_spatial"].median()),
        }
    )
    rows.append(
        {
            "panel": "a",
            "source_file": R68.relative_to(ROOT).as_posix(),
            "series": "71-city geometry residual f=0.005",
            "n": int(r68["road_minus_geometry_mean"].notna().sum()),
            "mean": float(r68["road_minus_geometry_mean"].mean()),
            "median": float(r68["road_minus_geometry_mean"].median()),
        }
    )
    for frac, sub in r81.groupby("swap_fraction"):
        rows.append(
            {
                "panel": "a",
                "source_file": R81.relative_to(ROOT).as_posix(),
                "series": f"71-city geometry residual f={float(frac):.3f}",
                "n": int(sub["road_minus_geometry_mean"].notna().sum()),
                "mean": float(sub["road_minus_geometry_mean"].mean()),
                "median": float(sub["road_minus_geometry_mean"].median()),
            }
        )
    for frac, sub in r72.groupby("swap_fraction"):
        rows.append(
            {
                "panel": "b",
                "source_file": R72.relative_to(ROOT).as_posix(),
                "series": f"21-city matched spatial f={float(frac):.3f}",
                "n": int(sub["road_minus_spatial"].notna().sum()),
                "mean": float(sub["road_minus_spatial"].mean()),
                "median": float(sub["road_minus_spatial"].median()),
            }
        )
        rows.append(
            {
                "panel": "b",
                "source_file": R72.relative_to(ROOT).as_posix(),
                "series": f"21-city matched geometry f={float(frac):.3f}",
                "n": int(sub["road_minus_geometry_mean"].notna().sum()),
                "mean": float(sub["road_minus_geometry_mean"].mean()),
                "median": float(sub["road_minus_geometry_mean"].median()),
            }
        )
    for null_type, sub in r90.groupby("null_type"):
        rows.append(
            {
                "panel": "c",
                "source_file": R90.relative_to(ROOT).as_posix(),
                "series": f"R90 fine-grid {null_type}",
                "n": int(sub["road_minus_null_fine"].notna().sum()),
                "mean": float(sub["road_minus_null_fine"].mean()),
                "median": float(sub["road_minus_null_fine"].median()),
            }
        )
    rows.append(
        {
            "panel": "d",
            "source_file": N99.relative_to(ROOT).as_posix(),
            "series": "N99 low-overlap residual",
            "n": int(len(n99)),
            "mean": float(n99["road_minus_geometry_fine"].mean()),
            "median": float(n99["road_minus_geometry_fine"].median()),
            "mean_edge_jaccard": float(n99["edge_jaccard"].mean()),
            "n_below_090": int((n99["edge_jaccard"] < 0.90).sum()),
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_FIG_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_OUT.mkdir(parents=True, exist_ok=True)

    r68 = pd.read_csv(R68)
    r81 = pd.read_csv(R81)
    r72 = pd.read_csv(R72)
    r90 = pd.read_csv(R90)
    n99 = pd.read_csv(N99)

    pub_style.apply()
    blue = pub_style.COLORS["geometry_null"]
    orange = pub_style.COLORS["spatial_null"]
    green = pub_style.COLORS["model"]
    purple = pub_style.COLORS["accent"]
    slate = pub_style.COLORS["annot"]

    fig, axes = plt.subplots(2, 2, figsize=(pub_style.FIG_WIDTH_2COL, 6.1), constrained_layout=True)

    ax = axes[0, 0]
    vals = [
        r68["road_minus_spatial"].to_numpy(float),
        r68["road_minus_geometry_mean"].to_numpy(float),
    ]
    labels = ["Spatial\n71-city", "Geometry\nf=0.005"]
    colors = [orange, blue]
    for frac in [0.01, 0.02]:
        sub = r81[r81["swap_fraction"].round(3).eq(frac)]
        vals.append(sub["road_minus_geometry_mean"].to_numpy(float))
        labels.append(f"Geometry\nf={frac:.2f}")
        colors.append(green if frac == 0.01 else purple)
    add_dot_box(ax, vals, labels, colors, 100)
    ax.set_ylabel("Road-minus-null threshold residual")
    pub_style.panel_title(ax, "a", "Full 71-city geometry-null coverage")
    pub_style.annot(ax, 0.02, 0.97, "n = 71 windows per group")

    ax = axes[0, 1]
    x_base = np.arange(3)
    fracs = [0.005, 0.01, 0.02]
    width = 0.28
    for i, frac in enumerate(fracs):
        sub = r72[np.isclose(r72["swap_fraction"], frac)]
        ax.scatter(
            jitter(i - width, len(sub), 200 + i, 0.025),
            sub["road_minus_spatial"],
            s=14,
            color=orange,
            alpha=0.45,
            linewidths=0,
        )
        ax.scatter(
            jitter(i + width, len(sub), 240 + i, 0.025),
            sub["road_minus_geometry_mean"],
            s=14,
            color=blue,
            alpha=0.50,
            linewidths=0,
        )
        ax.hlines(sub["road_minus_spatial"].mean(), i - width - 0.12, i - width + 0.12, color=orange, lw=2)
        ax.hlines(sub["road_minus_geometry_mean"].mean(), i + width - 0.12, i + width + 0.12, color=blue, lw=2)
    ax.axhline(0, color="#475569", lw=0.8, ls="--")
    ax.set_xticks(x_base)
    ax.set_xticklabels([f"{f:.3f}" if f < 0.01 else f"{f:.2f}" for f in fracs])
    ax.set_xlabel("Accepted-swap target")
    ax.set_ylabel("Road-minus-null threshold residual")
    pub_style.panel_title(ax, "b", "Matched-intensity contrast (21 cities)")
    ax.scatter([], [], color=orange, label="Spatial-scale null")
    ax.scatter([], [], color=blue, label="Strict geometry null")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")

    ax = axes[1, 0]
    r90_sp = r90[r90["null_type"].eq("spatial")]["road_minus_null_fine"].to_numpy(float)
    r90_geo = r90[r90["null_type"].eq("geometry")]["road_minus_null_fine"].to_numpy(float)
    add_dot_box(ax, [r90_sp, r90_geo], ["Spatial\nfine-grid", "Geometry\nfine-grid"], [orange, blue], 300)
    sep = r90_sp.mean() - r90_geo.mean()
    ax.set_ylabel("Fine-grid road-minus-null residual")
    pub_style.panel_title(ax, "c", "Fine-grid regenerated nulls (21 cities)")
    pub_style.annot(ax, 0.05, 0.95, f"Mean separation = {sep:.4f}")

    ax = axes[1, 1]
    frac_colors = {0.04: purple, 0.08: green}
    for frac, sub in n99.groupby("target_swap_fraction"):
        ax.scatter(
            sub["edge_jaccard"],
            sub["road_minus_geometry_fine"],
            s=28 + 80 * (sub["accepted_swaps_per_edge"] / sub["accepted_swaps_per_edge"].max()),
            color=frac_colors.get(round(float(frac), 2), slate),
            alpha=0.62,
            linewidths=0,
            label=f"f={float(frac):.2f}",
        )
    ax.axhline(0, color="#475569", lw=0.8, ls="--")
    ax.axvline(0.90, color="#B45309", lw=1.0, ls="--")
    ax.set_xlabel("Exact edge Jaccard with road graph")
    ax.set_ylabel("Road-minus-geometry residual")
    pub_style.panel_title(ax, "d", "Low-overlap audit (12 cities)")
    pub_style.annot(
        ax,
        0.03,
        0.96,
        f"{int((n99['edge_jaccard'] < 0.90).sum())}/{len(n99)} records below J = 0.90; mean J = {n99['edge_jaccard'].mean():.3f}",
    )
    pub_style.annot(ax, 0.97, 0.06, "lower J = less near-copy overlap", ha="right", va="bottom")
    ax.legend(frameon=False, fontsize=6.5, loc="lower left")

    pub_style.save(fig, FIG_STEM)
    for suffix in [".svg", ".pdf", ".png", ".tiff"]:
        shutil.copy2(FIG_STEM.with_suffix(suffix), TARGET_STEM.with_suffix(suffix))
    plt.close(fig)

    source = build_source_table()
    source.to_csv(SOURCE_OUT / "R100_geometry_null_contrast_source_data.csv", index=False)
    print(source.to_string(index=False))


if __name__ == "__main__":
    main()
