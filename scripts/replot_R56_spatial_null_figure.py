"""Recreate the public R56 spatial-null figure from derived source data."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "R56_spatial_length_constrained_nulls" / "spatial_length_constrained_nulls_city_summary.csv"
OUT = ROOT / "figures" / "Fig_R56_spatial_length_constrained_nulls_public"


def read_rows(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            parsed: dict[str, float | str] = {"city": row["city"]}
            for key in [
                "road_gap",
                "degree_null_gap_R41",
                "spatial_null_gap",
                "road_minus_spatial",
                "remaining_fraction_after_spatial",
                "accepted_swaps_per_edge_mean",
                "chord_length_bin_tvd_mean",
            ]:
                parsed[key] = float(row[key])
            rows.append(parsed)
    return rows


def main() -> int:
    rows = read_rows(SOURCE)
    cities = [str(r["city"]) for r in rows]
    x = np.arange(len(cities))
    road = np.asarray([float(r["road_gap"]) for r in rows])
    degree = np.asarray([float(r["degree_null_gap_R41"]) for r in rows])
    spatial = np.asarray([float(r["spatial_null_gap"]) for r in rows])
    remaining = np.asarray([float(r["remaining_fraction_after_spatial"]) for r in rows])
    swap = np.asarray([float(r["accepted_swaps_per_edge_mean"]) for r in rows])
    tvd = np.asarray([float(r["chord_length_bin_tvd_mean"]) for r in rows])

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
        }
    )
    colors = {"road": "#313A46", "degree": "#7C8DA6", "spatial": "#3F8F8C", "remaining": "#D9A441", "accent": "#B65C5A"}
    fig = plt.figure(figsize=(8.2, 5.1), constrained_layout=False)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1.0, 1.0], height_ratios=[1.12, 0.88], left=0.075, right=0.985, bottom=0.125, top=0.94, wspace=0.42, hspace=0.62)
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1:])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[1, 2])

    ax0.bar(x, degree, color=colors["degree"], label="Degree null", width=0.68)
    ax0.bar(x, spatial - degree, bottom=degree, color=colors["spatial"], label="+ spatial scale", width=0.68)
    ax0.bar(x, road - spatial, bottom=spatial, color=colors["remaining"], label="Residual", width=0.68)
    ax0.plot(x, road, marker="o", ms=3.2, lw=1.1, color=colors["road"], label="Observed road gap")
    ax0.set_ylabel("Threshold gap relative to CEBH")
    ax0.set_xticks(x)
    ax0.set_xticklabels([c.replace("Buenos Aires", "B. Aires") for c in cities], rotation=45, ha="right")
    ax0.set_title("a  Gap decomposition after a spatial null", loc="left", fontweight="bold")
    ax0.legend(loc="upper left")
    ax0.set_ylim(0, max(float(np.max(road)) * 1.18, 0.05))
    ax0.grid(axis="y", color="#ECECEC", lw=0.6)

    order = np.argsort(spatial)
    for rank, i in enumerate(order):
        alpha = 0.35 + 0.55 * (rank + 1) / len(order)
        ax1.plot([0, 1], [degree[i], spatial[i]], color=colors["spatial"], lw=1.25, alpha=alpha)
    ax1.scatter(np.zeros_like(degree), degree, s=32, color=colors["degree"], edgecolor="white", linewidth=0.6, zorder=3)
    ax1.scatter(np.ones_like(spatial), spatial, s=42, color=colors["spatial"], edgecolor="white", linewidth=0.6, zorder=3)
    label_y = spatial.copy()
    for pos in range(1, len(order)):
        if label_y[order[pos]] - label_y[order[pos - 1]] < 0.0042:
            label_y[order[pos]] = label_y[order[pos - 1]] + 0.0042
    ylim_top = max(float(np.max(spatial)) * 1.20, 0.05)
    overflow = max(0.0, float(np.max(label_y)) - ylim_top * 0.96)
    if overflow > 0:
        label_y[order] -= overflow
    for i in order:
        ax1.plot([1.005, 1.035], [spatial[i], label_y[i]], color="#97A9A7", lw=0.45)
        ax1.text(1.04, label_y[i], cities[i].replace("Buenos Aires", "B. Aires"), va="center", fontsize=6)
    ax1.set_xlim(-0.18, 1.38)
    ax1.set_ylim(0, ylim_top)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Degree\nonly", "Degree +\nspatial scale"])
    ax1.set_ylabel("Null threshold gap")
    ax1.set_title("b  Spatial scale shifts the null threshold", loc="left", fontweight="bold")
    ax1.grid(axis="y", color="#ECECEC", lw=0.6)

    order_res = np.argsort(remaining)[::-1]
    ax2.barh(np.arange(len(cities)), remaining[order_res], color=colors["remaining"], height=0.62)
    ax2.set_yticks(np.arange(len(cities)))
    ax2.set_yticklabels([cities[i].replace("Buenos Aires", "B. Aires") for i in order_res])
    ax2.set_xlim(0, 1.05)
    ax2.set_xlabel("Fraction of road gap not reproduced")
    ax2.set_title("c  Residual after spatial null", loc="left", fontweight="bold")
    ax2.grid(axis="x", color="#ECECEC", lw=0.6)

    ax3.bar([0, 1], [float(np.mean(swap)), float(np.mean(tvd))], color=[colors["spatial"], colors["accent"]], width=0.55, alpha=0.86)
    jitter = np.linspace(-0.14, 0.14, len(cities))
    ax3.scatter(np.zeros_like(swap) + jitter, swap, color=colors["spatial"], s=18, edgecolor="white", linewidth=0.35, zorder=3)
    ax3.scatter(np.ones_like(tvd) + jitter, tvd, color=colors["accent"], s=18, edgecolor="white", linewidth=0.35, zorder=3)
    ax3.text(0.98, 0.95, "degree drift = 0", transform=ax3.transAxes, ha="right", va="top", fontsize=6.5, color="#555B63")
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(["Swaps\nper edge", "Length-bin\nTVD"])
    ax3.set_ylim(0, max(float(np.max(swap)), float(np.max(tvd))) * 1.24)
    ax3.set_ylabel("Diagnostic value")
    ax3.set_title("d  Null diagnostics", loc="left", fontweight="bold")
    ax3.grid(axis="y", color="#ECECEC", lw=0.6)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUT}.png", dpi=450, bbox_inches="tight")
    fig.savefig(f"{OUT}.svg", bbox_inches="tight")
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight")
    print(f"Wrote {OUT}.png/.svg/.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
