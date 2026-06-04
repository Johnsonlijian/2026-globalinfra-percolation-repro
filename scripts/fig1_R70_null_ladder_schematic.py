"""Build Fig. 1 R70 null-ladder schematic for the road percolation paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

def find_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "figures").exists() and ((parent / "paper_figures").exists() or (parent / "README.md").exists()):
            return parent
    return current.parents[2]


ROOT = find_root()
PAPER_OUT = (ROOT / "paper_figures" / "output") if (ROOT / "paper_figures").exists() else None
FIG_DIR = ROOT / "figures"
BASE_NAME = "Fig_R70_null_ladder_schematic"


def setup() -> None:
    folders = [FIG_DIR]
    if PAPER_OUT is not None:
        folders.extend([PAPER_OUT / "svg", PAPER_OUT / "pdf", PAPER_OUT / "png"])
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.2,
            "axes.linewidth": 0.7,
        }
    )


def rounded_box(ax, xy, width, height, text, fc, ec="#2f3542", fontsize=7.0, weight="normal"):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=0.9,
        edgecolor=ec,
        facecolor=fc,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color="#20242a",
        linespacing=1.15,
    )
    return box


def arrow(ax, start, end, color="#4a5568", lw=1.1, rad=0.0):
    arr = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        clip_on=False,
    )
    ax.add_patch(arr)
    return arr


def draw_mini_road(ax, x0, y0, scale=1.0):
    pts = {
        "a": (x0, y0),
        "b": (x0 + 0.08 * scale, y0 + 0.08 * scale),
        "c": (x0 + 0.16 * scale, y0 + 0.03 * scale),
        "d": (x0 + 0.06 * scale, y0 - 0.06 * scale),
        "e": (x0 + 0.18 * scale, y0 - 0.08 * scale),
        "f": (x0 + 0.25 * scale, y0 + 0.02 * scale),
        "g": (x0 + 0.28 * scale, y0 - 0.10 * scale),
        "h": (x0 + 0.12 * scale, y0 - 0.14 * scale),
    }
    edges = [("a", "b"), ("b", "c"), ("a", "d"), ("d", "e"), ("c", "f"), ("f", "g"), ("e", "g"), ("d", "h"), ("h", "e"), ("b", "d")]
    for u, v in edges:
        ax.plot([pts[u][0], pts[v][0]], [pts[u][1], pts[v][1]], transform=ax.transAxes, color="#2b3037", lw=1.35, solid_capstyle="round")
    for p in pts.values():
        ax.add_patch(Circle(p, 0.0075 * scale, transform=ax.transAxes, facecolor="#ffffff", edgecolor="#2b3037", lw=0.8))


def draw_ladder(ax, x, y, rows, title, width=0.255):
    ax.text(x, y + 0.245, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.2, fontweight="bold", color="#15191f", linespacing=1.05)
    for idx, row in enumerate(rows):
        yy = y + 0.16 - idx * 0.092
        box_text = f"{row['label']}\n{row['value']}" if row.get("value") else row["label"]
        rounded_box(ax, (x, yy), width, 0.065, box_text, row["color"], ec=row.get("ec", "#2f3542"), fontsize=6.25, weight=row.get("weight", "normal"))
        if idx < len(rows) - 1:
            arrow(ax, (x + width / 2, yy - 0.004), (x + width / 2, yy - 0.025), color="#6a7280", lw=0.8)


def build() -> plt.Figure:
    setup()
    colors = {
        "road": "#252a31",
        "degree": "#a9c4df",
        "spatial": "#cfdbc0",
        "geometry": "#e9aca5",
        "neutral": "#f3f1ec",
        "boundary": "#fff2cc",
        "accent": "#a44c45",
    }
    fig = plt.figure(figsize=(7.2, 4.65))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    ax.text(0.04, 0.955, "a", fontsize=10.5, fontweight="bold", ha="left", va="top", transform=ax.transAxes)
    ax.text(0.07, 0.955, "Object and baseline", fontsize=9.5, fontweight="bold", ha="left", va="top", transform=ax.transAxes)
    draw_mini_road(ax, 0.06, 0.77, scale=0.93)
    rounded_box(ax, (0.055, 0.59), 0.25, 0.08, "Observed road graph\nfinite, embedded, nearly planar", "#ffffff", fontsize=6.55, weight="bold")
    rounded_box(ax, (0.055, 0.45), 0.25, 0.08, r"CEBH baseline" + "\n" + r"$p_c=\langle k\rangle/(\langle k^2\rangle-\langle k\rangle)$", colors["degree"], fontsize=6.35)
    arrow(ax, (0.18, 0.59), (0.18, 0.53), color="#4a5568")
    ax.text(0.055, 0.385, "Question: which constraints move\nnull thresholds toward roads?", transform=ax.transAxes, fontsize=6.55, color="#3d4652", ha="left", va="top")

    ax.text(0.36, 0.955, "b", fontsize=10.5, fontweight="bold", ha="left", va="top", transform=ax.transAxes)
    draw_ladder(
        ax,
        0.365,
        0.55,
        [
            {"label": "Observed road", "value": "8-city gap 0.210", "color": "#ffffff", "weight": "bold"},
            {"label": "Degree null", "value": "residual 0.026; 12.7% explained", "color": colors["degree"]},
            {"label": "Spatial-scale null", "value": "gap 0.069; remaining 0.140", "color": colors["spatial"]},
            {"label": "Post-spatial residual", "value": "embedded-organization target", "color": colors["neutral"], "ec": "#8b949e"},
        ],
        "Formal null ladder\n8 method-coherent cities",
        width=0.255,
    )

    ax.text(0.67, 0.955, "c", fontsize=10.5, fontweight="bold", ha="left", va="top", transform=ax.transAxes)
    draw_ladder(
        ax,
        0.69,
        0.55,
        [
            {"label": "Observed road", "value": "71-window gap 0.217", "color": "#ffffff", "weight": "bold"},
            {"label": "Spatial-scale null", "value": "mean residual 0.0965", "color": colors["spatial"]},
            {"label": "Strict geometry null", "value": "residual 0.0014; near-zero at grid resolution", "color": colors["geometry"], "weight": "bold"},
            {"label": "Boundary", "value": "0.005 swaps/edge; one record/city", "color": colors["boundary"], "ec": "#b58b00"},
        ],
        "Full-city geometry screen\n71 standardized windows",
        width=0.265,
    )

    rounded_box(
        ax,
        (0.055, 0.105),
        0.89,
        0.13,
        "Allowed claim: a model-conditional null ladder separates degree, edge-scale and strict embedded-geometry baselines.\n"
        "The full-city low-intensity layer tests coverage; the higher-intensity subset tests sensitivity.\n"
        "Forbidden overclaim: not a universal planar theorem, causal urban-design rule or high-intensity all-city planar null.",
        colors["boundary"],
        ec="#b58b00",
        fontsize=6.45,
        weight="normal",
    )
    ax.text(
        0.055,
        0.055,
        "Evidence domains are separated: full-city low-intensity coverage is not a planar-percolation proof.",
        transform=ax.transAxes,
        fontsize=5.8,
        color="#687386",
    )
    return fig


def save(fig: plt.Figure) -> None:
    outputs = {
        FIG_DIR / f"{BASE_NAME}.svg": {},
        FIG_DIR / f"{BASE_NAME}.pdf": {},
        FIG_DIR / f"{BASE_NAME}.png": {"dpi": 600},
        FIG_DIR / f"{BASE_NAME}.tiff": {"dpi": 600},
    }
    if PAPER_OUT is not None:
        outputs.update(
            {
                PAPER_OUT / "svg" / f"{BASE_NAME}.svg": {},
                PAPER_OUT / "pdf" / f"{BASE_NAME}.pdf": {},
                PAPER_OUT / "png" / f"{BASE_NAME}.png": {"dpi": 600},
            }
        )
    for path, kwargs in outputs.items():
        fig.savefig(path, bbox_inches="tight", **kwargs)


def main() -> None:
    fig = build()
    save(fig)
    plt.close(fig)
    print(f"wrote {FIG_DIR / (BASE_NAME + '.png')}")


if __name__ == "__main__":
    main()

