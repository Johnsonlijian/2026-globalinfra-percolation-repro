"""Unified publication style for all submission-facing figures.

Single source of truth for fonts, sizes, line weights, semantic colors and
export settings.  Every figure-generating script imports this module so that
the same entity (observed road, CEBH baseline, degree null, spatial-scale
null, strict geometry null, macro-region) uses the same colour in every
main-text and supplementary figure.

Style contract (Nature-family publication figures):
- white background, no card panels, no rounded boxes, no drop shadows,
  no gradients, no dashboard/infographic blocks;
- Arial/Helvetica sans-serif, 6.5-8 pt text;
- thin axes (0.7 pt), open top/right spines, restrained light grid;
- Okabe-Ito colourblind-safe palette;
- panel labels are bold lowercase letters with short neutral descriptors;
- vector SVG/PDF first; PNG/TIFF rasters exported at 600 dpi.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

# Standard journal column widths in inches (89 mm / 180 mm).
FIG_WIDTH_1COL = 3.50
FIG_WIDTH_2COL = 7.09

# Okabe-Ito colourblind-safe palette.
OKABE_ITO = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "black": "#000000",
}

# Semantic colours: the same scientific entity gets the same colour in every
# figure of the manuscript and SI.
COLORS = {
    "observed": "#1A1A1A",        # observed road graphs / thresholds
    "cebh": "#6E6E6E",            # CEBH degree-moment baseline
    "degree_null": "#999999",     # configuration-model degree nulls
    "spatial_null": "#D55E00",    # degree-preserving spatial-scale nulls
    "geometry_null": "#0072B2",   # strict non-crossing geometry nulls
    "residual": "#56B4E9",        # post-spatial residual (bounded by geometry)
    "nb": "#E69F00",              # non-backtracking spectral proxy
    "model": "#009E73",           # empirical correction / model predictions
    "accent": "#CC79A7",          # secondary comparisons
    "fit": "#1A1A1A",             # fitted lines
    "grid": "#E3E3E3",
    "zero": "#4D4D4D",
    "annot": "#4D4D4D",           # in-panel statistical annotations
}

# Macro-region palette (seven regions; yellow avoided for scatter visibility).
REGION_COLORS = {
    "Africa": "#D55E00",
    "Asia": "#0072B2",
    "Europe": "#009E73",
    "Latin America": "#CC79A7",
    "Middle East": "#E69F00",
    "North America": "#56B4E9",
    "Oceania": "#8C7B00",
    "Other": "#7F7F7F",
}


def apply() -> None:
    """Apply the shared publication rcParams."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "mathtext.fontset": "dejavusans",
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.0,
            "axes.titleweight": "normal",
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "lines.linewidth": 1.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def panel_title(ax, letter: str, text: str = "") -> None:
    """Bold panel letter plus a short neutral descriptor (no assertions)."""
    label = rf"$\mathbf{{{letter}}}$"
    ax.set_title(f"{label}  {text}" if text else label, loc="left", fontsize=8, pad=4)


def light_grid(ax, axis: str = "y") -> None:
    ax.grid(axis=axis, color=COLORS["grid"], lw=0.4)
    ax.set_axisbelow(True)


def annot(ax, x: float, y: float, text: str, ha: str = "left", va: str = "top", fontsize: float = 6.2) -> None:
    """Neutral dark-grey statistical annotation inside a panel."""
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=fontsize, color=COLORS["annot"])


def save(fig, stem: Path | str, formats: tuple[str, ...] = ("svg", "pdf", "png", "tiff"), dpi: int = 600) -> None:
    """Export vector-first; PNG/TIFF rasters at >= 600 dpi."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        kwargs = {"dpi": dpi} if ext in {"png", "tiff"} else {}
        fig.savefig(stem.with_suffix(f".{ext}"), bbox_inches="tight", **kwargs)
