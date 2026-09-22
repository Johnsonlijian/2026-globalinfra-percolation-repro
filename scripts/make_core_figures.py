"""Build the three core figures from frozen canonical and screening outputs.

Run with Python 3.12: python -B scripts/make_core_figures.py
Only fig1_*, fig2_* and fig3_* outputs are written. No analyses are rerun.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import fitz
import matplotlib
from common import Context, parser

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import MultipleLocator
from shapely.geometry import LineString, Point

CTX = None
OUT = SOURCE_DATA = None
try:
    fm.findfont("Arial", fallback_to_default=False)
    FONT = "Arial"
except ValueError:
    FONT = "DejaVu Sans"
SOURCES: dict[str, str] = {}
WIDTH = 6.8
SEED = 20260905
BAND = (0.15, 0.55)
PREDICTORS = {
    "CEBH": "pc_cebh",
    "2D": "pc_dimension",
    "junction": "pc_junction_anchor",
    "nonbacktracking": "pc_nonbacktracking",
}
LABELS = {"CEBH": "CEBH", "2D": "2D", "junction": "Junction", "nonbacktracking": "Leading NB"}
COLORS = {"CEBH": "#D55E00", "2D": "#0072B2", "junction": "#009E73", "nonbacktracking": "#8C6BB1"}
INK = "#252A2D"
GRAY = "#737B80"
LIGHT = "#E5E8EA"
REMOVED = "#CC503E"
ADDED = "#0072B2"


def source(path: Path) -> Path:
    path = path.resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    name = str(path.relative_to(CTX.root)).replace("\\", "/")
    if name in SOURCES and SOURCES[name] != digest:
        raise RuntimeError(f"Source changed during figure generation: {name}")
    SOURCES[name] = digest
    return path


def read_csv(relative: str) -> pd.DataFrame:
    return pd.read_csv(source(CTX.path(relative)))


def read_json(relative: str) -> dict:
    return json.loads(source(CTX.path(relative)).read_text(encoding="utf-8"))


def graph_hash(n: int, edges: np.ndarray) -> str:
    ordered = np.sort(np.asarray(edges, dtype="<i8"), axis=1)
    ordered = ordered[np.lexsort((ordered[:, 1], ordered[:, 0]))]
    return hashlib.sha256(np.array([n], dtype="<i8").tobytes() + ordered.tobytes()).hexdigest()


def mean_interval(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, float)
    rng = np.random.default_rng(SEED)
    draws = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def integrated_errors(predicted_f: np.ndarray, reference_f: np.ndarray):
    low, high = BAND
    unsafe = np.maximum(0, np.minimum(high, predicted_f) - np.maximum(low, reference_f)) / (
        high - low
    )
    conservative = np.maximum(0, np.minimum(high, reference_f) - np.maximum(low, predicted_f)) / (
        high - low
    )
    return unsafe, conservative


def setup_style():
    font = fm.findfont(FONT, fallback_to_default=False)
    plt.rcParams.update(
        {
            "font.family": FONT,
            "font.size": 9.5,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "axes.edgecolor": GRAY,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "core-figures",
            "mathtext.fontset": "custom",
            "mathtext.rm": FONT,
            "mathtext.it": FONT + ":italic",
            "mathtext.bf": FONT + ":bold",
            "mathtext.fallback": "stixsans",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.unicode_minus": True,
        }
    )
    return font


def axis(fig, bounds):
    width, height = fig.get_size_inches()
    left, bottom, w, h = bounds
    ax = fig.add_axes([left / width, bottom / height, w / width, h / height])
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def heading(fig, letter: str, title: str, x: float, y: float):
    width, height = fig.get_size_inches()
    fig.text(x / width, y / height, letter, weight="bold", fontsize=11, va="top")
    fig.text((x + 0.22) / width, y / height, title, fontsize=10, va="top")


def fig_text(fig, x: float, y: float, text: str, **kwargs):
    width, height = fig.get_size_inches()
    return fig.text(x / width, y / height, text, **kwargs)


def export_table(name: str, data):
    assert name.startswith(("fig1_", "fig2_", "fig3_"))
    pd.DataFrame(data).to_csv(SOURCE_DATA / name, index=False)


def export_json(name: str, data: dict | list):
    assert name.startswith(("fig1_", "fig2_", "fig3_"))
    (SOURCE_DATA / name).write_text(json.dumps(data, indent=2), encoding="utf-8")


def draw_vienna_map(ax, segments, spans):
    ax.add_collection(
        LineCollection(
            segments, colors="#515E66", linewidths=0.18, antialiased=True, rasterized=False
        )
    )
    pad = max(spans) * 0.025
    ax.set(xlim=(-pad, spans[0] + pad), ylim=(-pad, spans[1] + pad), aspect="equal")
    ax.set_axis_off()
    ax.plot([0.6, 2.6], [0.6, 0.6], color=INK, lw=1.6)
    ax.text(
        1.6,
        1.05,
        "2 km",
        ha="center",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
    )
    ax.annotate(
        "N",
        xy=(spans[0] - 1, spans[1] - 0.3),
        xytext=(spans[0] - 1, spans[1] - 3.5),
        ha="center",
        arrowprops={"arrowstyle": "-|>", "lw": 0.7, "color": INK},
        fontsize=9,
    )


def save_figure(fig, stem: str, data: dict, extra_checks: dict | None = None):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    clipped = []
    off_axis_ticks = set()
    for ax in fig.axes:
        for ticks in (ax.xaxis, ax.yaxis):
            low, high = sorted(ticks.get_view_interval())
            for tick in ticks.get_major_ticks() + ticks.get_minor_ticks():
                if tick.get_loc() < low - 1e-10 or tick.get_loc() > high + 1e-10:
                    off_axis_ticks.update((tick.label1, tick.label2))
    for item in fig.findobj(Text):
        if item in off_axis_ticks or not item.get_visible() or not item.get_text().strip():
            continue
        if item.axes is not None and not item.axes.axison:
            if item not in item.axes.texts:
                continue
        box = item.get_window_extent(renderer)
        if box.width == 0 or box.height == 0:
            continue
        if (
            box.x0 < canvas.x0 - 1
            or box.y0 < canvas.y0 - 1
            or box.x1 > canvas.x1 + 1
            or box.y1 > canvas.y1 + 1
        ):
            clipped.append(item.get_text())
    if clipped:
        raise RuntimeError(f"Text outside {stem} canvas: {clipped}")
    for suffix in ("pdf", "svg", "png"):
        kwargs = {"dpi": 320} if suffix == "png" else {}
        if suffix == "pdf":
            kwargs["metadata"] = {
                "Title": data["title"],
                "Author": None,
                "Creator": "Matplotlib",
                "Subject": data["description"],
                "CreationDate": None,
                "ModDate": None,
            }
        elif suffix == "svg":
            kwargs["metadata"] = {
                "Title": data["title"],
                "Description": data["description"],
                "Creator": "Matplotlib",
                "Date": None,
            }
        fig.savefig(OUT / f"{stem}.{suffix}", **kwargs)
    svg = (OUT / f"{stem}.svg").read_text(encoding="utf-8")
    if "<image" in svg or "<text" not in svg:
        raise RuntimeError(f"{stem}: expected all-vector paths and editable SVG text")
    data["sources_sha256"] = dict(SOURCES)
    data["qa"] = {
        "figure_inches": fig.get_size_inches().tolist(),
        "font": Path(fm.findfont(FONT, fallback_to_default=False)).name,
        "pdf_fonttype": plt.rcParams["pdf.fonttype"],
        "svg_editable_text_elements": svg.count("<text"),
        "svg_embedded_raster_images": svg.count("<image"),
        "text_outside_canvas": clipped,
        **(extra_checks or {}),
    }
    data["outputs_sha256"] = {
        suffix: hashlib.sha256((OUT / f"{stem}.{suffix}").read_bytes()).hexdigest()
        for suffix in ("pdf", "svg", "png")
    }
    (OUT / f"{stem}_data.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    plt.close(fig)
    print(f"Created {stem}.pdf/.svg/.png", flush=True)


def validate_metrics(observed: pd.DataFrame, metrics: pd.DataFrame):
    assert len(observed) == 91 and set(observed.n_orders) == {8192}
    assert not observed.duplicated(["domain", "instance"]).any()
    for domain in ("road", "power", "rail"):
        cohort = observed[observed.domain.eq(domain)]
        for name, column in PREDICTORS.items():
            u, c = integrated_errors(1 - cohort[column].to_numpy(), 1 - cohort.pc.to_numpy())
            frame = pd.DataFrame(
                {
                    "absolute_error": abs(cohort.pc - cohort[column]).to_numpy(),
                    "unsafe_integrated": u,
                    "safe_integrated": c,
                    "system_id": cohort.system_id.to_numpy(),
                }
            )
            units = (
                frame.groupby("system_id").mean(numeric_only=True) if domain == "power" else frame
            )
            row = metrics[metrics.domain.eq(domain) & metrics.predictor.eq(name)].iloc[0]
            assert len(cohort) == row.n_instances and len(units) == row.n_inference_units
            for field in ("absolute_error", "unsafe_integrated", "safe_integrated"):
                value, lo, hi = mean_interval(units[field].to_numpy())
                np.testing.assert_allclose(
                    [value, lo, hi],
                    [row[field], row[field + "_low"], row[field + "_high"]],
                    rtol=1e-11,
                    atol=1e-12,
                )


def figure_one(observed: pd.DataFrame):
    row = observed[observed.domain.eq("road") & observed.instance.eq("Vienna")].iloc[0]
    curve_data = read_csv("fig1_response")
    metadata = read_json("fig1_example")
    grid, q, derivative = (curve_data[key].to_numpy() for key in ("p", "Q", "derivative"))
    assert metadata["graph_hash"] == row.input_hash == row.realization_hash
    assert metadata["n_nodes"] == row.n_nodes and metadata["n_edges"] == row.n_edges
    p_star = grid[
        np.flatnonzero(np.isclose(derivative, derivative.max(), rtol=1e-12, atol=1e-12))[0]
    ]
    np.testing.assert_allclose(
        [p_star, metadata["p_star_low"], metadata["p_star_high"]], [row.pc, row.pc_low, row.pc_high]
    )
    np.testing.assert_allclose([q[0], q[-1]], [1 / row.n_nodes, row.mean_curve_p1])
    assert np.diff(q).min() >= -1e-12

    fig = plt.figure(figsize=(WIDTH, 4.85))
    heading(fig, "a", "Vienna road network", 0.16, 4.72)
    heading(fig, "b", "Independent-bond response", 3.72, 4.72)
    fig_text(
        fig,
        0.28,
        1.94,
        f"{int(row.n_nodes):,} nodes; {int(row.n_edges):,} edges",
        fontsize=9,
        va="top",
    )
    fig_text(fig, 0.28, 1.73, "\N{COPYRIGHT SIGN} OpenStreetMap contributors, ODbL",
             fontsize=9, va="top", color=GRAY)

    ax_q = axis(fig, (3.78, 2.33, 2.77, 2.00))
    ax_q.plot(grid, q, color=INK, lw=1.65)
    ax_q.axvspan(row.pc_low, row.pc_high, color=INK, alpha=0.11, lw=0)
    ax_q.axvline(row.pc_cebh, color=COLORS["CEBH"], ls=(0, (4, 2.5)), lw=1.3)
    ax_q.axvline(row.pc, color=INK, lw=1.0)
    ax_q.scatter(
        [row.pc_cebh, row.pc],
        np.interp([row.pc_cebh, row.pc], grid, q),
        color=[COLORS["CEBH"], INK],
        s=20,
        zorder=6,
    )
    ax_q.text(
        0.04,
        0.96,
        f"{int(row.n_orders):,} edge orders",
        va="top",
        transform=ax_q.transAxes,
        fontsize=9,
    )
    ax_q.annotate(
        r"$\widehat p_{\rm CEBH}$" + f" = {row.pc_cebh:.3f}",
        xy=(row.pc_cebh, 0.24),
        xytext=(0.06, 0.47),
        color=COLORS["CEBH"],
        arrowprops={"arrowstyle": "-", "color": COLORS["CEBH"], "lw": 0.7},
        fontsize=9,
        ha="left",
    )
    ax_q.annotate(
        r"$p_\star$" + f" = {row.pc:.4f}",
        xy=(row.pc, float(np.interp(row.pc, grid, q))),
        xytext=(0.69, 0.17),
        arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.7},
        fontsize=9,
    )
    ax_q.set(
        xlim=(0, 1),
        ylim=(0, 1.05),
        xlabel=r"Edge-retention probability $p$",
        ylabel=r"Expected LCC fraction $Q(p)$",
    )
    ax_q.xaxis.set_major_locator(MultipleLocator(0.25))
    ax_q.yaxis.set_major_locator(MultipleLocator(0.25))
    fig_text(fig, 3.79, 1.88, r"$p_\star$: maximum of $Q'(p)$", fontsize=9, va="top")

    heading(fig, "c", "Illustrative target on the same network", 0.16, 1.52)
    phi = 0.40
    reference_f, predicted_f = 1 - row.pc, 1 - row.pc_cebh
    assert reference_f < phi < predicted_f
    ax_d = axis(fig, (1.99, 0.48, 4.10, 0.79))
    ax_d.set(
        xlim=(0.15, 0.65), ylim=(-0.55, 1.55), yticks=[], xlabel=r"Target loss fraction $\phi$"
    )
    ax_d.spines[["left", "top", "right"]].set_visible(False)
    ax_d.set_xticks([0.15, 0.25, 0.35, 0.45, 0.55, 0.65])
    ax_d.axvspan(reference_f, predicted_f, color=COLORS["CEBH"], alpha=0.07, lw=0)
    ax_d.axvline(phi, color=GRAY, ls=(0, (2, 2)), lw=0.9)
    for y, value, color, name, decision in (
        (1, predicted_f, COLORS["CEBH"], "CEBH screen", "accepts"),
        (0, reference_f, INK, "Finite reference", "rejects"),
    ):
        ax_d.hlines(y, 0.15, value, color=color, lw=2.1)
        ax_d.plot(value, y, "o", ms=4.6, color=color)
        ax_d.text(
            -0.025,
            (y + 0.55) / 2.10,
            name,
            transform=ax_d.transAxes,
            ha="right",
            va="center",
            fontsize=9.5,
        )
        ax_d.text(0.654, y, decision, color=color, va="center", fontsize=9.5)
    ax_d.text(
        predicted_f,
        1.23,
        r"$\widehat f$" + f" = {predicted_f:.3f}",
        color=COLORS["CEBH"],
        ha="center",
        va="bottom",
        fontsize=9,
    )
    ax_d.text(
        reference_f - 0.004,
        -0.15,
        r"$f_\star$" + f" = {reference_f:.4f}",
        ha="right",
        va="top",
        fontsize=9,
    )
    ax_d.text(phi + 0.008, 0.50, r"$\phi=0.40$", va="center", fontsize=9)
    data = {
        "title": "From a network response to a screening decision",
        "description": "Real Vienna road geometry, canonical expected LCC, and an illustrative topological decision.",
        "city": "Vienna",
        "graph_hash": row.input_hash,
        "source_crs": metadata["source_crs"],
        "display_crs": "EPSG:32633",
        "cache_created_date": metadata.get("cache_created_date"),
        "osm_attribution": "OpenStreetMap contributors, ODbL; required in manuscript caption.",
        "geometry": "One source polyline per edge of the frozen simple undirected graph; no basemap or new crop.",
        "n_nodes": int(row.n_nodes),
        "n_edges": int(row.n_edges),
        "n_orders": int(row.n_orders),
        "p_star": float(row.pc),
        "p_star_low": float(row.pc_low),
        "p_star_high": float(row.pc_high),
        "cebh": float(row.pc_cebh),
        "grid": grid.tolist(),
        "canonical_q": q.tolist(),
        "canonical_derivative": derivative.tolist(),
        "illustrative_phi": phi,
        "reference_f": float(reference_f),
        "predicted_f": float(predicted_f),
        "decision": "unsafe-side disagreement",
        "interval": "95% order-bootstrap interval for the maximum-slope location, not a population interval.",
    }
    save_figure(
        fig,
        "fig1_screening_example",
        data,
        {"derived_curve_and_frozen_metadata_match": True, "fixed_map_layer_reused": True},
    )
    compose_fixed_map("fig1_screening_example")


def compose_fixed_map(stem):
    """Reuse produced vector artwork without distributing raw OSM coordinates."""
    pdf_map = source(CTX.path("map_pdf"))
    svg_map = source(CTX.path("map_svg"))
    pdf_path, svg_path = OUT / (stem + ".pdf"), OUT / (stem + ".svg")
    left, top, width, height = 0.20 * 72, 0.39 * 72, 2.94 * 72, 2.40 * 72
    with (
        fitz.open(stream=pdf_path.read_bytes(), filetype="pdf") as doc,
        fitz.open(pdf_map) as layer,
    ):
        rectangle = fitz.Rect(left, top, left + width, top + height)
        for block in doc[0].get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if (
                        span["text"].strip()
                        and (fitz.Rect(span["bbox"]) & rectangle).get_area() > 0.1
                    ):
                        raise RuntimeError("Map layer would cover a response label")
        doc[0].show_pdf_page(rectangle, layer, 0)
        pdf_path.write_bytes(doc.tobytes(garbage=4, deflate=True))
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(320 / 72, 320 / 72), alpha=False)
        pix.save(OUT / (stem + ".png"))
    root = ET.fromstring(svg_path.read_text(encoding="utf-8"))
    layer = ET.fromstring(svg_map.read_text(encoding="utf-8"))
    ids = {
        element.attrib["id"]: "map_" + element.attrib["id"]
        for element in layer.iter()
        if "id" in element.attrib
    }
    for element in layer.iter():
        for key, value in list(element.attrib.items()):
            if key == "id":
                element.set(key, ids[value])
            else:
                value = re.sub(
                    r"url\(#([^)]+)\)",
                    lambda match: "url(#" + ids.get(match[1], match[1]) + ")",
                    value,
                )
                if value.startswith("#"):
                    value = "#" + ids.get(value[1:], value[1:])
                element.set(key, value)
    layer.set("x", str(left))
    layer.set("y", str(top))
    layer.set("width", str(width))
    layer.set("height", str(height))
    root.append(layer)
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    svg_path.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    record_path = OUT / (stem + "_data.json")
    record = json.loads(record_path.read_text())
    record["sources_sha256"] = dict(SOURCES)
    record["qa"]["fixed_vector_layer_reused"] = True
    record["outputs_sha256"] = {
        suffix: hashlib.sha256((OUT / f"{stem}.{suffix}").read_bytes()).hexdigest()
        for suffix in ("pdf", "svg", "png")
    }
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def road_steps(roads: pd.DataFrame, column: str):
    predicted = 1 - roads[column].to_numpy()
    reference = 1 - roads.pc.to_numpy()
    low, high = BAND
    breaks = np.unique(np.r_[low, high, predicted, reference])
    breaks = breaks[(breaks >= low) & (breaks <= high)]
    midpoints = (breaks[:-1] + breaks[1:]) / 2
    p = predicted[:, None] - midpoints
    r = reference[:, None] - midpoints
    unsafe = ((p > 0) & (r < 0)).mean(axis=0)
    conservative = ((p < 0) & (r > 0)).mean(axis=0)
    u, c = integrated_errors(predicted, reference)
    np.testing.assert_allclose(
        [np.dot(unsafe, np.diff(breaks)), np.dot(conservative, np.diff(breaks))],
        [(high - low) * u.mean(), (high - low) * c.mean()],
        atol=1e-12,
    )
    return breaks, unsafe, conservative


def figure_two(observed: pd.DataFrame, metrics: pd.DataFrame):
    fig = plt.figure(figsize=(WIDTH, 5.97))
    heading(fig, "a", "Prediction error by network class", 0.16, 5.84)
    ax_a = axis(fig, (1.65, 3.91, 4.90, 1.49))
    shifts = np.array([0.24, 0.08, -0.08, -0.24])
    domains = ("road", "power", "rail")
    for i, (name, color) in enumerate(COLORS.items()):
        rows = metrics[metrics.predictor.eq(name)].set_index("domain").loc[list(domains)]
        y = np.arange(3)[::-1] + shifts[i]
        values = rows.absolute_error.to_numpy()
        error = np.vstack(
            [
                values - rows.absolute_error_low.to_numpy(),
                rows.absolute_error_high.to_numpy() - values,
            ]
        )
        ax_a.errorbar(
            values,
            y,
            xerr=error,
            fmt="o",
            ms=4.3,
            color=color,
            capsize=2.2,
            elinewidth=1.1,
            markeredgewidth=0.8,
        )
    ax_a.set(
        ylim=(-0.5, 2.5), xlim=(0, 0.48), xlabel=r"Mean absolute error in retained-edge fraction"
    )
    ax_a.set_yticks(
        [2, 1, 0], ["Roads\n71 cities", "Power\n6 families / 10 cases", "Rail\n6 networks"]
    )
    ax_a.tick_params(axis="y", length=0, pad=8)
    ax_a.spines["left"].set_visible(False)
    ax_a.xaxis.set_major_locator(MultipleLocator(0.1))
    ax_a.grid(axis="x", color=LIGHT, lw=0.6, zorder=0)
    ax_a.set_axisbelow(True)
    handles = [
        Line2D([], [], color=COLORS[k], marker="o", ls="none", ms=4, label=LABELS[k])
        for k in PREDICTORS
    ]
    ax_a.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(-0.01, 1.01),
        ncol=4,
        frameon=False,
        borderaxespad=0,
        handletextpad=0.35,
        columnspacing=1.25,
    )
    heading(fig, "b", "Road decision errors", 0.16, 3.15)
    heading(fig, "c", "Power penalty sensitivity", 4.08, 3.15)
    ax_b = axis(fig, (0.60, 0.52, 3.02, 1.83))
    roads = observed[observed.domain.eq("road")]
    curve_data = {}
    for name in ("nonbacktracking", "CEBH", "junction", "2D"):
        breaks, unsafe, conservative = road_steps(roads, PREDICTORS[name])
        ax_b.stairs(unsafe, breaks, color=COLORS[name], lw=1.35, baseline=None)
        ax_b.stairs(
            conservative, breaks, color=COLORS[name], lw=1.35, ls=(0, (4, 2.5)), baseline=None
        )
        curve_data[name] = {
            "interval_boundaries": breaks.tolist(),
            "unsafe": unsafe.tolist(),
            "conservative": conservative.tolist(),
        }
    ax_b.set(
        xlim=BAND,
        ylim=(-0.025, 1.04),
        xlabel=r"Target loss fraction $\phi$",
        ylabel="Disagreement fraction",
    )
    ax_b.set_xticks([0.15, 0.25, 0.35, 0.45, 0.55])
    ax_b.yaxis.set_major_locator(MultipleLocator(0.25))
    method_handles = [Line2D([], [], color=COLORS[k], lw=1.4, label=LABELS[k]) for k in PREDICTORS]
    style_handles = [
        Line2D([], [], color=INK, lw=1.4, label="Unsafe"),
        Line2D([], [], color=INK, lw=1.4, ls=(0, (4, 2.5)), label="Conservative"),
    ]
    legend = ax_b.legend(
        handles=method_handles + style_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.355),
        ncol=2,
        frameon=False,
        borderaxespad=0,
        handlelength=1.5,
        handletextpad=0.5,
        columnspacing=1.25,
        labelspacing=0.3,
    )

    power = metrics[metrics.domain.eq("power")].set_index("predictor")
    stored = read_json("loss_tradeoff")
    loss_table = read_csv("asymmetric_losses")
    two, junction = power.loc["2D"], power.loc["junction"]
    crossing = (two.safe_integrated - junction.safe_integrated) / (
        junction.unsafe_integrated - two.unsafe_integrated
    )
    np.testing.assert_allclose(
        crossing, stored["power_junction_2d_break_even_unsafe_cost"], atol=1e-12
    )
    ax_c = axis(fig, (4.39, 0.52, 2.16, 1.83))
    weights = np.linspace(0, 3, 301)
    losses = {}
    for name in ("2D", "junction"):
        row = power.loc[name]
        loss = weights * row.unsafe_integrated + row.safe_integrated
        ax_c.plot(weights, loss, color=COLORS[name], lw=1.7)
        losses[name] = loss.tolist()
        check = loss_table[loss_table.domain.eq("power") & loss_table.predictor.eq(name)]
        np.testing.assert_allclose(
            check.loss, check["lambda"] * row.unsafe_integrated + row.safe_integrated, atol=1e-12
        )
    crossing_y = crossing * two.unsafe_integrated + two.safe_integrated
    ax_c.axvline(crossing, color=GRAY, ls=(0, (2, 2)), lw=0.8)
    ax_c.plot(crossing, crossing_y, "o", color=INK, ms=3.8)
    ax_c.text(2.15, 0.325, "2D", color=COLORS["2D"], fontsize=9.5)
    ax_c.text(2.00, 0.60, "Junction", color=COLORS["junction"], fontsize=9.5)
    ax_c.annotate(
        r"$\lambda_\times$" + f" = {crossing:.2f}",
        xy=(crossing, crossing_y),
        xytext=(0.12, 0.65),
        arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.7},
        fontsize=9,
    )
    ax_c.set(
        xlim=(0, 3),
        ylim=(0, 0.8),
        xlabel=r"Relative unsafe penalty $\lambda$",
        ylabel=r"Mean loss $\lambda U+C$",
    )
    ax_c.xaxis.set_major_locator(MultipleLocator(1))
    ax_c.yaxis.set_major_locator(MultipleLocator(0.2))
    fig_text(fig, 4.39, 2.85, "6 equally weighted families", fontsize=9, va="top")
    fig_text(fig, 4.39, 2.63, r"Uniform $\phi\in[0.15,\,0.55]$", fontsize=9, va="top")
    fig.canvas.draw()
    legend_box = legend.get_window_extent(fig.canvas.get_renderer())
    ax_box = ax_b.get_window_extent()
    assert legend._loc == 2
    assert legend_box.y0 > ax_box.y1 + 3, "Legend must not overlap any road curve."
    assert abs(legend_box.x0 - ax_box.x0) < 1, "Legend must remain upper left."
    data = {
        "title": "Predictor accuracy and decision consequences",
        "description": "Cohort mean absolute error, road disagreement curves and family-weighted illustrative power losses.",
        "a_metrics": metrics[metrics.domain.isin(domains)].to_dict("records"),
        "a_intervals": "95% percentile bootstrap of 10000 city/network means; six equal family means for power.",
        "b_road_instances": len(roads),
        "b_target_band": BAND,
        "b_curves": curve_data,
        "b_curve_convention": "Exact piecewise-constant fractions between every decision boundary; endpoint ties have zero integration measure. Coincident curves are not displaced.",
        "c_lambda": weights.tolist(),
        "c_losses": losses,
        "c_crossing": float(crossing),
        "c_definition": "lambda * unsafe + conservative, family-weighted and uniformly integrated over the target band; no real accident-cost interpretation.",
        "omitted": "Four original water benchmarks are reported separately in the SI.",
        "nonbacktracking_boundary": "Leading spectral onset used as a graph-aware screening comparator; it does not target the same finite maximum-slope diagnostic.",
    }
    save_figure(
        fig,
        "fig2_accuracy",
        data,
        {
            "all_cohort_metrics_recomputed_and_matched": True,
            "step_curve_integrals_match_current_metrics": True,
            "panel_b_legend_loc": "upper left",
            "panel_b_legend_columns": 2,
            "panel_b_legend_clear_of_data_pixels": float(legend_box.y0 - ax_box.y1),
            "power_loss_matches_current_final_evidence": True,
        },
    )
    export_table(
        "fig2_cohort_metrics.csv",
        metrics[metrics.domain.isin(domains)][
            [
                "domain",
                "predictor",
                "n_instances",
                "n_inference_units",
                "absolute_error",
                "absolute_error_low",
                "absolute_error_high",
                "unsafe_integrated",
                "safe_integrated",
            ]
        ],
    )
    step_rows = []
    for predictor, curve in curve_data.items():
        edges = curve["interval_boundaries"]
        step_rows.extend(
            {"predictor": predictor, "phi_left": a, "phi_right": b, "unsafe": u, "conservative": c}
            for a, b, u, c in zip(
                edges[:-1], edges[1:], curve["unsafe"], curve["conservative"], strict=True
            )
        )
    export_table("fig2_road_decisions.csv", step_rows)
    export_table(
        "fig2_power_loss.csv",
        {"lambda": weights, "2D": losses["2D"], "junction": losses["junction"]},
    )
    export_json(
        "fig2_definitions.json",
        {
            "target_band": BAND,
            "power_crossing": float(crossing),
            "intervals": data["a_intervals"],
            "curve_convention": data["b_curve_convention"],
            "power_loss": data["c_definition"],
            "nonbacktracking": data["nonbacktracking_boundary"],
        },
    )


def edge_key(edge):
    return tuple(sorted(edge))


def crossings_and_contacts(positions: dict, edges: set):
    proper, forbidden = [], []
    for first, second in itertools.combinations(sorted(edges), 2):
        a, b = LineString([positions[n] for n in first]), LineString([positions[n] for n in second])
        intersection = a.intersection(b)
        shared = set(first) & set(second)
        if a.crosses(b):
            proper.append([first, second])
        if intersection.is_empty:
            continue
        if shared and intersection.geom_type == "Point":
            if any(intersection.distance(Point(positions[n])) < 1e-10 for n in shared):
                continue
        forbidden.append([first, second])
    node_on_edge = []
    for edge in sorted(edges):
        line = LineString([positions[n] for n in edge])
        for node in positions:
            if node not in edge and line.distance(Point(positions[node])) < 1e-10:
                node_on_edge.append([node, edge])
    return proper, forbidden, node_on_edge


def build_schematics():
    positions = {
        "A": (-1.0, -1.0),
        "B": (-1.0, 1.0),
        "C": (1.0, -1.0),
        "D": (1.0, 1.0),
        "E": (-2.0, 0.0),
        "F": (0.0, 2.0),
        "G": (2.0, 0.0),
        "H": (0.0, -2.0),
    }
    ring = ["A", "E", "B", "F", "D", "G", "C", "H", "A"]
    original = {edge_key(pair) for pair in zip(ring[:-1], ring[1:], strict=True)}
    original.update({("A", "B"), ("C", "D")})
    original_graph = nx.Graph(sorted(original))
    assert crossings_and_contacts(positions, original) == ([], [], [])
    boundaries = [0.0, 1.6, 3.0, 5.0]

    def lengths_and_bins(edges):
        lengths = [
            float(np.linalg.norm(np.subtract(positions[u], positions[v]))) for u, v in sorted(edges)
        ]
        bins = np.searchsorted(boundaries[1:-1], lengths, side="right")
        return lengths, np.bincount(bins, minlength=3).tolist()

    definitions = [
        ("Degree only", [("A", "E"), ("D", "G")], [("A", "D"), ("E", "G")], False, False),
        ("Length-bin matched", [("A", "B"), ("C", "D")], [("A", "D"), ("B", "C")], True, False),
        ("Strict noncrossing", [("A", "B"), ("C", "D")], [("A", "C"), ("B", "D")], True, True),
    ]
    result = []
    for name, remove, add, require_bins, require_geometry in definitions:
        deleted = {edge_key(e) for e in remove}
        added = {edge_key(e) for e in add}
        assert len(set(itertools.chain.from_iterable(deleted))) == 4
        assert deleted <= original and not (added & original)
        final = (original - deleted) | added
        graph = nx.Graph(sorted(final))
        assert len(final) == len(original) and nx.number_of_selfloops(graph) == 0
        assert dict(graph.degree()) == dict(original_graph.degree())
        assert nx.is_connected(graph)
        lengths_before, hist_before = lengths_and_bins(original)
        lengths_after, hist_after = lengths_and_bins(final)
        proper, contacts, through_node = crossings_and_contacts(positions, final)
        if require_bins:
            assert hist_before == hist_after
        if require_geometry:
            assert not proper and not contacts and not through_node
        result.append(
            {
                "name": name,
                "positions": positions,
                "original_edges": sorted(original),
                "removed_edges": sorted(deleted),
                "added_edges": sorted(added),
                "retained_edges": sorted(original - deleted),
                "final_edges": sorted(final),
                "degree_before": dict(original_graph.degree()),
                "degree_after": dict(graph.degree()),
                "bin_boundaries": boundaries,
                "original_lengths": lengths_before,
                "final_lengths": lengths_after,
                "bin_histogram_before": hist_before,
                "bin_histogram_after": hist_after,
                "bin_histogram_preserved": hist_before == hist_after,
                "proper_crossings_final": proper,
                "forbidden_contacts_final": contacts,
                "nodes_on_nonincident_edges_final": through_node,
                "connected_before_and_after": True,
                "fixed_positions": True,
                "status": "Illustrative deterministic construction; not an empirical null or a uniform ensemble sample.",
            }
        )
    return result


def figure_three(observed: pd.DataFrame, metrics: pd.DataFrame):
    schematic_data = build_schematics()
    roads = observed[observed.domain.eq("road")].set_index("instance")
    summary = read_csv("null_summary").set_index("kind")
    values = {"Observed": (roads.pc - roads.pc_cebh).to_numpy()}
    intervals = {"Observed": mean_interval(values["Observed"])}
    contrasts = {}
    null_sources = {}
    for kind, display, repeats in (
        ("degree", "Degree null", 4),
        ("connected", "Connected null", 2),
    ):
        frame = read_csv(f"{kind}_paired").set_index("instance").loc[roads.index]
        assert len(frame) == 71 and set(frame.n_null) == {repeats}
        assert frame.input_hash.equals(roads.input_hash)
        np.testing.assert_allclose(frame.pc, roads.pc)
        np.testing.assert_allclose(frame.pc_cebh, roads.pc_cebh)
        np.testing.assert_allclose(frame.null_gap, frame.pc_null - frame.pc_cebh)
        np.testing.assert_allclose(frame.organization_residual, frame.pc - frame.pc_null)
        values[display] = frame.null_gap.to_numpy()
        intervals[display] = mean_interval(values[display])
        contrasts[kind] = mean_interval(frame.organization_residual.to_numpy())
        row = summary.loc[kind]
        np.testing.assert_allclose(
            intervals[display], [row.null_gap, row.null_gap_low, row.null_gap_high]
        )
        np.testing.assert_allclose(
            contrasts[kind],
            [
                row.organization_residual,
                row.organization_residual_low,
                row.organization_residual_high,
            ],
        )
        null_sources[kind] = {
            "n_cities": len(frame),
            "n_realizations": int(frame.n_null.sum()),
            "orders_per_realization": 256,
        }
        raw_null = read_csv(f"null_{kind}")
        assert len(raw_null) == int(frame.n_null.sum()) and set(raw_null.n_orders) == {256}
        raw_means = raw_null.groupby("instance").pc.mean().loc[roads.index]
        np.testing.assert_allclose(raw_means, frame.pc_null)
    observed_mean = metrics[metrics.domain.eq("road") & metrics.predictor.eq("CEBH")].iloc[0]
    np.testing.assert_allclose(
        intervals["Observed"],
        [
            observed_mean.signed_error,
            observed_mean.signed_error_low,
            observed_mean.signed_error_high,
        ],
    )

    fig = plt.figure(figsize=(WIDTH, 5.75))
    positions = [0.14, 2.40, 4.65]
    for letter, x, construction in zip("abc", positions, schematic_data, strict=True):
        heading(fig, letter, construction["name"], x, 5.62)
        fig_text(fig, x + 0.22, 5.40, "Illustrative", fontsize=9, color=GRAY, va="top")
        ax = axis(fig, (x + 0.08, 3.32, 1.94, 1.88))
        pos = construction["positions"]
        for field, color, width, dash in (
            ("retained_edges", "#A4ADB3", 1.2, "-"),
            ("removed_edges", REMOVED, 1.7, (0, (3.2, 2.0))),
            ("added_edges", ADDED, 1.7, "-"),
        ):
            for u, v in construction[field]:
                a, b = pos[u], pos[v]
                ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=width, ls=dash, zorder=2)
        xy = np.array(list(pos.values()))
        ax.scatter(xy[:, 0], xy[:, 1], s=21, facecolor="white", edgecolor=INK, lw=0.9, zorder=5)
        ax.set(xlim=(-2.25, 2.25), ylim=(-2.25, 2.25), aspect="equal")
        ax.set_axis_off()
    legend_handles = [
        Line2D([], [], color="#A4ADB3", lw=1.3, label="Unchanged"),
        Line2D([], [], color=REMOVED, lw=1.6, ls=(0, (3.2, 2.0)), label="Deleted"),
        Line2D([], [], color=ADDED, lw=1.6, label="Added"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="center",
        bbox_to_anchor=(0.50, 3.11 / 5.75),
        ncol=3,
        frameon=False,
        handlelength=2.0,
        columnspacing=2.3,
        fontsize=9,
    )
    heading(fig, "d", "71 matched road networks", 0.16, 2.81)
    heading(fig, "e", "Paired contrasts", 4.61, 2.81)
    ax_d = axis(fig, (0.65, 0.64, 3.46, 1.91))
    group_colors = [INK, "#8F9DA8", ADDED]
    labels = list(values)
    array = np.column_stack(list(values.values()))
    rng = np.random.default_rng(314159)
    jitter = rng.uniform(-0.09, 0.09, len(roads))
    for city_i in range(len(roads)):
        ax_d.plot(
            np.arange(3) + jitter[city_i],
            array[city_i],
            color="#BEC4C8",
            lw=0.45,
            alpha=0.40,
            zorder=1,
        )
    for i, (label, color) in enumerate(zip(labels, group_colors, strict=True)):
        ax_d.scatter(
            i + jitter,
            values[label],
            s=10,
            color=color,
            alpha=0.7,
            edgecolor="white",
            lw=0.25,
            zorder=3,
        )
        mean, lo, hi = intervals[label]
        ax_d.errorbar(
            i + 0.25,
            mean,
            yerr=[[mean - lo], [hi - mean]],
            fmt="D",
            color=INK,
            mfc="white",
            ms=4.8,
            capsize=3,
            elinewidth=1.4,
            markeredgewidth=1.1,
            zorder=6,
        )
    ax_d.axhline(0, color=LIGHT, lw=0.8, zorder=0)
    ax_d.set(
        xlim=(-0.28, 2.47),
        ylim=(-0.013, max(array.max() + 0.015, 0.31)),
        ylabel=r"Gap $p_\star-\widehat p_{\rm CEBH}$",
    )
    ax_d.set_xticks(
        [0, 1, 2], ["Observed", "Degree null\n4 per city", "Connected null\n2 per city"]
    )
    ax_d.yaxis.set_major_locator(MultipleLocator(0.1))
    fig_text(fig, 0.66, 0.17, "Points: cities; diamonds: mean with 95% CI", fontsize=9, va="bottom")
    ax_e = axis(fig, (4.70, 0.85, 1.85, 1.65))
    for y, (kind, label) in zip(
        [1.0, 0.0],
        [("degree", "Observed - degree"), ("connected", "Observed - connected")],
        strict=True,
    ):
        mean, lo, hi = contrasts[kind]
        ax_e.text(0.004, y + 0.29, label, fontsize=9, va="bottom")
        ax_e.errorbar(
            mean,
            y,
            xerr=[[mean - lo], [hi - mean]],
            fmt="D",
            mfc="white",
            mec=INK,
            color=INK,
            ms=3,
            capsize=3,
            elinewidth=1.4,
            markeredgewidth=1.0,
        )
        ax_e.text(0.012, y, f"{mean:.3f}", va="center", fontsize=9)
    ax_e.set(xlim=(0, 0.25), ylim=(-0.37, 1.7), yticks=[], xlabel=r"Mean paired gap difference")
    ax_e.spines["left"].set_visible(False)
    ax_e.set_xticks([0, 0.1, 0.2])
    data = {
        "title": "Matched finite controls and constraints on geometric rewiring",
        "description": "Verified illustrative degree-preserving switches above; actual matched 71-city gaps and paired contrasts below.",
        "illustrative_constructions": schematic_data,
        "city_order": roads.index.tolist(),
        "matched_gap_values": {k: v.tolist() for k, v in values.items()},
        "gap_mean_intervals": intervals,
        "paired_contrast_mean_intervals": contrasts,
        "null_sources": null_sources,
        "observed_orders": 8192,
        "intervals": "10000-resample percentile city-bootstrap intervals for means; null realizations averaged within city before resampling.",
        "point_jitter": "Deterministic shared horizontal jitter per city; values unchanged and connected across all three groups.",
        "boundary": "Upper schematic states are not the empirical null realizations. Geometric sensitivity is illustrative here; lower contrasts jointly change network organization and do not isolate crossing causation.",
    }
    save_figure(
        fig,
        "fig3_nulls",
        data,
        {
            "all_toy_switches_preserve_nodewise_degree": True,
            "length_bin_example_preserves_histogram": True,
            "strict_example_preserves_bins_and_has_no_geometric_contacts": True,
            "null_means_recomputed_from_current_realizations": True,
            "all_city_bootstrap_intervals_match_current_tables": True,
        },
    )
    export_table(
        "fig3_matched_gaps.csv",
        {
            "city": roads.index,
            "observed": values["Observed"],
            "degree_null": values["Degree null"],
            "connected_null": values["Connected null"],
        },
    )
    export_table(
        "fig3_gap_summary.csv",
        [
            {"group": name, "mean": ci[0], "low_95": ci[1], "high_95": ci[2], "n_cities": 71}
            for name, ci in intervals.items()
        ],
    )
    export_table(
        "fig3_paired_contrasts.csv",
        [
            {"control": name, "mean": ci[0], "low_95": ci[1], "high_95": ci[2], "n_cities": 71}
            for name, ci in contrasts.items()
        ],
    )
    export_json("fig3_schematics.json", schematic_data)
    export_json(
        "fig3_definitions.json",
        {
            "intervals": data["intervals"],
            "point_jitter": data["point_jitter"],
            "boundary": data["boundary"],
            "null_sources": null_sources,
            "observed_orders": 8192,
        },
    )


def main(context):
    global CTX, OUT, SOURCE_DATA
    CTX = context
    OUT = context.output / "figures"
    SOURCE_DATA = OUT / "source_data"
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    setup_style()
    observed = read_csv("observed")
    metrics = read_csv("domain_metrics")
    validate_metrics(observed, metrics)
    figure_one(observed)
    figure_two(observed, metrics)
    figure_three(observed, metrics)
    for name, digest in SOURCES.items():
        assert hashlib.sha256((CTX.root / name).read_bytes()).hexdigest() == digest, name
    print("All input hashes stable; numerical and vector-output checks passed.", flush=True)


if __name__ == "__main__":
    main(Context(parser(__doc__).parse_args()))
