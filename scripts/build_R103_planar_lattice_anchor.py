"""R103: planar-lattice universality anchor for road bond-percolation thresholds.

Mechanistic upgrade. The earlier rounds established that strict non-crossing
geometry nulls reproduce observed road thresholds, but did not explain WHY.
This round supplies a parameter-free physical anchor:

  Road-intersection graphs are mixtures of 3-valent (T/Y) and 4-valent (cross)
  planar junctions. The two canonical planar lattices with these coordinations
  have exactly known bond-percolation thresholds - honeycomb p_c = 1 - 2 sin(pi/18)
  = 0.6527 (3-valent) and square p_c = 1/2 (4-valent). A zero-parameter mixture
  indexed only by the observed junction-degree composition,

      p_c^anchor = w3 * p_c^honeycomb + w4 * p_c^square,

  predicts the observed road thresholds far better than the degree-moment
  (CEBH) random-graph formula, with no fitted parameters.

The script is deliberately self-contained and reproducible:

1. Re-implements the SAME rank-order LCC-gradient estimator used for the road
   graphs, so any finite-size/estimator offset cancels between lattices and roads.
2. Calibrates the estimator on clean square and honeycomb lattices (must recover
   ~0.5 and ~0.6527).
3. Builds a vertex-split interpolation family (square -> honeycomb-like) that
   continuously tunes the 3-valent fraction while staying planar, and confirms
   p_c interpolates between the two lattice values.
4. Applies the zero-parameter anchor to all 71 standardized city windows and the
   eight method-coherent road graphs, comparing against CEBH and the strict
   geometry-null thresholds.

No third-party raw data are needed; city degree compositions and thresholds are
read from registered derived tables. Lattices are generated analytically.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import sys
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
import pub_style  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "R103_planar_lattice_anchor"
ROUND = ROOT / "rounds" / "R103_planar_lattice_anchor"
FIG_BASE = ROOT / "figures" / "Fig_R103_planar_lattice_anchor"

NPJ = ROOT / "submission" / "npj_complexity" / "target_submission"
NPJ_FIGS = NPJ / "figures"
NPJ_TABLES = NPJ / "source_data" / "tables"

COV = ROOT / "data" / "R62_urban_form_mechanism" / "urban_form_covariates.csv"
PRED = ROOT / "data" / "N99_prediction_correction_law" / "N99_prediction_city_predictions.csv"
R56 = ROOT / "data" / "R56_spatial_length_constrained_nulls" / "spatial_length_constrained_nulls_city_summary.csv"
R68 = ROOT / "data" / "R68_full71_geometry_null_ensemble" / "full71_geometry_null_city_summary.csv"

# Exact bond-percolation thresholds of the canonical planar lattices.
PC_HONEYCOMB = 1.0 - 2.0 * math.sin(math.pi / 18.0)   # 3-valent, 0.652704...
PC_SQUARE = 0.5                                         # 4-valent, exact
PC_TRIANGULAR = 2.0 * math.sin(math.pi / 18.0)         # 6-valent, 0.347296...


# --------------------------------------------------------------------------- #
# Estimator: same rank-order LCC-gradient threshold used for the road graphs.
# --------------------------------------------------------------------------- #
def _find(parent: np.ndarray, x: int) -> int:
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        parent[x], x = root, parent[x]
    return root


def lcc_curve(n: int, edges: np.ndarray, p_grid: np.ndarray, n_perm: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = len(edges)
    samples = np.zeros((len(p_grid), n_perm), dtype=float)
    if n == 0:
        return samples
    src = edges[:, 0]
    dst = edges[:, 1]
    target_ks = np.clip(np.rint(p_grid * m).astype(int), 0, m)
    for b in range(n_perm):
        parent = np.arange(n, dtype=np.int64)
        size = np.ones(n, dtype=np.int64)
        largest = 1
        t = 0
        while t < len(target_ks) and target_ks[t] == 0:
            samples[t, b] = largest / n
            t += 1
        order = rng.permutation(m)
        for step, eidx in enumerate(order, start=1):
            ru = _find(parent, int(src[eidx]))
            rv = _find(parent, int(dst[eidx]))
            if ru != rv:
                if size[ru] < size[rv]:
                    ru, rv = rv, ru
                parent[rv] = ru
                size[ru] += size[rv]
                if size[ru] > largest:
                    largest = int(size[ru])
            while t < len(target_ks) and step >= target_ks[t]:
                samples[t, b] = largest / n
                t += 1
            if t >= len(target_ks):
                break
        while t < len(target_ks):
            samples[t, b] = largest / n
            t += 1
    return samples


def estimate_pc(graph: nx.Graph, p_grid: np.ndarray, n_perm: int, seed: int) -> float:
    nodes = list(graph.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    edges = np.array([(idx[u], idx[v]) for u, v in graph.edges()], dtype=np.int64)
    samples = lcc_curve(len(nodes), edges, p_grid, n_perm, seed)
    mean_curve = samples.mean(axis=1)
    grad = np.gradient(mean_curve, p_grid)
    return float(p_grid[int(np.argmax(grad))])


# --------------------------------------------------------------------------- #
# Lattice builders.
# --------------------------------------------------------------------------- #
def square_lattice(L: int) -> nx.Graph:
    g = nx.grid_2d_graph(L, L, periodic=True)
    return nx.convert_node_labels_to_integers(g)


def honeycomb_lattice(L: int) -> nx.Graph:
    g = nx.hexagonal_lattice_graph(L, L, periodic=True)
    return nx.convert_node_labels_to_integers(g)


def triangular_lattice(L: int) -> nx.Graph:
    g = nx.triangular_lattice_graph(L, L, periodic=True)
    return nx.convert_node_labels_to_integers(g)


def vertex_split_family(L: int, frac: float, rng: np.random.Generator) -> nx.Graph:
    """Convert a fraction `frac` of degree-4 square-lattice nodes into pairs of
    degree-3 nodes by vertex splitting. Planarity and connectivity are preserved;
    the 3-valent node fraction rises from 0 (frac=0) toward ~1 (frac=1)."""
    g = square_lattice(L)
    g = nx.Graph(g)
    deg4 = [u for u in g.nodes() if g.degree(u) == 4]
    rng.shuffle(deg4)
    n_split = int(round(frac * len(deg4)))
    next_id = max(g.nodes()) + 1
    for u in deg4[:n_split]:
        nbrs = list(g.neighbors(u))
        if len(nbrs) != 4:
            continue
        # Split u into u (keeps 2 neighbours) and w (takes other 2), join u-w.
        w = next_id
        next_id += 1
        g.add_node(w)
        for x in nbrs[2:]:
            g.remove_edge(u, x)
            g.add_edge(w, x)
        g.add_edge(u, w)
    g.remove_nodes_from([u for u in g.nodes() if g.degree(u) == 0])
    return g


def degree3_fraction(graph: nx.Graph) -> float:
    degs = np.array([d for _, d in graph.degree()])
    junctions = degs[degs >= 3]
    if len(junctions) == 0:
        return float("nan")
    return float(np.mean(junctions == 3))


# --------------------------------------------------------------------------- #
# City anchor.
# --------------------------------------------------------------------------- #
def junction_weights(row: pd.Series) -> tuple[float, float]:
    """3-valent / 4+-valent weights among junction nodes (exclude dead ends and
    degree-2 simplification artifacts)."""
    s3 = float(row["degree3_share"])
    s4 = float(row["degree4plus_share"])
    tot = s3 + s4
    if tot <= 0:
        return float("nan"), float("nan")
    return s3 / tot, s4 / tot


def build_city_table() -> pd.DataFrame:
    cov = pd.read_csv(COV)
    pred = pd.read_csv(PRED)[["city", "macro_region", "pc_road_rank", "pc_cebh"]]
    r68 = pd.read_csv(R68)[["city", "geometry_null_gap_mean", "pc_cebh"]].rename(
        columns={"pc_cebh": "pc_cebh_r68"}
    )
    df = pred.merge(
        cov[["city", "degree3_share", "degree4plus_share", "dead_end_share",
             "edge_circuity_mean", "mean_degree", "kappa"]],
        on="city", how="left",
    )
    r68["pc_geometry_null"] = r68["pc_cebh_r68"] + r68["geometry_null_gap_mean"]
    df = df.merge(r68[["city", "pc_geometry_null"]], on="city", how="left")
    w3, w4 = zip(*df.apply(junction_weights, axis=1))
    df["w3"] = w3
    df["w4"] = w4
    df["pc_anchor"] = df["w3"] * PC_HONEYCOMB + df["w4"] * PC_SQUARE
    df["err_anchor"] = df["pc_anchor"] - df["pc_road_rank"]
    df["err_cebh"] = df["pc_cebh"] - df["pc_road_rank"]
    df["in_lattice_bracket"] = (df["pc_road_rank"] >= PC_SQUARE - 1e-9) & (
        df["pc_road_rank"] <= PC_HONEYCOMB + 1e-9
    )
    return df


def run_lattice_calibration(p_grid_fine: np.ndarray, n_perm: int, seed: int) -> dict:
    out = {}
    for name, builder, L, exact in [
        ("square", square_lattice, 90, PC_SQUARE),
        ("honeycomb", honeycomb_lattice, 60, PC_HONEYCOMB),
        ("triangular", triangular_lattice, 90, PC_TRIANGULAR),
    ]:
        g = builder(L)
        pc = estimate_pc(g, p_grid_fine, n_perm, seed)
        out[name] = {
            "nodes": g.number_of_nodes(),
            "edges": g.number_of_edges(),
            "mean_degree": float(2 * g.number_of_edges() / g.number_of_nodes()),
            "deg3_frac": degree3_fraction(g),
            "pc_estimated": pc,
            "pc_exact": exact,
            "abs_offset": abs(pc - exact),
        }
    return out


def run_interpolation(p_grid_fine: np.ndarray, n_perm: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for frac in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        g = vertex_split_family(80, frac, rng)
        pc = estimate_pc(g, p_grid_fine, n_perm, seed + int(frac * 100))
        f3 = degree3_fraction(g)
        pc_pred = f3 * PC_HONEYCOMB + (1 - f3) * PC_SQUARE
        rows.append({
            "split_fraction": frac,
            "deg3_fraction": f3,
            "nodes": g.number_of_nodes(),
            "edges": g.number_of_edges(),
            "pc_measured": pc,
            "pc_anchor_pred": pc_pred,
            "abs_err": abs(pc - pc_pred),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figure.
# --------------------------------------------------------------------------- #
def _hexagon(ax, cx, cy, r, color):
    ang = np.deg2rad(np.arange(0, 360, 60) + 30)
    xs, ys = cx + r * np.cos(ang), cy + r * np.sin(ang)
    ax.plot(np.append(xs, xs[0]), np.append(ys, ys[0]), color=color, lw=1.0, solid_capstyle="round")


def _honeycomb_patch(ax, x0, y0, color):
    """Draw honeycomb as unique edges so shared cell borders are not overdrawn."""
    r = 0.42
    dx = r * np.sqrt(3)
    dy = r * 1.5
    lw = 0.95
    seen = set()
    for j in range(2):
        for i in range(3):
            cx = x0 + i * dx + (j % 2) * dx / 2
            cy = y0 + j * dy
            ang = np.deg2rad(np.arange(0, 360, 60) + 30)
            verts = np.column_stack([cx + r * np.cos(ang), cy + r * np.sin(ang)])
            for k in range(6):
                a = tuple(np.round(verts[k], 4))
                b = tuple(np.round(verts[(k + 1) % 6], 4))
                key = tuple(sorted([a, b]))
                if key in seen:
                    continue
                seen.add(key)
                ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, solid_capstyle="round")


def _square_patch(ax, x0, y0, color):
    """Draw square lattice as unique grid edges, not overlapping cell boxes."""
    s = 0.58
    cols, rows = 3, 2
    lw = 0.95
    for i in range(cols + 1):
        x = x0 + i * s
        ax.plot([x, x], [y0, y0 + rows * s], color=color, lw=lw, solid_capstyle="round")
    for j in range(rows + 1):
        y = y0 + j * s
        ax.plot([x0, x0 + cols * s], [y, y], color=color, lw=lw, solid_capstyle="round")


def _random_blob(ax, cx, cy, color, seed=7):
    rng = np.random.default_rng(seed)
    pts = rng.normal(0, 0.62, size=(9, 2))
    pts[:, 0] += cx
    pts[:, 1] += cy
    # spanning tree only -> visibly loop-free / tree-like
    order = np.argsort(pts[:, 0])
    for k in range(1, len(order)):
        a, b = pts[order[k]], pts[order[rng.integers(0, k)]]
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=0.8, alpha=0.9, zorder=1)
    ax.scatter(pts[:, 0], pts[:, 1], s=11, color=color, zorder=2, linewidths=0)


def _junction(ax, cx, cy, kind, color):
    L = 0.27
    if kind == 3:  # Y / T three-way
        for ang in (90, 210, 330):
            a = np.deg2rad(ang)
            ax.plot([cx, cx + L * np.cos(a)], [cy, cy + L * np.sin(a)], color=color, lw=1.3, solid_capstyle="round")
    else:  # four-way cross
        for ang in (0, 90, 180, 270):
            a = np.deg2rad(ang)
            ax.plot([cx, cx + L * np.cos(a)], [cy, cy + L * np.sin(a)], color=color, lw=1.3, solid_capstyle="round")
    ax.scatter([cx], [cy], s=9, color=color, zorder=3, linewidths=0)


def _no_crossings_icon(ax, cx, cy, ink, bad):
    """Draw a prohibited free-crossing mark without using a red X glyph."""
    ax.plot([cx - 0.32, cx + 0.32], [cy - 0.18, cy + 0.18],
            color=ink, lw=1.0, solid_capstyle="round", zorder=2)
    ax.plot([cx - 0.32, cx + 0.32], [cy + 0.18, cy - 0.18],
            color=ink, lw=1.0, solid_capstyle="round", zorder=2)
    ax.add_patch(Circle((cx, cy), 0.38, fill=False, edgecolor=bad, lw=1.25, zorder=3))
    ax.plot([cx - 0.27, cx + 0.27], [cy - 0.27, cy + 0.27],
            color=bad, lw=1.25, solid_capstyle="round", zorder=4)


def draw_mechanism_schematic(ax) -> None:
    """Panel a: the physical idea - random graph fails; road junctions are the
    coordinations of the honeycomb and square lattices; the threshold is their
    junction-weighted mixture."""
    ink = pub_style.COLORS["observed"]
    hc = pub_style.COLORS["model"]       # honeycomb / 3-way
    sq = pub_style.COLORS["spatial_null"]  # square / 4-way
    road = pub_style.COLORS["geometry_null"]
    bad = "#C0392B"

    # Station 1: random graph (CEBH).
    _random_blob(ax, 1.35, 1.65, "#8A8F96")
    ax.text(1.35, 0.35, "Random graph\n(locally tree-like)", ha="center", va="top", fontsize=6.2, color=ink)
    ax.text(1.35, 2.95, r"CEBH $p_c\approx0.47$", ha="center", va="bottom", fontsize=6.6, color=bad, fontweight="bold")
    ax.text(1.35, 2.62, "overstates robustness", ha="center", va="bottom", fontsize=5.6, color=bad)

    # cross-out arrow to indicate "roads are not this".
    ax.annotate("", xy=(3.0, 1.6), xytext=(2.35, 1.6),
                arrowprops={"arrowstyle": "-|>", "color": "#9AA0A6", "lw": 1.0})
    ax.text(2.67, 1.95, "roads are\nplanar", ha="center", va="bottom", fontsize=5.6, color=pub_style.COLORS["annot"])

    # Station 2: the two lattice primitives with their junctions.
    _junction(ax, 3.55, 2.52, 3, hc)
    _honeycomb_patch(ax, 3.05, 0.98, hc)
    ax.text(3.95, 0.35, "3-way junction\n= honeycomb", ha="center", va="top", fontsize=6.0, color=hc)
    ax.text(3.95, 3.12, r"$p_c=0.653$", ha="center", va="bottom", fontsize=6.2, color=hc)

    _junction(ax, 6.35, 2.52, 4, sq)
    _square_patch(ax, 5.95, 0.98, sq)
    ax.text(6.65, 0.35, "4-way junction\n= square", ha="center", va="top", fontsize=6.0, color=sq)
    ax.text(6.65, 3.12, r"$p_c=0.500$", ha="center", va="bottom", fontsize=6.2, color=sq)

    # non-crossing constraint mark.
    _no_crossings_icon(ax, 8.42, 1.60, ink, bad)
    ax.text(8.42, 0.92, "planar embedding\n(no free crossings)",
            ha="center", va="top", fontsize=5.25, color=ink, linespacing=0.95)

    ax.annotate("", xy=(9.62, 1.6), xytext=(9.22, 1.6),
                arrowprops={"arrowstyle": "-|>", "color": "#9AA0A6", "lw": 1.0})

    # Station 3: the mixture / anchor.
    ax.text(10.58, 2.78, "Real road network", ha="center", va="bottom", fontsize=6.4, color=ink, fontweight="bold")
    ax.text(10.58, 2.20, r"$p_c=w_3\cdot0.653+w_4\cdot0.500$", ha="center", va="center", fontsize=6.55, color=ink)
    ax.text(10.58, 1.76, r"$w_3,w_4$: 3-way and 4-way shares",
            ha="center", va="center", fontsize=5.35, color=pub_style.COLORS["annot"])
    ax.text(10.58, 1.24, r"observed $p_c\approx0.62$-$0.69$", ha="center", va="center", fontsize=6.35,
            color=road, fontweight="bold")
    ax.text(10.58, 0.72, "predicted with no fitting", ha="center", va="center", fontsize=5.7,
            color=pub_style.COLORS["annot"])

    ax.set_xlim(0.2, 12.2)
    ax.set_ylim(0.0, 3.45)
    ax.set_aspect("equal")
    ax.axis("off")


def make_figure(city: pd.DataFrame, calib: dict, interp: pd.DataFrame, summary: dict) -> None:
    pub_style.apply()
    fig = plt.figure(figsize=(pub_style.FIG_WIDTH_2COL, 7.4), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.74, 1.0, 1.0])
    ax_s = fig.add_subplot(gs[0, :])
    draw_mechanism_schematic(ax_s)
    pub_style.panel_title(ax_s, "a", "Road junctions are planar-lattice coordinations")

    axes = np.array([[fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
                     [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]])

    # b: estimator calibration on clean lattices.
    ax = axes[0, 0]
    names = ["triangular", "square", "honeycomb"]
    exact = [calib[n]["pc_exact"] for n in names]
    est = [calib[n]["pc_estimated"] for n in names]
    ax.plot([0.3, 0.7], [0.3, 0.7], ls="--", lw=0.7, color="#1A1A1A")
    ax.scatter(exact, est, s=46, color=pub_style.COLORS["geometry_null"], zorder=3, linewidths=0)
    for n, x, y in zip(names, exact, est):
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(5, -2), fontsize=6.0,
                    color=pub_style.COLORS["annot"])
    ax.set_xlabel("Exact lattice $p_c$")
    ax.set_ylabel("Estimated $p_c$ (road estimator)")
    pub_style.panel_title(ax, "b", "Estimator reproduces lattice thresholds")
    pub_style.light_grid(ax, axis="both")
    pub_style.annot(ax, 0.04, 0.96,
                    f"max offset = {max(calib[n]['abs_offset'] for n in names):.3f}")

    # b: vertex-split interpolation family.
    ax = axes[0, 1]
    ax.axhline(PC_SQUARE, ls=":", lw=0.7, color=pub_style.COLORS["spatial_null"])
    ax.axhline(PC_HONEYCOMB, ls=":", lw=0.7, color=pub_style.COLORS["model"])
    ax.plot(interp["deg3_fraction"], interp["pc_anchor_pred"], "-", lw=1.0,
            color="#1A1A1A", label="Anchor (zero-parameter)")
    ax.scatter(interp["deg3_fraction"], interp["pc_measured"], s=34,
               color=pub_style.COLORS["geometry_null"], zorder=3, linewidths=0, label="Measured")
    ax.set_xlabel("3-valent junction fraction")
    ax.set_ylabel("Bond-percolation $p_c$")
    pub_style.panel_title(ax, "c", "Vertex-split lattice interpolation")
    pub_style.light_grid(ax, axis="both")
    ax.legend(frameon=False, fontsize=6.0, loc="lower right")
    pub_style.annot(ax, 0.04, 0.30, "honeycomb 0.653", fontsize=5.6)
    pub_style.annot(ax, 0.04, 0.12, "square 0.500", fontsize=5.6)

    # c: 71-city zero-parameter anchor vs observed.
    ax = axes[1, 0]
    lo, hi = 0.45, 0.95
    ax.plot([lo, hi], [lo, hi], ls="--", lw=0.7, color="#1A1A1A")
    ax.axhspan(PC_SQUARE, PC_HONEYCOMB, color="#0072B2", alpha=0.06, zorder=0)
    ax.scatter(city["pc_anchor"], city["pc_road_rank"], s=20,
               color=pub_style.COLORS["geometry_null"], alpha=0.8, linewidths=0,
               label="Lattice anchor")
    ax.scatter(city["pc_cebh"], city["pc_road_rank"], s=16,
               color=pub_style.COLORS["cebh"], alpha=0.6, linewidths=0, marker="x",
               label="CEBH")
    ax.set_xlabel("Predicted $p_c$")
    ax.set_ylabel("Observed road $p_c$")
    pub_style.panel_title(ax, "d", "Zero-parameter anchor (71 cities)")
    pub_style.light_grid(ax, axis="both")
    ax.legend(frameon=False, fontsize=6.0, loc="upper left")
    pub_style.annot(ax, 0.40, 0.16,
                    f"anchor MAE {summary['anchor_mae']:.3f}\nCEBH MAE {summary['cebh_mae']:.3f}")

    # d: absolute-error comparison.
    ax = axes[1, 1]
    data = [city["err_cebh"].abs().to_numpy(), city["err_anchor"].abs().to_numpy()]
    parts = ax.boxplot(data, tick_labels=["CEBH", "Lattice\nanchor"], widths=0.55,
                       patch_artist=True, showfliers=False)
    for patch, c in zip(parts["boxes"], [pub_style.COLORS["cebh"], pub_style.COLORS["geometry_null"]]):
        patch.set_facecolor(c)
        patch.set_alpha(0.25)
        patch.set_edgecolor(c)
    for key in ["whiskers", "caps", "medians"]:
        for art in parts[key]:
            art.set(color="#334155", lw=0.9)
    rng = np.random.default_rng(103)
    for i, vals in enumerate(data, start=1):
        ax.scatter(np.full(len(vals), i) + rng.normal(0, 0.04, len(vals)), vals,
                   s=10, color="#515b66", alpha=0.4, linewidths=0)
    ax.set_ylabel("Absolute threshold error")
    pub_style.panel_title(ax, "e", "Error: degree-moment vs lattice")
    pub_style.light_grid(ax, axis="y")
    pub_style.annot(ax, 0.5, 0.96,
                    f"{summary['error_reduction']:.1f}x lower", ha="center")

    pub_style.save(fig, FIG_BASE)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ROUND.mkdir(parents=True, exist_ok=True)

    p_grid_fine = np.round(np.arange(0.30, 0.760, 0.01), 4)
    n_perm = 24
    seed = 20260611

    print("[R103] calibrating estimator on clean lattices ...", flush=True)
    calib = run_lattice_calibration(p_grid_fine, n_perm, seed)
    for k, v in calib.items():
        print(f"   {k:11s} est={v['pc_estimated']:.4f} exact={v['pc_exact']:.4f} "
              f"offset={v['abs_offset']:.4f}", flush=True)

    print("[R103] vertex-split interpolation family ...", flush=True)
    interp = run_interpolation(p_grid_fine, n_perm, seed)
    print(interp.to_string(index=False), flush=True)

    print("[R103] applying zero-parameter anchor to cities ...", flush=True)
    city = build_city_table()

    anchor_mae = float(city["err_anchor"].abs().mean())
    cebh_mae = float(city["err_cebh"].abs().mean())
    summary = {
        "pc_honeycomb_exact": PC_HONEYCOMB,
        "pc_square_exact": PC_SQUARE,
        "estimator_calibration": calib,
        "interpolation_max_abs_err": float(interp["abs_err"].max()),
        "interpolation_mean_abs_err": float(interp["abs_err"].mean()),
        "n_cities": int(len(city)),
        "anchor_mae": anchor_mae,
        "anchor_bias": float(city["err_anchor"].mean()),
        "cebh_mae": cebh_mae,
        "cebh_bias": float(city["err_cebh"].mean()),
        "error_reduction": cebh_mae / anchor_mae,
        "anchor_spearman_vs_observed": float(spearmanr(city["pc_anchor"], city["pc_road_rank"]).statistic),
        "cebh_spearman_vs_observed": float(spearmanr(city["pc_cebh"], city["pc_road_rank"]).statistic),
        "anchor_pearson_vs_observed": float(pearsonr(city["pc_anchor"], city["pc_road_rank"]).statistic),
        "fraction_in_lattice_bracket": float(city["in_lattice_bracket"].mean()),
        "residual_dead_end_spearman": float(spearmanr(city["dead_end_share"], city["err_anchor"]).statistic),
        "residual_deg4_spearman": float(spearmanr(city["degree4plus_share"], city["err_anchor"]).statistic),
        "residual_circuity_spearman": float(spearmanr(city["edge_circuity_mean"], city["err_anchor"]).statistic),
        "geometry_null_anchor_mae": float((city["pc_anchor"] - city["pc_geometry_null"]).abs().mean()),
    }

    city.to_csv(OUT / "R103_city_lattice_anchor.csv", index=False)
    interp.to_csv(OUT / "R103_vertex_split_interpolation.csv", index=False)
    (OUT / "R103_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    make_figure(city, calib, interp, summary)

    # Copy display figure (manuscript Fig. 3) and source tables into the npj package.
    if NPJ_FIGS.exists():
        for ext in [".svg", ".pdf", ".png", ".tiff"]:
            src = FIG_BASE.with_suffix(ext)
            if src.exists():
                shutil.copy2(src, NPJ_FIGS / f"Fig3_planar_lattice_anchor{ext}")
        NPJ_TABLES.mkdir(parents=True, exist_ok=True)
        for f in ["R103_city_lattice_anchor.csv", "R103_vertex_split_interpolation.csv", "R103_summary.json"]:
            shutil.copy2(OUT / f, NPJ_TABLES / f)
        print("[R103] copied Fig. 3 display figure and source tables into npj package.", flush=True)

    print("\n[R103] SUMMARY", flush=True)
    for k in ["anchor_mae", "cebh_mae", "error_reduction", "anchor_spearman_vs_observed",
              "cebh_spearman_vs_observed", "fraction_in_lattice_bracket",
              "interpolation_max_abs_err", "residual_dead_end_spearman", "residual_deg4_spearman"]:
        print(f"   {k}: {summary[k]:.4f}", flush=True)


if __name__ == "__main__":
    main()
