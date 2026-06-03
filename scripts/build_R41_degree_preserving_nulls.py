"""R41 degree-sequence null models for the road-layer offset mechanism.

This round asks whether the road-layer offset is mainly a degree-distribution
effect or whether spatial road organization adds a threshold penalty beyond a
degree-preserving random null. It uses cached live-OSMnx road graphs only.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_R39 = ROOT / "data" / "R39_threshold_sensitivity" / "threshold_sensitivity_source_data.csv"
DEFAULT_CACHE = ROOT / "cache" / "private_graphs"
DEFAULT_OUT = ROOT / "data" / "R41_degree_preserving_nulls"
DEFAULT_FIGURE = ROOT / "figures" / "Fig_R41_degree_preserving_nulls"


@dataclass
class NullReplicate:
    city: str
    replicate: int
    n_nodes: int
    n_edges: int
    lcc_fraction_p1: float
    pc_cebh_null: float
    pc_rank_null: float
    pc_rank_ci_low: float
    pc_rank_ci_high: float
    threshold_boundary_hit: bool


@dataclass
class CityNullSummary:
    city: str
    graph_cache_file: str
    n_nodes_road: int
    n_edges_road: int
    pc_cebh_road: float
    pc_road_fine_grid: float
    road_gap: float
    pc_null_rank_mean: float
    pc_null_rank_ci_low: float
    pc_null_rank_ci_high: float
    pc_null_cebh_mean: float
    null_gap: float
    road_minus_null: float
    organization_fraction_of_gap: float
    n_null: int
    n_perm: int
    p_grid_min: float
    p_grid_max: float
    p_grid_step: float
    boundary_hit_count: int
    status: str
    notes: str


def safe_slug(text: str) -> str:
    keep: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            keep.append(ch)
        elif ch in {" ", "-", "_", ","}:
            keep.append("_")
    slug = "".join(keep).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "unknown"


def find_cached_graph(cache_dir: Path, city: str) -> Path:
    matches = sorted(cache_dir.glob(f"{safe_slug(city)}_*__road.pkl"))
    if not matches:
        raise FileNotFoundError(f"No cached road graph found for city={city!r}")
    return matches[0]


def load_graph(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def read_r39(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        for key in ["pc_cebh", "pc_fine_grid", "delta_fine_grid"]:
            row[key] = float(row[key])
    return rows


def graph_arrays(G) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    import networkx as nx

    if G.is_directed():
        G = G.to_undirected()
    G.remove_edges_from(nx.selfloop_edges(G))
    H = nx.convert_node_labels_to_integers(G, ordering="default")
    degrees = np.asarray([d for _, d in H.degree()], dtype=np.int64)
    edges = np.asarray(list(H.edges()), dtype=np.int64)
    if edges.size == 0:
        return H.number_of_nodes(), np.array([], dtype=np.int64), np.array([], dtype=np.int64), degrees
    return H.number_of_nodes(), edges[:, 0], edges[:, 1], degrees


def cebh_from_degrees(degrees: np.ndarray) -> float:
    if len(degrees) == 0:
        return float("nan")
    k = float(np.mean(degrees))
    k2 = float(np.mean(degrees.astype(float) ** 2))
    denom = k2 - k
    if denom <= 0:
        return float("nan")
    return k / denom


def _find(parent: np.ndarray, x: int) -> int:
    while int(parent[x]) != x:
        parent[x] = parent[int(parent[x])]
        x = int(parent[x])
    return x


def rank_order_samples(
    n: int,
    src: np.ndarray,
    dst: np.ndarray,
    p_grid: np.ndarray,
    n_perm: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = len(src)
    samples = np.zeros((len(p_grid), n_perm), dtype=float)
    if n == 0:
        return samples
    if m == 0:
        samples[:, :] = 1.0 / n
        return samples
    target_ks = np.clip(np.rint(p_grid * m).astype(int), 0, m)
    for b in range(n_perm):
        parent = np.arange(n, dtype=np.int32)
        size = np.ones(n, dtype=np.int32)
        largest = 1
        t = 0
        while t < len(target_ks) and target_ks[t] == 0:
            samples[t, b] = largest / n
            t += 1
        order = rng.permutation(m)
        for step, edge_idx in enumerate(order, start=1):
            u = int(src[edge_idx])
            v = int(dst[edge_idx])
            ru = _find(parent, u)
            rv = _find(parent, v)
            if ru != rv:
                if size[ru] < size[rv]:
                    ru, rv = rv, ru
                parent[rv] = ru
                size[ru] += size[rv]
                if int(size[ru]) > largest:
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


def estimate_pc_index(p_grid: np.ndarray, mean_curve: np.ndarray) -> int:
    if len(p_grid) < 3:
        return int(np.argmax(mean_curve))
    gradient = np.gradient(mean_curve, p_grid)
    return int(np.argmax(gradient))


def bootstrap_pc_ci(
    p_grid: np.ndarray,
    samples: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float, bool]:
    rng = np.random.default_rng(seed)
    mean_curve = samples.mean(axis=1)
    pc_idx = estimate_pc_index(p_grid, mean_curve)
    pc = float(p_grid[pc_idx])
    n_rep = samples.shape[1]
    boot = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_rep, size=n_rep)
        curve = samples[:, idx].mean(axis=1)
        boot.append(float(p_grid[estimate_pc_index(p_grid, curve)]))
    lo, hi = np.percentile(np.asarray(boot), [2.5, 97.5])
    return pc, float(lo), float(hi), pc_idx in {0, len(p_grid) - 1}


def lcc_fraction(G) -> float:
    import networkx as nx

    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    return max((len(c) for c in nx.connected_components(G)), default=0) / n


def generate_configuration_projection(degrees: np.ndarray, seed: int):
    import networkx as nx

    multi = nx.configuration_model([int(x) for x in degrees], seed=seed)
    G = nx.Graph(multi)
    G.remove_edges_from(nx.selfloop_edges(G))
    G.add_nodes_from(range(len(degrees)))
    return G


def null_p_grid(pc_cebh: float, width: float, step: float) -> np.ndarray:
    p_min = max(step, pc_cebh - width)
    p_max = min(0.98, pc_cebh + width)
    grid = np.arange(p_min, p_max + step / 2, step)
    return np.unique(np.round(grid, 6))


def run_city(
    city_row: dict[str, Any],
    cache_dir: Path,
    n_null: int,
    n_perm: int,
    n_bootstrap: int,
    p_step: float,
    window: float,
    seed: int,
) -> tuple[CityNullSummary, list[NullReplicate]]:
    city = str(city_row["city"])
    pc_cebh = float(city_row["pc_cebh"])
    pc_road = float(city_row["pc_fine_grid"])
    graph_path = find_cached_graph(cache_dir, city)
    G_road = load_graph(graph_path)
    n_road, src_road, _dst_road, degrees = graph_arrays(G_road)
    m_road = len(src_road)
    p_grid = null_p_grid(pc_cebh, window, p_step)
    reps: list[NullReplicate] = []
    for rep in range(n_null):
        null_seed = seed + 1009 * rep
        G_null = generate_configuration_projection(degrees, null_seed)
        n_null_nodes, src, dst, null_degrees = graph_arrays(G_null)
        samples = rank_order_samples(
            n_null_nodes,
            src,
            dst,
            p_grid,
            n_perm=n_perm,
            seed=null_seed + 101,
        )
        pc, lo, hi, boundary = bootstrap_pc_ci(
            p_grid,
            samples,
            n_bootstrap=n_bootstrap,
            seed=null_seed + 202,
        )
        reps.append(
            NullReplicate(
                city=city,
                replicate=rep + 1,
                n_nodes=n_null_nodes,
                n_edges=len(src),
                lcc_fraction_p1=lcc_fraction(G_null),
                pc_cebh_null=cebh_from_degrees(null_degrees),
                pc_rank_null=pc,
                pc_rank_ci_low=lo,
                pc_rank_ci_high=hi,
                threshold_boundary_hit=boundary,
            )
        )
        del G_null, samples
        gc.collect()

    null_pcs = np.asarray([rep.pc_rank_null for rep in reps], dtype=float)
    null_cebh = np.asarray([rep.pc_cebh_null for rep in reps], dtype=float)
    pc_null_mean = float(null_pcs.mean())
    ci_low, ci_high = np.percentile(null_pcs, [2.5, 97.5])
    pc_null_cebh = float(np.nanmean(null_cebh))
    road_gap = pc_road - pc_cebh
    null_gap = pc_null_mean - pc_null_cebh
    road_minus_null = pc_road - pc_null_mean
    org_fraction = road_minus_null / road_gap if abs(road_gap) > 1e-12 else float("nan")
    boundary_hits = int(sum(rep.threshold_boundary_hit for rep in reps))
    if boundary_hits:
        status = "risk_boundary_hit"
        notes = "At least one null replicate selected the edge of the null p-grid."
    elif road_minus_null <= 0:
        status = "risk_no_spatial_penalty"
        notes = "Degree-preserving null threshold is not below the observed road threshold."
    else:
        status = "pass"
        notes = "Observed road threshold exceeds degree-preserving null threshold."

    return (
        CityNullSummary(
            city=city,
            graph_cache_file=str(graph_path.relative_to(ROOT)),
            n_nodes_road=n_road,
            n_edges_road=m_road,
            pc_cebh_road=pc_cebh,
            pc_road_fine_grid=pc_road,
            road_gap=road_gap,
            pc_null_rank_mean=pc_null_mean,
            pc_null_rank_ci_low=float(ci_low),
            pc_null_rank_ci_high=float(ci_high),
            pc_null_cebh_mean=pc_null_cebh,
            null_gap=null_gap,
            road_minus_null=road_minus_null,
            organization_fraction_of_gap=org_fraction,
            n_null=n_null,
            n_perm=n_perm,
            p_grid_min=float(p_grid.min()),
            p_grid_max=float(p_grid.max()),
            p_grid_step=p_step,
            boundary_hit_count=boundary_hits,
            status=status,
            notes=notes,
        ),
        reps,
    )


def round_float(value: Any) -> Any:
    if isinstance(value, float):
        if math.isfinite(value):
            return round(value, 6)
        return None
    return value


def write_csv(items: list[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(items[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            writer.writerow({key: round_float(value) for key, value in asdict(item).items()})


def bootstrap_mean(values: np.ndarray, n_bootstrap: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    boot = np.zeros(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, len(values), size=len(values))
        boot[i] = values[idx].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def plot_figure(rows: list[CityNullSummary], out_stem: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    plt.rcParams.update(
        {
            "font.size": 9,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    rows = sorted(rows, key=lambda row: row.pc_cebh_road)
    cities = [row.city for row in rows]
    x = np.arange(len(rows))
    color_cebh = "#4a6f2a"
    color_null = "#6f6f96"
    color_road = "#b04a2f"
    color_penalty = "#d7b365"

    fig = plt.figure(figsize=(13.2, 8.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[0.95, 1.25], height_ratios=[0.85, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # Panel a: null-model design.
    ax_a.axis("off")
    boxes = [
        (0.08, 0.64, "Observed road graph\nfixed spatial organization", "#f4eadf"),
        (0.49, 0.75, "Keep degree\nsequence", "#eef3df"),
        (0.49, 0.47, "Randomize\nconnections", "#e5eff7"),
        (0.82, 0.61, "Degree-preserving\nnull threshold", "#efe6c8"),
    ]
    for x0, y0, text, color in boxes:
        ax_a.text(
            x0,
            y0,
            text,
            ha="center",
            va="center",
            transform=ax_a.transAxes,
            bbox=dict(boxstyle="round,pad=0.35", fc=color, ec="#4f4f4f", lw=0.9),
        )
    for start, end in [
        ((0.23, 0.64), (0.38, 0.73)),
        ((0.23, 0.64), (0.38, 0.49)),
        ((0.61, 0.73), (0.70, 0.63)),
        ((0.61, 0.49), (0.70, 0.59)),
    ]:
        ax_a.add_patch(
            FancyArrowPatch(start, end, transform=ax_a.transAxes, arrowstyle="->", mutation_scale=12, lw=1.4, color="#555555")
        )
    ax_a.text(
        0.04,
        0.15,
        "If CEBH is accurate on the degree-preserving null but not on the road graph,\nthe missing term is organizational rather than degree-only.",
        transform=ax_a.transAxes,
        fontsize=9,
        color="#333333",
    )
    ax_a.set_title("a  Degree-preserving null-model test")

    # Panel b: threshold ladder by city.
    for i, row in enumerate(rows):
        ax_b.plot([i, i], [row.pc_cebh_road, row.pc_road_fine_grid], color="#c9c9c9", lw=1.2, zorder=1)
        ax_b.scatter(i - 0.16, row.pc_cebh_road, marker="_", s=220, color=color_cebh, lw=2.2, label="CEBH road" if i == 0 else None, zorder=4)
        ax_b.errorbar(
            i,
            row.pc_null_rank_mean,
            yerr=[
                [row.pc_null_rank_mean - row.pc_null_rank_ci_low],
                [row.pc_null_rank_ci_high - row.pc_null_rank_mean],
            ],
            fmt="o",
            color=color_null,
            capsize=3,
            ms=5,
            label="Degree-preserving null" if i == 0 else None,
            zorder=5,
        )
        ax_b.scatter(i + 0.16, row.pc_road_fine_grid, marker="D", color=color_road, s=36, label="Observed road" if i == 0 else None, zorder=5)
    ax_b.set_xticks(x, cities, rotation=35, ha="right")
    ax_b.set_ylabel("Bond threshold estimate")
    ax_b.set_title("b  Randomizing organization collapses most of the road gap")
    ax_b.grid(axis="y", color="#ececec")
    ax_b.legend(loc="upper left", ncols=3, fontsize=8)

    # Panel c: aggregate gap decomposition.
    road_gaps = np.asarray([row.road_gap for row in rows])
    null_gaps = np.asarray([row.null_gap for row in rows])
    penalties = np.asarray([row.road_minus_null for row in rows])
    labels = ["Road gap\nroad - CEBH", "Null gap\nnull - null CEBH", "Organization\nroad - null"]
    values = [road_gaps.mean(), null_gaps.mean(), penalties.mean()]
    spreads = [road_gaps.std(ddof=1), null_gaps.std(ddof=1), penalties.std(ddof=1)]
    colors = [color_road, color_null, color_penalty]
    ax_c.bar(np.arange(3), values, yerr=spreads, color=colors, alpha=0.86, capsize=4)
    for i, vals in enumerate([road_gaps, null_gaps, penalties]):
        jitter = np.linspace(-0.08, 0.08, len(vals))
        ax_c.scatter(np.full(len(vals), i) + jitter, vals, color="white", edgecolor="#333333", s=28, zorder=5)
    ax_c.set_xticks(np.arange(3), labels)
    ax_c.set_ylabel("Threshold difference")
    ax_c.set_title("c  Gap decomposition across eight cities")
    ax_c.grid(axis="y", color="#ececec")

    # Panel d: per-city organizational penalty fraction.
    fractions = np.asarray([row.organization_fraction_of_gap for row in rows])
    ax_d.axhline(0, color="#444444", lw=0.8)
    ax_d.axhline(1, color="#b8b8b8", lw=0.9, linestyle="--")
    ax_d.plot(x, fractions, "o-", color=color_penalty, lw=1.5, ms=5)
    for i, row in enumerate(rows):
        ax_d.text(i, fractions[i], row.city, fontsize=7, ha="left", va="bottom")
    ax_d.set_xticks(x, cities, rotation=35, ha="right")
    ax_d.set_ylabel("(road threshold - null threshold) / road gap")
    ax_d.set_title("d  Share of road gap beyond degree sequence")
    ax_d.set_ylim(max(-0.05, np.nanmin(fractions) - 0.1), min(1.35, np.nanmax(fractions) + 0.18))
    ax_d.grid(axis="y", color="#ececec")

    fig.suptitle("R41 degree-preserving nulls separate degree sequence from road organization", fontsize=13, fontweight="bold")
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=320 if ext == "png" else None)
    plt.close(fig)


def write_report(payload: dict[str, Any], path: Path) -> None:
    agg = payload["aggregate"]
    lines = [
        "# R41 Degree-Preserving Null Model Report",
        "",
        "## Purpose",
        "",
        "R41 tests whether the road-layer offset is mostly degree-sequence driven or whether road organization adds a threshold penalty beyond a degree-preserving random null.",
        "",
        "## Main Results",
        "",
        f"- Mean observed road gap: {agg['road_gap_mean']:.3f} [95% bootstrap CI {agg['road_gap_ci_95'][0]:.3f}-{agg['road_gap_ci_95'][1]:.3f}].",
        f"- Mean degree-preserving null gap: {agg['null_gap_mean']:.3f} [95% bootstrap CI {agg['null_gap_ci_95'][0]:.3f}-{agg['null_gap_ci_95'][1]:.3f}].",
        f"- Mean road-minus-null organizational penalty: {agg['organization_penalty_mean']:.3f} [95% bootstrap CI {agg['organization_penalty_ci_95'][0]:.3f}-{agg['organization_penalty_ci_95'][1]:.3f}].",
        f"- Mean fraction of road gap beyond degree sequence: {agg['organization_fraction_mean']:.3f}.",
        "",
        "## City-Level Summary",
        "",
        "| City | CEBH | Road fine-grid | Null rank | Road gap | Null gap | Road-null | Org fraction | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["records"]:
        lines.append(
            f"| {row['city']} | {row['pc_cebh_road']:.3f} | {row['pc_road_fine_grid']:.3f} | "
            f"{row['pc_null_rank_mean']:.3f} | {row['road_gap']:.3f} | {row['null_gap']:.3f} | "
            f"{row['road_minus_null']:.3f} | {row['organization_fraction_of_gap']:.3f} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["recommended_interpretation"],
            "",
            "## Boundary",
            "",
            "- Allowed: argue that degree sequence alone does not reproduce the observed road threshold scale.",
            "- Not allowed: claim that the null model proves planarity, causality or universality.",
            "- Required next: add controlled planar ensembles or expand the method-coherent road matrix before mechanistic finality.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.time()
    city_rows = read_r39(args.r39_source)
    summaries: list[CityNullSummary] = []
    reps: list[NullReplicate] = []
    for city_index, city_row in enumerate(city_rows):
        summary, city_reps = run_city(
            city_row,
            args.graph_cache_dir,
            n_null=args.n_null,
            n_perm=args.n_perm,
            n_bootstrap=args.n_bootstrap,
            p_step=args.p_step,
            window=args.window,
            seed=args.seed + 100_003 * city_index,
        )
        summaries.append(summary)
        reps.extend(city_reps)

    road_gap = np.asarray([row.road_gap for row in summaries], dtype=float)
    null_gap = np.asarray([row.null_gap for row in summaries], dtype=float)
    penalty = np.asarray([row.road_minus_null for row in summaries], dtype=float)
    fraction = np.asarray([row.organization_fraction_of_gap for row in summaries], dtype=float)
    road_mean, road_lo, road_hi = bootstrap_mean(road_gap, args.n_bootstrap, args.seed + 77)
    null_mean, null_lo, null_hi = bootstrap_mean(null_gap, args.n_bootstrap, args.seed + 88)
    penalty_mean, penalty_lo, penalty_hi = bootstrap_mean(penalty, args.n_bootstrap, args.seed + 99)

    if penalty_mean > 0 and null_mean < road_mean / 2:
        interpretation = (
            "R41 supports a mechanism upgrade: the observed road thresholds remain "
            "well above degree-preserving random null thresholds, while the null "
            "gap is much smaller than the road gap. This points to road organization "
            "as the missing term beyond degree sequence, but it remains a null-model "
            "mechanism test rather than a proof of universality."
        )
    elif penalty_mean > 0:
        interpretation = (
            "R41 supports an organizational contribution beyond degree sequence, "
            "but the degree-preserving null still retains a substantial gap. The "
            "mechanism claim should remain bounded."
        )
    else:
        interpretation = (
            "R41 does not support an organizational penalty: degree-preserving nulls "
            "match or exceed observed road thresholds. The mechanism claim should be "
            "downgraded."
        )

    payload = {
        "_schema": "R41-degree-preserving-nulls-v1",
        "parameters": {
            "r39_source": str(args.r39_source.relative_to(ROOT)),
            "graph_cache_dir": str(args.graph_cache_dir.relative_to(ROOT)),
            "n_null": args.n_null,
            "n_perm": args.n_perm,
            "n_bootstrap": args.n_bootstrap,
            "p_step": args.p_step,
            "window": args.window,
            "seed": args.seed,
        },
        "aggregate": {
            "n_cities": len(summaries),
            "road_gap_mean": round(road_mean, 6),
            "road_gap_ci_95": [round(road_lo, 6), round(road_hi, 6)],
            "null_gap_mean": round(null_mean, 6),
            "null_gap_ci_95": [round(null_lo, 6), round(null_hi, 6)],
            "organization_penalty_mean": round(penalty_mean, 6),
            "organization_penalty_ci_95": [round(penalty_lo, 6), round(penalty_hi, 6)],
            "organization_fraction_mean": round(float(np.nanmean(fraction)), 6),
            "boundary_hit_cities": [row.city for row in summaries if row.boundary_hit_count],
            "elapsed_seconds": round(time.time() - t0, 3),
        },
        "records": [
            {key: round_float(value) for key, value in asdict(row).items()}
            for row in summaries
        ],
        "replicates": [
            {key: round_float(value) for key, value in asdict(rep).items()}
            for rep in reps
        ],
        "recommended_interpretation": interpretation,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "degree_preserving_nulls_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(summaries, args.out_dir / "degree_preserving_nulls_city_summary.csv")
    write_csv(reps, args.out_dir / "degree_preserving_nulls_replicates.csv")
    write_report(payload, args.out_dir / "degree_preserving_nulls_report.md")
    plot_figure(summaries, args.figure_stem)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r39_source", type=Path, default=DEFAULT_R39)
    parser.add_argument("--graph_cache_dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--figure_stem", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--n_null", type=int, default=6)
    parser.add_argument("--n_perm", type=int, default=20)
    parser.add_argument("--n_bootstrap", type=int, default=300)
    parser.add_argument("--p_step", type=float, default=0.01)
    parser.add_argument("--window", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260531)
    args = parser.parse_args()
    payload = run(args)
    print(
        json.dumps(
            {
                "n_cities": payload["aggregate"]["n_cities"],
                "road_gap_mean": payload["aggregate"]["road_gap_mean"],
                "null_gap_mean": payload["aggregate"]["null_gap_mean"],
                "organization_penalty_mean": payload["aggregate"]["organization_penalty_mean"],
                "organization_fraction_mean": payload["aggregate"]["organization_fraction_mean"],
                "out_dir": str(args.out_dir),
                "figure": str(args.figure_stem.with_suffix(".png")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
