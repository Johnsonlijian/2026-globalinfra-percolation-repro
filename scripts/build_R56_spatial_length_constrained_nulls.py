"""R56 degree-preserving, spatial-scale-constrained null models.

This round tests whether the R41 degree-preserving null gap changes after
adding a real spatial constraint. The null keeps the road degree sequence
exactly, rewires edges by double-edge swaps, and accepts only swaps whose new
Euclidean chord lengths remain in the original length slots up to a small
bin tolerance. It then re-estimates rank-order bond-percolation thresholds.
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
DEFAULT_R41 = ROOT / "data" / "R41_degree_preserving_nulls" / "degree_preserving_nulls_city_summary.csv"
DEFAULT_CACHE = ROOT / "cache" / "private_graphs"
DEFAULT_OUT = ROOT / "data" / "R56_spatial_length_constrained_nulls"
DEFAULT_FIGURE = ROOT / "figures" / "Fig_R56_spatial_length_constrained_nulls"


@dataclass
class SpatialNullReplicate:
    city: str
    replicate: int
    n_nodes: int
    n_edges: int
    pc_cebh_null: float
    pc_rank_null: float
    pc_rank_ci_low: float
    pc_rank_ci_high: float
    lcc_fraction_p1: float
    accepted_swaps: int
    attempted_swaps: int
    acceptance_rate: float
    accepted_swaps_per_edge: float
    degree_max_abs_drift: int
    chord_length_mean_ratio: float
    chord_length_median_ratio: float
    chord_length_bin_tvd: float
    length_rejects: int
    topology_rejects: int
    threshold_boundary_hit: bool


@dataclass
class SpatialNullSummary:
    city: str
    graph_cache_file: str
    n_nodes_road: int
    n_edges_road: int
    pc_cebh_road: float
    pc_road_fine_grid: float
    road_gap: float
    degree_null_gap_R41: float
    degree_null_rank_mean_R41: float
    spatial_null_rank_mean: float
    spatial_null_rank_ci_low: float
    spatial_null_rank_ci_high: float
    spatial_null_cebh_mean: float
    spatial_null_gap: float
    spatial_minus_degree_gap: float
    road_minus_spatial: float
    spatial_fraction_of_road_gap: float
    remaining_fraction_after_spatial: float
    accepted_swaps_per_edge_mean: float
    acceptance_rate_mean: float
    chord_length_bin_tvd_mean: float
    degree_max_abs_drift_max: int
    n_null: int
    n_perm: int
    length_bins: int
    bin_tolerance: int
    swap_fraction_target: float
    p_grid_min: float
    p_grid_max: float
    p_grid_step: float
    boundary_hit_count: int
    status: str
    notes: str


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    numeric = {
        "n_nodes_road",
        "n_edges_road",
        "pc_cebh_road",
        "pc_road_fine_grid",
        "road_gap",
        "pc_null_rank_mean",
        "null_gap",
    }
    for row in rows:
        for key in list(row):
            if key in numeric and row[key] != "":
                row[key] = float(row[key])
    return rows


def stable_city_seed(city: str) -> int:
    total = 0
    for i, ch in enumerate(city.lower(), start=1):
        total += i * ord(ch)
    return total


def load_graph(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def chord_length_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle chord distance in meters from OSM lon/lat node coordinates."""
    r = 6_371_008.8
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = phi2 - phi1
    d_lam = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2.0) ** 2
    return 2.0 * r * math.asin(min(1.0, math.sqrt(max(0.0, a))))


def to_simple_spatial_graph(G):
    import networkx as nx

    if G.is_directed():
        G = G.to_undirected()
    H = nx.Graph(G)
    H.remove_edges_from(nx.selfloop_edges(H))
    H = nx.convert_node_labels_to_integers(H, ordering="default")
    n = H.number_of_nodes()
    lon = np.full(n, np.nan, dtype=float)
    lat = np.full(n, np.nan, dtype=float)
    for node, data in H.nodes(data=True):
        lon[int(node)] = float(data.get("x", np.nan))
        lat[int(node)] = float(data.get("y", np.nan))
    if np.isnan(lon).any() or np.isnan(lat).any():
        missing = int(np.isnan(lon).sum() + np.isnan(lat).sum())
        raise ValueError(f"Missing node coordinates for {missing} node-coordinate fields")
    edges = np.asarray(list(H.edges()), dtype=np.int64)
    if edges.size == 0:
        edges = np.empty((0, 2), dtype=np.int64)
    degrees = np.asarray([d for _, d in H.degree()], dtype=np.int64)
    return H, edges, degrees, lon, lat


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


def null_p_grid(pc_cebh: float, width: float, step: float) -> np.ndarray:
    p_min = max(step, pc_cebh - width)
    p_max = min(0.98, pc_cebh + width)
    grid = np.arange(p_min, p_max + step / 2, step)
    return np.unique(np.round(grid, 6))


def make_length_bins(lengths: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    finite = lengths[np.isfinite(lengths) & (lengths > 0)]
    if len(finite) == 0:
        raise ValueError("No positive chord lengths")
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(finite, qs)
    edges[0] = 0.0
    edges[-1] = max(float(edges[-1]) * 1.000001, float(edges[-1]) + 1e-6)
    edges = np.unique(edges)
    if len(edges) < 3:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
        edges = np.linspace(max(0.0, lo * 0.999), hi * 1.001 + 1e-6, min(n_bins, len(finite)) + 1)
    bins = np.searchsorted(edges, lengths, side="right") - 1
    bins = np.clip(bins, 0, len(edges) - 2).astype(np.int16)
    return edges, bins


def edge_key(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def chord_lengths_for_edges(edges: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    out = np.zeros(len(edges), dtype=float)
    for i, (u, v) in enumerate(edges):
        out[i] = chord_length_m(float(lon[u]), float(lat[u]), float(lon[v]), float(lat[v]))
    return out


def rewire_spatial_slots(
    edges: np.ndarray,
    degrees: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    bin_edges: np.ndarray,
    target_bins: np.ndarray,
    rng: np.random.Generator,
    swap_fraction: float,
    bin_tolerance: int,
    max_attempt_factor: int,
) -> tuple[np.ndarray, dict[str, float]]:
    m = len(edges)
    work = np.asarray(edges, dtype=np.int64).copy()
    edge_set = {edge_key(int(u), int(v)) for u, v in work}
    target_accept = int(round(max(0.0, swap_fraction) * m))
    max_attempts = max(target_accept * max_attempt_factor, target_accept + 1)
    accepted = 0
    attempts = 0
    topology_rejects = 0
    length_rejects = 0

    while accepted < target_accept and attempts < max_attempts:
        attempts += 1
        i = int(rng.integers(0, m))
        j = int(rng.integers(0, m - 1))
        if j >= i:
            j += 1

        a, b = int(work[i, 0]), int(work[i, 1])
        c, d = int(work[j, 0]), int(work[j, 1])
        if len({a, b, c, d}) < 4:
            topology_rejects += 1
            continue

        if bool(rng.integers(0, 2)):
            e1 = edge_key(a, d)
            e2 = edge_key(c, b)
        else:
            e1 = edge_key(a, c)
            e2 = edge_key(b, d)

        if e1[0] == e1[1] or e2[0] == e2[1] or e1 == e2:
            topology_rejects += 1
            continue
        old1 = edge_key(a, b)
        old2 = edge_key(c, d)
        if (e1 != old1 and e1 != old2 and e1 in edge_set) or (e2 != old1 and e2 != old2 and e2 in edge_set):
            topology_rejects += 1
            continue

        len1 = chord_length_m(float(lon[e1[0]]), float(lat[e1[0]]), float(lon[e1[1]]), float(lat[e1[1]]))
        len2 = chord_length_m(float(lon[e2[0]]), float(lat[e2[0]]), float(lon[e2[1]]), float(lat[e2[1]]))
        bin1 = int(np.searchsorted(bin_edges, len1, side="right") - 1)
        bin2 = int(np.searchsorted(bin_edges, len2, side="right") - 1)
        bin1 = int(np.clip(bin1, 0, len(bin_edges) - 2))
        bin2 = int(np.clip(bin2, 0, len(bin_edges) - 2))
        if abs(bin1 - int(target_bins[i])) > bin_tolerance or abs(bin2 - int(target_bins[j])) > bin_tolerance:
            length_rejects += 1
            continue

        edge_set.remove(old1)
        edge_set.remove(old2)
        edge_set.add(e1)
        edge_set.add(e2)
        work[i, :] = e1
        work[j, :] = e2
        accepted += 1

    # Diagnostics.
    deg_after = np.zeros(len(degrees), dtype=np.int64)
    np.add.at(deg_after, work[:, 0], 1)
    np.add.at(deg_after, work[:, 1], 1)
    chord_after = chord_lengths_for_edges(work, lon, lat)
    bins_after = np.searchsorted(bin_edges, chord_after, side="right") - 1
    bins_after = np.clip(bins_after, 0, len(bin_edges) - 2)
    hist_target = np.bincount(target_bins.astype(int), minlength=len(bin_edges) - 1).astype(float)
    hist_after = np.bincount(bins_after.astype(int), minlength=len(bin_edges) - 1).astype(float)
    if hist_target.sum() > 0:
        hist_target /= hist_target.sum()
    if hist_after.sum() > 0:
        hist_after /= hist_after.sum()
    tvd = float(0.5 * np.abs(hist_after - hist_target).sum())
    chord_before = chord_lengths_for_edges(edges, lon, lat)
    diagnostics = {
        "accepted_swaps": float(accepted),
        "attempted_swaps": float(attempts),
        "acceptance_rate": float(accepted / attempts) if attempts else 0.0,
        "accepted_swaps_per_edge": float(accepted / m) if m else 0.0,
        "degree_max_abs_drift": float(np.max(np.abs(deg_after - degrees))) if len(degrees) else 0.0,
        "chord_length_mean_ratio": float(np.mean(chord_after) / np.mean(chord_before)),
        "chord_length_median_ratio": float(np.median(chord_after) / np.median(chord_before)),
        "chord_length_bin_tvd": tvd,
        "length_rejects": float(length_rejects),
        "topology_rejects": float(topology_rejects),
    }
    return work, diagnostics


def lcc_fraction_from_edges(n: int, edges: np.ndarray) -> float:
    import networkx as nx

    H = nx.Graph()
    H.add_nodes_from(range(n))
    H.add_edges_from((int(u), int(v)) for u, v in edges)
    if n == 0:
        return 0.0
    return max((len(c) for c in nx.connected_components(H)), default=0) / n


def run_city(
    row: dict[str, Any],
    cache_dir: Path,
    n_null: int,
    n_perm: int,
    n_bootstrap: int,
    p_step: float,
    window: float,
    length_bins: int,
    bin_tolerance: int,
    swap_fraction: float,
    max_attempt_factor: int,
    seed: int,
) -> tuple[SpatialNullSummary, list[SpatialNullReplicate]]:
    city = str(row["city"])
    graph_path = ROOT / str(row["graph_cache_file"])
    if not graph_path.exists():
        graph_path = cache_dir / Path(str(row["graph_cache_file"])).name
    G = load_graph(graph_path)
    H, edges, degrees, lon, lat = to_simple_spatial_graph(G)
    n = H.number_of_nodes()
    m = len(edges)
    pc_cebh = float(row["pc_cebh_road"])
    pc_road = float(row["pc_road_fine_grid"])
    road_gap = float(row["road_gap"])
    degree_null_gap = float(row["null_gap"])
    degree_null_rank = float(row["pc_null_rank_mean"])
    p_grid = null_p_grid(pc_cebh, window, p_step)
    chord_before = chord_lengths_for_edges(edges, lon, lat)
    bin_edges, target_bins = make_length_bins(chord_before, length_bins)

    reps: list[SpatialNullReplicate] = []
    for rep in range(n_null):
        rep_seed = seed + 100_003 * rep + 977 * stable_city_seed(city)
        rng = np.random.default_rng(rep_seed)
        rewired, diag = rewire_spatial_slots(
            edges=edges,
            degrees=degrees,
            lon=lon,
            lat=lat,
            bin_edges=bin_edges,
            target_bins=target_bins,
            rng=rng,
            swap_fraction=swap_fraction,
            bin_tolerance=bin_tolerance,
            max_attempt_factor=max_attempt_factor,
        )
        src = rewired[:, 0]
        dst = rewired[:, 1]
        samples = rank_order_samples(n, src, dst, p_grid, n_perm, rep_seed + 71)
        pc_rank, ci_low, ci_high, boundary = bootstrap_pc_ci(p_grid, samples, n_bootstrap, rep_seed + 313)
        reps.append(
            SpatialNullReplicate(
                city=city,
                replicate=rep + 1,
                n_nodes=n,
                n_edges=m,
                pc_cebh_null=cebh_from_degrees(degrees),
                pc_rank_null=pc_rank,
                pc_rank_ci_low=ci_low,
                pc_rank_ci_high=ci_high,
                lcc_fraction_p1=lcc_fraction_from_edges(n, rewired),
                accepted_swaps=int(diag["accepted_swaps"]),
                attempted_swaps=int(diag["attempted_swaps"]),
                acceptance_rate=float(diag["acceptance_rate"]),
                accepted_swaps_per_edge=float(diag["accepted_swaps_per_edge"]),
                degree_max_abs_drift=int(diag["degree_max_abs_drift"]),
                chord_length_mean_ratio=float(diag["chord_length_mean_ratio"]),
                chord_length_median_ratio=float(diag["chord_length_median_ratio"]),
                chord_length_bin_tvd=float(diag["chord_length_bin_tvd"]),
                length_rejects=int(diag["length_rejects"]),
                topology_rejects=int(diag["topology_rejects"]),
                threshold_boundary_hit=boundary,
            )
        )
        gc.collect()

    pc_values = np.asarray([r.pc_rank_null for r in reps], dtype=float)
    ci_lows = np.asarray([r.pc_rank_ci_low for r in reps], dtype=float)
    ci_highs = np.asarray([r.pc_rank_ci_high for r in reps], dtype=float)
    spatial_rank = float(np.mean(pc_values))
    spatial_gap = spatial_rank - pc_cebh
    road_minus_spatial = pc_road - spatial_rank
    remaining = road_minus_spatial / road_gap if road_gap else float("nan")
    spatial_fraction = spatial_gap / road_gap if road_gap else float("nan")
    drift_max = int(max(r.degree_max_abs_drift for r in reps))
    boundary_hits = int(sum(1 for r in reps if r.threshold_boundary_hit))
    if drift_max == 0 and boundary_hits == 0 and float(np.mean([r.accepted_swaps_per_edge for r in reps])) >= swap_fraction * 0.95:
        status = "pass"
    elif drift_max == 0:
        status = "diagnostic"
    else:
        status = "fail"
    notes = (
        "Degree sequence is exact; spatial constraint preserves chord-length slots within bin tolerance. "
        "Use as a spatial-scale constrained null, not as a full planar null."
    )
    summary = SpatialNullSummary(
        city=city,
        graph_cache_file=str(graph_path.relative_to(ROOT)) if graph_path.is_relative_to(ROOT) else str(graph_path),
        n_nodes_road=n,
        n_edges_road=m,
        pc_cebh_road=pc_cebh,
        pc_road_fine_grid=pc_road,
        road_gap=road_gap,
        degree_null_gap_R41=degree_null_gap,
        degree_null_rank_mean_R41=degree_null_rank,
        spatial_null_rank_mean=spatial_rank,
        spatial_null_rank_ci_low=float(np.mean(ci_lows)),
        spatial_null_rank_ci_high=float(np.mean(ci_highs)),
        spatial_null_cebh_mean=float(np.mean([r.pc_cebh_null for r in reps])),
        spatial_null_gap=spatial_gap,
        spatial_minus_degree_gap=spatial_gap - degree_null_gap,
        road_minus_spatial=road_minus_spatial,
        spatial_fraction_of_road_gap=spatial_fraction,
        remaining_fraction_after_spatial=remaining,
        accepted_swaps_per_edge_mean=float(np.mean([r.accepted_swaps_per_edge for r in reps])),
        acceptance_rate_mean=float(np.mean([r.acceptance_rate for r in reps])),
        chord_length_bin_tvd_mean=float(np.mean([r.chord_length_bin_tvd for r in reps])),
        degree_max_abs_drift_max=drift_max,
        n_null=n_null,
        n_perm=n_perm,
        length_bins=length_bins,
        bin_tolerance=bin_tolerance,
        swap_fraction_target=swap_fraction,
        p_grid_min=float(p_grid.min()),
        p_grid_max=float(p_grid.max()),
        p_grid_step=p_step,
        boundary_hit_count=boundary_hits,
        status=status,
        notes=notes,
    )
    return summary, reps


def write_dataclass_csv(path: Path, rows: list[Any]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_report(out_dir: Path, summaries: list[SpatialNullSummary], elapsed_s: float) -> None:
    vals = {
        "road_gap": np.mean([s.road_gap for s in summaries]),
        "degree_gap": np.mean([s.degree_null_gap_R41 for s in summaries]),
        "spatial_gap": np.mean([s.spatial_null_gap for s in summaries]),
        "remaining": np.mean([s.remaining_fraction_after_spatial for s in summaries]),
        "spatial_fraction": np.mean([s.spatial_fraction_of_road_gap for s in summaries]),
        "tvd": np.mean([s.chord_length_bin_tvd_mean for s in summaries]),
        "swap": np.mean([s.accepted_swaps_per_edge_mean for s in summaries]),
        "acceptance": np.mean([s.acceptance_rate_mean for s in summaries]),
    }
    lines = [
        "# R56 spatial-length constrained null models",
        "",
        "## Purpose",
        "",
        "R56 adds a real spatial-scale constraint to the R41 degree-preserving null. "
        "Each null replicate preserves the degree sequence exactly and uses double-edge "
        "swaps that are accepted only when new edge chord lengths remain in the original "
        "length slots within the configured bin tolerance.",
        "",
        "This is an empirical spatial null, not a theorem-level planar null. It tests "
        "whether adding spatial edge-scale information materially narrows the gap between "
        "configuration-like degree nulls and observed road percolation thresholds.",
        "",
        "## Aggregate result",
        "",
        f"- Cities: {len(summaries)}.",
        f"- Mean observed road gap: {vals['road_gap']:.4f}.",
        f"- Mean R41 degree-null gap: {vals['degree_gap']:.4f}.",
        f"- Mean R56 degree + spatial-scale null gap: {vals['spatial_gap']:.4f}.",
        f"- Mean increment over degree-only null: {vals['spatial_gap'] - vals['degree_gap']:.4f}.",
        f"- Mean remaining road gap after spatial null: {vals['remaining']:.1%}.",
        f"- Mean spatial-null fraction of road gap: {vals['spatial_fraction']:.1%}.",
        f"- Mean accepted swaps per edge: {vals['swap']:.3f}.",
        f"- Mean acceptance rate: {vals['acceptance']:.3%}.",
        f"- Mean chord-length bin total-variation distance: {vals['tvd']:.4f}.",
        f"- Runtime: {elapsed_s / 60:.1f} min.",
        "",
        "## Manuscript-safe interpretation",
        "",
        "The safe claim is that degree sequence alone is insufficient and that adding a "
        "spatial edge-scale constraint shifts the null threshold toward the observed road "
        "threshold, but does not in itself license a universal law, a causal mechanism, "
        "or a strict planar-null equivalence. The residual after this null should be "
        "described as the part not reproduced by degree sequence plus the tested spatial "
        "scale constraint.",
        "",
        "## City summary",
        "",
        "| City | road gap | degree null gap | spatial null gap | remaining after spatial | swaps/edge | length TVD | status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s.city} | {s.road_gap:.4f} | {s.degree_null_gap_R41:.4f} | "
            f"{s.spatial_null_gap:.4f} | {s.remaining_fraction_after_spatial:.1%} | "
            f"{s.accepted_swaps_per_edge_mean:.3f} | {s.chord_length_bin_tvd_mean:.4f} | {s.status} |"
        )
    lines += [
        "",
        "## Boundary",
        "",
        "- Uses cached live-OSMnx road graphs for the eight cities with complete node coordinates.",
        "- The spatial constraint is based on node-to-node chord length because rewired null edges have no road geometry.",
        "- The algorithm preserves degree exactly; it preserves the empirical length-bin distribution approximately.",
        "- The result should not be phrased as a full planar constrained null or as proof of a mechanism.",
    ]
    (out_dir / "spatial_length_constrained_nulls_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_figure(out_stem: Path, summaries: list[SpatialNullSummary]) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    import pub_style

    pub_style.apply()

    cities = [s.city for s in summaries]
    x = np.arange(len(cities))
    road = np.asarray([s.road_gap for s in summaries])
    degree = np.asarray([s.degree_null_gap_R41 for s in summaries])
    spatial = np.asarray([s.spatial_null_gap for s in summaries])
    remaining = np.asarray([s.road_minus_spatial for s in summaries])
    swap = np.asarray([s.accepted_swaps_per_edge_mean for s in summaries])
    tvd = np.asarray([s.chord_length_bin_tvd_mean for s in summaries])

    fig = plt.figure(figsize=(pub_style.FIG_WIDTH_2COL, 4.7), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.35, 1.0, 1.0],
        height_ratios=[1.12, 0.88],
        left=0.075,
        right=0.985,
        bottom=0.125,
        top=0.94,
        wspace=0.42,
        hspace=0.62,
    )
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1:])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[1, 2])

    colors = {
        "road": pub_style.COLORS["observed"],
        "degree": pub_style.COLORS["degree_null"],
        "spatial": pub_style.COLORS["spatial_null"],
        "remaining": pub_style.COLORS["residual"],
        "accent": pub_style.COLORS["accent"],
    }

    ax0.bar(x, degree, color=colors["degree"], label="Degree null", width=0.68)
    ax0.bar(x, spatial - degree, bottom=degree, color=colors["spatial"], label="+ spatial scale", width=0.68)
    ax0.bar(x, road - spatial, bottom=spatial, color=colors["remaining"], label="Residual", width=0.68)
    ax0.plot(x, road, marker="o", ms=3.2, lw=1.1, color=colors["road"], label="Observed road gap")
    ax0.set_ylabel("Threshold gap relative to CEBH")
    ax0.set_xticks(x)
    ax0.set_xticklabels([c.replace("Buenos Aires", "B. Aires") for c in cities], rotation=45, ha="right")
    pub_style.panel_title(ax0, "a", "Eight-city null-ladder decomposition")
    ax0.legend(ncol=1, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    ax0.set_ylim(0, max(float(np.max(road)) * 1.18, 0.05))
    ax0.grid(axis="y", color="#ECECEC", lw=0.6)

    order_shift = np.argsort(spatial)
    for rank, i in enumerate(order_shift):
        alpha = 0.35 + 0.55 * (rank + 1) / len(order_shift)
        ax1.plot([0, 1], [degree[i], spatial[i]], color=colors["spatial"], lw=1.25, alpha=alpha)
    ax1.scatter(np.zeros_like(degree), degree, s=32, color=colors["degree"], edgecolor="white", linewidth=0.6, zorder=3)
    ax1.scatter(np.ones_like(spatial), spatial, s=42, color=colors["spatial"], edgecolor="white", linewidth=0.6, zorder=3)
    label_y = spatial.copy()
    min_sep = 0.0042
    for pos in range(1, len(order_shift)):
        prev_i = order_shift[pos - 1]
        curr_i = order_shift[pos]
        if label_y[curr_i] - label_y[prev_i] < min_sep:
            label_y[curr_i] = label_y[prev_i] + min_sep
    ylim_top = max(float(np.max(spatial)) * 1.20, 0.05)
    overflow = max(0.0, float(np.max(label_y)) - ylim_top * 0.96)
    if overflow > 0:
        label_y[order_shift] -= overflow
    for i in order_shift:
        ax1.plot([1.005, 1.035], [spatial[i], label_y[i]], color="#97A9A7", lw=0.45)
        ax1.text(1.04, label_y[i], cities[i].replace("Buenos Aires", "B. Aires"), va="center", fontsize=6)
    ax1.set_xlim(-0.18, 1.38)
    ax1.set_ylim(0, ylim_top)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Degree\nonly", "Degree +\nspatial scale"])
    ax1.set_ylabel("Null threshold gap")
    pub_style.panel_title(ax1, "b", "Degree versus degree + edge-scale nulls")
    ax1.grid(axis="y", color="#ECECEC", lw=0.6)

    order = np.argsort(remaining)
    ax2.barh(np.arange(len(cities)), remaining[order] / road[order], color=colors["remaining"], height=0.62)
    ax2.set_yticks(np.arange(len(cities)))
    ax2.set_yticklabels([cities[i].replace("Buenos Aires", "B. Aires") for i in order])
    ax2.set_xlim(0, 1.05)
    ax2.set_xlabel("Fraction of road gap not reproduced")
    pub_style.panel_title(ax2, "c", "Post-spatial residual fraction")
    ax2.grid(axis="x", color="#ECECEC", lw=0.6)

    means = np.asarray([float(np.mean(swap)), float(np.mean(tvd))])
    ax3.bar([0, 1], means, color=[colors["spatial"], colors["accent"]], width=0.55, alpha=0.86)
    jitter = np.linspace(-0.14, 0.14, len(cities))
    ax3.scatter(np.zeros_like(swap) + jitter, swap, color=colors["spatial"], s=18, edgecolor="white", linewidth=0.35, zorder=3)
    ax3.scatter(np.ones_like(tvd) + jitter, tvd, color=colors["accent"], s=18, edgecolor="white", linewidth=0.35, zorder=3)
    ax3.text(0.98, 0.95, "degree drift = 0", transform=ax3.transAxes, ha="right", va="top", fontsize=6.5, color="#555B63")
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(["Swaps\nper edge", "Length-bin\nTVD"])
    ax3.set_ylim(0, max(float(np.max(swap)), float(np.max(tvd))) * 1.24)
    ax3.set_ylabel("Diagnostic value")
    pub_style.panel_title(ax3, "d", "Null-fidelity diagnostics")
    ax3.grid(axis="y", color="#ECECEC", lw=0.6)

    pub_style.save(fig, out_stem)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r41", type=Path, default=DEFAULT_R41)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--n-null", type=int, default=4)
    parser.add_argument("--n-perm", type=int, default=16)
    parser.add_argument("--n-bootstrap", type=int, default=300)
    parser.add_argument("--p-step", type=float, default=0.01)
    parser.add_argument("--window", type=float, default=0.25)
    parser.add_argument("--length-bins", type=int, default=12)
    parser.add_argument("--bin-tolerance", type=int, default=1)
    parser.add_argument("--swap-fraction", type=float, default=0.30)
    parser.add_argument("--max-attempt-factor", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260603)
    args = parser.parse_args()

    start = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.r41)
    summaries: list[SpatialNullSummary] = []
    reps_all: list[SpatialNullReplicate] = []

    for idx, row in enumerate(rows, start=1):
        city = str(row["city"])
        print(f"[R56] {idx}/{len(rows)} {city}: spatial length-constrained nulls", flush=True)
        summary, reps = run_city(
            row=row,
            cache_dir=args.cache_dir,
            n_null=args.n_null,
            n_perm=args.n_perm,
            n_bootstrap=args.n_bootstrap,
            p_step=args.p_step,
            window=args.window,
            length_bins=args.length_bins,
            bin_tolerance=args.bin_tolerance,
            swap_fraction=args.swap_fraction,
            max_attempt_factor=args.max_attempt_factor,
            seed=args.seed + idx * 10_000,
        )
        summaries.append(summary)
        reps_all.extend(reps)
        write_dataclass_csv(args.out_dir / "spatial_length_constrained_nulls_city_summary.partial.csv", summaries)
        write_dataclass_csv(args.out_dir / "spatial_length_constrained_nulls_replicates.partial.csv", reps_all)
        gc.collect()

    write_dataclass_csv(args.out_dir / "spatial_length_constrained_nulls_city_summary.csv", summaries)
    write_dataclass_csv(args.out_dir / "spatial_length_constrained_nulls_replicates.csv", reps_all)
    elapsed = time.time() - start
    payload = {
        "created_by": "scripts/build_R56_spatial_length_constrained_nulls.py",
        "elapsed_seconds": elapsed,
        "parameters": {
            "n_null": args.n_null,
            "n_perm": args.n_perm,
            "n_bootstrap": args.n_bootstrap,
            "p_step": args.p_step,
            "window": args.window,
            "length_bins": args.length_bins,
            "bin_tolerance": args.bin_tolerance,
            "swap_fraction": args.swap_fraction,
            "max_attempt_factor": args.max_attempt_factor,
            "seed": args.seed,
        },
        "aggregate": {
            "mean_road_gap": float(np.mean([s.road_gap for s in summaries])),
            "mean_degree_null_gap_R41": float(np.mean([s.degree_null_gap_R41 for s in summaries])),
            "mean_spatial_null_gap": float(np.mean([s.spatial_null_gap for s in summaries])),
            "mean_spatial_minus_degree_gap": float(np.mean([s.spatial_minus_degree_gap for s in summaries])),
            "mean_remaining_fraction_after_spatial": float(np.mean([s.remaining_fraction_after_spatial for s in summaries])),
            "mean_chord_length_bin_tvd": float(np.mean([s.chord_length_bin_tvd_mean for s in summaries])),
            "mean_accepted_swaps_per_edge": float(np.mean([s.accepted_swaps_per_edge_mean for s in summaries])),
        },
    }
    (args.out_dir / "spatial_length_constrained_nulls_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    write_report(args.out_dir, summaries, elapsed)
    plot_figure(args.figure, summaries)
    print(f"[R56] complete in {elapsed / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
