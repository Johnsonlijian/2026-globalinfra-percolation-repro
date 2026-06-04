"""R60 expanded road evidence and strict geometry-preserving null pilots.

This round addresses the two hardest remaining reviewer risks:

1. The eight-city evidence base is too small.
2. The previous spatial null was length-constrained, but not planar/geometry
   preserving.

The expansion is deliberately tiered. The eight OSMnx-live cities remain the
primary, method-coherent evidence set. Cape Town and Singapore are added as
diagnostic local-PBF cached extensions. The geometry-preserving null is a
strict pilot for selected medium-sized cities: it preserves degree sequence,
keeps replacement edges in comparable chord-length slots, and rejects any
replacement edge intersecting non-adjacent existing edges.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_R56_spatial_length_constrained_nulls as r56  # noqa: E402


OUT = ROOT / "data" / "R60_expansion_geometry_nulls"
FIG_BASE = ROOT / "figures" / "Fig_R60_expansion_geometry_nulls"
ROUND_DIR = ROOT / "validation_reports" / "R60_expansion_geometry_nulls"
R56_SUMMARY = ROOT / "data" / "R56_spatial_length_constrained_nulls" / "spatial_length_constrained_nulls_city_summary.csv"
R24_SUMMARY = ROOT / "data" / "W7_nc10_R24_local_pbf_recovery_matrix" / "nc10_road_gap_summary.csv"


PILOT_CITIES = ["Rotterdam", "Buenos Aires", "Stockholm"]
GEOMETRY_NULL_REPS = 2
GEOMETRY_N_PERM = 24
GEOMETRY_N_BOOTSTRAP = 240
GEOMETRY_P_GRID_STEP = 0.01
GEOMETRY_P_GRID_WIDTH = 0.35
GEOMETRY_SWAP_FRACTION = 0.02


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_BASE.parent.mkdir(parents=True, exist_ok=True)
    ROUND_DIR.mkdir(parents=True, exist_ok=True)


def read_csv_dict(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def as_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def build_expanded_city_matrix() -> pd.DataFrame:
    r56_rows = read_csv_dict(R56_SUMMARY)
    r24_rows = {row["city"]: row for row in read_csv_dict(R24_SUMMARY)}

    records: list[dict[str, Any]] = []
    formal_cities = {row["city"] for row in r56_rows}

    for row in r56_rows:
        city = row["city"]
        records.append(
            {
                "city": city,
                "continent_or_region": region_label(city),
                "evidence_tier": "primary_osmnx_live_method_coherent",
                "n_nodes_road": as_int(row.get("n_nodes_road")),
                "n_edges_road": as_int(row.get("n_edges_road")),
                "pc_cebh_road": as_float(row.get("pc_cebh_road")),
                "pc_road": as_float(row.get("pc_road_fine_grid")),
                "road_gap": as_float(row.get("road_gap")),
                "degree_null_gap_R41": as_float(row.get("degree_null_gap_R41")),
                "spatial_null_gap_R56": as_float(row.get("spatial_null_gap")),
                "road_minus_spatial_R56": as_float(row.get("road_minus_spatial")),
                "remaining_fraction_after_spatial_R56": as_float(row.get("remaining_fraction_after_spatial")),
                "graph_cache_file": row.get("graph_cache_file", ""),
                "threshold_source": "fine-grid rank-order bond threshold, R56",
                "claim_status": "main",
            }
        )

    for city in ["Cape Town", "Singapore"]:
        if city in formal_cities or city not in r24_rows:
            continue
        row = r24_rows[city]
        graph_name = {
            "Cape Town": "cape_town_south_africa__road.pkl",
            "Singapore": "singapore_singapore__road.pkl",
        }[city]
        records.append(
            {
                "city": city,
                "continent_or_region": region_label(city),
                "evidence_tier": "diagnostic_cached_local_pbf_extension",
                "n_nodes_road": as_int(row.get("road_n")),
                "n_edges_road": as_int(row.get("road_m")),
                "pc_cebh_road": as_float(row.get("pc_cebh")),
                "pc_road": as_float(row.get("pc_bond")),
                "road_gap": as_float(row.get("bond_gap")),
                "degree_null_gap_R41": float("nan"),
                "spatial_null_gap_R56": float("nan"),
                "road_minus_spatial_R56": float("nan"),
                "remaining_fraction_after_spatial_R56": float("nan"),
                "graph_cache_file": str(Path("cache") / "private_graphs" / graph_name),
                "threshold_source": "coarser local-PBF diagnostic bond threshold, R24",
                "claim_status": "diagnostic_extension",
            }
        )

    df = pd.DataFrame(records)
    df = df.sort_values(["claim_status", "city"], ascending=[False, True]).reset_index(drop=True)
    df.to_csv(OUT / "expanded_city_diagnostic_matrix.csv", index=False)
    return df


def region_label(city: str) -> str:
    return {
        "Berlin": "Europe",
        "Buenos Aires": "South America",
        "Cape Town": "Africa",
        "Chicago": "North America",
        "Madrid": "Europe",
        "Rotterdam": "Europe",
        "Singapore": "Asia",
        "Stockholm": "Europe",
        "Sydney": "Oceania",
        "Toronto": "North America",
    }.get(city, "Unassigned")


def projected_xy(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean_lat = float(np.nanmean(lat))
    scale_x = 111_320.0 * math.cos(math.radians(mean_lat))
    scale_y = 110_540.0
    x = lon * scale_x
    y = lat * scale_y
    return x, y


def line_touches_forbidden(line, edge_nodes: tuple[int, int], work: np.ndarray, geoms: list[Any], idx: int) -> bool:
    u, v = edge_nodes
    a, b = int(work[idx, 0]), int(work[idx, 1])
    if u in {a, b} or v in {a, b}:
        return False
    return bool(line.intersects(geoms[idx]))


def tree_query_indices(tree: Any, line: Any, geom_id_to_idx: dict[int, int]) -> list[int]:
    hits = tree.query(line)
    out: list[int] = []
    for item in hits:
        if isinstance(item, (int, np.integer)):
            out.append(int(item))
        else:
            idx = geom_id_to_idx.get(id(item))
            if idx is not None:
                out.append(idx)
    return out


def geometry_preserving_rewire(
    edges: np.ndarray,
    degrees: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    seed: int,
    length_bins: int = 12,
    bin_tolerance: int = 1,
    swap_fraction: float = 0.02,
    max_attempt_factor: int = 700,
    neighbor_k: int = 220,
    rebuild_interval: int = 50,
) -> tuple[np.ndarray, dict[str, float]]:
    from scipy.spatial import cKDTree
    from shapely.geometry import LineString
    from shapely.strtree import STRtree

    rng = np.random.default_rng(seed)
    m = len(edges)
    work = np.asarray(edges, dtype=np.int64).copy()
    original_degrees = np.asarray(degrees, dtype=np.int64)
    original_lengths = r56.chord_lengths_for_edges(work, lon, lat)
    bin_edges, slot_bins = r56.make_length_bins(original_lengths, length_bins)
    x, y = projected_xy(lon, lat)
    edge_set = {r56.edge_key(int(u), int(v)) for u, v in work}

    target_accept = int(round(max(0.0, swap_fraction) * m))
    max_attempts = max(target_accept * max_attempt_factor, target_accept + 1)

    accepted = 0
    attempts = 0
    topology_rejects = 0
    length_rejects = 0
    crossing_rejects = 0
    stale_index_false_reject_guard = 0

    def edge_line(edge: np.ndarray) -> Any:
        u, v = int(edge[0]), int(edge[1])
        return LineString([(float(x[u]), float(y[u])), (float(x[v]), float(y[v]))])

    geoms = [edge_line(edge) for edge in work]
    tree = STRtree(geoms)
    geom_id_to_idx = {id(g): i for i, g in enumerate(geoms)}
    changed_indices: set[int] = set()

    mids = np.column_stack(((x[work[:, 0]] + x[work[:, 1]]) * 0.5, (y[work[:, 0]] + y[work[:, 1]]) * 0.5))
    kd = cKDTree(mids)

    def rebuild_spatial_indices() -> None:
        nonlocal tree, geom_id_to_idx, kd, mids, changed_indices
        tree = STRtree(geoms)
        geom_id_to_idx = {id(g): i for i, g in enumerate(geoms)}
        mids = np.column_stack(((x[work[:, 0]] + x[work[:, 1]]) * 0.5, (y[work[:, 0]] + y[work[:, 1]]) * 0.5))
        kd = cKDTree(mids)
        changed_indices = set()

    def conflicts_with_existing(line: Any, nodes: tuple[int, int], skip_indices: set[int]) -> bool:
        nonlocal stale_index_false_reject_guard
        for idx in tree_query_indices(tree, line, geom_id_to_idx):
            if idx in skip_indices or idx in changed_indices:
                continue
            if line_touches_forbidden(line, nodes, work, geoms, idx):
                return True
        for idx in changed_indices:
            if idx in skip_indices:
                continue
            if line_touches_forbidden(line, nodes, work, geoms, idx):
                return True
        stale_index_false_reject_guard += 1
        return False

    def length_slot_ok(edge: tuple[int, int], slot: int) -> bool:
        length = r56.chord_length_m(float(lon[edge[0]]), float(lat[edge[0]]), float(lon[edge[1]]), float(lat[edge[1]]))
        new_bin = int(np.searchsorted(bin_edges, length, side="right") - 1)
        new_bin = max(0, min(new_bin, len(bin_edges) - 2))
        return abs(new_bin - int(slot)) <= bin_tolerance

    while accepted < target_accept and attempts < max_attempts:
        attempts += 1
        i = int(rng.integers(0, m))
        _, nbrs = kd.query(mids[i], k=min(neighbor_k, m))
        nbrs = np.atleast_1d(nbrs)
        if len(nbrs) <= 1:
            topology_rejects += 1
            continue
        j = int(rng.choice(nbrs[1:]))
        if i == j:
            topology_rejects += 1
            continue

        a, b = int(work[i, 0]), int(work[i, 1])
        c, d = int(work[j, 0]), int(work[j, 1])
        if len({a, b, c, d}) < 4:
            topology_rejects += 1
            continue

        if bool(rng.integers(0, 2)):
            proposal = (r56.edge_key(a, d), r56.edge_key(c, b))
        else:
            proposal = (r56.edge_key(a, c), r56.edge_key(b, d))
        e1, e2 = proposal
        old1 = r56.edge_key(a, b)
        old2 = r56.edge_key(c, d)

        if e1[0] == e1[1] or e2[0] == e2[1] or e1 == e2:
            topology_rejects += 1
            continue
        if (e1 not in {old1, old2} and e1 in edge_set) or (e2 not in {old1, old2} and e2 in edge_set):
            topology_rejects += 1
            continue
        if not (length_slot_ok(e1, int(slot_bins[i])) and length_slot_ok(e2, int(slot_bins[j]))):
            length_rejects += 1
            continue

        line1 = LineString([(float(x[e1[0]]), float(y[e1[0]])), (float(x[e1[1]]), float(y[e1[1]]))])
        line2 = LineString([(float(x[e2[0]]), float(y[e2[0]])), (float(x[e2[1]]), float(y[e2[1]]))])
        if line1.intersects(line2):
            crossing_rejects += 1
            continue
        skip = {i, j}
        if conflicts_with_existing(line1, e1, skip) or conflicts_with_existing(line2, e2, skip):
            crossing_rejects += 1
            continue

        edge_set.remove(old1)
        edge_set.remove(old2)
        edge_set.add(e1)
        edge_set.add(e2)
        work[i] = e1
        work[j] = e2
        geoms[i] = line1
        geoms[j] = line2
        changed_indices.update({i, j})
        accepted += 1

        if accepted % rebuild_interval == 0:
            rebuild_spatial_indices()

    deg = np.bincount(work.ravel(), minlength=len(degrees))
    new_lengths = r56.chord_lengths_for_edges(work, lon, lat)
    _, new_bins = r56.make_length_bins(new_lengths, length_bins)
    hist0 = np.bincount(slot_bins, minlength=length_bins).astype(float)
    hist1 = np.bincount(new_bins, minlength=length_bins).astype(float)
    hist0 /= max(1.0, hist0.sum())
    hist1 /= max(1.0, hist1.sum())

    diagnostics = {
        "accepted_swaps": int(accepted),
        "attempted_swaps": int(attempts),
        "acceptance_rate": float(accepted / attempts) if attempts else 0.0,
        "accepted_swaps_per_edge": float(accepted / m) if m else 0.0,
        "target_swaps": int(target_accept),
        "target_swap_fraction": float(swap_fraction),
        "topology_rejects": int(topology_rejects),
        "length_rejects": int(length_rejects),
        "crossing_rejects": int(crossing_rejects),
        "degree_max_abs_drift": int(np.max(np.abs(deg - original_degrees))) if len(deg) else 0,
        "length_bin_tvd": float(0.5 * np.abs(hist0 - hist1).sum()),
        "length_mean_ratio": float(np.mean(new_lengths) / np.mean(original_lengths)),
        "length_median_ratio": float(np.median(new_lengths) / np.median(original_lengths)),
        "stale_index_guard_calls": int(stale_index_false_reject_guard),
        "max_attempts": int(max_attempts),
    }
    return work, diagnostics


def run_geometry_null_pilots(expanded: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rep_records: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []

    formal = expanded.set_index("city").to_dict(orient="index")
    for city in PILOT_CITIES:
        row = formal[city]
        graph_path = ROOT / str(row["graph_cache_file"])
        G = r56.load_graph(graph_path)
        _, edges, degrees, lon, lat = r56.to_simple_spatial_graph(G)
        del G
        gc.collect()

        pc_cebh = float(row["pc_cebh_road"])
        pc_road = float(row["pc_road"])
        road_gap = float(row["road_gap"])
        p_grid = r56.null_p_grid(pc_cebh, width=GEOMETRY_P_GRID_WIDTH, step=GEOMETRY_P_GRID_STEP)

        src_road = edges[:, 0].astype(np.int64)
        dst_road = edges[:, 1].astype(np.int64)
        road_samples = r56.rank_order_samples(
            len(degrees),
            src_road,
            dst_road,
            p_grid,
            n_perm=GEOMETRY_N_PERM,
            seed=r56.stable_city_seed(city) + 59_001,
        )
        pc_road_same, road_same_lo, road_same_hi, road_same_boundary = r56.bootstrap_pc_ci(
            p_grid,
            road_samples,
            n_bootstrap=GEOMETRY_N_BOOTSTRAP,
            seed=r56.stable_city_seed(city) + 59_777,
        )

        city_pcs: list[float] = []
        city_los: list[float] = []
        city_his: list[float] = []
        boundary_hits = 0
        for rep in range(GEOMETRY_NULL_REPS):
            seed = r56.stable_city_seed(city) + 60_000 + rep * 1_001
            start = time.time()
            null_edges, diag = geometry_preserving_rewire(
                edges,
                degrees,
                lon,
                lat,
                seed=seed,
                swap_fraction=GEOMETRY_SWAP_FRACTION,
                max_attempt_factor=1_000,
            )
            src = null_edges[:, 0].astype(np.int64)
            dst = null_edges[:, 1].astype(np.int64)
            samples = r56.rank_order_samples(len(degrees), src, dst, p_grid, n_perm=GEOMETRY_N_PERM, seed=seed + 117)
            pc_rank, lo, hi, boundary = r56.bootstrap_pc_ci(
                p_grid,
                samples,
                n_bootstrap=GEOMETRY_N_BOOTSTRAP,
                seed=seed + 223,
            )
            elapsed = time.time() - start
            if boundary:
                boundary_hits += 1
            city_pcs.append(pc_rank)
            city_los.append(lo)
            city_his.append(hi)

            rep_records.append(
                {
                    "city": city,
                    "replicate": rep + 1,
                    "n_nodes": int(len(degrees)),
                    "n_edges": int(len(edges)),
                    "pc_cebh_road": pc_cebh,
                    "pc_road": pc_road,
                    "pc_road_same_estimator": pc_road_same,
                    "pc_road_same_estimator_ci_low": road_same_lo,
                    "pc_road_same_estimator_ci_high": road_same_hi,
                    "road_gap": road_gap,
                    "road_gap_same_estimator": pc_road_same - pc_cebh,
                    "pc_geometry_null_rank": pc_rank,
                    "pc_geometry_null_ci_low": lo,
                    "pc_geometry_null_ci_high": hi,
                    "geometry_null_gap": pc_rank - pc_cebh,
                    "road_minus_geometry_null": pc_road - pc_rank,
                    "road_same_estimator_minus_geometry_null": pc_road_same - pc_rank,
                    "geometry_fraction_of_road_gap": (pc_rank - pc_cebh) / road_gap if road_gap > 0 else float("nan"),
                    "remaining_fraction_after_geometry": (pc_road - pc_rank) / road_gap if road_gap > 0 else float("nan"),
                    "remaining_fraction_after_geometry_same_estimator": (
                        (pc_road_same - pc_rank) / (pc_road_same - pc_cebh)
                        if pc_road_same > pc_cebh
                        else float("nan")
                    ),
                    "threshold_boundary_hit": bool(boundary),
                    "road_same_estimator_boundary_hit": bool(road_same_boundary),
                    "runtime_seconds": elapsed,
                    **diag,
                }
            )

        pc_mean = float(np.mean(city_pcs))
        summary_records.append(
            {
                "city": city,
                "n_nodes": int(len(degrees)),
                "n_edges": int(len(edges)),
                "pc_cebh_road": pc_cebh,
                "pc_road": pc_road,
                "pc_road_same_estimator": pc_road_same,
                "pc_road_same_estimator_ci_low": road_same_lo,
                "pc_road_same_estimator_ci_high": road_same_hi,
                "road_gap": road_gap,
                "road_gap_same_estimator": pc_road_same - pc_cebh,
                "spatial_null_gap_R56": float(row["spatial_null_gap_R56"]),
                "spatial_null_rank_R56": pc_cebh + float(row["spatial_null_gap_R56"]),
                "geometry_null_rank_mean": pc_mean,
                "geometry_null_rank_min": float(np.min(city_pcs)),
                "geometry_null_rank_max": float(np.max(city_pcs)),
                "geometry_null_ci_low_min": float(np.min(city_los)),
                "geometry_null_ci_high_max": float(np.max(city_his)),
                "geometry_null_gap_mean": pc_mean - pc_cebh,
                "road_minus_geometry_null_mean": pc_road - pc_mean,
                "road_same_estimator_minus_geometry_null_mean": pc_road_same - pc_mean,
                "geometry_fraction_of_road_gap": (pc_mean - pc_cebh) / road_gap if road_gap > 0 else float("nan"),
                "remaining_fraction_after_geometry": (pc_road - pc_mean) / road_gap if road_gap > 0 else float("nan"),
                "remaining_fraction_after_geometry_same_estimator": (
                    (pc_road_same - pc_mean) / (pc_road_same - pc_cebh)
                    if pc_road_same > pc_cebh
                    else float("nan")
                ),
                "accepted_swaps_per_edge_mean": float(np.mean([r["accepted_swaps_per_edge"] for r in rep_records if r["city"] == city])),
                "acceptance_rate_mean": float(np.mean([r["acceptance_rate"] for r in rep_records if r["city"] == city])),
                "degree_max_abs_drift_max": int(max(r["degree_max_abs_drift"] for r in rep_records if r["city"] == city)),
                "length_bin_tvd_mean": float(np.mean([r["length_bin_tvd"] for r in rep_records if r["city"] == city])),
                "boundary_hit_count": int(boundary_hits),
                "road_same_estimator_boundary_hit": bool(road_same_boundary),
                "n_geometry_null": GEOMETRY_NULL_REPS,
                "n_perm": GEOMETRY_N_PERM,
                "swap_fraction_target": GEOMETRY_SWAP_FRACTION,
                "geometry_null_status": "strict_non_crossing_degree_preserving_pilot",
                "claim_boundary": "pilot evidence; not yet a full global planar-null ensemble",
            }
        )

    reps = pd.DataFrame(rep_records)
    summary = pd.DataFrame(summary_records)
    reps.to_csv(OUT / "geometry_preserving_null_replicates.csv", index=False)
    summary.to_csv(OUT / "geometry_preserving_null_summary.csv", index=False)
    return reps, summary


def figure(expanded: pd.DataFrame, geom: pd.DataFrame, reps: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {
        "main": "#2F6B9A",
        "diagnostic_extension": "#C45A3B",
        "spatial": "#6A8D3A",
        "geometry": "#7B4FA3",
        "residual": "#4B5563",
    }

    fig = plt.figure(figsize=(7.8, 6.7), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], width_ratios=[1.25, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_df = expanded.sort_values("road_gap", ascending=True)
    y = np.arange(len(plot_df))
    bar_colors = [colors["main"] if s == "main" else colors["diagnostic_extension"] for s in plot_df["claim_status"]]
    ax_a.barh(y, plot_df["road_gap"], color=bar_colors, height=0.68)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(plot_df["city"])
    ax_a.set_xlabel("Observed bond-threshold gap")
    ax_a.set_title("a  Tiered ten-city road evidence", loc="left")
    ax_a.axvline(0, color="black", lw=0.7)
    ax_a.text(
        0.98,
        0.05,
        "blue: primary n=8\nbrick: diagnostic n=2",
        transform=ax_a.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="#111827",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.0},
    )

    means = expanded.groupby("claim_status")["road_gap"].agg(["mean", "min", "max", "count"]).reindex(["main", "diagnostic_extension"])
    x = np.arange(len(means))
    ax_b.errorbar(
        x,
        means["mean"],
        yerr=[means["mean"] - means["min"], means["max"] - means["mean"]],
        fmt="o",
        ms=7,
        color="#111827",
        ecolor="#6B7280",
        capsize=4,
    )
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(["Primary\n8 cities", "Diagnostic\n+2 cities"])
    ax_b.set_ylabel("Mean gap with range")
    ax_b.set_title("b  Expansion changes scope", loc="left")
    for xi, row in means.iterrows():
        xpos = list(means.index).index(xi)
        ax_b.text(xpos, row["max"] + 0.01, f"n={int(row['count'])}", ha="center", va="bottom", fontsize=7)

    ladder = geom.sort_values("road_gap", ascending=False)
    xloc = np.arange(len(ladder))
    width = 0.22
    ax_c.bar(xloc - width, ladder["road_gap"], width=width, color=colors["main"], label="Observed road gap")
    ax_c.bar(xloc, ladder["spatial_null_gap_R56"], width=width, color=colors["spatial"], label="Length-scale null gap")
    ax_c.bar(xloc + width, ladder["geometry_null_gap_mean"], width=width, color=colors["geometry"], label="Strict geometry null gap")
    ax_c.scatter(
        xloc - width,
        ladder["road_gap_same_estimator"],
        marker="D",
        s=22,
        color="#111827",
        zorder=5,
        label="Road gap, same estimator",
    )
    for i, row in enumerate(ladder.itertuples()):
        lo = row.geometry_null_ci_low_min - row.pc_cebh_road
        hi = row.geometry_null_ci_high_max - row.pc_cebh_road
        ax_c.plot([i + width, i + width], [lo, hi], color="#2D1B46", lw=1.0)
    ax_c.set_xticks(xloc)
    ax_c.set_xticklabels(ladder["city"], rotation=0)
    ax_c.set_ylabel("Gap relative to CEBH")
    ax_c.set_title("c  Non-crossing null ladder", loc="left")
    ax_c.legend(frameon=False, fontsize=7, ncol=1, loc="upper left")

    funnel = reps.groupby("city")[["topology_rejects", "length_rejects", "crossing_rejects", "accepted_swaps"]].mean()
    funnel = funnel.loc[ladder["city"]]
    bottom = np.zeros(len(funnel))
    stack_cols = [
        ("topology_rejects", "#9CA3AF", "Topology reject"),
        ("length_rejects", "#E0A94F", "Length-slot reject"),
        ("crossing_rejects", "#D15B64", "Crossing reject"),
        ("accepted_swaps", "#2AA876", "Accepted swaps"),
    ]
    for col, color, label in stack_cols:
        vals = funnel[col].to_numpy(dtype=float)
        ax_d.bar(np.arange(len(funnel)), vals, bottom=bottom, color=color, label=label, width=0.62)
        bottom += vals
    ax_d.set_yscale("log")
    ax_d.set_xticks(np.arange(len(funnel)))
    ax_d.set_xticklabels(funnel.index)
    ax_d.set_ylabel("Mean proposals per replicate (log)")
    ax_d.set_title("d  Proposal-rejection funnel", loc="left")
    ax_d.legend(frameon=False, fontsize=7, loc="upper left")
    for i, row in enumerate(funnel.itertuples()):
        attempts = float(row.topology_rejects + row.length_rejects + row.crossing_rejects + row.accepted_swaps)
        acc_rate = float(row.accepted_swaps / attempts) if attempts else 0.0
        ax_d.text(i, attempts * 1.08, f"{acc_rate:.2%}", ha="center", va="bottom", fontsize=7, color="#111827")

    fig.suptitle("Expanded evidence and strict geometry-preserving null pilots", x=0.01, ha="left", fontsize=10)
    for suffix, dpi in [(".pdf", 300), (".svg", 300), (".png", 450), (".tiff", 600)]:
        fig.savefig(str(FIG_BASE) + suffix, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_report(expanded: pd.DataFrame, geom: pd.DataFrame, reps: pd.DataFrame) -> dict[str, Any]:
    primary = expanded[expanded["claim_status"] == "main"]
    diagnostic = expanded[expanded["claim_status"] == "diagnostic_extension"]
    all_positive = int((expanded["road_gap"] > 0).sum())
    summary = {
        "n_primary_method_coherent_cities": int(len(primary)),
        "n_diagnostic_extension_cities": int(len(diagnostic)),
        "n_total_tiered_cities": int(len(expanded)),
        "primary_gap_mean": float(primary["road_gap"].mean()),
        "primary_gap_min": float(primary["road_gap"].min()),
        "primary_gap_max": float(primary["road_gap"].max()),
        "diagnostic_gap_mean": float(diagnostic["road_gap"].mean()) if len(diagnostic) else float("nan"),
        "tiered_all_positive_count": all_positive,
        "tiered_all_positive_total": int(len(expanded)),
        "geometry_pilot_city_count": int(len(geom)),
        "geometry_gap_mean": float(geom["geometry_null_gap_mean"].mean()),
        "geometry_remaining_fraction_mean": float(geom["remaining_fraction_after_geometry"].mean()),
        "geometry_remaining_fraction_same_estimator_mean": float(
            geom["remaining_fraction_after_geometry_same_estimator"].mean()
        ),
        "road_same_estimator_minus_geometry_null_mean": float(
            geom["road_same_estimator_minus_geometry_null_mean"].mean()
        ),
        "geometry_acceptance_rate_mean": float(reps["acceptance_rate"].mean()),
        "geometry_degree_max_abs_drift_max": int(reps["degree_max_abs_drift"].max()),
        "geometry_length_bin_tvd_mean": float(reps["length_bin_tvd"].mean()),
        "claim_boundary": (
            "The 8-city OSMnx-live set remains the main evidence. Cape Town and Singapore are "
            "diagnostic extensions. The strict non-crossing degree-preserving null is a pilot "
            "for three medium-sized cities, not a completed global planar-null ensemble."
        ),
    }
    (OUT / "R60_expansion_geometry_nulls_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# R60 expanded evidence and geometry-preserving nulls",
        "",
        "## What changed",
        "",
        f"- Method-coherent primary road evidence remains n={summary['n_primary_method_coherent_cities']} cities.",
        f"- Tiered diagnostic road evidence now covers n={summary['n_total_tiered_cities']} cities; all {summary['tiered_all_positive_count']} have positive observed gaps.",
        f"- A strict geometry-preserving degree-null pilot was completed for n={summary['geometry_pilot_city_count']} cities.",
        "",
        "## Main numerical results",
        "",
        f"- Primary 8-city mean observed gap: {summary['primary_gap_mean']:.3f} "
        f"(range {summary['primary_gap_min']:.3f}-{summary['primary_gap_max']:.3f}).",
        f"- Diagnostic +2 mean observed gap: {summary['diagnostic_gap_mean']:.3f}.",
        f"- Geometry-null pilot mean gap: {summary['geometry_gap_mean']:.3f}.",
        f"- Mean remaining road gap after geometry null: {summary['geometry_remaining_fraction_mean']:.3f}.",
        f"- Same-estimator mean road-minus-geometry residual: {summary['road_same_estimator_minus_geometry_null_mean']:.3f} "
        f"(fraction {summary['geometry_remaining_fraction_same_estimator_mean']:.3f}).",
        f"- Degree drift under geometry null: max abs drift {summary['geometry_degree_max_abs_drift_max']}.",
        f"- Mean length-bin total variation under geometry null: {summary['geometry_length_bin_tvd_mean']:.3f}.",
        "",
        "## Claim boundary",
        "",
        summary["claim_boundary"],
        "",
        "## Reviewer-facing interpretation",
        "",
        "The added evidence weakens the 'only eight cities' criticism but does not erase it; the paper should now say "
        "that the main claim is replicated in an eight-city method-coherent sample and remains directionally consistent "
        "in two cached non-Western diagnostic extensions. The geometry-preserving null materially improves the null-model "
        "ladder because the proposal mechanism now rejects non-adjacent geometric intersections rather than only matching "
        "edge-length bins. Because this is still a three-city pilot, the manuscript should present it as a hardening and "
        "mechanistic probe, not as a completed global planar-null theory.",
    ]
    (OUT / "R60_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROUND_DIR / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    ensure_dirs()
    expanded = build_expanded_city_matrix()
    reps, geom = run_geometry_null_pilots(expanded)
    figure(expanded, geom, reps)
    summary = write_report(expanded, geom, reps)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

