"""N99 low-overlap strict-geometry surrogate pilot.

R90 established that the implemented strict non-crossing null is a local
geometry-preserving perturbation with high exact edge overlap.  This pilot asks
the next reviewer question directly: can the same strict degree, length-bin and
non-crossing constraints be pushed to materially lower edge overlap while still
producing usable threshold estimates?

The output is claim-bounded.  A successful record is a low-overlap pilot under
the implemented local-swap algorithm, not a full planar-ensemble mixing proof.
If the edge overlap remains high or acceptance collapses, the result is reported
as a constraint-strength diagnostic rather than as completed low-overlap
resampling.
"""

from __future__ import annotations

import argparse
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
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_R56_spatial_length_constrained_nulls as r56  # noqa: E402
import build_R60_expansion_geometry_nulls as r60  # noqa: E402
import build_R67_geometry_null_sensitivity as r67  # noqa: E402
import build_R90_finegrid_edge_audit as r90  # noqa: E402
import pub_style  # noqa: E402


SUBSET = ROOT / "data" / "R62_urban_form_mechanism" / "geometry_null_subset_registry.csv"
R64 = ROOT / "data" / "R64_71city_spatial_null_screen" / "spatial_null_screen_summary.csv"
OUT = ROOT / "data" / "N99_low_overlap_geometry_surrogate"
ROUND = ROOT / "rounds" / "N99_low_overlap_geometry_surrogate"
FIG_BASE = ROOT / "figures" / "Fig_N99_low_overlap_geometry_surrogate"

RECORDS_CSV = OUT / "N99_low_overlap_surrogate_records.csv"
SUMMARY_CSV = OUT / "N99_low_overlap_surrogate_city_summary.csv"
SUMMARY_JSON = OUT / "N99_low_overlap_surrogate_summary.json"
REPORT_MD = OUT / "N99_low_overlap_surrogate_report.md"
GATE_REPORT_MD = ROUND / "gate_report.md"
CLOSEOUT_MD = ROUND / "substantive_increment_closeout.md"

DEFAULT_CITIES = ["Singapore", "Dubai", "Taipei", "Seattle", "Nairobi", "Sydney", "Hong Kong", "Rio de Janeiro"]


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ROUND.mkdir(parents=True, exist_ok=True)
    FIG_BASE.parent.mkdir(parents=True, exist_ok=True)


def parse_float_list(value: str) -> list[float]:
    return sorted(set(float(x.strip()) for x in value.split(",") if x.strip()))


def parse_city_list(value: str | None) -> list[str]:
    if not value:
        return DEFAULT_CITIES
    return [x.strip() for x in value.split(",") if x.strip()]


def as_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_existing(force: bool) -> tuple[list[dict[str, Any]], set[tuple[str, float, int]]]:
    if force or not RECORDS_CSV.exists():
        return [], set()
    records = pd.read_csv(RECORDS_CSV).to_dict("records")
    done = {
        (str(r["city"]), round(float(r["target_swap_fraction"]), 6), int(r["replicate"]))
        for r in records
        if str(r.get("status", "")) not in {"fail_before_write"}
    }
    return records, done


def write_partial(records: list[dict[str, Any]]) -> None:
    pd.DataFrame(records).to_csv(RECORDS_CSV, index=False)


def record_threshold(
    n: int,
    edges: np.ndarray,
    grid: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[float, float, float, bool, np.ndarray]:
    return r90.threshold_from_edges(n, edges, grid, args.n_perm, args.n_bootstrap, seed)


def run(args: argparse.Namespace) -> pd.DataFrame:
    registry = pd.read_csv(SUBSET)
    city_order = parse_city_list(args.cities)
    registry["_order"] = registry["city"].map({city: i for i, city in enumerate(city_order)})
    registry = registry[registry["_order"].notna()].sort_values("_order").reset_index(drop=True)
    if args.max_cities:
        registry = registry.head(args.max_cities).copy()
    fractions = parse_float_list(args.swap_fractions)

    records, done = load_existing(args.force)
    for city_idx, row in registry.iterrows():
        city = str(row["city"])
        graph_path = as_project_path(str(row["graph_cache_file"]))
        print(f"[N99-low-overlap] city {city}, edges={int(row['n_edges'])}", flush=True)
        G = r56.load_graph(graph_path)
        _H, edges, degrees, lon, lat = r56.to_simple_spatial_graph(G)
        del G
        gc.collect()

        n = len(degrees)
        pc_cebh = float(row["pc_cebh"])
        pc_road_coarse = float(row["pc_rank"])
        grid = r90.fine_grid(pc_road_coarse, args.p_window, args.p_step)
        road_pc, road_lo, road_hi, road_boundary, road_curve = record_threshold(
            n,
            edges,
            grid,
            args,
            args.seed + 1_003_003 * (city_idx + 1),
        )
        original_edges = r90.edge_set(edges)
        original_metrics = r67.embedded_metrics(edges, lon, lat, include_connectivity=True)

        for fraction in fractions:
            for rep in range(args.replicates):
                key = (city, round(float(fraction), 6), int(rep))
                if key in done:
                    continue
                seed = int(args.seed + 100_003 * (city_idx + 1) + 11_017 * rep + round(fraction * 1_000_000))
                t0 = time.time()
                try:
                    null_edges, diag = r60.geometry_preserving_rewire(
                        edges,
                        degrees,
                        lon,
                        lat,
                        seed=seed,
                        length_bins=args.length_bins,
                        bin_tolerance=args.bin_tolerance,
                        swap_fraction=float(fraction),
                        max_attempt_factor=args.max_attempt_factor,
                        neighbor_k=args.neighbor_k,
                        rebuild_interval=args.rebuild_interval,
                    )
                    pc, lo, hi, boundary, null_curve = record_threshold(n, null_edges, grid, args, seed + 55_001)
                    null_edges_set = r90.edge_set(null_edges)
                    inter = len(original_edges & null_edges_set)
                    union = len(original_edges | null_edges_set)
                    null_metrics = r67.embedded_metrics(null_edges, lon, lat, include_connectivity=True)
                    achieved = float(diag.get("accepted_swaps_per_edge", 0.0))
                    edge_jaccard = float(inter / union) if union else float("nan")
                    status = "pass"
                    if achieved < 0.8 * float(fraction):
                        status = "partial_acceptance"
                    if edge_jaccard > args.low_overlap_jaccard:
                        status = "high_overlap_" + status
                    record = {
                        "city": city,
                        "country": row["country"],
                        "macro_region": row["macro_region"],
                        "replicate": int(rep),
                        "target_swap_fraction": float(fraction),
                        "n_nodes": int(n),
                        "n_edges": int(len(edges)),
                        "pc_cebh": pc_cebh,
                        "pc_road_fine": float(road_pc),
                        "pc_road_fine_ci_low": float(road_lo),
                        "pc_road_fine_ci_high": float(road_hi),
                        "pc_geometry_fine": float(pc),
                        "pc_geometry_fine_ci_low": float(lo),
                        "pc_geometry_fine_ci_high": float(hi),
                        "road_gap_fine": float(road_pc - pc_cebh),
                        "geometry_gap_fine": float(pc - pc_cebh),
                        "road_minus_geometry_fine": float(road_pc - pc),
                        "threshold_boundary_hit": bool(boundary or road_boundary),
                        "accepted_swaps": int(diag.get("accepted_swaps", 0)),
                        "target_swaps": int(diag.get("target_swaps", 0)),
                        "accepted_swaps_per_edge": achieved,
                        "attempted_swaps": int(diag.get("attempted_swaps", 0)),
                        "acceptance_rate": float(diag.get("acceptance_rate", float("nan"))),
                        "degree_max_abs_drift": float(diag.get("degree_max_abs_drift", float("nan"))),
                        "length_bin_tvd": float(diag.get("length_bin_tvd", float("nan"))),
                        "length_mean_ratio": float(diag.get("length_mean_ratio", float("nan"))),
                        "length_median_ratio": float(diag.get("length_median_ratio", float("nan"))),
                        "topology_rejects": int(diag.get("topology_rejects", 0)),
                        "length_rejects": int(diag.get("length_rejects", 0)),
                        "crossing_rejects": int(diag.get("crossing_rejects", 0)),
                        "exact_edge_overlap": int(inter),
                        "edge_jaccard": edge_jaccard,
                        "edge_turnover_share": float(1.0 - inter / max(1, len(original_edges))),
                        "lcc_curve_l2": float(np.sqrt(np.mean((road_curve - null_curve) ** 2))),
                        "delta_connected_components": float(
                            null_metrics.get("connected_components", np.nan)
                            - original_metrics.get("connected_components", np.nan)
                        ),
                        "delta_bridge_edge_share": float(
                            null_metrics.get("bridge_edge_share", np.nan)
                            - original_metrics.get("bridge_edge_share", np.nan)
                        ),
                        "delta_articulation_point_share": float(
                            null_metrics.get("articulation_point_share", np.nan)
                            - original_metrics.get("articulation_point_share", np.nan)
                        ),
                        "delta_largest_biconnected_node_fraction": float(
                            null_metrics.get("largest_biconnected_node_fraction", np.nan)
                            - original_metrics.get("largest_biconnected_node_fraction", np.nan)
                        ),
                        "delta_cycle_density": float(
                            null_metrics.get("cyclomatic_density", np.nan)
                            - original_metrics.get("cyclomatic_density", np.nan)
                        ),
                        "status": status,
                        "runtime_seconds": float(time.time() - t0),
                    }
                    records.append(record)
                    done.add(key)
                    write_partial(records)
                    print(
                        f"[N99-low-overlap] {city} f={fraction:.3f} rep={rep}: "
                        f"edgeJ={edge_jaccard:.3f}, accepted={achieved:.3f}, residual={record['road_minus_geometry_fine']:.3f}, "
                        f"status={status}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    records.append(
                        {
                            "city": city,
                            "country": row.get("country", ""),
                            "macro_region": row.get("macro_region", ""),
                            "replicate": int(rep),
                            "target_swap_fraction": float(fraction),
                            "n_nodes": int(n),
                            "n_edges": int(len(edges)),
                            "pc_cebh": pc_cebh,
                            "pc_road_fine": float(road_pc),
                            "status": "fail",
                            "notes": repr(exc),
                            "runtime_seconds": float(time.time() - t0),
                        }
                    )
                    done.add(key)
                    write_partial(records)
                    print(f"[N99-low-overlap] fail {city} f={fraction:.3f} rep={rep}: {exc}", flush=True)
        gc.collect()
    return pd.DataFrame(records)


def mean_ci(values: pd.Series) -> tuple[float, float, float]:
    arr = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    if len(arr) == 1:
        return mean, mean, mean
    se = float(np.std(arr, ddof=1) / math.sqrt(len(arr)))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def summarize(records: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    passed = records[records["status"].astype(str).str.contains("pass|high_overlap|partial_acceptance", regex=True)].copy()
    spatial = pd.read_csv(R64)
    city_rows: list[dict[str, Any]] = []
    if len(passed):
        for keys, group in passed.groupby(["city", "country", "macro_region", "target_swap_fraction"]):
            mean_res, lo_res, hi_res = mean_ci(group["road_minus_geometry_fine"])
            mean_j, lo_j, hi_j = mean_ci(group["edge_jaccard"])
            city_rows.append(
                {
                    "city": keys[0],
                    "country": keys[1],
                    "macro_region": keys[2],
                    "target_swap_fraction": float(keys[3]),
                    "n_records": int(len(group)),
                    "mean_edge_jaccard": mean_j,
                    "ci95_edge_jaccard_low": lo_j,
                    "ci95_edge_jaccard_high": hi_j,
                    "min_edge_jaccard": float(group["edge_jaccard"].min()),
                    "mean_edge_turnover_share": float(group["edge_turnover_share"].mean()),
                    "mean_accepted_swaps_per_edge": float(group["accepted_swaps_per_edge"].mean()),
                    "mean_acceptance_rate": float(group["acceptance_rate"].mean()),
                    "mean_length_bin_tvd": float(group["length_bin_tvd"].mean()),
                    "max_degree_drift": float(group["degree_max_abs_drift"].max()),
                    "mean_road_minus_geometry_fine": mean_res,
                    "ci95_road_minus_geometry_fine_low": lo_res,
                    "ci95_road_minus_geometry_fine_high": hi_res,
                    "n_below_low_overlap_jaccard": int((group["edge_jaccard"] <= args.low_overlap_jaccard).sum()),
                    "n_below_090_jaccard": int((group["edge_jaccard"] <= 0.90).sum()),
                    "n_below_085_jaccard": int((group["edge_jaccard"] <= 0.85).sum()),
                }
            )
    summary_df = pd.DataFrame(city_rows)
    if len(summary_df):
        summary_df = summary_df.merge(
            spatial[["city", "pc_spatial_null", "road_minus_spatial", "spatial_null_gap"]],
            on="city",
            how="left",
        )
        summary_df["spatial_minus_low_overlap_geometry"] = (
            summary_df["road_minus_spatial"] - summary_df["mean_road_minus_geometry_fine"]
        )
    summary_df.to_csv(SUMMARY_CSV, index=False)

    low = passed[pd.to_numeric(passed.get("edge_jaccard"), errors="coerce") <= args.low_overlap_jaccard]
    res_mean, res_lo, res_hi = mean_ci(passed["road_minus_geometry_fine"] if len(passed) else pd.Series(dtype=float))
    j_mean, j_lo, j_hi = mean_ci(passed["edge_jaccard"] if len(passed) else pd.Series(dtype=float))
    status = "pass_low_overlap_stratified_audit" if len(low) and passed["city"].nunique() >= 12 else (
        "pass_low_overlap_pilot" if len(low) else "partial_high_overlap_only"
    )
    summary = {
        "round": "N99_low_overlap_geometry_surrogate",
        "status": status,
        "n_records_total": int(len(records)),
        "n_records_usable": int(len(passed)),
        "n_cities": int(passed["city"].nunique()) if len(passed) else 0,
        "target_swap_fractions": sorted(float(v) for v in passed["target_swap_fraction"].dropna().unique())
        if len(passed)
        else [],
        "low_overlap_jaccard_threshold": float(args.low_overlap_jaccard),
        "n_records_below_low_overlap_jaccard": int(len(low)),
        "min_edge_jaccard": float(passed["edge_jaccard"].min()) if len(passed) else float("nan"),
        "mean_edge_jaccard": j_mean,
        "ci95_edge_jaccard_low": j_lo,
        "ci95_edge_jaccard_high": j_hi,
        "mean_road_minus_geometry_fine": res_mean,
        "ci95_road_minus_geometry_fine_low": res_lo,
        "ci95_road_minus_geometry_fine_high": res_hi,
        "mean_accepted_swaps_per_edge": float(passed["accepted_swaps_per_edge"].mean()) if len(passed) else float("nan"),
        "mean_length_bin_tvd": float(passed["length_bin_tvd"].mean()) if len(passed) else float("nan"),
        "max_degree_drift": float(passed["degree_max_abs_drift"].max()) if len(passed) else float("nan"),
        "claim_boundary": (
            "Stratified audit only. The test asks whether the implemented strict non-crossing local-swap algorithm can be "
            "pushed toward lower exact edge overlap while preserving degree and length-bin constraints. It is not "
            "a full planar-ensemble mixing proof."
        ),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_df, summary


def build_figure(records: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    if records.empty:
        return
    usable = records[records["status"].astype(str).ne("fail")].copy()
    if usable.empty:
        return
    pub_style.apply()
    fig, axes = plt.subplots(2, 2, figsize=(pub_style.FIG_WIDTH_2COL, 5.9), constrained_layout=True)
    colors = pub_style.REGION_COLORS

    ax = axes[0, 0]
    rng = np.random.default_rng(99)
    for region, sub in usable.groupby("macro_region"):
        xj = sub["target_swap_fraction"] + rng.normal(0, 0.0011, len(sub))
        ax.scatter(
            xj,
            sub["edge_jaccard"],
            s=26,
            color=colors.get(region, colors["Other"]),
            alpha=0.8,
            linewidths=0,
            label=region,
        )
    ax.axhline(0.90, color="#4D4D4D", lw=0.7, ls="--")
    ax.set_xlabel("Target accepted swaps per edge")
    ax.set_ylabel("Exact edge Jaccard")
    pub_style.panel_title(ax, "a", "Edge overlap at higher swap targets")
    pub_style.light_grid(ax, axis="y")
    ax.legend(frameon=False, fontsize=5.8, ncol=2, handletextpad=0.25, columnspacing=0.8)

    ax = axes[0, 1]
    ax.scatter(
        usable["target_swap_fraction"],
        usable["accepted_swaps_per_edge"],
        s=24,
        color=pub_style.COLORS["geometry_null"],
        alpha=0.75,
        linewidths=0,
    )
    lim = max(float(usable["target_swap_fraction"].max()), float(usable["accepted_swaps_per_edge"].max())) * 1.08
    ax.plot([0, lim], [0, lim], color="#1A1A1A", lw=0.7, ls="--")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Target accepted swaps per edge")
    ax.set_ylabel("Achieved accepted swaps per edge")
    pub_style.panel_title(ax, "b", "Accepted-swap fidelity")
    pub_style.light_grid(ax, axis="both")

    ax = axes[1, 0]
    for region, sub in usable.groupby("macro_region"):
        ax.scatter(
            sub["edge_jaccard"],
            sub["road_minus_geometry_fine"],
            s=26,
            color=colors.get(region, colors["Other"]),
            alpha=0.8,
            linewidths=0,
        )
    ax.axhline(0, color="#1A1A1A", lw=0.7)
    ax.set_xlabel("Exact edge Jaccard")
    ax.set_ylabel("Road-minus-geometry residual")
    pub_style.panel_title(ax, "c", "Residual versus edge overlap")
    pub_style.light_grid(ax, axis="both")

    ax = axes[1, 1]
    if not summary_df.empty and "spatial_minus_low_overlap_geometry" in summary_df.columns:
        sub = summary_df.dropna(subset=["road_minus_spatial", "mean_road_minus_geometry_fine"]).reset_index(drop=True)
        rng_d = np.random.default_rng(991)
        x_sp = rng_d.normal(1.0, 0.05, len(sub))
        x_geo = rng_d.normal(2.0, 0.05, len(sub))
        for i in range(len(sub)):
            ax.plot(
                [x_sp[i], x_geo[i]],
                [sub.loc[i, "road_minus_spatial"], sub.loc[i, "mean_road_minus_geometry_fine"]],
                color="#C7C7C7",
                lw=0.6,
                zorder=1,
            )
        ax.scatter(x_sp, sub["road_minus_spatial"], s=22, color=pub_style.COLORS["spatial_null"], alpha=0.85, linewidths=0, zorder=2, label="Spatial-scale null")
        ax.scatter(x_geo, sub["mean_road_minus_geometry_fine"], s=22, color=pub_style.COLORS["geometry_null"], alpha=0.85, linewidths=0, zorder=2, label="Strict geometry audit")
        for xc, col in ((1.0, "road_minus_spatial"), (2.0, "mean_road_minus_geometry_fine")):
            ax.hlines(float(sub[col].mean()), xc - 0.18, xc + 0.18, color="#1A1A1A", lw=1.4, zorder=3)
        ax.axhline(0, color="#4D4D4D", lw=0.7, ls="--")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Spatial-scale\nnull", "Strict geometry\naudit"])
        ax.set_ylabel("Road-minus-null residual")
        pub_style.annot(ax, 0.97, 0.96, "paired by city-fraction record", ha="right")
        ax.legend(frameon=False, fontsize=6.0, loc="upper left")
    pub_style.panel_title(ax, "d", "Spatial versus geometry residuals")
    pub_style.light_grid(ax, axis="y")

    pub_style.save(fig, FIG_BASE)
    plt.close(fig)


def write_reports(summary: dict[str, Any], summary_df: pd.DataFrame) -> None:
    table = summary_df.to_markdown(index=False, floatfmt=".4f") if not summary_df.empty else "No usable records."
    REPORT_MD.write_text(
        f"""# N99 Low-Overlap Strict-Geometry Surrogate Audit

## Purpose

This stratified audit tests whether the implemented strict non-crossing
local-swap null can be pushed toward lower exact edge overlap while preserving
degree and edge-length-bin constraints.

## Status

{summary['status']}

## Headline diagnostics

- Usable records: {summary['n_records_usable']} of {summary['n_records_total']}
- Cities: {summary['n_cities']}
- Minimum exact edge Jaccard: {summary['min_edge_jaccard']:.4f}
- Mean exact edge Jaccard: {summary['mean_edge_jaccard']:.4f}
- Records below low-overlap threshold ({summary['low_overlap_jaccard_threshold']:.2f}): {summary['n_records_below_low_overlap_jaccard']}
- Mean road-minus-geometry residual: {summary['mean_road_minus_geometry_fine']:.4f}
- Mean accepted swaps per edge: {summary['mean_accepted_swaps_per_edge']:.4f}
- Maximum degree drift: {summary['max_degree_drift']:.4f}

## City-level summary

{table}

## Claim boundary

{summary['claim_boundary']}
""",
        encoding="utf-8",
    )
    GATE_REPORT_MD.write_text(
        f"""# N99 Gate Report: Low-Overlap Strict-Geometry Surrogate

## Gate status

{summary['status']}

## Evidence

- Records: `{RECORDS_CSV.relative_to(ROOT)}`
- Summary: `{SUMMARY_CSV.relative_to(ROOT)}`
- Figure: `{FIG_BASE.with_suffix('.svg').relative_to(ROOT)}`

## Interpretation

If low-overlap records exist, they are stratified audit evidence under the
implemented strict local-swap chain. If they do not, the result supports the
manuscript boundary that the present strict geometry null is a local
counterfactual perturbation and not a fully mixed low-overlap planar surrogate.
""",
        encoding="utf-8",
    )
    CLOSEOUT_MD.write_text(
        f"""# N99 Substantive Increment Closeout

| Item | Status | Evidence | Boundary |
|---|---|---|---|
| Low-overlap strict-geometry audit | {'Done' if summary['status'] in {'pass_low_overlap_pilot', 'pass_low_overlap_stratified_audit'} else 'Partial'} | `{RECORDS_CSV.relative_to(ROOT)}`; min edge Jaccard {summary['min_edge_jaccard']:.4f} | Stratified audit only; no mixing theorem |
| Degree/length/non-crossing diagnostics | Done | acceptance, degree drift, length-bin TVD and crossing rejections in records | Applies to implemented local-swap algorithm |
| Full 71-city low-overlap planar ensemble | Deferred | Not claimed | Computational and theoretical mixing boundary |
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", default=None)
    parser.add_argument("--max-cities", type=int, default=6)
    parser.add_argument("--swap-fractions", default="0.04,0.08")
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--length-bins", type=int, default=12)
    parser.add_argument("--bin-tolerance", type=int, default=1)
    parser.add_argument("--max-attempt-factor", type=int, default=900)
    parser.add_argument("--neighbor-k", type=int, default=320)
    parser.add_argument("--rebuild-interval", type=int, default=50)
    parser.add_argument("--n-perm", type=int, default=8)
    parser.add_argument("--n-bootstrap", type=int, default=120)
    parser.add_argument("--p-window", type=float, default=0.06)
    parser.add_argument("--p-step", type=float, default=0.005)
    parser.add_argument("--low-overlap-jaccard", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=99111)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-compute", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    if args.skip_compute and RECORDS_CSV.exists():
        records = pd.read_csv(RECORDS_CSV)
    else:
        records = run(args)
    summary_df, summary = summarize(records, args)
    build_figure(records, summary_df)
    write_reports(summary, summary_df)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
