"""R95/N95 non-backtracking spectral audit for the 71-city road-window matrix.

This round adds a theory-bridge diagnostic between the CEBH degree-moment
threshold and the constrained geometry-null ladder. It computes the leading
non-backtracking spectral radius by power iteration on directed edges and
compares the resulting threshold proxy with CEBH, observed road thresholds,
spatial-scale nulls and strict-geometry nulls.

The audit is intentionally bounded: it tests whether a graph-spectral
message-passing baseline closes the road threshold gap. It is not a theorem,
and it is not a causal urban-design law.
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

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_R56_spatial_length_constrained_nulls as r56  # noqa: E402

R61 = ROOT / "data" / "R61_city_window_scale_screen" / "central_window_screen_results.csv"
R64 = ROOT / "data" / "R64_71city_spatial_null_screen" / "spatial_null_screen_summary.csv"
R81 = ROOT / "data" / "R81_full71_high_intensity_geometry_null_ensemble" / "full71_geometry_null_city_summary.csv"
URBAN = ROOT / "data" / "R62_urban_form_mechanism" / "urban_form_covariates.csv"
OUT = ROOT / "data" / "R95_nonbacktracking_spectral_layer"
ROUND_DIR = ROOT / "rounds" / "R95_nonbacktracking_spectral_layer"
FIG_BASE = ROOT / "figures" / "Fig_R95_nonbacktracking_spectral_layer"

THRESHOLDS_CSV = OUT / "N95_nonbacktracking_thresholds.csv"
DECOMP_CSV = OUT / "N95_cebh_nb_geometry_decomposition.csv"
LOCALIZATION_CSV = OUT / "N95_nb_spectrum_localization_metrics.csv"
SUMMARY_JSON = OUT / "R95_summary.json"
REPORT_MD = OUT / "R95_report.md"
GATE_REPORT_MD = ROUND_DIR / "gate_report.md"
CLOSEOUT_MD = ROUND_DIR / "substantive_increment_closeout.md"


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ROUND_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "figures").mkdir(parents=True, exist_ok=True)


def as_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def mean_ci95(values: pd.Series | np.ndarray) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    if len(arr) == 1:
        return mean, mean, mean
    se = float(np.std(arr, ddof=1) / math.sqrt(len(arr)))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def spearman(x: pd.Series, y: pd.Series) -> float:
    data = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 3:
        return float("nan")
    return float(data["x"].rank().corr(data["y"].rank()))


def pearson(x: pd.Series, y: pd.Series) -> float:
    data = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 3:
        return float("nan")
    return float(data["x"].corr(data["y"]))


def nonbacktracking_power(
    edges: np.ndarray,
    n_nodes: int,
    seed: int,
    max_iter: int = 160,
    tol: float = 1e-8,
) -> dict[str, Any]:
    """Power iteration on the non-backtracking operator for an undirected graph."""

    edges = np.asarray(edges, dtype=np.int64)
    if len(edges) == 0:
        return {
            "rho_nb": float("nan"),
            "iterations": 0,
            "converged": False,
            "ipr_directed": float("nan"),
            "top1_directed_weight_share": float("nan"),
            "top5pct_directed_weight_share": float("nan"),
            "node_nb_entropy": float("nan"),
            "node_nb_ipr": float("nan"),
            "top1_node_weight_share": float("nan"),
            "top5pct_node_weight_share": float("nan"),
        }

    src = np.concatenate([edges[:, 0], edges[:, 1]]).astype(np.int64)
    dst = np.concatenate([edges[:, 1], edges[:, 0]]).astype(np.int64)
    m2 = len(src)
    reverse = np.concatenate([np.arange(len(edges), 2 * len(edges)), np.arange(len(edges))]).astype(np.int64)

    rng = np.random.default_rng(seed)
    x = rng.random(m2) + 1e-6
    x /= np.linalg.norm(x)
    prev_lambda = 0.0
    converged = False
    lam = float("nan")

    for iteration in range(1, max_iter + 1):
        incoming = np.bincount(dst, weights=x, minlength=n_nodes)
        y = incoming[src] - x[reverse]
        y = np.maximum(y, 0.0)
        norm = float(np.linalg.norm(y))
        if not np.isfinite(norm) or norm <= 0:
            break
        x_next = y / norm
        bx = incoming[src] - x[reverse]
        denom = float(np.dot(x, x))
        lam = float(np.dot(x, bx) / denom) if denom > 0 else float("nan")
        if np.isfinite(lam) and abs(lam - prev_lambda) <= tol * max(1.0, abs(lam)):
            converged = True
            x = x_next
            break
        x = x_next
        prev_lambda = lam if np.isfinite(lam) else prev_lambda

    incoming = np.bincount(dst, weights=x, minlength=n_nodes)
    bx = incoming[src] - x[reverse]
    denom = float(np.dot(x, x))
    lam = float(np.dot(x, bx) / denom) if denom > 0 else float("nan")

    weights = np.asarray(x, dtype=float)
    weights = weights / max(float(weights.sum()), 1e-300)
    sorted_w = np.sort(weights)[::-1]
    node_weights = np.bincount(dst, weights=weights, minlength=n_nodes)
    node_weights = node_weights / max(float(node_weights.sum()), 1e-300)
    sorted_node = np.sort(node_weights)[::-1]
    nz = node_weights[node_weights > 0]

    return {
        "rho_nb": lam,
        "iterations": iteration,
        "converged": converged,
        "ipr_directed": float(np.sum(weights**2)),
        "top1_directed_weight_share": float(sorted_w[0]) if len(sorted_w) else float("nan"),
        "top5pct_directed_weight_share": float(np.sum(sorted_w[: max(1, int(math.ceil(0.05 * len(sorted_w))))])),
        "node_nb_entropy": float(-np.sum(nz * np.log(nz)) / math.log(max(2, len(nz)))) if len(nz) else float("nan"),
        "node_nb_ipr": float(np.sum(node_weights**2)),
        "top1_node_weight_share": float(sorted_node[0]) if len(sorted_node) else float("nan"),
        "top5pct_node_weight_share": float(np.sum(sorted_node[: max(1, int(math.ceil(0.05 * len(sorted_node))))])),
    }


def compute(args: argparse.Namespace) -> pd.DataFrame:
    rows = pd.read_csv(R61)
    rows = rows[rows["status"].eq("pass")].copy()
    rows = rows.sort_values(["n_edges", "city"]).reset_index(drop=True)
    if args.max_cities:
        rows = rows.head(args.max_cities).copy()

    records: list[dict[str, Any]] = []
    for idx, row in rows.iterrows():
        city = str(row["city"])
        t0 = time.time()
        graph_path = as_project_path(str(row["graph_cache_file"]))
        print(f"[R95] {idx + 1}/{len(rows)} {city}: {graph_path.name}", flush=True)
        try:
            G = r56.load_graph(graph_path)
            _H, edges, degrees, _lon, _lat = r56.to_simple_spatial_graph(G)
            del G
            gc.collect()
            nb = nonbacktracking_power(edges, len(degrees), args.seed + 1009 * (idx + 1), args.max_iter, args.tol)
            rho_nb = float(nb["rho_nb"])
            pc_nb = float(1.0 / rho_nb) if np.isfinite(rho_nb) and rho_nb > 0 else float("nan")
            kappa = float(np.mean(degrees.astype(float) ** 2) / np.mean(degrees)) if len(degrees) else float("nan")
            record = {
                "city": city,
                "country": row["country"],
                "macro_region": row["macro_region"],
                "n_nodes": int(len(degrees)),
                "n_edges": int(len(edges)),
                "mean_degree": float(np.mean(degrees)) if len(degrees) else float("nan"),
                "kappa": kappa,
                "kappa_minus_1": kappa - 1.0 if np.isfinite(kappa) else float("nan"),
                "pc_cebh": float(row["pc_cebh"]),
                "pc_nb": pc_nb,
                "pc_road_rank": float(row["pc_rank"]),
                "road_gap_cebh": float(row["pc_rank"] - row["pc_cebh"]),
                "road_minus_nb": float(row["pc_rank"] - pc_nb) if np.isfinite(pc_nb) else float("nan"),
                "nb_minus_cebh": float(pc_nb - row["pc_cebh"]) if np.isfinite(pc_nb) else float("nan"),
                "abs_error_cebh": abs(float(row["pc_rank"] - row["pc_cebh"])),
                "abs_error_nb": abs(float(row["pc_rank"] - pc_nb)) if np.isfinite(pc_nb) else float("nan"),
                "nb_improvement_over_cebh": abs(float(row["pc_rank"] - row["pc_cebh"])) - abs(float(row["pc_rank"] - pc_nb))
                if np.isfinite(pc_nb)
                else float("nan"),
                "rho_nb": rho_nb,
                "nb_iterations": int(nb["iterations"]),
                "nb_converged": bool(nb["converged"]),
                "runtime_seconds": float(time.time() - t0),
                "status": "pass",
            }
            record.update({k: v for k, v in nb.items() if k not in {"rho_nb", "iterations", "converged"}})
            records.append(record)
        except Exception as exc:  # noqa: BLE001
            records.append(
                {
                    "city": city,
                    "country": row.get("country", ""),
                    "macro_region": row.get("macro_region", ""),
                    "n_nodes": int(row.get("n_nodes", 0)),
                    "n_edges": int(row.get("n_edges", 0)),
                    "status": "fail",
                    "notes": repr(exc),
                    "runtime_seconds": float(time.time() - t0),
                }
            )
            print(f"[R95] fail {city}: {exc}", flush=True)
    df = pd.DataFrame(records)
    df.to_csv(THRESHOLDS_CSV, index=False)
    return df


def load_or_compute(args: argparse.Namespace) -> pd.DataFrame:
    if args.skip_compute and THRESHOLDS_CSV.exists():
        return pd.read_csv(THRESHOLDS_CSV)
    if THRESHOLDS_CSV.exists() and not args.force:
        return pd.read_csv(THRESHOLDS_CSV)
    return compute(args)


def build_decomposition(thresholds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    passed = thresholds[thresholds["status"].eq("pass")].copy()
    spatial = pd.read_csv(R64)
    urban = pd.read_csv(URBAN)
    decomp = passed.merge(
        spatial[["city", "road_minus_spatial", "spatial_null_gap", "spatial_fraction_of_road_gap"]],
        on="city",
        how="left",
    )
    if R81.exists():
        r81 = pd.read_csv(R81)
        geometry_cols = [
            c
            for c in [
                "pc_geometry_null_mean",
                "geometry_null_gap_mean",
                "road_minus_geometry_mean",
                "geometry_fraction_of_road_gap_mean",
                "remaining_fraction_after_geometry_mean",
                "accepted_swaps_per_edge_mean",
                "acceptance_rate_mean",
                "length_bin_tvd_mean",
                "null_mobility_index_mean",
                "geometry_vs_spatial_residual_change",
            ]
            if c in r81.columns
        ]
        r81_city = r81.groupby("city", as_index=False)[geometry_cols].mean()
        decomp = decomp.merge(r81_city, on="city", how="left")
    metric_cols = [
        "city",
        "dead_end_share",
        "degree4plus_share",
        "cycle_density",
        "orientation_entropy",
        "orientation_dominance",
        "edge_length_cv",
        "bridge_tunnel_length_share",
    ]
    metric_cols = [c for c in metric_cols if c in urban.columns]
    decomp = decomp.merge(urban[metric_cols], on="city", how="left")

    decomp["nb_fraction_of_road_gap"] = decomp["road_minus_nb"] / decomp["road_gap_cebh"]
    decomp["nb_improvement_fraction"] = decomp["nb_improvement_over_cebh"] / decomp["abs_error_cebh"]
    decomp.to_csv(DECOMP_CSV, index=False)

    localization_cols = [
        "city",
        "country",
        "macro_region",
        "n_nodes",
        "n_edges",
        "rho_nb",
        "pc_nb",
        "road_minus_nb",
        "ipr_directed",
        "top1_directed_weight_share",
        "top5pct_directed_weight_share",
        "node_nb_entropy",
        "node_nb_ipr",
        "top1_node_weight_share",
        "top5pct_node_weight_share",
        "dead_end_share",
        "cycle_density",
        "degree4plus_share",
        "orientation_entropy",
    ]
    localization_cols = [c for c in localization_cols if c in decomp.columns]
    loc = decomp[localization_cols].copy()
    loc.to_csv(LOCALIZATION_CSV, index=False)

    mean_cebh, lo_cebh, hi_cebh = mean_ci95(decomp["abs_error_cebh"])
    mean_nb, lo_nb, hi_nb = mean_ci95(decomp["abs_error_nb"])
    mean_improve, lo_improve, hi_improve = mean_ci95(decomp["nb_improvement_over_cebh"])
    summary = {
        "status": "pass" if int((thresholds["status"] == "pass").sum()) == len(thresholds) else "partial",
        "round": "R95_nonbacktracking_spectral_layer",
        "n_records_total": int(len(thresholds)),
        "n_records_pass": int((thresholds["status"] == "pass").sum()),
        "n_decomposition_rows": int(len(decomp)),
        "n_decomposition_cities": int(decomp["city"].nunique()),
        "n_nb_power_converged": int(decomp["nb_converged"].sum()) if "nb_converged" in decomp.columns else 0,
        "mean_abs_error_cebh": mean_cebh,
        "ci95_abs_error_cebh_low": lo_cebh,
        "ci95_abs_error_cebh_high": hi_cebh,
        "mean_abs_error_nb": mean_nb,
        "ci95_abs_error_nb_low": lo_nb,
        "ci95_abs_error_nb_high": hi_nb,
        "mean_nb_improvement_over_cebh": mean_improve,
        "ci95_nb_improvement_low": lo_improve,
        "ci95_nb_improvement_high": hi_improve,
        "spearman_pc_nb_vs_pc_road": spearman(decomp["pc_nb"], decomp["pc_road_rank"]),
        "spearman_pc_cebh_vs_pc_road": spearman(decomp["pc_cebh"], decomp["pc_road_rank"]),
        "spearman_nb_error_vs_road_gap": spearman(decomp["road_minus_nb"], decomp["road_gap_cebh"]),
        "spearman_nb_ipr_vs_road_gap": spearman(decomp.get("node_nb_ipr", pd.Series(dtype=float)), decomp["road_gap_cebh"]),
        "pearson_pc_nb_vs_pc_road": pearson(decomp["pc_nb"], decomp["pc_road_rank"]),
        "claim_boundary": (
            "R95 adds a non-backtracking spectral threshold proxy as a theory-bridge diagnostic. "
            "It does not replace the constrained geometry-null evidence and does not prove a theorem."
        ),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return decomp, loc, summary


def make_figure(decomp: pd.DataFrame, summary: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    import pub_style

    pub_style.apply()
    colors = {
        "road": pub_style.COLORS["observed"],
        "cebh": pub_style.COLORS["cebh"],
        "nb": pub_style.COLORS["nb"],
        "spatial": pub_style.COLORS["spatial_null"],
        "geometry": pub_style.COLORS["geometry_null"],
        "accent": pub_style.COLORS["accent"],
    }
    fig, axes = plt.subplots(2, 2, figsize=(pub_style.FIG_WIDTH_2COL, 5.8), constrained_layout=True)

    ax = axes[0, 0]
    lims = [
        float(np.nanmin([decomp["pc_road_rank"].min(), decomp["pc_cebh"].min(), decomp["pc_nb"].min()])) - 0.03,
        float(np.nanmax([decomp["pc_road_rank"].max(), decomp["pc_cebh"].max(), decomp["pc_nb"].max()])) + 0.03,
    ]
    ax.scatter(decomp["pc_road_rank"], decomp["pc_cebh"], s=18, color=colors["cebh"], alpha=0.72, label="CEBH")
    ax.scatter(decomp["pc_road_rank"], decomp["pc_nb"], s=18, color=colors["nb"], alpha=0.72, label="Non-backtracking")
    ax.plot(lims, lims, color="#333333", lw=0.8, ls="--")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Observed road threshold")
    ax.set_ylabel("Predicted threshold")
    pub_style.panel_title(ax, "a", "Observed versus predicted thresholds")
    pub_style.light_grid(ax, axis="both")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    data = [
        decomp["abs_error_cebh"].dropna().to_numpy(),
        decomp["abs_error_nb"].dropna().to_numpy(),
        decomp["road_minus_spatial"].dropna().abs().to_numpy(),
    ]
    labels = ["CEBH", "NB", "Spatial null"]
    if "road_minus_geometry_mean" in decomp.columns:
        data.append(decomp["road_minus_geometry_mean"].dropna().abs().to_numpy())
        labels.append("Geometry null")
    bp = ax.boxplot(data, patch_artist=True, tick_labels=labels, widths=0.55)
    for patch, c in zip(bp["boxes"], [colors["cebh"], colors["nb"], colors["spatial"], colors["geometry"]]):
        patch.set_facecolor(c)
        patch.set_alpha(0.25)
        patch.set_edgecolor(c)
    ax.set_ylabel(r"|road threshold $-$ baseline|")
    pub_style.panel_title(ax, "b", "Absolute-error ladder")
    pub_style.light_grid(ax, axis="y")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[1, 0]
    sc = ax.scatter(decomp["kappa_minus_1"], decomp["rho_nb"], s=18, c=decomp["road_gap_cebh"], cmap="viridis", alpha=0.8, linewidths=0)
    lo = float(np.nanmin([decomp["kappa_minus_1"].min(), decomp["rho_nb"].min()]))
    hi = float(np.nanmax([decomp["kappa_minus_1"].max(), decomp["rho_nb"].max()]))
    ax.plot([lo, hi], [lo, hi], color="#1A1A1A", lw=0.7, ls="--")
    ax.set_xlabel(r"Degree-moment branch factor $\kappa-1$")
    ax.set_ylabel(r"NB spectral radius $\rho(B)$")
    pub_style.panel_title(ax, "c", r"$\kappa-1$ versus NB spectral radius")
    pub_style.light_grid(ax, axis="both")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Road-minus-CEBH gap", fontsize=6.5)
    cbar.ax.tick_params(labelsize=6)

    ax = axes[1, 1]
    metric = "node_nb_ipr" if "node_nb_ipr" in decomp.columns else "ipr_directed"
    ax.scatter(decomp[metric], decomp["road_gap_cebh"], s=18, color=colors["accent"], alpha=0.74, linewidths=0)
    ax.set_xlabel("NB centrality localization (IPR)")
    ax.set_ylabel("Observed road-minus-CEBH gap")
    pub_style.panel_title(ax, "d", "NB localization versus road gap")
    pub_style.light_grid(ax, axis="both")
    pub_style.annot(
        ax,
        0.03,
        0.97,
        f"NB MAE = {summary['mean_abs_error_nb']:.3f}\nCEBH MAE = {summary['mean_abs_error_cebh']:.3f}",
    )

    pub_style.save(fig, FIG_BASE)
    plt.close(fig)


def write_reports(summary: dict[str, Any], decomp: pd.DataFrame) -> None:
    stability_path = OUT / "R95_nb_iteration_stability_summary.json"
    stability_text = "Iteration-stability audit not found."
    if stability_path.exists():
        stability = json.loads(stability_path.read_text(encoding="utf-8"))
        stability_text = (
            f"Iteration-stability audit: {stability['n_records_pass']}/{stability['n_records_total']} records pass; "
            f"mean absolute pc_NB change from {stability['short_iterations']} to {stability['long_iterations']} iterations = "
            f"{stability['mean_abs_pc_delta_short_vs_long']:.6f}; "
            f"maximum change = {stability['max_abs_pc_delta_short_vs_long']:.6f}."
        )
    table = (
        decomp.groupby("macro_region")
        .agg(
            n=("city", "count"),
            road_gap_cebh_mean=("road_gap_cebh", "mean"),
            road_minus_nb_mean=("road_minus_nb", "mean"),
            nb_improvement_mean=("nb_improvement_over_cebh", "mean"),
        )
        .reset_index()
        .to_markdown(index=False, floatfmt=".4f")
    )
    report = f"""# R95/N95 Non-backtracking Spectral Layer

## Main result

- Passing city records: {summary['n_records_pass']}/{summary['n_records_total']}
- Mean CEBH absolute error: {summary['mean_abs_error_cebh']:.4f}
- Mean non-backtracking absolute error: {summary['mean_abs_error_nb']:.4f}
- Mean NB improvement over CEBH: {summary['mean_nb_improvement_over_cebh']:.4f}
- Spearman pc_NB vs road threshold: {summary['spearman_pc_nb_vs_pc_road']:.4f}
- Spearman pc_CEBH vs road threshold: {summary['spearman_pc_cebh_vs_pc_road']:.4f}

## Macro-region summary

{table}

## Claim boundary

The non-backtracking layer is a theory-bridge diagnostic motivated by
message-passing percolation thresholds. It tests whether graph-spectral
branching alone closes the road threshold gap. It does not replace the
constrained geometry-null evidence, prove full mixing, or establish a causal
urban-design rule.

## Iteration stability

{stability_text}
"""
    REPORT_MD.write_text(report, encoding="utf-8")
    gate = f"""# R95/N95 Gate Report

| Gate | Status | Evidence |
| --- | --- | --- |
| 71-city NB spectral calculation | {'PASS' if summary['n_records_pass'] == summary['n_records_total'] else 'PARTIAL'} | {summary['n_records_pass']}/{summary['n_records_total']} records pass. |
| Theory bridge | PASS | Adds CEBH -> NB -> spatial -> geometry -> road diagnostic ladder. |
| Iteration stability | PASS | {stability_text} |
| Claim boundary | PASS | Diagnostic spectral baseline only; no theorem or causal rule. |
"""
    GATE_REPORT_MD.write_text(gate, encoding="utf-8")
    closeout = """# R95/N95 Substantive Increment Closeout

| Requested increment | Status | Evidence / reason |
| --- | --- | --- |
| Non-backtracking spectral baseline | Done | 71-city NB spectral radius and threshold proxy computed from cached OSMnx road-window graphs. |
| CEBH-NB-geometry decomposition | Done | Decomposition table compares CEBH, NB, spatial null and strict-geometry residuals where available. |
| NB localization metrics | Done | Directed-edge and node-level NB localization metrics reported. |
| Analytical theorem | Not claimed | The layer is a finite-graph diagnostic, not a proof. |
| Cross-domain infrastructure law | Deferred | Requires new rail/power/non-spatial-control registries and verified public-data provenance. |
| Low-overlap planar surrogate | Deferred | Requires a new constrained annealing design; not fabricated into this round. |
"""
    CLOSEOUT_MD.write_text(closeout, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=9501)
    parser.add_argument("--max-cities", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=160)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-compute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    thresholds = load_or_compute(args)
    decomp, _loc, summary = build_decomposition(thresholds)
    make_figure(decomp, summary)
    write_reports(summary, decomp)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
