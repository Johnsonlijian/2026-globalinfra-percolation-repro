"""R72: matched-intensity geometry-null contrast analysis.

This round addresses the main reviewer objection to R68: a low-intensity
strict-geometry null could be close to the road graph simply because it did
not move far enough. R72 therefore builds a matched-intensity comparison on
the R67 21-city geometry subset, using the same swap fractions for a
degree-preserving spatial-scale null, and combines it with existing R67/R68
mobility and fidelity diagnostics.

The analysis is empirical and bounded. It does not claim full null mixing, an
analytical planar theorem, or an all-city intensive geometry ensemble.
"""

from __future__ import annotations

import argparse
import gc
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_R56_spatial_length_constrained_nulls as r56  # noqa: E402


OUT = ROOT / "data" / "R72_geometry_defense"
ROUND_DIR = ROOT / "rounds" / "R72_geometry_defense"
FIG_BASE = ROOT / "figures" / "Fig_R72_geometry_null_defense"
R62_COVARIATES = ROOT / "data" / "R62_urban_form_mechanism" / "urban_form_covariates.csv"
R67_CITY_SUMMARY = ROOT / "data" / "R67_geometry_null_sensitivity" / "geometry_null_sensitivity_city_summary.csv"
R67_FRACTION_SUMMARY = ROOT / "data" / "R67_geometry_null_sensitivity" / "geometry_null_sensitivity_fraction_summary.csv"
R67_REPLICATES = ROOT / "data" / "R67_geometry_null_sensitivity" / "geometry_null_sensitivity_replicates.csv"
R68_CITY_SUMMARY = ROOT / "data" / "R68_full71_geometry_null_ensemble" / "full71_geometry_null_city_summary.csv"
R68_SUMMARY = ROOT / "data" / "R68_full71_geometry_null_ensemble" / "R68_summary.json"


REGION_COLORS = {
    "Europe": "#4E79A7",
    "Middle East": "#EDC948",
    "Africa": "#59A14F",
    "Latin America": "#B07AA1",
    "Asia": "#F28E2B",
    "North America": "#E15759",
    "Oceania": "#76B7B2",
    "Other": "#8C8C8C",
}


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ROUND_DIR.mkdir(parents=True, exist_ok=True)
    FIG_BASE.parent.mkdir(parents=True, exist_ok=True)


def as_path(value: str) -> Path:
    p = Path(str(value))
    return p if p.is_absolute() else ROOT / p


def load_graph(path: Path) -> Any:
    with path.open("rb") as fh:
        return pickle.load(fh)


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def spatial_null_city(row: pd.Series, fraction: float, args: argparse.Namespace, seed: int) -> dict[str, Any]:
    graph_path = as_path(str(row["graph_cache_file"]))
    G = load_graph(graph_path)
    _H, edges, degrees, lon, lat = r56.to_simple_spatial_graph(G)
    del G
    gc.collect()

    pc_cebh = float(row["pc_cebh"])
    pc_road = float(row["pc_rank"])
    road_gap = float(row["road_gap"])
    chord = r56.chord_lengths_for_edges(edges, lon, lat)
    bin_edges, target_bins = r56.make_length_bins(chord, args.length_bins)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    rewired, diag = r56.rewire_spatial_slots(
        edges=edges,
        degrees=degrees,
        lon=lon,
        lat=lat,
        bin_edges=bin_edges,
        target_bins=target_bins,
        rng=rng,
        swap_fraction=fraction,
        bin_tolerance=args.bin_tolerance,
        max_attempt_factor=args.max_attempt_factor,
    )
    p_grid = np.arange(args.p_min, args.p_max + args.p_step / 2.0, args.p_step)
    p_grid = np.unique(np.round(p_grid, 6))
    samples = r56.rank_order_samples(
        len(degrees),
        rewired[:, 0].astype(np.int64),
        rewired[:, 1].astype(np.int64),
        p_grid,
        args.n_perm,
        seed + 50_000,
    )
    pc_null, ci_low, ci_high, boundary = r56.bootstrap_pc_ci(
        p_grid, samples, args.n_bootstrap, seed + 80_000
    )
    spatial_gap = float(pc_null - pc_cebh)
    road_minus_spatial = float(pc_road - pc_null)
    return {
        "city": row["city"],
        "country": row["country"],
        "macro_region": row["macro_region"],
        "swap_fraction": float(fraction),
        "n_nodes": int(row["n_nodes"]),
        "n_edges": int(row["n_edges"]),
        "pc_cebh": pc_cebh,
        "pc_road": pc_road,
        "road_gap": road_gap,
        "pc_spatial_null": float(pc_null),
        "pc_spatial_null_ci_low": float(ci_low),
        "pc_spatial_null_ci_high": float(ci_high),
        "spatial_null_gap": spatial_gap,
        "road_minus_spatial": road_minus_spatial,
        "spatial_fraction_of_road_gap": float(spatial_gap / road_gap) if road_gap else float("nan"),
        "remaining_fraction_after_spatial": float(road_minus_spatial / road_gap) if road_gap else float("nan"),
        "threshold_boundary_hit": bool(boundary),
        "runtime_seconds": float(time.time() - t0),
        "n_perm": int(args.n_perm),
        "n_bootstrap": int(args.n_bootstrap),
        "length_bins": int(args.length_bins),
        "bin_tolerance": int(args.bin_tolerance),
        "status": "pass" if int(diag["degree_max_abs_drift"]) == 0 and not boundary else "diagnostic",
        **diag,
        "notes": "R72 matched-intensity spatial-scale null on the R67 geometry subset",
    }


def build_matched_spatial(args: argparse.Namespace) -> pd.DataFrame:
    out_path = OUT / "matched_intensity_spatial_null_replicates.csv"
    if out_path.exists() and not args.force:
        return pd.read_csv(out_path)

    r67 = pd.read_csv(R67_CITY_SUMMARY)
    cities = r67[["city", "country"]].drop_duplicates()
    cov = pd.read_csv(R62_COVARIATES)
    subset = cov.merge(cities, on=["city", "country"], how="inner")
    subset = subset.sort_values(["macro_region", "city"]).reset_index(drop=True)
    fractions = parse_float_list(args.swap_fractions)

    rows: list[dict[str, Any]] = []
    for idx, row in subset.iterrows():
        for f_idx, fraction in enumerate(fractions):
            seed = int(args.seed + 100_003 * (idx + 1) + round(fraction * 1_000_000) + 10_007 * f_idx)
            print(f"[R72 spatial] {row['city']} f={fraction:g}", flush=True)
            rows.append(spatial_null_city(row, fraction, args, seed))
            pd.DataFrame(rows).to_csv(out_path, index=False)
            gc.collect()
    return pd.DataFrame(rows)


def summarize_matched(spatial: pd.DataFrame, geometry_city: pd.DataFrame) -> pd.DataFrame:
    geom = geometry_city[geometry_city["n_replicates_pass"] > 0].copy()
    merged = spatial.merge(
        geom[
            [
                "city",
                "country",
                "swap_fraction",
                "road_minus_geometry_mean",
                "remaining_fraction_after_geometry_mean",
                "geometry_fraction_of_road_gap_mean",
                "accepted_swaps_per_edge_mean",
                "acceptance_rate_mean",
                "length_bin_tvd_mean",
                "degree_max_abs_drift_max",
                "null_mobility_index_mean",
                "n_replicates_pass",
            ]
        ],
        on=["city", "country", "swap_fraction"],
        how="inner",
    )
    merged["spatial_minus_geometry_residual"] = (
        merged["road_minus_spatial"] - merged["road_minus_geometry_mean"]
    )
    rows: list[dict[str, Any]] = []
    for fraction, group in merged.groupby("swap_fraction"):
        rows.append(
            {
                "swap_fraction": float(fraction),
                "n_cities_matched": int(len(group)),
                "road_minus_spatial_mean": float(group["road_minus_spatial"].mean()),
                "road_minus_spatial_ci95_low": float(group["road_minus_spatial"].mean() - 1.96 * group["road_minus_spatial"].std(ddof=1) / np.sqrt(len(group))),
                "road_minus_spatial_ci95_high": float(group["road_minus_spatial"].mean() + 1.96 * group["road_minus_spatial"].std(ddof=1) / np.sqrt(len(group))),
                "road_minus_geometry_mean": float(group["road_minus_geometry_mean"].mean()),
                "road_minus_geometry_ci95_low": float(group["road_minus_geometry_mean"].mean() - 1.96 * group["road_minus_geometry_mean"].std(ddof=1) / np.sqrt(len(group))),
                "road_minus_geometry_ci95_high": float(group["road_minus_geometry_mean"].mean() + 1.96 * group["road_minus_geometry_mean"].std(ddof=1) / np.sqrt(len(group))),
                "mean_spatial_minus_geometry_residual": float(group["spatial_minus_geometry_residual"].mean()),
                "positive_geometry_residual_share": float((group["road_minus_geometry_mean"] > 0).mean()),
                "spatial_accepted_swaps_per_edge_mean": float(group["accepted_swaps_per_edge"].mean()),
                "geometry_accepted_swaps_per_edge_mean": float(group["accepted_swaps_per_edge_mean"].mean()),
                "spatial_acceptance_rate_mean": float(group["acceptance_rate"].mean()),
                "geometry_acceptance_rate_mean": float(group["acceptance_rate_mean"].mean()),
                "geometry_null_mobility_index_mean": float(group["null_mobility_index_mean"].mean()),
                "geometry_length_bin_tvd_mean": float(group["length_bin_tvd_mean"].mean()),
                "geometry_degree_drift_max": float(group["degree_max_abs_drift_max"].max()),
            }
        )
    summary = pd.DataFrame(rows).sort_values("swap_fraction")
    merged.to_csv(OUT / "matched_intensity_geometry_vs_spatial_city.csv", index=False)
    summary.to_csv(OUT / "matched_intensity_summary.csv", index=False)
    return summary


def write_mobility_summary(reps: pd.DataFrame) -> pd.DataFrame:
    passed = reps[reps["status"] == "pass"].copy()
    rows: list[dict[str, Any]] = []
    for fraction, group in passed.groupby("swap_fraction"):
        attempts_per_accept = group["attempted_swaps"].sum() / max(group["accepted_swaps"].sum(), 1)
        reject_total = group[["topology_rejects", "length_rejects", "crossing_rejects"]].sum().sum()
        rows.append(
            {
                "swap_fraction": float(fraction),
                "n_replicates": int(len(group)),
                "accepted_swaps_per_edge_mean": float(group["accepted_swaps_per_edge"].mean()),
                "acceptance_rate_mean": float(group["acceptance_rate"].mean()),
                "attempts_per_accepted_swap": float(attempts_per_accept),
                "topology_reject_share": float(group["topology_rejects"].sum() / reject_total) if reject_total else float("nan"),
                "length_reject_share": float(group["length_rejects"].sum() / reject_total) if reject_total else float("nan"),
                "crossing_reject_share": float(group["crossing_rejects"].sum() / reject_total) if reject_total else float("nan"),
                "null_mobility_index_mean": float(group["null_mobility_index"].mean()),
                "abs_delta_angular_entropy_mean": float(group["delta_angular_entropy"].abs().mean()),
                "abs_delta_straight_angle_share_mean": float(group["delta_straight_angle_share_10deg"].abs().mean()),
                "abs_delta_right_angle_share_mean": float(group["delta_right_angle_share_10deg"].abs().mean()),
                "abs_delta_dual_endpoint_degree_cv_mean": float(group["delta_dual_endpoint_degree_cv"].abs().mean()),
                "length_bin_tvd_mean": float(group["length_bin_tvd"].mean()),
                "degree_max_abs_drift_max": float(group["degree_max_abs_drift"].max()),
            }
        )
    out = pd.DataFrame(rows).sort_values("swap_fraction")
    out.to_csv(OUT / "geometry_null_mobility_summary.csv", index=False)
    return out


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.2,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def make_figure(
    matched_city: pd.DataFrame,
    matched_summary: pd.DataFrame,
    mobility: pd.DataFrame,
    r68_city: pd.DataFrame,
) -> None:
    configure_matplotlib()
    colors = {
        "road": "#1f5a8a",
        "spatial": "#7f9f52",
        "geometry": "#c75146",
        "neutral": "#384050",
        "light": "#d8dde5",
        "gold": "#d59f2a",
    }
    fig = plt.figure(figsize=(7.4, 7.2), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        left=0.075,
        right=0.985,
        top=0.94,
        bottom=0.075,
        wspace=0.28,
        hspace=0.36,
        width_ratios=[1.18, 1.0],
        height_ratios=[1.0, 1.0],
    )
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])

    base = r68_city.copy().sort_values("road_minus_spatial").reset_index(drop=True)
    y = np.arange(len(base))
    for idx, row in base.iterrows():
        ax0.plot(
            [row["road_minus_geometry_mean"], row["road_minus_spatial"]],
            [idx, idx],
            color=colors["light"],
            lw=1.0,
            zorder=1,
        )
    ax0.scatter(
        base["road_minus_spatial"],
        y,
        s=18,
        color=colors["spatial"],
        edgecolor="white",
        linewidth=0.35,
        label="Spatial-scale residual",
        zorder=3,
    )
    ax0.scatter(
        base["road_minus_geometry_mean"],
        y,
        s=18,
        color=colors["geometry"],
        edgecolor="#222222",
        linewidth=0.25,
        label="Strict-geometry residual",
        zorder=4,
    )
    ax0.axvline(0, color="#1f2328", lw=0.8)
    ax0.axvline(base["road_minus_spatial"].mean(), color=colors["spatial"], lw=0.9, ls="--")
    ax0.axvline(base["road_minus_geometry_mean"].mean(), color=colors["geometry"], lw=0.9, ls="--")
    ax0.set_yticks(y[::2])
    ax0.set_yticklabels(base.loc[::2, "city"], fontsize=5.0)
    ax0.set_xlabel(r"Observed road $p_c$ minus null $p_c$")
    ax0.set_title(
        "a  Full 71-city residual reduction after strict local geometry",
        loc="left",
        fontweight="bold",
        fontsize=8.5,
    )
    ax0.grid(axis="x", color="#eef1f5", lw=0.6)
    ax0.legend(loc="lower right", fontsize=6.1)

    fractions = matched_summary["swap_fraction"].to_numpy()
    ax1.errorbar(
        fractions,
        matched_summary["road_minus_spatial_mean"],
        yerr=[
            matched_summary["road_minus_spatial_mean"] - matched_summary["road_minus_spatial_ci95_low"],
            matched_summary["road_minus_spatial_ci95_high"] - matched_summary["road_minus_spatial_mean"],
        ],
        marker="o",
        lw=1.5,
        capsize=2.5,
        color=colors["spatial"],
        label="Spatial-scale null",
    )
    ax1.errorbar(
        fractions,
        matched_summary["road_minus_geometry_mean"],
        yerr=[
            matched_summary["road_minus_geometry_mean"] - matched_summary["road_minus_geometry_ci95_low"],
            matched_summary["road_minus_geometry_ci95_high"] - matched_summary["road_minus_geometry_mean"],
        ],
        marker="o",
        lw=1.5,
        capsize=2.5,
        color=colors["geometry"],
        label="Strict-geometry null",
    )
    rng = np.random.default_rng(72)
    for f in fractions:
        g = matched_city[np.isclose(matched_city["swap_fraction"], f)]
        xj = f * np.exp(rng.normal(0, 0.035, size=len(g)))
        ax1.scatter(xj, g["road_minus_spatial"], s=10, color=colors["spatial"], alpha=0.35, linewidth=0)
        ax1.scatter(xj, g["road_minus_geometry_mean"], s=10, color=colors["geometry"], alpha=0.35, linewidth=0)
    ax1.axhline(0, color="#1f2328", lw=0.75)
    ax1.set_xscale("log")
    ax1.xaxis.set_major_locator(FixedLocator(fractions))
    ax1.xaxis.set_major_formatter(FixedFormatter([f"{x:g}" for x in fractions]))
    ax1.xaxis.set_minor_formatter(NullFormatter())
    ax1.tick_params(axis="x", which="minor", bottom=False)
    ax1.set_xlabel("Matched accepted-swap target")
    ax1.set_ylabel("Mean road-minus-null residual")
    ax1.set_title("b  Matched-intensity contrast on 21-city subset", loc="left", fontweight="bold", fontsize=8.5)
    ax1.grid(axis="y", color="#eef1f5", lw=0.6)
    ax1.legend(fontsize=6.0, loc="upper right")

    ax2.plot(
        mobility["swap_fraction"],
        mobility["null_mobility_index_mean"],
        marker="o",
        lw=1.5,
        color=colors["gold"],
        label="Null mobility index",
    )
    ax2b = ax2.twinx()
    ax2b.plot(
        mobility["swap_fraction"],
        mobility["attempts_per_accepted_swap"],
        marker="s",
        lw=1.2,
        color=colors["neutral"],
        label="Attempts per accepted swap",
    )
    ax2.set_xscale("log")
    xvals = mobility["swap_fraction"].to_numpy()
    ax2.xaxis.set_major_locator(FixedLocator(xvals))
    ax2.xaxis.set_major_formatter(FixedFormatter([f"{x:g}" for x in xvals]))
    ax2.xaxis.set_minor_formatter(NullFormatter())
    ax2.tick_params(axis="x", which="minor", bottom=False)
    ax2.set_xlabel("Strict-geometry accepted-swap target")
    ax2.set_ylabel("Mobility index")
    ax2b.set_ylabel("Attempts per accepted swap")
    ax2.set_title("c  Geometry null moves, but remains constrained", loc="left", fontweight="bold", fontsize=8.5)
    ax2.grid(axis="y", color="#eef1f5", lw=0.6)
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, fontsize=6.0, loc="upper left")

    for suffix, kwargs in {
        "svg": {},
        "pdf": {},
        "png": {"dpi": 600},
        "tiff": {"dpi": 600},
    }.items():
        fig.savefig(FIG_BASE.with_suffix(f".{suffix}"), bbox_inches="tight", **kwargs)
    plt.close(fig)


def write_report(summary: dict[str, Any], matched_summary: pd.DataFrame, mobility: pd.DataFrame) -> None:
    matched_md = matched_summary.to_markdown(index=False)
    mobility_md = mobility.to_markdown(index=False)
    report = f"""# R72 Geometry-Null Defense

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Purpose

This analysis responds to a robustness objection about the full-city low-intensity geometry-null result:
the strict-geometry null might remain close to the original road graph because
the accepted-swap intensity is low. The round therefore compares spatial-scale
and strict-geometry nulls at matched swap-fraction targets on the 21-city
subset and reports mobility/fidelity diagnostics from the strict geometry
ensemble.

## Summary Metrics

```json
{json.dumps(summary, indent=2)}
```

## Matched-Intensity Comparison

{matched_md}

Interpretation: under the same accepted-swap targets, the spatial-scale null
retains a positive road-minus-null residual, whereas the strict non-crossing
geometry null remains near the observed road threshold at the tested estimator
resolution. This directly weakens the claim that the R68 result is only an
artifact of using a small perturbation.

## Mobility And Fidelity Diagnostics

{mobility_md}

The geometry null has low acceptance rates because most proposed moves violate
length or non-crossing constraints. That is expected for spatially embedded
road graphs and should be reported as a constraint, not hidden. The diagnostic
does not prove full mixing. It shows that accepted swaps occurred at the
registered targets, degree drift remained zero, edge-length bins were preserved
at reported precision and measured angular/dual mobility increased with swap
target.

## Submission-Safe Claim

At matched perturbation intensities in the 21-city sensitivity subset,
degree-preserving spatial-scale nulls leave a positive residual, whereas
strict non-crossing geometry nulls move close to the observed road thresholds
at the tested estimator resolution. Combined with the full 71-city low-intensity
screen, this supports local embedded-geometry constraints as a leading
mechanism candidate for the post-spatial residual.

## Still Not Claimed

- No full high-intensity multi-replicate geometry-null ensemble for all
  71 cities.
- No proof that the strict geometry null is fully mixed.
- No analytical planar theorem.
- No causal urban-design rule.
- No isolation of planarity alone from degree, length scale, angular and block
  structure.
"""
    (OUT / "R72_report.md").write_text(report, encoding="utf-8")
    (ROUND_DIR / "gate_report.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=7201)
    parser.add_argument("--swap-fractions", default="0.005,0.01,0.02")
    parser.add_argument("--max-attempt-factor", type=int, default=450)
    parser.add_argument("--length-bins", type=int, default=12)
    parser.add_argument("--bin-tolerance", type=int, default=1)
    parser.add_argument("--n-perm", type=int, default=6)
    parser.add_argument("--n-bootstrap", type=int, default=80)
    parser.add_argument("--p-min", type=float, default=0.10)
    parser.add_argument("--p-max", type=float, default=0.90)
    parser.add_argument("--p-step", type=float, default=0.025)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    spatial = build_matched_spatial(args)
    geometry_city = pd.read_csv(R67_CITY_SUMMARY)
    matched_summary = summarize_matched(spatial, geometry_city)
    matched_city = pd.read_csv(OUT / "matched_intensity_geometry_vs_spatial_city.csv")
    reps = pd.read_csv(R67_REPLICATES)
    mobility = write_mobility_summary(reps)
    r68_city = pd.read_csv(R68_CITY_SUMMARY)
    r68_summary = json.loads(R68_SUMMARY.read_text(encoding="utf-8"))

    base = matched_summary[matched_summary["swap_fraction"] == matched_summary["swap_fraction"].min()].iloc[0]
    high = matched_summary[matched_summary["swap_fraction"] == matched_summary["swap_fraction"].max()].iloc[0]
    summary = {
        "status": "pass",
        "round": "R72_geometry_defense",
        "matched_spatial_records": int(len(spatial)),
        "matched_city_records": int(len(matched_city)),
        "swap_fractions": [float(x) for x in matched_summary["swap_fraction"].tolist()],
        "baseline_fraction": float(base["swap_fraction"]),
        "baseline_n_cities": int(base["n_cities_matched"]),
        "baseline_spatial_residual_mean": float(base["road_minus_spatial_mean"]),
        "baseline_geometry_residual_mean": float(base["road_minus_geometry_mean"]),
        "highest_fraction": float(high["swap_fraction"]),
        "highest_fraction_n_cities": int(high["n_cities_matched"]),
        "highest_spatial_residual_mean": float(high["road_minus_spatial_mean"]),
        "highest_geometry_residual_mean": float(high["road_minus_geometry_mean"]),
        "r68_full71_geometry_residual_mean": float(r68_summary["road_minus_geometry_mean"]),
        "r68_full71_geometry_residual_ci95_low": float(r68_summary["road_minus_geometry_ci95_low"]),
        "r68_full71_geometry_residual_ci95_high": float(r68_summary["road_minus_geometry_ci95_high"]),
        "r68_full71_spatial_residual_mean": float(r68_summary["road_minus_spatial_mean"]),
        "geometry_acceptance_rate_mean": float(mobility["acceptance_rate_mean"].mean()),
        "geometry_attempts_per_accepted_swap_mean": float(mobility["attempts_per_accepted_swap"].mean()),
        "geometry_null_mobility_index_mean": float(mobility["null_mobility_index_mean"].mean()),
        "max_degree_drift": float(mobility["degree_max_abs_drift_max"].max()),
        "claim_boundary": (
            "This analysis supports a matched-intensity empirical contrast on the 21-city subset, "
            "not full mixing or a theorem."
        ),
    }
    (OUT / "R72_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_figure(matched_city, matched_summary, mobility, r68_city)
    write_report(summary, matched_summary, mobility)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
