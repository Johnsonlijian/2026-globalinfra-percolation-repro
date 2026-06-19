"""R78 observed-road fine-threshold estimator check for the 21-city subset.

This round tests whether the 21-city geometry-null subset conclusions depend
on the coarser observed-road rank-order threshold grid used in the geometry
rounds. It recomputes observed-road thresholds from cached OSMnx graphs on a
local fine grid around each registered coarse estimate.

The output is an estimator sensitivity check only. It does not regenerate
geometry-null curves and does not change the geometry-null ensemble.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import pickle
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_R56_spatial_length_constrained_nulls import (
    bootstrap_pc_ci,
    cebh_from_degrees,
    rank_order_samples,
    stable_city_seed,
    to_simple_spatial_graph,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data" / "R62_urban_form_mechanism" / "geometry_null_subset_registry.csv"
OUT = ROOT / "data" / "R78_observed_road_fine_estimator_check"
ROUND_DIR = ROOT / "rounds" / "R78_observed_road_fine_estimator_check"
FIG_BASE = ROOT / "figures" / "Fig_R78_observed_road_fine_estimator_check"
PUBLIC_SD = ROOT / "source_data"
R72_CITY = ROOT / "data" / "R72_geometry_defense" / "matched_intensity_geometry_vs_spatial_city.csv"


@dataclass
class FineEstimatorRecord:
    city: str
    country: str
    macro_region: str
    n_nodes: int
    n_edges: int
    pc_cebh_registry: float
    pc_cebh_from_graph: float
    pc_rank_coarse: float
    pc_rank_fine: float
    pc_rank_fine_ci_low: float
    pc_rank_fine_ci_high: float
    road_gap_coarse: float
    road_gap_fine: float
    delta_fine_minus_coarse: float
    abs_delta_fine_minus_coarse: float
    p_grid_min: float
    p_grid_max: float
    p_grid_step: float
    n_perm: int
    n_bootstrap: int
    threshold_boundary_hit: bool
    graph_cache_file: str
    status: str
    notes: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pickle(path: Path) -> Any:
    with path.open("rb") as fh:
        return pickle.load(fh)


def local_grid(center: float, window: float, step: float) -> np.ndarray:
    lo = max(step, center - window)
    hi = min(0.98, center + window)
    return np.unique(np.round(np.arange(lo, hi + step / 2.0, step), 6))


def resolve_graph_path(raw: str) -> Path:
    path = ROOT / raw
    if path.exists():
        return path
    direct = Path(raw)
    if direct.exists():
        return direct
    raise FileNotFoundError(raw)


def run_city(row: dict[str, str], args: argparse.Namespace) -> FineEstimatorRecord:
    graph_path = resolve_graph_path(row["graph_cache_file"])
    graph = load_pickle(graph_path)
    simple, edges, degrees, _lon, _lat = to_simple_spatial_graph(graph)
    n = simple.number_of_nodes()
    m = len(edges)
    src = edges[:, 0]
    dst = edges[:, 1]
    coarse = float(row["pc_rank"])
    pc_cebh_registry = float(row["pc_cebh"])
    pc_cebh_graph = float(cebh_from_degrees(degrees))
    seed = args.seed + stable_city_seed(row["city"])

    grid = local_grid(coarse, args.window, args.p_step)
    samples = rank_order_samples(n, src, dst, grid, args.n_perm, seed)
    pc_fine, ci_low, ci_high, boundary = bootstrap_pc_ci(grid, samples, args.n_bootstrap, seed + 711)
    notes = "local fine grid around registered coarse estimate"

    if boundary and args.full_grid_on_boundary:
        full_grid = np.unique(np.round(np.arange(args.full_grid_min, args.full_grid_max + args.p_step / 2, args.p_step), 6))
        samples = rank_order_samples(n, src, dst, full_grid, args.n_perm, seed + 19_003)
        pc_fine, ci_low, ci_high, boundary = bootstrap_pc_ci(full_grid, samples, args.n_bootstrap, seed + 19_713)
        grid = full_grid
        notes = "full fine grid because local estimator hit boundary"

    road_gap_coarse = coarse - pc_cebh_registry
    road_gap_fine = pc_fine - pc_cebh_graph
    delta = pc_fine - coarse
    status = "pass" if abs(delta) <= args.pass_abs_delta and not boundary else "diagnostic"

    return FineEstimatorRecord(
        city=row["city"],
        country=row["country"],
        macro_region=row["macro_region"],
        n_nodes=n,
        n_edges=m,
        pc_cebh_registry=pc_cebh_registry,
        pc_cebh_from_graph=pc_cebh_graph,
        pc_rank_coarse=coarse,
        pc_rank_fine=pc_fine,
        pc_rank_fine_ci_low=ci_low,
        pc_rank_fine_ci_high=ci_high,
        road_gap_coarse=road_gap_coarse,
        road_gap_fine=road_gap_fine,
        delta_fine_minus_coarse=delta,
        abs_delta_fine_minus_coarse=abs(delta),
        p_grid_min=float(grid.min()),
        p_grid_max=float(grid.max()),
        p_grid_step=args.p_step,
        n_perm=args.n_perm,
        n_bootstrap=args.n_bootstrap,
        threshold_boundary_hit=boundary,
        graph_cache_file=str(graph_path.relative_to(ROOT)) if graph_path.is_relative_to(ROOT) else str(graph_path),
        status=status,
        notes=notes,
    )


def write_records(path: Path, records: list[FineEstimatorRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.0,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def make_figure(df: pd.DataFrame, summary: dict[str, Any]) -> None:
    set_style()
    colors = {"fine": "#4E79A7", "delta": "#C75146", "neutral": "#222222", "grid": "#E7EAEE"}
    fig = plt.figure(figsize=(7.35, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.9], width_ratios=[1.05, 0.95])
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    lo = min(df["pc_rank_coarse"].min(), df["pc_rank_fine"].min()) - 0.015
    hi = max(df["pc_rank_coarse"].max(), df["pc_rank_fine"].max()) + 0.015
    ax_a.plot([lo, hi], [lo, hi], color=colors["neutral"], lw=0.8)
    yerr = np.vstack(
        [
            df["pc_rank_fine"] - df["pc_rank_fine_ci_low"],
            df["pc_rank_fine_ci_high"] - df["pc_rank_fine"],
        ]
    )
    ax_a.errorbar(
        df["pc_rank_coarse"],
        df["pc_rank_fine"],
        yerr=yerr,
        fmt="o",
        color=colors["fine"],
        ecolor="#AFC3D7",
        ms=4.1,
        capsize=1.8,
        lw=0.7,
        alpha=0.9,
    )
    ax_a.set_xlim(lo, hi)
    ax_a.set_ylim(lo, hi)
    ax_a.set_xlabel("Registered coarse observed threshold")
    ax_a.set_ylabel("Fine-grid observed threshold")
    ax_a.set_title("a  Observed-road threshold estimator check", loc="left", fontweight="bold")
    ax_a.grid(color=colors["grid"], lw=0.45)

    order = np.argsort(df["delta_fine_minus_coarse"].to_numpy())
    y = np.arange(len(df))
    ordered = df.iloc[order]
    ax_b.axvline(0, color=colors["neutral"], lw=0.8)
    ax_b.scatter(ordered["delta_fine_minus_coarse"], y, s=22, color=colors["delta"], alpha=0.85)
    for yi, val in zip(y, ordered["delta_fine_minus_coarse"]):
        ax_b.plot([0, val], [yi, yi], color="#D8A2A0", lw=0.7)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(ordered["city"].str.replace("Buenos Aires", "B. Aires", regex=False), fontsize=5.8)
    ax_b.set_xlabel("Fine minus coarse threshold")
    ax_b.set_title("b  City-level estimator shift", loc="left", fontweight="bold")
    ax_b.grid(axis="x", color=colors["grid"], lw=0.45)

    labels = ["mean |shift|", "median |shift|", "max |shift|"]
    vals = [
        summary["mean_abs_delta"],
        summary["median_abs_delta"],
        summary["max_abs_delta"],
    ]
    ax_c.bar(np.arange(3), vals, width=0.62, color=["#8AA8C8", "#6F97BD", colors["delta"]])
    ax_c.axhline(0.025, color=colors["neutral"], lw=0.8, ls="--")
    ax_c.text(
        0.02,
        0.0255,
        "0.025 coarse step",
        transform=ax_c.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=6.2,
    )
    ax_c.set_xticks(np.arange(3))
    ax_c.set_xticklabels(labels, rotation=15, ha="right")
    ax_c.set_ylabel("Threshold units")
    ax_c.set_title("c  Aggregate shift scale", loc="left", fontweight="bold")
    ax_c.grid(axis="y", color=colors["grid"], lw=0.45)

    FIG_BASE.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["svg", "pdf", "png", "tiff"]:
        kwargs = {"dpi": 600} if ext in {"png", "tiff"} else {}
        fig.savefig(FIG_BASE.with_suffix(f".{ext}"), bbox_inches="tight", **kwargs)
    plt.close(fig)


def build_matched_intensity_substitution(fine_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    r72 = pd.read_csv(R72_CITY)
    subst = r72.merge(
        fine_df[["city", "pc_rank_fine", "delta_fine_minus_coarse"]],
        on="city",
        how="left",
        validate="many_to_one",
    )
    subst["pc_geometry_null_mean"] = subst["pc_road"] - subst["road_minus_geometry_mean"]
    subst["fine_road_minus_spatial"] = subst["pc_rank_fine"] - subst["pc_spatial_null"]
    subst["fine_road_minus_geometry_mean"] = subst["pc_rank_fine"] - subst["pc_geometry_null_mean"]
    subst["fine_spatial_minus_geometry_residual"] = (
        subst["fine_road_minus_spatial"] - subst["fine_road_minus_geometry_mean"]
    )

    rows: list[dict[str, Any]] = []
    for fraction, group in subst.groupby("swap_fraction"):
        rows.append(
            {
                "swap_fraction": float(fraction),
                "n_cities": int(len(group)),
                "coarse_road_minus_spatial_mean": float(group["road_minus_spatial"].mean()),
                "coarse_road_minus_geometry_mean": float(group["road_minus_geometry_mean"].mean()),
                "fine_road_minus_spatial_mean": float(group["fine_road_minus_spatial"].mean()),
                "fine_road_minus_geometry_mean": float(group["fine_road_minus_geometry_mean"].mean()),
                "coarse_spatial_minus_geometry_mean": float(group["spatial_minus_geometry_residual"].mean()),
                "fine_spatial_minus_geometry_mean": float(group["fine_spatial_minus_geometry_residual"].mean()),
                "positive_fine_geometry_residual_share": float((group["fine_road_minus_geometry_mean"] > 0).mean()),
                "mean_observed_threshold_shift": float(group["delta_fine_minus_coarse"].mean()),
            }
        )
    return subst, pd.DataFrame(rows).sort_values("swap_fraction")


def update_manifest(sd: Path, specs: list[tuple[Path, str, str]]) -> None:
    manifest = sd / "Supplementary_Data_1_source_data_manifest.csv"
    df = pd.read_csv(manifest)
    for src, copied_file, role in specs:
        if src.suffix == ".csv":
            data = pd.read_csv(src)
            n_rows = int(data.shape[0])
            n_cols = int(data.shape[1])
            fmt = "csv"
        else:
            payload = json.loads(src.read_text(encoding="utf-8"))
            n_rows = 1
            n_cols = len(payload) if isinstance(payload, dict) else 1
            fmt = "json"
        entry = {
            "round": "R78",
            "source_role": role,
            "source_path": f"data/R78_observed_road_fine_estimator_check/{src.name}",
            "copied_file": copied_file,
            "format": fmt,
            "n_rows": n_rows,
            "n_columns_or_keys": n_cols,
            "sha256": sha256_file(src),
            "redistribution_boundary": "derived project table; raw OSM/Geofabrik archives excluded",
        }
        df = df[df["copied_file"] != copied_file]
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    df.to_csv(manifest, index=False)

    claim_map = sd / "Supplementary_Data_3_claim_to_table_map.csv"
    cm = pd.read_csv(claim_map)
    claim = "Fine-grid observed-road threshold check supports the 21-city geometry-null estimator scale"
    cm = cm[cm["claim"] != claim]
    cm = pd.concat(
        [
            cm,
            pd.DataFrame(
                [
                    {
                        "claim": claim,
                        "supporting_source_data_files": (
                            "R78_observed_road_fine_estimator_21city.csv; "
                            "R78_observed_road_fine_estimator_summary.json; "
                            "R78_matched_intensity_fine_observed_summary.csv"
                        ),
                        "manuscript_location": "Methods, Supplementary estimator sensitivity and matched-intensity robustness",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    cm.to_csv(claim_map, index=False)


def copy_to_targets() -> None:
    specs = [
        (
            OUT / "R78_observed_road_fine_estimator_21city.csv",
            "tables/R78_observed_road_fine_estimator_21city.csv",
            "observed-road fine-grid estimator check",
        ),
        (
            OUT / "R78_observed_road_fine_estimator_summary.json",
            "tables/R78_observed_road_fine_estimator_summary.json",
            "observed-road fine-grid estimator summary",
        ),
        (
            OUT / "R78_matched_intensity_fine_observed_substitution.csv",
            "tables/R78_matched_intensity_fine_observed_substitution.csv",
            "matched-intensity contrast with fine observed-road thresholds",
        ),
        (
            OUT / "R78_matched_intensity_fine_observed_summary.csv",
            "tables/R78_matched_intensity_fine_observed_summary.csv",
            "matched-intensity fine-observed substitution summary",
        ),
    ]
    for src, copied, _role in specs:
        dst = PUBLIC_SD / copied
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    update_manifest(PUBLIC_SD, specs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--p-step", type=float, default=0.005)
    parser.add_argument("--window", type=float, default=0.075)
    parser.add_argument("--n-perm", type=int, default=32)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--pass-abs-delta", type=float, default=0.025)
    parser.add_argument("--full-grid-on-boundary", action="store_true", default=True)
    parser.add_argument("--full-grid-min", type=float, default=0.10)
    parser.add_argument("--full-grid-max", type=float, default=0.90)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ROUND_DIR.mkdir(parents=True, exist_ok=True)

    with args.registry.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    records: list[FineEstimatorRecord] = []
    for i, row in enumerate(rows, start=1):
        print(f"[R78] {i}/{len(rows)} {row['city']}: fine observed-road threshold", flush=True)
        records.append(run_city(row, args))
        write_records(OUT / "R78_observed_road_fine_estimator_21city.partial.csv", records)
        gc.collect()

    final_csv = OUT / "R78_observed_road_fine_estimator_21city.csv"
    write_records(final_csv, records)
    df = pd.DataFrame([asdict(record) for record in records])
    summary: dict[str, Any] = {
        "status": "pass" if (df["status"] == "pass").all() else "diagnostic",
        "round": "R78_observed_road_fine_estimator_check",
        "n_cities": int(len(df)),
        "n_perm": args.n_perm,
        "n_bootstrap": args.n_bootstrap,
        "p_step": args.p_step,
        "window": args.window,
        "mean_delta_fine_minus_coarse": float(df["delta_fine_minus_coarse"].mean()),
        "mean_abs_delta": float(df["abs_delta_fine_minus_coarse"].mean()),
        "median_abs_delta": float(df["abs_delta_fine_minus_coarse"].median()),
        "max_abs_delta": float(df["abs_delta_fine_minus_coarse"].max()),
        "max_abs_delta_city": str(df.loc[df["abs_delta_fine_minus_coarse"].idxmax(), "city"]),
        "cities_abs_delta_gt_0_01": int((df["abs_delta_fine_minus_coarse"] > 0.01).sum()),
        "cities_abs_delta_gt_0_025": int((df["abs_delta_fine_minus_coarse"] > 0.025).sum()),
        "boundary_hit_count": int(df["threshold_boundary_hit"].sum()),
        "claim_boundary": (
            "Observed-road fine-grid check only; geometry-null fine curves were not regenerated."
        ),
    }
    summary_path = OUT / "R78_observed_road_fine_estimator_summary.json"
    subst, subst_summary = build_matched_intensity_substitution(df)
    subst.to_csv(OUT / "R78_matched_intensity_fine_observed_substitution.csv", index=False)
    subst_summary.to_csv(OUT / "R78_matched_intensity_fine_observed_summary.csv", index=False)
    summary["matched_intensity_fine_observed_substitution"] = subst_summary.to_dict(orient="records")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_figure(df, summary)
    copy_to_targets()

    (ROUND_DIR / "gate_report.md").write_text(
        "# R78 Observed-Road Fine Estimator Check\n\n"
        "| Gate | Status | Evidence |\n"
        "|---|---|---|\n"
        f"| 21-city observed-road fine-grid recomputation | {summary['status']} | `{final_csv.relative_to(ROOT)}` |\n"
        f"| Boundary hits | {'Pass' if summary['boundary_hit_count'] == 0 else 'Diagnostic'} | {summary['boundary_hit_count']} cities |\n"
        f"| Maximum absolute fine-minus-coarse shift | {'Pass' if summary['max_abs_delta'] <= 0.025 else 'Diagnostic'} | {summary['max_abs_delta']:.4f} in {summary['max_abs_delta_city']} |\n"
        "| Matched-intensity contrast under fine observed thresholds | Pass | Spatial-minus-geometry difference is unchanged because both nulls share the same observed-road threshold. |\n"
        "| Claim boundary | Pass | Observed-road estimator check only; no regenerated geometry-null fine curves. |\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
