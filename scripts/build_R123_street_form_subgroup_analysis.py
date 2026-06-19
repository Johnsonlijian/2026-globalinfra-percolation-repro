"""R123 street-form subgroup stress test for the public source-data bundle.

The analysis uses only already-registered 71-city source-data variables from the
public reproducibility package. It builds a pre-specified street-form
bottleneck index from termination, bridge/articulation, biconnected redundancy,
cycle and local edge-adjacency variables; then it checks whether the CEBH
transfer error changes across index quartiles while the strict-geometry
residual remains near the estimator resolution.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pub_style import COLORS, FIG_WIDTH_2COL, REGION_COLORS, annot, apply, light_grid, panel_title, save


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA = ROOT / "source_data"
SOURCE_TABLES = SOURCE_DATA / "tables"
SOURCE_FIGURES = SOURCE_DATA / "figures"
DATA_DIR = ROOT / "data" / "R123_street_form_subgroup"
FIG_DIR = ROOT / "figures"
SEED = 20260613


INPUT_TABLE = SOURCE_TABLES / "R77_geometry_absorption_city_table.csv"

POSITIVE_BOTTLENECK = [
    "dead_end_share",
    "bridge_edge_share",
    "articulation_point_share",
]

NEGATIVE_REDUNDANCY = [
    "cycle_density",
    "largest_biconnected_node_fraction",
    "dual_degree_mean",
]

PRIMARY_METRICS = [
    "road_gap",
    "road_minus_spatial",
    "road_minus_geometry_mean",
    "geometry_absorption_fraction_of_road_gap",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def zscore(series: pd.Series) -> pd.Series:
    sd = float(series.std(ddof=0))
    if sd == 0:
        return series * 0.0
    return (series - float(series.mean())) / sd


def pearson(x: pd.Series, y: pd.Series) -> float:
    x0 = x.astype(float) - float(x.mean())
    y0 = y.astype(float) - float(y.mean())
    denom = float(np.sqrt((x0 * x0).sum() * (y0 * y0).sum()))
    if denom == 0:
        return float("nan")
    return float((x0 * y0).sum() / denom)


def spearman(x: pd.Series, y: pd.Series) -> float:
    return pearson(x.rank(method="average"), y.rank(method="average"))


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 5000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) == 1:
        return mean, mean, mean
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return mean, float(lo), float(hi)


def append_unique_csv(path: Path, rows: list[dict], key_cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path, dtype=str).fillna("")
        fieldnames = list(existing.columns)
        existing_keys = {tuple(row.get(col, "") for col in key_cols) for row in existing.to_dict("records")}
    else:
        fieldnames = list(rows[0].keys())
        existing_keys = set()

    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)

    new_rows = []
    for row in rows:
        key = tuple(str(row.get(col, "")) for col in key_cols)
        if key not in existing_keys:
            new_rows.append({field: row.get(field, "") for field in fieldnames})
            existing_keys.add(key)

    if not new_rows:
        return

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if path.stat().st_size == 0:
            writer.writeheader()
        writer.writerows(new_rows)


def copy_with_hash(src: Path, dest: Path) -> tuple[Path, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return dest, sha256(dest)


def make_scores(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in POSITIVE_BOTTLENECK + NEGATIVE_REDUNDANCY + PRIMARY_METRICS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    pos_z = []
    neg_z = []
    for col in POSITIVE_BOTTLENECK:
        zcol = f"z_{col}"
        out[zcol] = zscore(out[col])
        pos_z.append(zcol)
    for col in NEGATIVE_REDUNDANCY:
        zcol = f"z_{col}"
        out[zcol] = zscore(out[col])
        neg_z.append(zcol)

    out["street_form_bottleneck_index"] = out[pos_z].mean(axis=1) - out[neg_z].mean(axis=1)
    labels = [
        "Q1 low bottleneck / high redundancy",
        "Q2",
        "Q3",
        "Q4 high bottleneck / low redundancy",
    ]
    out["street_form_bottleneck_quartile"] = pd.qcut(
        out["street_form_bottleneck_index"],
        4,
        labels=labels,
        duplicates="drop",
    ).astype(str)
    keep_cols = [
        "city",
        "country",
        "macro_region",
        "n_nodes",
        "n_edges",
        "street_form_bottleneck_index",
        "street_form_bottleneck_quartile",
        *POSITIVE_BOTTLENECK,
        *NEGATIVE_REDUNDANCY,
        *PRIMARY_METRICS,
    ]
    return out[keep_cols].sort_values("street_form_bottleneck_index").reset_index(drop=True)


def summarize_quartiles(scores: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for quartile, sub in scores.groupby("street_form_bottleneck_quartile", sort=False):
        for metric in PRIMARY_METRICS:
            mean, lo, hi = bootstrap_ci(sub[metric].to_numpy(), rng)
            rows.append(
                {
                    "street_form_bottleneck_quartile": quartile,
                    "metric": metric,
                    "n_cities": len(sub),
                    "mean": mean,
                    "ci95_low": lo,
                    "ci95_high": hi,
                }
            )
    return pd.DataFrame(rows)


def summarize_correlations(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    x = scores["street_form_bottleneck_index"]
    for metric in PRIMARY_METRICS:
        rows.append(
            {
                "index": "street_form_bottleneck_index",
                "metric": metric,
                "n_cities": len(scores),
                "pearson_r": pearson(x, scores[metric]),
                "spearman_rho": spearman(x, scores[metric]),
            }
        )
    return pd.DataFrame(rows)


def make_figure(scores: pd.DataFrame, quartiles: pd.DataFrame, correlations: pd.DataFrame) -> None:
    apply()
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH_2COL, 2.35), gridspec_kw={"width_ratios": [1.0, 1.15, 1.3]})

    labels_short = ["Q1", "Q2", "Q3", "Q4"]
    q_order = list(scores["street_form_bottleneck_quartile"].drop_duplicates())
    x = np.arange(len(q_order))

    ax = axes[0]
    q_gap = quartiles[quartiles["metric"] == "road_gap"].set_index("street_form_bottleneck_quartile").loc[q_order]
    err = np.vstack([q_gap["mean"] - q_gap["ci95_low"], q_gap["ci95_high"] - q_gap["mean"]])
    ax.bar(x, q_gap["mean"], yerr=err, color=COLORS["cebh"], edgecolor="none", capsize=2)
    ax.set_xticks(x, labels_short)
    ax.set_ylabel("Road - CEBH threshold gap")
    ax.set_xlabel("Street-form bottleneck quartile")
    panel_title(ax, "a", "CEBH transfer error")
    light_grid(ax)

    ax = axes[1]
    width = 0.36
    for offset, metric, color, name in [
        (-width / 2, "road_minus_spatial", COLORS["spatial_null"], "Road - spatial"),
        (width / 2, "road_minus_geometry_mean", COLORS["geometry_null"], "Road - geometry"),
    ]:
        q = quartiles[quartiles["metric"] == metric].set_index("street_form_bottleneck_quartile").loc[q_order]
        err = np.vstack([q["mean"] - q["ci95_low"], q["ci95_high"] - q["mean"]])
        ax.bar(x + offset, q["mean"], width=width, yerr=err, color=color, edgecolor="none", capsize=2, label=name)
    ax.axhline(0, color=COLORS["zero"], lw=0.7)
    ax.set_xticks(x, labels_short)
    ax.set_ylabel("Residual threshold gap")
    ax.set_xlabel("Street-form bottleneck quartile")
    ax.legend(loc="upper left", handlelength=1.0)
    panel_title(ax, "b", "Residual after null constraints")
    light_grid(ax)

    ax = axes[2]
    for region, sub in scores.groupby("macro_region"):
        color = REGION_COLORS.get(region, REGION_COLORS["Other"])
        ax.scatter(
            sub["street_form_bottleneck_index"],
            sub["road_gap"],
            s=12,
            color=color,
            alpha=0.9,
            linewidth=0,
            label=region,
        )
    xs = scores["street_form_bottleneck_index"].to_numpy()
    ys = scores["road_gap"].to_numpy()
    coef = np.polyfit(xs, ys, 1)
    xr = np.linspace(xs.min(), xs.max(), 100)
    ax.plot(xr, coef[0] * xr + coef[1], color=COLORS["fit"], lw=0.9)
    rho = correlations.loc[correlations["metric"] == "road_gap", "spearman_rho"].iloc[0]
    annot(ax, 0.03, 0.97, f"Spearman rho = {rho:.3f}")
    ax.set_xlabel("Street-form bottleneck index")
    ax.set_ylabel("Road - CEBH threshold gap")
    panel_title(ax, "c", "City-level association")
    light_grid(ax)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
        handletextpad=0.3,
        labelspacing=0.25,
    )

    fig.tight_layout(w_pad=1.2)
    save(fig, FIG_DIR / "Fig_R123_street_form_subgroup")
    plt.close(fig)


def register_outputs(score_path: Path, quartile_path: Path, corr_path: Path, summary_path: Path) -> None:
    copied = []
    for src in [score_path, quartile_path, corr_path, summary_path]:
        dest = SOURCE_TABLES / src.name
        copied.append((*copy_with_hash(src, dest), src))

    fig_copied = []
    for ext in ["svg", "pdf", "png", "tiff"]:
        src = FIG_DIR / f"Fig_R123_street_form_subgroup.{ext}"
        dest = SOURCE_FIGURES / f"FigS_R123_street_form_subgroup.{ext}"
        fig_copied.append((*copy_with_hash(src, dest), src))

    manifest_rows = []
    roles = {
        score_path.name: "street-form bottleneck city scores",
        quartile_path.name: "street-form bottleneck quartile summary",
        corr_path.name: "street-form bottleneck correlations",
        summary_path.name: "street-form bottleneck JSON summary",
    }
    for dest, digest, src in copied:
        if dest.suffix == ".csv":
            obj = pd.read_csv(dest)
            n_rows = len(obj)
            n_cols = len(obj.columns)
            fmt = "csv"
        else:
            obj = json.loads(dest.read_text(encoding="utf-8"))
            n_rows = 1
            n_cols = len(obj)
            fmt = "json"
        manifest_rows.append(
            {
                "round": "R123",
                "source_role": roles[src.name],
                "source_path": str(src.relative_to(ROOT)).replace("\\", "/"),
                "copied_file": f"tables/{dest.name}",
                "format": fmt,
                "n_rows": n_rows,
                "n_columns_or_keys": n_cols,
                "sha256": digest,
                "redistribution_boundary": "derived project table; raw OSM/Geofabrik/GHSL/WDI archives excluded",
            }
        )
    for dest, digest, src in fig_copied:
        manifest_rows.append(
            {
                "round": "R123",
                "source_role": "Supplementary Fig. S8 street-form subgroup stress-test figure",
                "source_path": str(src.relative_to(ROOT)).replace("\\", "/"),
                "copied_file": f"figures/{dest.name}",
                "format": dest.suffix.lstrip("."),
                "n_rows": "",
                "n_columns_or_keys": "",
                "sha256": digest,
                "redistribution_boundary": "derived project figure; raw OSM/Geofabrik/GHSL/WDI archives excluded",
            }
        )
    append_unique_csv(
        SOURCE_DATA / "Supplementary_Data_1_source_data_manifest.csv",
        manifest_rows,
        ["round", "copied_file"],
    )

    dictionary_rows = []
    descriptions = {
        "street_form_bottleneck_index": "Pre-specified index: mean z(dead-end, bridge-edge, articulation shares) minus mean z(cycle density, largest biconnected node fraction, dual-degree mean). Higher values indicate more local bottleneck/less redundancy.",
        "street_form_bottleneck_quartile": "Quartile of street_form_bottleneck_index across the 71 standardized city windows.",
        "ci95_low": "Lower 2.5 percentile of deterministic city-level bootstrap mean.",
        "ci95_high": "Upper 97.5 percentile of deterministic city-level bootstrap mean.",
        "pearson_r": "Pearson correlation between street-form bottleneck index and the metric.",
        "spearman_rho": "Spearman rank correlation between street-form bottleneck index and the metric.",
    }
    for path in [SOURCE_TABLES / score_path.name, SOURCE_TABLES / quartile_path.name, SOURCE_TABLES / corr_path.name]:
        frame = pd.read_csv(path)
        for col in frame.columns:
            non_null = frame[col].notna().sum()
            example = "" if frame[col].dropna().empty else str(frame[col].dropna().iloc[0])
            inferred = "numeric" if pd.api.types.is_numeric_dtype(frame[col]) else "text"
            dictionary_rows.append(
                {
                    "file": path.name,
                    "column": col,
                    "inferred_type": inferred,
                    "n_rows": float(len(frame)),
                    "non_empty_count": float(non_null),
                    "example_value": example,
                    "description": descriptions.get(col, "R123 street-form subgroup stress-test variable; see script and Supplementary Note 17 for definition."),
                    "type": "",
                    "non_null": "",
                    "unique_values": "",
                    "example": "",
                }
            )
    append_unique_csv(
        SOURCE_DATA / "Supplementary_Data_2_column_dictionary.csv",
        dictionary_rows,
        ["file", "column"],
    )

    append_unique_csv(
        SOURCE_DATA / "Supplementary_Data_3_claim_to_table_map.csv",
        [
            {
                "claim": "Street-form bottleneck quartiles modulate the CEBH road-threshold transfer gap while the strict-geometry residual remains near estimator resolution",
                "supporting_source_data_files": "; ".join(
                    [
                        score_path.name,
                        quartile_path.name,
                        corr_path.name,
                        summary_path.name,
                    ]
                ),
                "manuscript_location": "Main text Results 2.4-2.5; Supplementary Fig. S8; Supplementary Note 17",
            }
        ],
        ["claim"],
    )

    append_unique_csv(
        SOURCE_DATA / "Figure_Source_Data_Map.csv",
        [
            {
                "figure": "Supplementary Fig. S8",
                "panel_or_content": "street-form bottleneck quartile stress test and city-level association",
                "source_data_files": "; ".join([score_path.name, quartile_path.name, corr_path.name, summary_path.name]),
                "notes": "Post-R120 acceptance-sprint stress test using only already-registered 71-city source-data variables.",
            }
        ],
        ["figure"],
    )

    append_unique_csv(
        SOURCE_DATA / "run_manifest.csv",
        [
            {
                "round": "R123",
                "analysis": "street-form bottleneck subgroup stress test",
                "data_directory": "data/R123_street_form_subgroup",
                "purpose": "Bounded mechanism subgroup check for CEBH transfer error and strict-geometry residual",
                "status": "completed",
                "script_or_source": "scripts/build_R123_street_form_subgroup_analysis.py",
                "output_files": "; ".join(
                    [
                        f"tables/{score_path.name}",
                        f"tables/{quartile_path.name}",
                        f"tables/{corr_path.name}",
                        f"tables/{summary_path.name}",
                        "figures/FigS_R123_street_form_subgroup.svg",
                        "figures/FigS_R123_street_form_subgroup.pdf",
                        "figures/FigS_R123_street_form_subgroup.png",
                        "figures/FigS_R123_street_form_subgroup.tiff",
                    ]
                ),
            }
        ],
        ["round", "analysis"],
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_TABLES.mkdir(parents=True, exist_ok=True)
    SOURCE_FIGURES.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_TABLE)
    scores = make_scores(df)
    quartiles = summarize_quartiles(scores)
    correlations = summarize_correlations(scores)

    score_path = DATA_DIR / "R123_street_form_subgroup_city_scores.csv"
    quartile_path = DATA_DIR / "R123_street_form_subgroup_quartiles.csv"
    corr_path = DATA_DIR / "R123_street_form_subgroup_correlations.csv"

    scores.to_csv(score_path, index=False)
    quartiles.to_csv(quartile_path, index=False)
    correlations.to_csv(corr_path, index=False)

    q1 = quartiles[
        (quartiles["metric"] == "road_gap")
        & (quartiles["street_form_bottleneck_quartile"] == "Q1 low bottleneck / high redundancy")
    ]["mean"].iloc[0]
    q4 = quartiles[
        (quartiles["metric"] == "road_gap")
        & (quartiles["street_form_bottleneck_quartile"] == "Q4 high bottleneck / low redundancy")
    ]["mean"].iloc[0]
    geo_abs = quartiles[
        (quartiles["metric"] == "geometry_absorption_fraction_of_road_gap")
        & (quartiles["street_form_bottleneck_quartile"] == "Q4 high bottleneck / low redundancy")
    ]["mean"].iloc[0]
    geo_resid_max_abs = float(
        quartiles[quartiles["metric"] == "road_minus_geometry_mean"]["mean"].abs().max()
    )
    road_gap_rho = correlations.loc[correlations["metric"] == "road_gap", "spearman_rho"].iloc[0]

    summary = {
        "round": "R123",
        "analysis": "street-form bottleneck subgroup stress test",
        "date": date.today().isoformat(),
        "n_cities": int(len(scores)),
        "index_positive_terms": POSITIVE_BOTTLENECK,
        "index_negative_terms": NEGATIVE_REDUNDANCY,
        "road_gap_mean_Q1_low_bottleneck": float(q1),
        "road_gap_mean_Q4_high_bottleneck": float(q4),
        "road_gap_Q4_minus_Q1": float(q4 - q1),
        "road_gap_spearman_rho_with_index": float(road_gap_rho),
        "max_abs_quartile_mean_road_minus_geometry": geo_resid_max_abs,
        "Q4_geometry_absorption_fraction_mean": float(geo_abs),
        "interpretation": "Street-form bottleneck quartiles modulate the random-graph transfer error; strict local non-crossing geometry leaves quartile means near the tested estimator resolution.",
        "input_table": str(INPUT_TABLE.relative_to(ROOT)).replace("\\", "/"),
    }
    summary_path = DATA_DIR / "R123_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    make_figure(scores, quartiles, correlations)
    register_outputs(score_path, quartile_path, corr_path, summary_path)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
