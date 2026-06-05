"""R75 submission hardening source tables.

This script creates submission-facing helper tables requested by the final
npj/CP package audit. It does not create new scientific measurements. It
renames and reorganizes already derived R72/R73 outputs, and it records the
boundary that exact rewired-edge overlap measures cannot be recovered from archived
summary tables alone.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "R75_submission_hardening"
CP_SD = ROOT / "submission" / "communications_physics" / "target_submission" / "source_data"
NPJ_SD = ROOT / "submission" / "npj_complexity" / "target_submission" / "source_data"


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for sd in [CP_SD, NPJ_SD]:
        (sd / "tables").mkdir(parents=True, exist_ok=True)
        (sd / "figures").mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dimensions(path: Path) -> tuple[int, int]:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        return len(df), len(df.columns)
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            return 1, len(obj)
        if isinstance(obj, list):
            return len(obj), 0
    return 0, 0


def write_feature_groups(nested: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for _, row in nested.iterrows():
        features = str(row["features_used"]).split(";")
        for i, feature in enumerate(features, start=1):
            key = (row["target"], row["model_group"], feature)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "target": row["target"],
                    "model_group": row["model_group"],
                    "model_label": row["model_label"],
                    "feature_order": i,
                    "feature_name": feature,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "R73_feature_groups.csv", index=False)
    return out


def write_geometry_proxy_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    city = pd.read_csv(CP_SD / "tables" / "R72_matched_intensity_geometry_vs_spatial_city.csv")
    mobility = pd.read_csv(CP_SD / "tables" / "R72_geometry_null_mobility_summary.csv")

    proxy = city[
        [
            "city",
            "country",
            "macro_region",
            "swap_fraction",
            "n_edges",
            "accepted_swaps",
            "attempted_swaps",
            "accepted_swaps_per_edge",
            "acceptance_rate",
            "road_minus_spatial",
            "road_minus_geometry_mean",
            "spatial_minus_geometry_residual",
            "length_bin_tvd_mean",
            "degree_max_abs_drift_max",
            "null_mobility_index_mean",
        ]
    ].copy()
    proxy["edge_turnover_proxy_fraction"] = (2.0 * proxy["accepted_swaps"] / proxy["n_edges"]).clip(upper=1.0)
    proxy["edge_jaccard_distance"] = pd.NA
    proxy["edge_jaccard_status"] = "not_computable_from_archived_summary_tables_edge_sets_not_stored"
    proxy["interpretation"] = (
        "Accepted-swap-derived turnover proxy only; it is not an exact rewired-edge overlap measure."
    )
    proxy.to_csv(OUT / "R72_geometry_edge_jaccard_by_city.csv", index=False)

    graph_summary = (
        proxy.groupby("swap_fraction", as_index=False)
        .agg(
            n_city_fraction_records=("city", "count"),
            accepted_swaps_per_edge_mean=("accepted_swaps_per_edge", "mean"),
            edge_turnover_proxy_fraction_mean=("edge_turnover_proxy_fraction", "mean"),
            acceptance_rate_mean=("acceptance_rate", "mean"),
            null_mobility_index_mean=("null_mobility_index_mean", "mean"),
            length_bin_tvd_mean=("length_bin_tvd_mean", "mean"),
            degree_max_abs_drift_max=("degree_max_abs_drift_max", "max"),
            road_minus_spatial_mean=("road_minus_spatial", "mean"),
            road_minus_geometry_mean=("road_minus_geometry_mean", "mean"),
        )
    )
    graph_summary["edge_jaccard_available"] = False
    graph_summary["edge_jaccard_boundary"] = (
        "Exact rewired-edge overlap requires archived rewired edge sets; this table reports a turnover proxy."
    )
    graph_summary.to_csv(OUT / "R72_geometry_graph_distance_summary.csv", index=False)

    rejection = mobility[
        [
            "swap_fraction",
            "n_replicates",
            "accepted_swaps_per_edge_mean",
            "acceptance_rate_mean",
            "attempts_per_accepted_swap",
            "topology_reject_share",
            "length_reject_share",
            "crossing_reject_share",
            "length_bin_tvd_mean",
            "degree_max_abs_drift_max",
        ]
    ].copy()
    rejection["rejection_categories_logged"] = True
    rejection["category_note"] = "Shares summarize proposal rejections into topology, length-bin and crossing classes."
    rejection.to_csv(OUT / "R72_geometry_rejection_reason_summary.csv", index=False)
    return proxy, graph_summary, rejection


def write_aliases_and_figure_sources() -> None:
    nested = pd.read_csv(CP_SD / "tables" / "R73_nested_model_summary.csv")
    pred = pd.read_csv(CP_SD / "tables" / "R73_nested_model_predictions.csv")
    lro = pd.read_csv(CP_SD / "tables" / "R73_leave_region_out_region_summary.csv")

    nested.to_csv(OUT / "R73_nested_model_ladder_summary.csv", index=False)
    pred.to_csv(OUT / "R73_nested_model_ladder_city_predictions.csv", index=False)
    pred[["city", "country", "macro_region", "target", "model_group", "observed", "predicted_leave_region_out"]].to_csv(
        OUT / "R73_leave_region_out_predictions.csv",
        index=False,
    )
    lro.to_csv(OUT / "R73_leave_region_out_region_summary.csv", index=False)
    write_feature_groups(nested)

    fig5 = nested[
        [
            "target",
            "model_group",
            "model_label",
            "n",
            "n_features_used",
            "cv5_r2",
            "cv5_mae",
            "leave_region_out_r2",
            "leave_region_out_mae",
            "in_sample_r2",
            "alpha",
        ]
    ].copy()
    fig5.to_csv(OUT / "Fig5_nested_validation_source_data.csv", index=False)


def copy_outputs_to_submission_packages() -> None:
    for sd in [CP_SD, NPJ_SD]:
        for path in OUT.glob("*.csv"):
            target_dir = sd / ("figures" if path.name.startswith("Fig") else "tables")
            path_target = target_dir / path.name
            path_target.write_bytes(path.read_bytes())


def upsert_manifest(sd: Path) -> None:
    manifest = sd / "Supplementary_Data_1_source_data_manifest.csv"
    rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8-sig")))
    fieldnames = list(rows[0].keys())
    existing = {row["copied_file"]: row for row in rows}
    specs = []
    for path in sorted((sd / "tables").glob("R72_geometry_*summary.csv")):
        specs.append(("R75", "geometry mobility and graph-distance proxy", f"data/R75_submission_hardening/{path.name}", f"tables/{path.name}", "csv"))
    for path in sorted((sd / "tables").glob("R72_geometry_edge_jaccard_by_city.csv")):
        specs.append(("R75", "edge-overlap availability and turnover proxy", f"data/R75_submission_hardening/{path.name}", f"tables/{path.name}", "csv"))
    for path in sorted((sd / "tables").glob("R73_*")):
        if path.name in {
            "R73_nested_model_ladder_summary.csv",
            "R73_nested_model_ladder_city_predictions.csv",
            "R73_leave_region_out_predictions.csv",
            "R73_feature_groups.csv",
            "R73_leave_region_out_region_summary.csv",
        }:
            specs.append(("R75", "nested validation submission alias", f"data/R75_submission_hardening/{path.name}", f"tables/{path.name}", "csv"))
    for path in sorted((sd / "figures").glob("Fig5_nested_validation_source_data.csv")):
        specs.append(("R75", "figure source data", f"data/R75_submission_hardening/{path.name}", f"figures/{path.name}", "csv"))

    for round_id, role, source, copied, fmt in specs:
        path = sd / copied
        n_rows, n_cols = dimensions(path)
        record = {
            "round": round_id,
            "source_role": role,
            "source_path": source,
            "copied_file": copied,
            "format": fmt,
            "n_rows": str(n_rows),
            "n_columns_or_keys": str(n_cols),
            "sha256": sha256(path),
            "redistribution_boundary": "derived project table; raw OSM/Geofabrik archives excluded",
        }
        existing[copied] = record

    ordered = [existing.get(row["copied_file"], row) for row in rows]
    known = {row["copied_file"] for row in rows}
    ordered.extend(record for copied, record in existing.items() if copied not in known)
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered)


def upsert_claim_map(sd: Path) -> None:
    path = sd / "Supplementary_Data_3_claim_to_table_map.csv"
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8-sig")))
    fieldnames = list(rows[0].keys())
    existing = {row["claim"]: row for row in rows}
    additions = [
        {
            "claim": "Nested validation feature groups and fold-level predictions are source-data backed",
            "supporting_source_data_files": "R73_feature_groups.csv; R73_nested_model_ladder_summary.csv; R73_nested_model_ladder_city_predictions.csv; R73_leave_region_out_predictions.csv; Fig5_nested_validation_source_data.csv",
            "manuscript_location": "Main text Results 2.4; Supplementary Note 7; Fig. 5",
        },
        {
            "claim": "Geometry-null mobility diagnostics support perturbation fidelity without proving full mixing",
            "supporting_source_data_files": "R72_geometry_graph_distance_summary.csv; R72_geometry_rejection_reason_summary.csv; R72_geometry_edge_jaccard_by_city.csv; R72_geometry_null_mobility_summary.csv",
            "manuscript_location": "Main text Results 2.5; Supplementary Note 10; Fig. 4",
        },
    ]
    for row in additions:
        existing[row["claim"]] = row
    ordered = [existing.get(row["claim"], row) for row in rows]
    known = {row["claim"] for row in rows}
    ordered.extend(row for claim, row in existing.items() if claim not in known)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered)


def upsert_column_dictionary(sd: Path) -> None:
    path = sd / "Supplementary_Data_2_column_dictionary.csv"
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8-sig")))
    fieldnames = list(rows[0].keys())
    existing = {(row["file"], row["column"]) for row in rows}
    descriptions = {
        "target": "Prediction target: road_gap or road_minus_spatial.",
        "model_group": "Nested model family identifier.",
        "model_label": "Human-readable nested model label.",
        "feature_order": "Feature order within the model family.",
        "feature_name": "Predictor included in the nested feature group.",
        "predicted_cv5": "Prediction from fixed five-fold cross-validation.",
        "predicted_leave_region_out": "Prediction when the city's macro-region is held out.",
        "edge_turnover_proxy_fraction": "Accepted-swap-derived proxy equal to two accepted swaps per edge count, clipped at one; not an exact edge-overlap measure.",
        "edge_jaccard_distance": "Exact rewired-edge overlap; unavailable when rewired edge sets are not archived.",
        "edge_jaccard_status": "Availability status for exact rewired-edge overlap.",
        "topology_reject_share": "Share of rejected proposals caused by topology constraints such as self-loops or duplicate edges.",
        "length_reject_share": "Share of rejected proposals caused by length-bin constraints.",
        "crossing_reject_share": "Share of rejected proposals caused by non-crossing geometry constraints.",
        "rejection_categories_logged": "Whether rejection categories were logged for this diagnostic.",
    }
    files = [
        "R73_feature_groups.csv",
        "R73_nested_model_ladder_summary.csv",
        "R73_nested_model_ladder_city_predictions.csv",
        "R73_leave_region_out_predictions.csv",
        "R73_leave_region_out_region_summary.csv",
        "R72_geometry_graph_distance_summary.csv",
        "R72_geometry_rejection_reason_summary.csv",
        "R72_geometry_edge_jaccard_by_city.csv",
        "Fig5_nested_validation_source_data.csv",
    ]
    for filename in files:
        location = sd / ("figures" if filename.startswith("Fig") else "tables") / filename
        if not location.exists():
            continue
        df = pd.read_csv(location)
        for column in df.columns:
            key = (filename, column)
            if key in existing:
                continue
            series = df[column]
            non_empty = int(series.notna().sum())
            vals = series.dropna()
            example = str(vals.iloc[0])[:80] if len(vals) else ""
            rows.append(
                {
                    "file": filename,
                    "column": column,
                    "inferred_type": "numeric" if pd.api.types.is_numeric_dtype(series) else "text",
                    "n_rows": str(len(df)),
                    "non_empty_count": str(non_empty),
                    "example_value": example,
                    "description": descriptions.get(
                        column,
                        "Project-derived submission-hardening variable; see manuscript Methods and R75 report.",
                    ),
                }
            )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def upsert_run_manifest(sd: Path) -> None:
    path = sd / "run_manifest.csv"
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8-sig")))
    fieldnames = list(rows[0].keys())
    if not any(row["round"] == "R75" for row in rows):
        rows.append(
            {
                "round": "R75",
                "analysis": "submission hardening source tables",
                "data_directory": "data/R75_submission_hardening",
                "purpose": "submission-facing aliases, feature groups, figure source data and geometry mobility proxy tables",
                "status": "completed",
            }
        )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ensure_dirs()
    write_aliases_and_figure_sources()
    write_geometry_proxy_tables()
    copy_outputs_to_submission_packages()
    for sd in [CP_SD, NPJ_SD]:
        upsert_manifest(sd)
        upsert_claim_map(sd)
        upsert_column_dictionary(sd)
        upsert_run_manifest(sd)
    summary = {
        "status": "pass",
        "round": "R75_submission_hardening",
        "outputs": sorted(path.name for path in OUT.glob("*.csv")),
        "claim_boundary": "True exact rewired-edge overlap is not computed because archived summary tables do not store rewired edge sets.",
    }
    (OUT / "R75_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
