"""Register N99 analyses in the npj Complexity target submission package."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NPJ = ROOT / "submission" / "npj_complexity" / "target_submission"
SD = NPJ / "source_data"
TABLES = SD / "tables"
FIGS = SD / "figures"
DISPLAY_FIGS = NPJ / "figures"

N99_PRED = ROOT / "data" / "N99_prediction_correction_law"
N99_LOW = ROOT / "data" / "N99_low_overlap_geometry_surrogate"
R100_FIG = ROOT / "data" / "R100_main_figure_hardening"

SOURCES = [
    (
        N99_PRED / "N99_prediction_city_predictions.csv",
        TABLES / "N99_prediction_city_predictions.csv",
        "N99",
        "71-city compact transfer-error correction predictions",
    ),
    (
        N99_PRED / "N99_prediction_model_comparison.csv",
        TABLES / "N99_prediction_model_comparison.csv",
        "N99",
        "prediction-correction model comparison",
    ),
    (
        N99_PRED / "N99_prediction_feature_group_comparison.csv",
        TABLES / "N99_prediction_feature_group_comparison.csv",
        "N99",
        "candidate feature-family comparison for the compact correction",
    ),
    (
        N99_PRED / "N99_prediction_feature_coefficients.csv",
        TABLES / "N99_prediction_feature_coefficients.csv",
        "N99",
        "standardized compact-correction coefficients",
    ),
    (
        N99_PRED / "N99_prediction_region_holdout.csv",
        TABLES / "N99_prediction_region_holdout.csv",
        "N99",
        "leave-region-out diagnostics for the compact correction",
    ),
    (
        N99_PRED / "N99_prediction_permutation_baseline.csv",
        TABLES / "N99_prediction_permutation_baseline.csv",
        "N99",
        "shuffled-label permutation audit for the compact correction",
    ),
    (
        N99_PRED / "N99_prediction_coefficient_stability.csv",
        TABLES / "N99_prediction_coefficient_stability.csv",
        "N99",
        "leave-one-city-out coefficient-stability audit for the compact correction",
    ),
    (
        N99_PRED / "N99_prediction_summary.json",
        TABLES / "N99_prediction_summary.json",
        "N99",
        "prediction-correction summary metrics",
    ),
    (
        N99_LOW / "N99_low_overlap_surrogate_records.csv",
        TABLES / "N99_low_overlap_surrogate_records.csv",
        "N99",
        "low-overlap strict-geometry surrogate audit records",
    ),
    (
        N99_LOW / "N99_low_overlap_surrogate_city_summary.csv",
        TABLES / "N99_low_overlap_surrogate_city_summary.csv",
        "N99",
        "low-overlap strict-geometry surrogate city summary",
    ),
    (
        N99_LOW / "N99_low_overlap_surrogate_summary.json",
        TABLES / "N99_low_overlap_surrogate_summary.json",
        "N99",
        "low-overlap strict-geometry surrogate summary metrics",
    ),
    (
        R100_FIG / "R100_geometry_null_contrast_source_data.csv",
        TABLES / "R100_geometry_null_contrast_source_data.csv",
        "R100",
        "source-data summary for the R100 main geometry contrast (manuscript Fig. 5)",
    ),
]

FIGURE_SOURCES = [
    (
        ROOT / "figures" / "Fig_N99_compact_correction",
        FIGS / "FigS_N99_compact_correction",
        DISPLAY_FIGS / "FigS_N99_compact_correction",
        "N99",
        "supplementary compact transfer-error correction and permutation-audit figure",
    ),
    (
        ROOT / "figures" / "Fig_N99_low_overlap_geometry_surrogate",
        FIGS / "FigS_N99_low_overlap_geometry_surrogate",
        DISPLAY_FIGS / "FigS_N99_low_overlap_geometry_surrogate",
        "N99",
        "supplementary low-overlap strict-geometry surrogate audit figure",
    ),
]

EXTRA_REGISTER_ONLY = [
    (
        FIGS / "FigS_geometry_null_distance_audit.pdf",
        "R80",
        "registered derivative of Supplementary Fig. S1",
    ),
    (
        FIGS / "FigS_geometry_null_distance_audit.png",
        "R80",
        "registered derivative of Supplementary Fig. S1",
    ),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(path: Path) -> tuple[int, int]:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        return int(len(df)), int(len(df.columns))
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return 1, int(len(data))
    return 1, 1


def project_rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def sd_rel(path: Path) -> str:
    return path.relative_to(SD).as_posix()


def copy_assets() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    DISPLAY_FIGS.mkdir(parents=True, exist_ok=True)
    for src, dst, _round, _role in SOURCES:
        shutil.copy2(src, dst)
    for stem_src, stem_sd, stem_display, _round, _role in FIGURE_SOURCES:
        for suffix in [".svg", ".pdf", ".png", ".tiff"]:
            src = stem_src.with_suffix(suffix)
            if src.exists():
                shutil.copy2(src, stem_sd.with_suffix(suffix))
                shutil.copy2(src, stem_display.with_suffix(suffix))


def variable_description(name: str) -> str:
    lower = name.lower()
    if lower in {"city", "country", "macro_region"}:
        return "City, country or macro-region identifier."
    if "jaccard" in lower or "overlap" in lower or "turnover" in lower:
        return "Exact edge-overlap or edge-turnover diagnostic for regenerated strict-geometry surrogates."
    if "accepted_swap" in lower or "acceptance" in lower:
        return "Accepted-swap intensity or acceptance diagnostic for a null-model chain."
    if "degree_drift" in lower:
        return "Maximum absolute degree drift after rewiring."
    if "length" in lower or "tvd" in lower:
        return "Edge-length-bin fidelity or edge-scale diagnostic."
    if "prediction" in lower or "pred" in lower or "loo" in lower or "lro" in lower:
        return "Cross-validated prediction or holdout diagnostic for the transfer-error correction."
    if "mae" in lower or "rmse" in lower or "r2" in lower or "error" in lower:
        return "Prediction-error or goodness-of-fit metric."
    if "kappa" in lower:
        return "Degree-moment branching metric kappa."
    if "cebh" in lower:
        return "CEBH degree-moment threshold baseline or related error."
    if "geometry" in lower:
        return "Strict non-crossing geometry-null threshold or residual metric."
    if "spatial" in lower:
        return "Degree-preserving spatial-scale null threshold or residual metric."
    if "pc_" in lower or "threshold" in lower:
        return "Estimated bond-percolation threshold or threshold proxy."
    if "gap" in lower or "residual" in lower or "road_minus" in lower:
        return "Road-minus-baseline threshold gap or residual."
    return "Project-derived N99 variable; see N99 report and manuscript Methods."


def upsert_manifest() -> None:
    path = SD / "Supplementary_Data_1_source_data_manifest.csv"
    df = pd.read_csv(path)
    remove_round = df["round"].astype(str).isin({"N99", "R100"})
    remove_map = df["copied_file"].astype(str).eq("Figure_Source_Data_Map.csv")
    remove_extra = df["copied_file"].astype(str).isin([sd_rel(p) for p, _r, _role in EXTRA_REGISTER_ONLY if p.exists()])
    df = df[~(remove_round | remove_map | remove_extra)].copy()
    rows: list[dict[str, Any]] = []
    all_items = list(SOURCES)
    for stem_src, stem_sd, _stem_display, round_id, role in FIGURE_SOURCES:
        for suffix in [".svg", ".pdf", ".png", ".tiff"]:
            src = stem_src.with_suffix(suffix)
            dst = stem_sd.with_suffix(suffix)
            if src.exists() and dst.exists():
                all_items.append((src, dst, round_id, role))
    for path_extra, round_id, role in EXTRA_REGISTER_ONLY:
        if path_extra.exists():
            all_items.append((path_extra, path_extra, round_id, role))
    for src, dst, round_id, role in all_items:
        n_rows, n_cols = describe(dst)
        rows.append(
            {
                "round": round_id,
                "source_role": role,
                "source_path": project_rel(src) if src.is_relative_to(ROOT) else sd_rel(src),
                "copied_file": sd_rel(dst),
                "format": dst.suffix.lower().lstrip("."),
                "n_rows": n_rows,
                "n_columns_or_keys": n_cols,
                "sha256": sha256(dst),
                "redistribution_boundary": "derived project table/figure; raw OSM/Geofabrik/GHSL/WDI archives excluded",
            }
        )
    fmap = write_figure_source_map()
    rows.append(
        {
            "round": "N99",
            "source_role": "figure-to-source-data map",
            "source_path": project_rel(fmap),
            "copied_file": sd_rel(fmap),
            "format": "csv",
            "n_rows": int(len(pd.read_csv(fmap))),
            "n_columns_or_keys": int(len(pd.read_csv(fmap).columns)),
            "sha256": sha256(fmap),
            "redistribution_boundary": "derived project table; raw OSM/Geofabrik/GHSL/WDI archives excluded",
        }
    )
    pd.concat([df, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)


def upsert_column_dictionary() -> None:
    path = SD / "Supplementary_Data_2_column_dictionary.csv"
    df = pd.read_csv(path)
    files = [dst.name for _src, dst, _round, _role in SOURCES if dst.suffix.lower() in {".csv", ".json"}]
    files.append("Figure_Source_Data_Map.csv")
    df = df[~df["file"].astype(str).isin(files)].copy()
    rows: list[dict[str, Any]] = []
    for _src, dst, _round, _role in SOURCES + [(SD / "Figure_Source_Data_Map.csv", SD / "Figure_Source_Data_Map.csv", "N99", "map")]:
        if dst.suffix.lower() == ".csv":
            data = pd.read_csv(dst)
            for col in data.columns:
                series = data[col]
                rows.append(
                    {
                        "file": dst.name,
                        "column": col,
                        "inferred_type": "numeric" if pd.api.types.is_numeric_dtype(series) else "text",
                        "n_rows": int(len(series)),
                        "non_empty_count": int(series.notna().sum()),
                        "example_value": "" if series.dropna().empty else str(series.dropna().iloc[0]),
                        "description": variable_description(col),
                        "type": "",
                        "non_null": "",
                        "unique_values": "",
                        "example": "",
                    }
                )
        elif dst.suffix.lower() == ".json":
            data = json.loads(dst.read_text(encoding="utf-8"))
            for key, value in data.items():
                rows.append(
                    {
                        "file": dst.name,
                        "column": key,
                        "inferred_type": type(value).__name__,
                        "n_rows": 1,
                        "non_empty_count": 1,
                        "example_value": str(value),
                        "description": variable_description(key),
                        "type": "",
                        "non_null": "",
                        "unique_values": "",
                        "example": "",
                    }
                )
    pd.concat([df, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)


def upsert_claim_map() -> None:
    path = SD / "Supplementary_Data_3_claim_to_table_map.csv"
    df = pd.read_csv(path)
    claims = {
        "A compact 71-city correction strongly reduces the CEBH transfer error without using null thresholds as predictors": (
            "N99_prediction_city_predictions.csv; N99_prediction_model_comparison.csv; "
            "N99_prediction_feature_group_comparison.csv; N99_prediction_region_holdout.csv; "
            "N99_prediction_permutation_baseline.csv; N99_prediction_coefficient_stability.csv; "
            "N99_prediction_summary.json"
        ),
        "A 12-city low-overlap strict-geometry audit preserves degree and length bins while reducing most exact edge Jaccard values below 0.90": (
            "N99_low_overlap_surrogate_records.csv; N99_low_overlap_surrogate_city_summary.csv; "
            "N99_low_overlap_surrogate_summary.json"
        ),
        "Figure panels and source-data files are mapped for the submission package": "Figure_Source_Data_Map.csv",
    }
    stale_claims = {
        "A compact 71-city correction law strongly reduces the CEBH transfer error without using null thresholds as predictors",
        "A six-city low-overlap strict-geometry pilot preserves degree and length bins while reducing exact edge Jaccard below 0.90",
    }
    df = df[~df["claim"].isin(set(claims) | stale_claims)].copy()
    rows = []
    for claim, files in claims.items():
        location = (
            "Main text Results 2.4, 2.5, 2.6; Methods; Supplementary Notes 14-15; Supplementary Figs. S5-S6"
            if "Figure panels" not in claim
            else "Figure legends; Source Data"
        )
        rows.append(
            {
                "claim": claim,
                "supporting_source_data_files": files,
                "manuscript_location": location,
            }
        )
    pd.concat([df, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)


def upsert_run_manifest() -> None:
    path = SD / "run_manifest.csv"
    df = pd.read_csv(path)
    df = df[~df["round"].astype(str).isin({"N99", "R100"})].copy()
    rows = [
        {
            "round": "N99",
            "analysis": "compact transfer-error correction and permutation audit",
            "data_directory": "data/N99_prediction_correction_law",
            "purpose": "Cross-validated empirical correction of the CEBH road-threshold transfer error",
            "status": "complete",
            "script_or_source": "scripts/build_N99_prediction_correction_law.py",
            "output_files": (
                "data/N99_prediction_correction_law/N99_prediction_city_predictions.csv; "
                "data/N99_prediction_correction_law/N99_prediction_permutation_baseline.csv; "
                "figures/Fig_N99_compact_correction.svg"
            ),
        },
        {
            "round": "N99",
            "analysis": "12-city low-overlap strict-geometry surrogate audit",
            "data_directory": "data/N99_low_overlap_geometry_surrogate",
            "purpose": "Test whether the strict non-crossing local-swap null can be pushed to lower exact edge overlap",
            "status": "complete_stratified_audit",
            "script_or_source": "scripts/build_N99_low_overlap_geometry_surrogate.py",
            "output_files": (
                "data/N99_low_overlap_geometry_surrogate/N99_low_overlap_surrogate_records.csv; "
                "figures/Fig_N99_low_overlap_geometry_surrogate.svg"
            ),
        },
        {
            "round": "R100",
            "analysis": "main Fig. 4 geometry-null contrast hardening",
            "data_directory": "data/R100_main_figure_hardening",
            "purpose": "Promote the 12-city low-overlap evidence into the main geometry-null figure",
            "status": "complete",
            "script_or_source": "scripts/build_R100_main_fig4_geometry_null_contrast.py",
            "output_files": (
                "data/R100_main_figure_hardening/R100_geometry_null_contrast_source_data.csv; "
                "figures/Fig_R100_geometry_null_contrast.svg; "
                "submission/npj_complexity/target_submission/figures/Fig5_geometry_null_contrast.svg"
            ),
        },
    ]
    pd.concat([df, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)
    manifest_df = pd.read_csv(path).astype(object).where(pd.notna(pd.read_csv(path)), None)
    (SD / "run_manifest.json").write_text(
        json.dumps(manifest_df.to_dict("records"), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def write_figure_source_map() -> Path:
    path = SD / "Figure_Source_Data_Map.csv"
    rows = [
        {
            "figure": "Fig. 1",
            "panel_or_content": "null-ladder schematic, matched spatial-versus-geometry contrast and kappa/urban-form links",
            "source_data_files": "figures/R77_matched_intensity_main_result_source_data.csv; tables/R77_kappa_predictive_relation.csv; tables/R62_urban_form_correlations.csv",
            "notes": "Main head figure; panel a is an explanatory schematic of the null ladder (not extracted road data); panels b-d use matched-intensity summary, kappa relation and urban-form correlations.",
        },
        {
            "figure": "Fig. 2",
            "panel_or_content": "eight-city road, degree-null and spatial-null decomposition",
            "source_data_files": "tables/R41_degree_preserving_nulls_city_summary.csv; tables/R56_spatial_length_constrained_nulls_city_summary.csv",
            "notes": "Eight-city formal multi-replicate null ladder.",
        },
        {
            "figure": "Fig. 3",
            "panel_or_content": "71-city window screen and urban-form associations",
            "source_data_files": "tables/R61_central_window_screen_results.csv; tables/R62_urban_form_covariates.csv; tables/R62_urban_form_correlations.csv; tables/R63_high_order_embedded_metrics.csv",
            "notes": "Geographic scale screen and embedded road-form signal.",
        },
        {
            "figure": "Fig. 4",
            "panel_or_content": "nested validation, public controls, high-order metrics and leave-region-out checks",
            "source_data_files": "figures/Fig4_combined_nested_public_source_data.csv; figures/Fig4_nested_validation_source_data.csv; tables/R73_nested_model_ladder_summary.csv; tables/R65_R65_model_comparison.csv; tables/R63_high_order_external_correlations.csv",
            "notes": "Urban-form validation figure (renumbered from Fig. 5 in R101 for sequential citation order); claim boundaries are stated in the caption.",
        },
        {
            "figure": "Fig. 5",
            "panel_or_content": "full-city strict-geometry coverage, matched-intensity residual reduction, fine-grid regeneration and low-overlap audit",
            "source_data_files": "tables/R100_geometry_null_contrast_source_data.csv; tables/R68_full71_geometry_null_city_summary.csv; tables/R81_full71_high_intensity_geometry_null_city_summary.csv; tables/R72_matched_intensity_geometry_vs_spatial_city.csv; tables/R80_geometry_null_city_distance_audit.csv; tables/R90_finegrid_null_thresholds.csv; tables/N99_low_overlap_surrogate_records.csv",
            "notes": "R100 geometry-null main figure (renumbered from Fig. 4 in R101); strict geometry-null coverage, matched contrast, fine-grid diagnostics and 12-city low-overlap audit.",
        },
        {
            "figure": "Supplementary Fig. S1",
            "panel_or_content": "geometry-null distance audit",
            "source_data_files": "tables/R80_geometry_null_fraction_distance_audit.csv; tables/R80_geometry_null_metric_distance_audit.csv; tables/R80_geometry_null_city_distance_audit.csv",
            "notes": "Archived replicate metrics, not exact edge-overlap reconstruction.",
        },
        {
            "figure": "Supplementary Fig. S2",
            "panel_or_content": "R81 full 71-city high-intensity geometry ensemble",
            "source_data_files": "tables/R81_full71_high_intensity_geometry_null_city_summary.csv; tables/R81_full71_high_intensity_geometry_null_macro_region_summary.csv; tables/R81_full71_high_intensity_geometry_null_replicates.csv",
            "notes": "426 passing high-intensity strict-geometry records.",
        },
        {
            "figure": "Supplementary Fig. S3",
            "panel_or_content": "R90 fine-grid and exact edge-overlap audit",
            "source_data_files": "tables/R90_finegrid_null_thresholds.csv; tables/R90_edge_overlap_audit.csv; tables/R90_spatial_geometry_pair_summary.csv",
            "notes": "21-city regenerated matched nulls.",
        },
        {
            "figure": "Supplementary Fig. S4",
            "panel_or_content": "R95 non-backtracking spectral audit",
            "source_data_files": "tables/N95_nonbacktracking_thresholds.csv; tables/N95_cebh_nb_geometry_decomposition.csv; tables/N95_nb_spectrum_localization_metrics.csv",
            "notes": "71-city negative graph-spectral control.",
        },
        {
            "figure": "Supplementary Fig. S5",
            "panel_or_content": "N99 compact transfer-error correction and permutation audit",
            "source_data_files": "tables/N99_prediction_city_predictions.csv; tables/N99_prediction_model_comparison.csv; tables/N99_prediction_feature_coefficients.csv; tables/N99_prediction_region_holdout.csv; tables/N99_prediction_permutation_baseline.csv; tables/N99_prediction_coefficient_stability.csv",
            "notes": "Descriptive cross-validated empirical correction, not a causal law.",
        },
        {
            "figure": "Supplementary Fig. S6",
            "panel_or_content": "N99 12-city low-overlap strict-geometry surrogate audit",
            "source_data_files": "tables/N99_low_overlap_surrogate_records.csv; tables/N99_low_overlap_surrogate_city_summary.csv",
            "notes": "Twelve-city audit under the implemented strict local-swap chain; not a 71-city mixing proof.",
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main() -> None:
    copy_assets()
    upsert_manifest()
    upsert_column_dictionary()
    upsert_claim_map()
    upsert_run_manifest()


if __name__ == "__main__":
    main()
