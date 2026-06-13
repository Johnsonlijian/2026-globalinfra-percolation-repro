"""R101: regenerate all submission-facing figures in the unified publication style.

This round is a figure-style correction pass, not a data round:

- every figure script now imports ``pub_style`` (Arial, 6.5-8 pt, Okabe-Ito
  semantic colours, thin axes, no card/dashboard/poster elements);
- the dashboard-style "claim boundary" text panel was removed from the
  validation figure;
- manuscript Fig. 1 gained an explanatory null-ladder schematic panel;
- npj figure numbering: validation figure = Fig. 4, geometry contrast = Fig. 5
  (sequential first in-text citations);
- the N99 figure stem was renamed from "correction_law" to
  "compact_correction" to match the demoted claim language;
- PNG/TIFF rasters are re-exported at 600 dpi (fixes the 160-170 dpi exports
  of the old geometry-contrast and compact-correction figures).

No simulation, null-model chain, threshold estimate or statistical result is
recomputed: every panel is redrawn from the registered derived tables.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

NPJ = ROOT / "submission" / "npj_complexity" / "target_submission"
NPJ_FIGS = NPJ / "figures"
NPJ_SD = NPJ / "source_data"
DATA = ROOT / "data"
FIGURES = ROOT / "figures"


def log(msg: str) -> None:
    print(f"[R101] {msg}", flush=True)


def redraw_r56() -> None:
    import build_R56_spatial_length_constrained_nulls as r56

    df = pd.read_csv(DATA / "R56_spatial_length_constrained_nulls" / "spatial_length_constrained_nulls_city_summary.csv")
    summaries = [SimpleNamespace(**row) for row in df.to_dict("records")]
    r56.plot_figure(r56.DEFAULT_FIGURE, summaries)
    for ext in [".svg", ".pdf", ".png", ".tiff"]:
        src = r56.DEFAULT_FIGURE.with_suffix(ext)
        shutil.copy2(src, NPJ_FIGS / f"Fig3_eight_city_null_decomposition{ext}")
    log("Fig. 3 (R56 eight-city decomposition) redrawn and copied.")


def redraw_r62() -> None:
    import build_R62_urban_form_mechanism as r62

    d = DATA / "R62_urban_form_mechanism"
    cov = pd.read_csv(d / "urban_form_covariates.csv")
    corr = pd.read_csv(d / "urban_form_correlations.csv")
    pca = pd.read_csv(d / "urban_form_pca_scores.csv")
    selected = pd.read_csv(d / "geometry_null_subset_registry.csv")
    geom = pd.read_csv(d / "geometry_null_pilot_summary.csv")
    r62.make_figure(cov, corr, pca, selected, geom)
    for ext in [".svg", ".pdf", ".png", ".tiff"]:
        src = Path(f"{r62.FIG_BASE}{ext}")
        shutil.copy2(src, NPJ_FIGS / f"Fig4_city_window_urban_form{ext}")
    log("Fig. 4 (R62 urban-form screen) redrawn and copied.")


def redraw_r81() -> None:
    import build_R81_full71_high_intensity_geometry_null_ensemble as r81

    d = DATA / "R81_full71_high_intensity_geometry_null_ensemble"
    city = pd.read_csv(d / "full71_geometry_null_city_summary.csv")
    macro = pd.read_csv(d / "full71_geometry_null_macro_region_summary.csv")
    summary = json.loads((d / "R81_summary.json").read_text(encoding="utf-8"))
    r81.make_figure(city, macro, summary)
    log("Supplementary Fig. S7 (R81 high-intensity ensemble) redrawn.")


def redraw_r90() -> None:
    import build_R90_finegrid_edge_audit as r90

    d = DATA / "R90_finegrid_edge_audit"
    reps = pd.read_csv(d / "R90_finegrid_null_thresholds.csv")
    audits = pd.read_csv(d / "R90_edge_overlap_audit.csv")
    pair_df = pd.read_csv(d / "R90_spatial_geometry_pair_summary.csv")
    summary = json.loads((d / "R90_summary.json").read_text(encoding="utf-8"))
    r90.make_figure(reps, audits, pair_df, summary)
    log("Supplementary Fig. S8 (R90 fine-grid edge audit) redrawn.")


def redraw_r95() -> None:
    import build_R95_nonbacktracking_spectral_layer as r95

    d = DATA / "R95_nonbacktracking_spectral_layer"
    decomp = pd.read_csv(d / "N95_cebh_nb_geometry_decomposition.csv")
    summary = json.loads((d / "R95_summary.json").read_text(encoding="utf-8"))
    r95.make_figure(decomp, summary)
    log("Supplementary Fig. S9 (R95 non-backtracking audit) redrawn.")


def redraw_n99_prediction() -> None:
    import build_N99_prediction_correction_law as n99p

    d = DATA / "N99_prediction_correction_law"
    pred = pd.read_csv(d / "N99_prediction_city_predictions.csv")
    coef = pd.read_csv(d / "N99_prediction_feature_coefficients.csv")
    metrics = pd.read_csv(d / "N99_prediction_model_comparison.csv")
    perm = pd.read_csv(d / "N99_prediction_permutation_baseline.csv")
    n99p.build_figures(pred, coef, metrics, perm)
    log("Supplementary Fig. S10 (N99 compact correction) redrawn under the renamed stem.")


def redraw_n99_low_overlap() -> None:
    import build_N99_low_overlap_geometry_surrogate as n99l

    d = DATA / "N99_low_overlap_geometry_surrogate"
    records = pd.read_csv(d / "N99_low_overlap_surrogate_records.csv")
    summary_df = pd.read_csv(d / "N99_low_overlap_surrogate_city_summary.csv")
    n99l.build_figure(records, summary_df)
    log("Supplementary Fig. S11 (N99 low-overlap audit) redrawn.")


def cleanup_npj_package() -> None:
    """Retire stale display figures and renamed source-data files."""
    legacy = NPJ_FIGS / "legacy"
    legacy.mkdir(exist_ok=True)
    stale_stems = [
        "Fig4_geometry_null_contrast",
        "Fig5_public_controls_boundary_checks",
        "FigS_N99_prediction_correction_law",
        "Supplementary_Fig_nested_urban_form_validation",
    ]
    for stem in stale_stems:
        for old in NPJ_FIGS.glob(f"{stem}.*"):
            shutil.move(str(old), str(legacy / old.name))
            log(f"moved stale display figure to legacy: {old.name}")

    sd_figs = NPJ_SD / "figures"
    for old in sd_figs.glob("FigS_N99_prediction_correction_law.*"):
        old.unlink()
        log(f"removed stale source-data figure: {old.name}")
    old_combined = sd_figs / "Fig5_combined_nested_public_source_data.csv"
    if old_combined.exists():
        old_combined.unlink()
        log("removed stale Fig5_combined_nested_public_source_data.csv")
    old_nested = sd_figs / "Fig5_nested_validation_source_data.csv"
    new_nested = sd_figs / "Fig4_nested_validation_source_data.csv"
    if old_nested.exists():
        if new_nested.exists():
            old_nested.unlink()
        else:
            old_nested.rename(new_nested)
        log("renamed Fig5_nested_validation_source_data.csv -> Fig4_nested_validation_source_data.csv")

    old_table = NPJ_SD / "tables" / "Fig4_R100_geometry_null_contrast_source_data.csv"
    if old_table.exists():
        old_table.unlink()
        log("removed stale tables/Fig4_R100_geometry_null_contrast_source_data.csv")


def fix_source_data_references() -> None:
    """Update renamed file references and drop manifest rows for retired files."""
    renames = {
        "Fig5_nested_validation_source_data.csv": "Fig4_nested_validation_source_data.csv",
        "Fig5_combined_nested_public_source_data.csv": "Fig4_combined_nested_public_source_data.csv",
        "Fig4_R100_geometry_null_contrast_source_data.csv": "R100_geometry_null_contrast_source_data.csv",
        "FigS_N99_prediction_correction_law": "FigS_N99_compact_correction",
        "Fig_N99_prediction_correction_law": "Fig_N99_compact_correction",
    }
    for name in [
        "Supplementary_Data_1_source_data_manifest.csv",
        "Supplementary_Data_2_column_dictionary.csv",
        "Supplementary_Data_3_claim_to_table_map.csv",
        "run_manifest.csv",
        "run_manifest.json",
    ]:
        path = NPJ_SD / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for old, new in renames.items():
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")

    manifest = NPJ_SD / "Supplementary_Data_1_source_data_manifest.csv"
    df = pd.read_csv(manifest)
    keep = []
    for _, row in df.iterrows():
        copied = str(row["copied_file"])
        keep.append((NPJ_SD / copied).exists())
    removed = int(len(df) - sum(keep))
    df[pd.Series(keep, index=df.index)].to_csv(manifest, index=False)
    log(f"manifest hygiene: removed {removed} rows pointing at retired files.")


def verify() -> None:
    expected = [
        "Fig1_matched_geometry_partition",
        "Fig2_planar_lattice_anchor",
        "Fig3_eight_city_null_decomposition",
        "Fig4_city_window_urban_form",
        "Fig5_urban_form_validation",
        "Fig6_geometry_null_contrast",
        "FigS_geometry_null_distance_audit",
        "FigS_R81_full71_high_intensity_geometry_null_ensemble",
        "FigS_R90_finegrid_edge_audit",
        "FigS_R95_nonbacktracking_spectral_layer",
        "FigS_N99_compact_correction",
        "FigS_N99_low_overlap_geometry_surrogate",
    ]
    missing = []
    for stem in expected:
        for ext in [".svg", ".pdf", ".png", ".tiff"]:
            if not (NPJ_FIGS / f"{stem}{ext}").exists():
                missing.append(f"{stem}{ext}")
    if missing:
        raise SystemExit(f"[R101] missing expected figure files: {missing}")
    log("verification passed: all 11 submission figure stems present in SVG/PDF/PNG/TIFF.")


def main() -> None:
    import build_R76_fig5_nested_submission_figure as r76
    import build_R100_main_fig4_geometry_null_contrast as r100
    import build_R77_physics_takeaway as r77
    import build_R80_npj_complexity_hardening as r80
    import update_R81_npj_submission_package as u81
    import update_R90_npj_submission_package as u90
    import update_R95_npj_submission_package as u95
    import update_N99_npj_submission_package as un99

    log("running R76 (validation figure, npj Fig. 4) ...")
    r76.main()
    log("running R100 (geometry contrast, npj Fig. 5) ...")
    r100.main()
    log("running R77 (head figure with null-ladder schematic, Fig. 1) ...")
    r77.main()
    log("running R80 (distance audit, Supplementary Fig. S6) ...")
    r80.main()

    redraw_r56()
    redraw_r62()
    redraw_r81()
    redraw_r90()
    redraw_r95()
    redraw_n99_prediction()
    redraw_n99_low_overlap()

    log("re-running npj package update scripts ...")
    u81.main()
    u90.main()
    u95.main()
    un99.main()

    cleanup_npj_package()
    fix_source_data_references()
    verify()
    log("done.")


if __name__ == "__main__":
    main()
