"""Register R95 non-backtracking spectral audit in the npj Complexity package."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "R95_nonbacktracking_spectral_layer"
NPJ = ROOT / "submission" / "npj_complexity" / "target_submission"
SD = NPJ / "source_data"
TABLES = SD / "tables"
FIGS = SD / "figures"
DISPLAY_FIGS = NPJ / "figures"


SOURCES = [
    (
        OUT / "N95_nonbacktracking_thresholds.csv",
        TABLES / "N95_nonbacktracking_thresholds.csv",
        "R95",
        "71-city non-backtracking spectral threshold proxy",
    ),
    (
        OUT / "N95_cebh_nb_geometry_decomposition.csv",
        TABLES / "N95_cebh_nb_geometry_decomposition.csv",
        "R95",
        "CEBH, non-backtracking, spatial-null and geometry-null decomposition",
    ),
    (
        OUT / "N95_nb_spectrum_localization_metrics.csv",
        TABLES / "N95_nb_spectrum_localization_metrics.csv",
        "R95",
        "non-backtracking spectral localization metrics",
    ),
    (
        OUT / "R95_nb_iteration_stability_audit.csv",
        TABLES / "R95_nb_iteration_stability_audit.csv",
        "R95",
        "160-versus-640 iteration stability audit for the non-backtracking proxy",
    ),
    (
        OUT / "R95_summary.json",
        TABLES / "R95_R95_summary.json",
        "R95",
        "non-backtracking spectral audit summary metrics",
    ),
    (
        OUT / "R95_nb_iteration_stability_summary.json",
        TABLES / "R95_nb_iteration_stability_summary.json",
        "R95",
        "non-backtracking iteration-stability summary metrics",
    ),
    (
        ROOT / "figures" / "Fig_R95_nonbacktracking_spectral_layer.svg",
        FIGS / "FigS_R95_nonbacktracking_spectral_layer.svg",
        "R95",
        "supplementary non-backtracking spectral audit figure",
    ),
    (
        ROOT / "figures" / "Fig_R95_nonbacktracking_spectral_layer.pdf",
        FIGS / "FigS_R95_nonbacktracking_spectral_layer.pdf",
        "R95",
        "supplementary non-backtracking spectral audit figure",
    ),
    (
        ROOT / "figures" / "Fig_R95_nonbacktracking_spectral_layer.png",
        FIGS / "FigS_R95_nonbacktracking_spectral_layer.png",
        "R95",
        "supplementary non-backtracking spectral audit figure",
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


def copied_rel(path: Path) -> str:
    return path.relative_to(SD).as_posix()


def project_rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def copy_assets() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    DISPLAY_FIGS.mkdir(parents=True, exist_ok=True)
    for src, dst, _round, _role in SOURCES:
        shutil.copy2(src, dst)
    for suffix in [".svg", ".pdf", ".png", ".tiff"]:
        src = ROOT / "figures" / f"Fig_R95_nonbacktracking_spectral_layer{suffix}"
        if src.exists():
            shutil.copy2(src, DISPLAY_FIGS / f"FigS_R95_nonbacktracking_spectral_layer{suffix}")


def variable_description(name: str) -> str:
    lower = name.lower()
    if lower in {"city", "country", "macro_region"}:
        return "City, country or macro-region identifier."
    if "nonbacktracking" in lower or "_nb" in lower or "rho_nb" in lower:
        return "Non-backtracking spectral-radius, threshold-proxy or error diagnostic."
    if "kappa" in lower:
        return "Degree-moment branching metric kappa or kappa-derived quantity."
    if "cebh" in lower:
        return "CEBH degree-moment threshold baseline or related error metric."
    if "geometry" in lower:
        return "Strict non-crossing geometry-null threshold or residual metric."
    if "spatial" in lower:
        return "Degree-preserving spatial-scale null threshold or residual metric."
    if "ipr" in lower or "localization" in lower or "entropy" in lower or "weight_share" in lower:
        return "Non-backtracking centrality localization or entropy metric."
    if "pc_" in lower or "threshold" in lower:
        return "Estimated bond-percolation threshold or threshold proxy."
    if "gap" in lower or "residual" in lower or "road_minus" in lower or "error" in lower:
        return "Road-minus-baseline residual or prediction-error metric."
    if "delta" in lower or "change" in lower:
        return "Difference or stability-diagnostic quantity."
    if "converged" in lower or "iterations" in lower or "runtime" in lower:
        return "Power-iteration convergence or runtime diagnostic."
    if "degree" in lower:
        return "Degree-sequence or degree-composition metric."
    if "edge" in lower or "length" in lower:
        return "Edge-count, edge-scale or edge-length diagnostic."
    return "Project-derived variable; see R95 report and manuscript Methods for definition."


def upsert_manifest() -> None:
    path = SD / "Supplementary_Data_1_source_data_manifest.csv"
    df = pd.read_csv(path)
    df = df[df["round"] != "R95"].copy()
    rows = []
    for src, dst, round_id, role in SOURCES:
        n_rows, n_cols = describe(dst)
        rows.append(
            {
                "round": round_id,
                "source_role": role,
                "source_path": project_rel(src),
                "copied_file": copied_rel(dst),
                "format": dst.suffix.lower().lstrip("."),
                "n_rows": n_rows,
                "n_columns_or_keys": n_cols,
                "sha256": sha256(dst),
                "redistribution_boundary": "derived project table/figure; raw OSM/Geofabrik archives excluded",
            }
        )
    pd.concat([df, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)


def upsert_column_dictionary() -> None:
    path = SD / "Supplementary_Data_2_column_dictionary.csv"
    df = pd.read_csv(path)
    names = [dst.name for _src, dst, _round, _role in SOURCES if dst.suffix.lower() in {".csv", ".json"}]
    df = df[~df["file"].isin(names)].copy()
    rows = []
    for _src, dst, _round, _role in SOURCES:
        if dst.suffix.lower() == ".csv":
            data = pd.read_csv(dst)
            for col in data.columns:
                series = data[col]
                rows.append(
                    {
                        "file": dst.name,
                        "column": col,
                        "type": "numeric" if pd.api.types.is_numeric_dtype(series) else "text",
                        "non_null": int(series.notna().sum()),
                        "unique_values": int(series.nunique(dropna=True)),
                        "example": "" if series.dropna().empty else str(series.dropna().iloc[0]),
                        "description": variable_description(col),
                    }
                )
        elif dst.suffix.lower() == ".json":
            data = json.loads(dst.read_text(encoding="utf-8"))
            for key, value in data.items():
                rows.append(
                    {
                        "file": dst.name,
                        "column": key,
                        "type": type(value).__name__,
                        "non_null": 1,
                        "unique_values": 1,
                        "example": str(value),
                        "description": variable_description(key),
                    }
                )
    pd.concat([df, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)


def upsert_claim_map() -> None:
    path = SD / "Supplementary_Data_3_claim_to_table_map.csv"
    df = pd.read_csv(path)
    claim = "R95 non-backtracking spectral audit shows graph-spectral branching alone does not close the road threshold gap"
    df = df[df["claim"] != claim].copy()
    row = {
        "claim": claim,
        "supporting_source_data_files": (
            "N95_nonbacktracking_thresholds.csv; N95_cebh_nb_geometry_decomposition.csv; "
            "N95_nb_spectrum_localization_metrics.csv; R95_nb_iteration_stability_audit.csv; "
            "R95_R95_summary.json; R95_nb_iteration_stability_summary.json"
        ),
        "manuscript_location": "Main text Results 2.6; Methods; Supplementary Note 13; Supplementary Fig. S9",
    }
    pd.concat([df, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)


def upsert_run_manifest() -> None:
    path = SD / "run_manifest.csv"
    df = pd.read_csv(path)
    df = df[df["round"] != "R95"].copy()
    row = {
        "round": "R95",
        "analysis": "non-backtracking spectral audit",
        "data_directory": "data/R95_nonbacktracking_spectral_layer",
        "script_or_source": (
            "scripts/build_R95_nonbacktracking_spectral_layer.py; "
            "scripts/build_R95_nb_iteration_stability_audit.py"
        ),
        "output_files": (
            "data/R95_nonbacktracking_spectral_layer/N95_nonbacktracking_thresholds.csv; "
            "data/R95_nonbacktracking_spectral_layer/N95_cebh_nb_geometry_decomposition.csv; "
            "data/R95_nonbacktracking_spectral_layer/R95_nb_iteration_stability_audit.csv; "
            "figures/Fig_R95_nonbacktracking_spectral_layer.svg"
        ),
        "purpose": "Test whether a non-backtracking spectral threshold proxy closes the road threshold gap",
        "status": "complete",
    }
    pd.concat([df, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)
    manifest_df = pd.read_csv(path).astype(object)
    manifest_df = manifest_df.where(pd.notna(manifest_df), None)
    (SD / "run_manifest.json").write_text(
        json.dumps(manifest_df.to_dict("records"), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def main() -> None:
    copy_assets()
    upsert_manifest()
    upsert_column_dictionary()
    upsert_claim_map()
    upsert_run_manifest()


if __name__ == "__main__":
    main()
