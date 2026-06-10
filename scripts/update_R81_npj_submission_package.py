"""Register R81 source data in the npj Complexity target package."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "R81_full71_high_intensity_geometry_null_ensemble"
NPJ = ROOT / "submission" / "npj_complexity" / "target_submission"
SD = NPJ / "source_data"
TABLES = SD / "tables"
FIGS = SD / "figures"
DISPLAY_FIGS = NPJ / "figures"


SOURCES = [
    (
        OUT / "full71_geometry_null_city_summary.csv",
        TABLES / "R81_full71_high_intensity_geometry_null_city_summary.csv",
        "R81",
        "full 71-city high-intensity geometry null city summary",
    ),
    (
        OUT / "full71_geometry_null_macro_region_summary.csv",
        TABLES / "R81_full71_high_intensity_geometry_null_macro_region_summary.csv",
        "R81",
        "full 71-city high-intensity geometry null macro-region summary",
    ),
    (
        OUT / "full71_geometry_null_replicates.csv",
        TABLES / "R81_full71_high_intensity_geometry_null_replicates.csv",
        "R81",
        "full 71-city high-intensity geometry null replicates",
    ),
    (
        OUT / "R81_summary.json",
        TABLES / "R81_R81_summary.json",
        "R81",
        "full 71-city high-intensity geometry null summary metrics",
    ),
    (
        ROOT / "figures" / "Fig_R81_full71_high_intensity_geometry_null_ensemble.svg",
        FIGS / "FigS_R81_full71_high_intensity_geometry_null_ensemble.svg",
        "R81",
        "supplementary full-city high-intensity geometry-null figure",
    ),
    (
        ROOT / "figures" / "Fig_R81_full71_high_intensity_geometry_null_ensemble.pdf",
        FIGS / "FigS_R81_full71_high_intensity_geometry_null_ensemble.pdf",
        "R81",
        "supplementary full-city high-intensity geometry-null figure",
    ),
    (
        ROOT / "figures" / "Fig_R81_full71_high_intensity_geometry_null_ensemble.png",
        FIGS / "FigS_R81_full71_high_intensity_geometry_null_ensemble.png",
        "R81",
        "supplementary full-city high-intensity geometry-null figure",
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
        src = ROOT / "figures" / f"Fig_R81_full71_high_intensity_geometry_null_ensemble{suffix}"
        if src.exists():
            shutil.copy2(src, DISPLAY_FIGS / f"FigS_R81_full71_high_intensity_geometry_null_ensemble{suffix}")


def upsert_manifest() -> None:
    path = SD / "Supplementary_Data_1_source_data_manifest.csv"
    df = pd.read_csv(path)
    df = df[df["round"] != "R81"].copy()
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


def variable_description(name: str) -> str:
    lower = name.lower()
    if lower in {"city", "country", "macro_region"}:
        return "City, country or macro-region identifier."
    if "degree" in lower:
        return "Degree-sequence or degree-composition metric."
    if "length" in lower or "edge" in lower:
        return "Edge-length or edge-scale diagnostic."
    if "swap" in lower or "accept" in lower or "reject" in lower:
        return "Accepted-swap, attempted-swap or rejection diagnostic."
    if "pc_" in lower or "threshold" in lower:
        return "Estimated bond-percolation threshold or threshold diagnostic."
    if "gap" in lower or "residual" in lower or "road_minus" in lower:
        return "Road-minus-baseline or road-minus-null residual metric."
    if "geometry" in lower or "null" in lower:
        return "Strict non-crossing geometry-null metric or diagnostic."
    if "spatial" in lower:
        return "Spatial-scale null metric or diagnostic."
    if "ci95" in lower or "ci_" in lower:
        return "Bootstrap or normal-approximation confidence interval bound."
    return "Project-derived variable; see round report and manuscript Methods for definition."


def upsert_column_dictionary() -> None:
    path = SD / "Supplementary_Data_2_column_dictionary.csv"
    df = pd.read_csv(path)
    r81_names = [dst.name for _src, dst, _round, _role in SOURCES if dst.suffix.lower() in {".csv", ".json"}]
    df = df[~df["file"].isin(r81_names)].copy()
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
    claim = "R81 full 71-city high-intensity geometry null residual is near zero"
    df = df[df["claim"] != claim].copy()
    row = {
        "claim": claim,
        "supporting_source_data_files": (
            "R81_full71_high_intensity_geometry_null_city_summary.csv; "
            "R81_full71_high_intensity_geometry_null_macro_region_summary.csv; "
            "R81_full71_high_intensity_geometry_null_replicates.csv; R81_R81_summary.json"
        ),
        "manuscript_location": "Main text Results 2.5; Methods; Supplementary Note 9; Supplementary Fig. S7",
    }
    pd.concat([df, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)


def upsert_run_manifest() -> None:
    path = SD / "run_manifest.csv"
    df = pd.read_csv(path)
    df = df[df["round"] != "R81"].copy()
    row = {
        "round": "R81",
        "script_or_source": "scripts/build_R81_full71_high_intensity_geometry_null_ensemble.py",
        "output_files": (
            "data/R81_full71_high_intensity_geometry_null_ensemble/full71_geometry_null_replicates.csv; "
            "figures/Fig_R81_full71_high_intensity_geometry_null_ensemble.svg"
        ),
        "purpose": "Full 71-city high-intensity strict non-crossing geometry-null ensemble",
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
