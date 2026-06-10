"""Register R90 fine-grid and edge-overlap audit in the npj Complexity package."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "R90_finegrid_edge_audit"
NPJ = ROOT / "submission" / "npj_complexity" / "target_submission"
SD = NPJ / "source_data"
TABLES = SD / "tables"
FIGS = SD / "figures"
DISPLAY_FIGS = NPJ / "figures"


SOURCES = [
    (
        OUT / "R90_finegrid_null_thresholds.csv",
        TABLES / "R90_finegrid_null_thresholds.csv",
        "R90",
        "fine-grid matched spatial and strict-geometry null thresholds",
    ),
    (
        OUT / "R90_edge_overlap_audit.csv",
        TABLES / "R90_edge_overlap_audit.csv",
        "R90",
        "exact edge-overlap and threshold-curve distance audit",
    ),
    (
        OUT / "R90_spatial_geometry_pair_summary.csv",
        TABLES / "R90_spatial_geometry_pair_summary.csv",
        "R90",
        "matched spatial-versus-geometry fine-grid pair summary",
    ),
    (
        OUT / "R90_summary.json",
        TABLES / "R90_R90_summary.json",
        "R90",
        "fine-grid and edge-overlap audit summary metrics",
    ),
    (
        ROOT / "figures" / "Fig_R90_finegrid_edge_audit.svg",
        FIGS / "FigS_R90_finegrid_edge_audit.svg",
        "R90",
        "supplementary fine-grid and edge-overlap audit figure",
    ),
    (
        ROOT / "figures" / "Fig_R90_finegrid_edge_audit.pdf",
        FIGS / "FigS_R90_finegrid_edge_audit.pdf",
        "R90",
        "supplementary fine-grid and edge-overlap audit figure",
    ),
    (
        ROOT / "figures" / "Fig_R90_finegrid_edge_audit.png",
        FIGS / "FigS_R90_finegrid_edge_audit.png",
        "R90",
        "supplementary fine-grid and edge-overlap audit figure",
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
        src = ROOT / "figures" / f"Fig_R90_finegrid_edge_audit{suffix}"
        if src.exists():
            shutil.copy2(src, DISPLAY_FIGS / f"FigS_R90_finegrid_edge_audit{suffix}")


def variable_description(name: str) -> str:
    lower = name.lower()
    if lower in {"city", "country", "macro_region"}:
        return "City, country or macro-region identifier."
    if "null_type" in lower:
        return "Null-model family identifier: spatial-scale or strict geometry."
    if "jaccard" in lower or "overlap" in lower or "turnover" in lower:
        return "Exact edge-overlap, edge-turnover or line-graph similarity diagnostic."
    if "lcc_curve" in lower or "path_length" in lower:
        return "Largest-connected-component curve or sampled shortest-path distance diagnostic."
    if "degree" in lower:
        return "Degree-sequence or degree-drift metric."
    if "length" in lower or "edge" in lower:
        return "Edge-length, edge-scale or edge-count diagnostic."
    if "swap" in lower or "accept" in lower or "reject" in lower:
        return "Accepted-swap, attempted-swap or rejection diagnostic."
    if "pc_" in lower or "threshold" in lower:
        return "Estimated bond-percolation threshold or threshold diagnostic."
    if "gap" in lower or "residual" in lower or "road_minus" in lower:
        return "Road-minus-baseline or road-minus-null residual metric."
    if "ci95" in lower or "ci_" in lower:
        return "Bootstrap or normal-approximation confidence interval bound."
    if lower.startswith("delta_"):
        return "Null-minus-observed change in an embedded graph metric."
    return "Project-derived variable; see R90 report and manuscript Methods for definition."


def upsert_manifest() -> None:
    path = SD / "Supplementary_Data_1_source_data_manifest.csv"
    df = pd.read_csv(path)
    df = df[df["round"] != "R90"].copy()
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
    claim = "R90 fine-grid and exact edge-overlap audit preserves spatial-versus-geometry separation"
    df = df[df["claim"] != claim].copy()
    row = {
        "claim": claim,
        "supporting_source_data_files": (
            "R90_finegrid_null_thresholds.csv; R90_edge_overlap_audit.csv; "
            "R90_spatial_geometry_pair_summary.csv; R90_R90_summary.json"
        ),
        "manuscript_location": "Main text Results 2.5; Methods; Supplementary Note 12; Supplementary Fig. S8",
    }
    pd.concat([df, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)


def upsert_run_manifest() -> None:
    path = SD / "run_manifest.csv"
    df = pd.read_csv(path)
    df = df[df["round"] != "R90"].copy()
    row = {
        "round": "R90",
        "analysis": "fine-grid null and exact edge-overlap audit",
        "data_directory": "data/R90_finegrid_edge_audit",
        "script_or_source": "scripts/build_R90_finegrid_edge_audit.py",
        "output_files": (
            "data/R90_finegrid_edge_audit/R90_finegrid_null_thresholds.csv; "
            "data/R90_finegrid_edge_audit/R90_edge_overlap_audit.csv; "
            "figures/Fig_R90_finegrid_edge_audit.svg"
        ),
        "purpose": "Fine-grid matched null-threshold and exact edge-overlap audit",
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
