"""Refresh source-data registry hashes after the R123 final sprint.

This script is intentionally narrow: it updates copied-file hashes and row/column
counts inside the npj Complexity target submission source-data manifest, syncs
the CSV run manifest to JSON, and reports missing/mismatched copied files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "submission" / "npj_complexity" / "target_submission"
SOURCE_DATA = TARGET / "source_data"
MANIFEST = SOURCE_DATA / "Supplementary_Data_1_source_data_manifest.csv"
RUN_CSV = SOURCE_DATA / "run_manifest.csv"
RUN_JSON = SOURCE_DATA / "run_manifest.json"
FIGURE_MAP = SOURCE_DATA / "Figure_Source_Data_Map.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_shape(path: Path) -> tuple[str, str, str]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        try:
            df = pd.read_csv(path)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
        return "csv", str(len(df)), str(len(df.columns))
    if suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return "json", str(len(obj)), str(len(obj[0]) if obj and isinstance(obj[0], dict) else "")
        if isinstance(obj, dict):
            return "json", "1", str(len(obj))
        return "json", "1", ""
    return suffix.lstrip(".") or "file", "", ""


def normalize_figure_map() -> None:
    if not FIGURE_MAP.exists():
        return
    df = pd.read_csv(FIGURE_MAP)
    mask = df["figure"].astype(str).eq("Supplementary Fig. S8")
    if mask.any():
        df.loc[mask, "source_data_files"] = (
            "tables/R123_street_form_subgroup_city_scores.csv; "
            "tables/R123_street_form_subgroup_quartiles.csv; "
            "tables/R123_street_form_subgroup_correlations.csv; "
            "tables/R123_summary.json"
        )
    df.to_csv(FIGURE_MAP, index=False)


def refresh_run_manifest_json() -> None:
    if not RUN_CSV.exists():
        return
    df = pd.read_csv(RUN_CSV)
    records = []
    for row in df.to_dict("records"):
        clean = {}
        for key, value in row.items():
            clean[key] = None if pd.isna(value) else value
        records.append(clean)
    RUN_JSON.write_text(json.dumps(records, indent=2), encoding="utf-8")


def refresh_manifest() -> dict:
    df = pd.read_csv(MANIFEST, dtype=str).fillna("")
    missing: list[str] = []
    for i, row in df.iterrows():
        copied = row.get("copied_file", "")
        if not copied:
            continue
        path = SOURCE_DATA / copied
        if not path.exists():
            missing.append(copied)
            continue
        fmt, n_rows, n_cols = file_shape(path)
        df.at[i, "format"] = fmt
        df.at[i, "n_rows"] = n_rows
        df.at[i, "n_columns_or_keys"] = n_cols
        df.at[i, "sha256"] = sha256(path)
    df.to_csv(MANIFEST, index=False)

    reloaded = pd.read_csv(MANIFEST, dtype=str).fillna("")
    mismatches: list[str] = []
    for _, row in reloaded.iterrows():
        copied = row.get("copied_file", "")
        if not copied:
            continue
        path = SOURCE_DATA / copied
        if not path.exists():
            continue
        if row.get("sha256", "").lower() != sha256(path).lower():
            mismatches.append(copied)
    return {
        "manifest_rows": int(len(reloaded)),
        "missing_copied_files": missing,
        "checksum_mismatches_after_refresh": mismatches,
        "run_manifest_json_sha256": sha256(RUN_JSON) if RUN_JSON.exists() else None,
        "supplementary_code_zip_sha256": sha256(TARGET / "Supplementary_Code_1_reproducibility_package.zip")
        if (TARGET / "Supplementary_Code_1_reproducibility_package.zip").exists()
        else None,
    }


def main() -> None:
    normalize_figure_map()
    refresh_run_manifest_json()
    summary = refresh_manifest()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
