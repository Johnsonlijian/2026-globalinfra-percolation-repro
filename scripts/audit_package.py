"""Audit packaged files, derived-data hashes and figure rendering."""

from __future__ import annotations

import csv
import json
import re
import xml.etree.ElementTree as ET

import fitz
import numpy as np
from common import Context, parser, sha


def audit(context):
    root = context.root
    required = [
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "requirements.txt",
        "DATASETS_AND_LINKS.csv",
        "DATA_MANIFEST.json",
        "REPRODUCIBLE_RUNBOOK.md",
        "config/reproduction.json",
        "data/source_registry.csv",
    ]
    for name in required:
        if not (root / name).is_file():
            raise FileNotFoundError(name)
    forbidden_suffixes = {".pkl", ".pickle", ".npz", ".inp", ".pbf", ".osm", ".zip", ".pyc", ".log"}
    forbidden_columns = {
        "private_file",
        "graph_cache_file",
        "source_note",
        "pc_r211",
        "pc_rank",
        "elapsed_seconds",
        "stage_edge_snapshot",
        "source_pickle",
        "private_snapshot_file",
    }
    windows_path = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/](?!/)")
    legacy_import = re.compile(r"(?:from|import)\s+R(?:105|211)\w*")
    problems = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() in forbidden_suffixes:
            problems.append((relative, "excluded raw/archive/cache type"))
        if any(
            part in {"rounds", "logs", "manuscript", "private_graphs", ".venv"}
            for part in path.relative_to(root).parts
        ):
            problems.append((relative, "excluded content directory"))
        if path.suffix.lower() in {".py", ".json", ".csv", ".md", ".txt", ".tex", ".cff", ".svg"}:
            text = path.read_text(encoding="utf-8-sig")
            if windows_path.search(text) or re.search(r"/(?:Users|home)/[A-Za-z0-9_]", text):
                problems.append((relative, "private absolute path"))
            if legacy_import.search(text):
                problems.append((relative, "historical module import dependency"))
        if path.suffix == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as handle:
                fields = set(next(csv.reader(handle), []))
            if fields & forbidden_columns:
                problems.append((relative, "excluded metadata/old-estimate column"))
    data = json.loads((root / "DATA_MANIFEST.json").read_text())
    for item in data["datasets"].values():
        if sha(root / item["path"]) != item["sha256"]:
            problems.append((item["path"], "derived input hash changed"))
    figure_checks = []
    for stem in [
        "fig1_screening_example",
        "fig2_accuracy",
        "fig3_nulls",
        "fig4_local_control",
        "fig5_validation",
        "fig6_geometry",
    ]:
        pdf, svg = (
            context.output / "figures" / (stem + ".pdf"),
            context.output / "figures" / (stem + ".svg"),
        )
        tree = ET.parse(svg).getroot()
        if tree.findall(".//{http://www.w3.org/2000/svg}image"):
            problems.append((stem, "unexpected embedded raster"))
        with fitz.open(pdf) as doc:
            page = doc[0]
            if (
                len(doc) != 1
                or not page.get_fonts()
                or any(f[2] == "Type3" for f in page.get_fonts())
            ):
                problems.append((stem, "PDF page/font contract"))
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        box = fitz.Rect(span["bbox"])
                        if (
                            box.x0 < -1
                            or box.y0 < -1
                            or box.x1 > page.rect.width + 1
                            or box.y1 > page.rect.height + 1
                        ):
                            problems.append((stem, "PDF text outside page"))
            pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
            ink = float(np.mean(np.frombuffer(pix.samples, dtype=np.uint8) < 245))
            if ink < 0.01:
                problems.append((stem, "blank or near-blank render"))
            figure_checks.append(
                {
                    "figure": stem,
                    "pdf_sha256": sha(pdf),
                    "svg_sha256": sha(svg),
                    "pages": len(doc),
                    "nonwhite_sample_fraction": ink,
                    "vector_svg": True,
                }
            )
    report = {
        "status": "PASS" if not problems else "FAIL",
        "problems": problems,
        "dataset_count": len(data["datasets"]),
        "figures": figure_checks,
        "release_state": "ACCOMPANYING_ARCHIVE_NOT_PUBLICLY_RELEASED",
        "raw_graphs_distributed": False,
        "scope": "File/plot hygiene and input hashes; not original-data acquisition or publication approval",
    }
    context.write_json("audit_report.json", report)
    if problems:
        raise RuntimeError(json.dumps(problems, indent=2))
    rows = [
        {"path": p.relative_to(root).as_posix(), "sha256": sha(p), "bytes": p.stat().st_size}
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "FILES_SHA256.csv"
    ]
    with (root / "FILES_SHA256.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "sha256", "bytes"])
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"PASS: {len(rows)} files, {len(data['datasets'])} bound derived inputs, six nonblank vector figures"
    )


if __name__ == "__main__":
    audit(Context(parser(__doc__).parse_args()))
