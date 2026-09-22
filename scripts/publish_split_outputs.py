"""Publish the derived split and comparator outputs into the evidence directory.

`external_split.py` writes its derived tables into `generated/` and `data/`.
The supplementary-table builder reads the derived evidence directory, so this
step copies the same bytes there and records their hashes. It performs no
calculation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT.parent / "analysis" / "final_evidence"
PAIRS = [
    "external_cohort_split_summary.csv",
    "external_primary_predictors.csv",
    "external_shift_comparator.csv",
    "external_predictors.csv",
    "external_shift_summary.json",
    "external_shift_equal_rank.csv",
    "external_split_note.json",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    generated, data, evidence = ROOT / "generated", ROOT / "data", EVIDENCE
    evidence.mkdir(parents=True, exist_ok=True)
    published = {}
    for name in PAIRS:
        source = generated / name
        if not source.is_file():
            raise FileNotFoundError(f"Run external_split.py first: missing {source}")
        if digest(source) != digest(data / name):
            raise RuntimeError(f"Derived copy mismatch for {name}")
        destination = evidence / name
        # Byte-for-byte write; the network share rejects atomic replace.
        destination.write_bytes(source.read_bytes())
        published[name] = digest(destination)
    (evidence / "external_split_publication.json").write_text(
        json.dumps(
            {
                "status": "DERIVED_OUTPUTS_PUBLISHED_NO_RECALCULATION",
                "source_script": "scripts/external_split.py",
                "files": published,
                "note": "Copied byte-identical derived outputs into the evidence directory.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(published, indent=2))


if __name__ == "__main__":
    main()
