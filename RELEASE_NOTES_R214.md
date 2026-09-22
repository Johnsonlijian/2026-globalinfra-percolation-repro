# Release notes: v1.1.0-r214

Date: 22 September 2026

This release supersedes `v1.0.2-r209` and accompanies the revised manuscript
under review at *Reliability Engineering & System Safety*. The concept DOI
`10.5281/zenodo.21102470` covers every version of this package.

## What changed since v1.0.2-r209

- **External cohort split.** The acquired external cities are reported as a
  19-city primary cohort plus two cities used in earlier project work, which are
  disclosed and reported separately. The reference allowance, predictor and rule
  were not changed by the split, and the split is recorded in
  `data/external_split_note.json`.
- **Primary-cohort evaluation and constant-offset comparator.** The primary
  cohort outcomes and a post hoc reference-only constant-offset comparator are
  included as derived tables, with the comparator labelled as a diagnostic of the
  predictor choice rather than a candidate rule.
- **Response-definition robustness.** The reported error direction was re-tested
  under four alternative finite-graph peak definitions on one shared ensemble per
  graph; the records are in `data/response_definition_check.csv`.
- **Control budget ladders.** The degree-preserving controls were regenerated and
  re-estimated across Monte Carlo order budgets and across rewiring depths; see
  `data/control_ladders.csv` and `data/control_mixing_archive.csv`.
- **Window and failure-process sensitivities.** Within-window outward diagnostic
  records, nested-window results and the non-random removal scenarios are
  included with their numerical bounds.
- **Additional measured-coordinate power diagnostic.** A mapped transmission
  model diagnostic and the published high-voltage layer boundary check are
  included as scalar derived records.
- **Declarations and provenance.** The package carries `DATA_MANIFEST.json`,
  `CODE_PROVENANCE.json`, `FILES_SHA256.csv` and the timestamped acquisition
  protocol records.

## Scope and limits

- The package regenerates the reported tables and figures from the included
  derived outputs. Rebuilding a historical graph result from raw source requires
  the matching author-held snapshot.
- Raw third-party data are not redistributed. Retrieval locations, versions and
  use conditions are listed in `DATASETS_AND_LINKS.csv`.
- No manuscript text, reviewer response, internal round log, credential, private
  path or raw graph is included.
- The protocol records are local timestamped analysis records, not a public
  preregistration.

## Verification performed for this release

- Core regression tests and the packaged replay commands were executed from the
  archive; the results are recorded in the manuscript's submission package.
- `FILES_SHA256.csv` binds every member of this release by SHA-256.
- `scripts/audit_package.py` reports the file, manifest and figure checks that
  apply inside the archive.
