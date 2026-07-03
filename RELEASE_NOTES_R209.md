# Release notes: v1.0.2-r209

Date: 2026-07-03

This release aligns the public reproducibility package with the R209
submission package for:

**A measured validity boundary for percolation threshold screening in
infrastructure networks**

## Added

- R205 measured crossing-density and dose-response derived tables.
- R205 scripts for crossing-density measurement, grade-separation validation,
  confound controls, crossing-length decomposition, controlled dials,
  cross-domain non-planarity and decision experiment summaries.
- R206/R208 RESS reframing derived tables for fine-grid dials, rail artifact
  validation, pass/fail decision sweeps, reference verification, breadth checks,
  window sensitivity, review-response statistics and mid-range dials.
- R206/R208 scripts used to build the RESS-oriented evidence and figure layers.
- R209 final six-main-figure package under `figures/R209_RESS_main/` as PDF,
  SVG and PNG.
- Copies of the R205/R206/R208 derived tables under `source_data/tables/` so
  the public source-data path matches the manuscript data-availability
  statement.

## Updated

- README title, scope and package inventory updated from R198 to R209.
- `CITATION.cff` and `.zenodo.json` updated to release `v1.0.2-r209`.
- `DATASETS_AND_LINKS.csv`, `run_manifest.csv` and `run_manifest.json`
  updated with R205/R206/R208/R209 package entries.
- Two figure-builder label edits are included to clarify
  shortcut/crossing wording in exponent and effective-dimension figures.

## Not included

- Active submission manuscript PDF/DOCX.
- Cover letter.
- Reviewer-response drafts, portal metadata and internal round notes.
- Raw third-party OSM, Geofabrik, GHSL, WDI, pandapower or WNTR source
  archives.

## DOI

Use the Zenodo concept DOI for the package family:

<https://doi.org/10.5281/zenodo.21102470>

If Zenodo creates a new version DOI for this GitHub release, it can be added to
the manuscript portal metadata after the archive job completes.
