# 2026 Global Infrastructure Percolation Reproducibility Package

This repository contains public, non-sensitive materials for reproducing the
derived evidence behind the road-network percolation null-model analysis.

## Core Result

Across eight method-coherent OSMnx road graphs, observed bond-percolation
thresholds exceed CEBH degree-moment predictions by about 0.2096. R41
degree-preserving nulls explain a small component of the gap. R56
degree-preserving spatial-scale nulls increase the explained component to a
mean gap of 0.0694, leaving about 66.8% of the road gap unreproduced by the
tested degree-plus-edge-scale control.

The package also includes the R65 public-covariate control increment. R65
matches the 71-city road-window matrix to JRC/GHSL UCDB R2024A urban-centre
covariates and World Bank WDI country controls. In five-fold RidgeCV screens,
road-form variables outperform external-only GHSL/WDI controls for both the
observed CEBH gap and the post-spatial residual.

The package now also includes the R67 geometry-null sensitivity increment. R67
uses the 21-city strict non-crossing subset, completes 162/162 geometry-null
replicates with zero degree drift, and shows that the mean road-minus-geometry
residual is 0.0056 at 0.005 accepted swaps per edge and approximately zero in
the feasible 0.02-swap subset. This is a sensitivity-tested subset, not a full
71-city planar-null ensemble.

## What Is Included

- `scripts/build_R41_degree_preserving_nulls.py`: full R41 analysis script.
- `scripts/build_R56_spatial_length_constrained_nulls.py`: full R56 analysis
  script. It requires cached road graphs with node coordinates.
- `scripts/replot_R56_spatial_null_figure.py`: public-data-only script that
  recreates the R56 figure from included derived source tables.
- `scripts/build_R65_public_covariates_and_controls.py`: public-data script
  that downloads official GHSL/WDI sources, matches the 71-city matrix, runs
  external-control screens and recreates the R65 figure.
- `scripts/build_R60_expansion_geometry_nulls.py`: strict non-crossing
  geometry-preserving rewiring helper used by R67.
- `scripts/build_R67_geometry_null_sensitivity.py`: R67 sensitivity script.
  With `--skip-compute`, it recreates R67 summaries and figures from included
  derived tables; a full rerun requires local cached OSMnx graph objects.
- `data/R41_degree_preserving_nulls/`: derived degree-null source tables.
- `data/R56_spatial_length_constrained_nulls/`: derived spatial-null source
  tables and report.
- `data/R63_high_order_external_covariates/`: derived high-order 71-city input
  table used by R65.
- `data/R64_71city_spatial_null_screen/`: derived all-city spatial-null input
  table used by R65.
- `data/R65_public_covariates_and_controls/`: derived public-covariate match,
  model-comparison and source-registry tables. Raw GHSL zip packages are not
  redistributed.
- `data/R67_geometry_null_sensitivity/`: derived R67 replicate, city-summary,
  fraction-summary and report tables.
- `figures/Fig_R56_spatial_length_constrained_nulls.*`: exported figure.
- `figures/Fig_R65_public_covariate_controls.*`: exported R65 public-control
  figure.
- `figures/Fig_R67_geometry_null_sensitivity.*`: exported R67 null-ladder and
  geometry-null sensitivity figure.

## What Is Not Included

This package intentionally excludes raw OpenStreetMap downloads, cached graph
pickles, active manuscripts, cover letters, reviewer drafts, internal rounds,
logs and private author/funding material. It also excludes raw GHSL zip
packages and large WDI all-year caches; the R65 script records official source
URLs and can download the public sources again.

## Quick Start

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python scripts/replot_R56_spatial_null_figure.py
python scripts/build_R65_public_covariates_and_controls.py
python scripts/build_R67_geometry_null_sensitivity.py --skip-compute
```

The full R56 null-model regeneration requires local graph caches or a
fresh OSMnx extraction workflow. The included derived tables are sufficient to
recreate the submitted R56 evidence figure. R65 can be recreated from the
included R63/R64 derived inputs plus official GHSL/WDI downloads.
R67 can be redrawn from included derived tables with `--skip-compute`; the full
geometry-null regeneration requires local cached OSMnx graph objects.

## Boundary

The R56 null preserves degree sequence exactly and constrains edge chord-length
slots. It is not a full planar, block-preserving or road-geometry-preserving
null model.

R67 adds a stricter non-crossing geometry-null sensitivity test for 21 cities.
It preserves degree sequence and rejects replacement edges that cross
non-adjacent existing segments, but it still does not preserve all road
semantics, planning history, directed traffic rules or block structure. It is
not a complete 71-city planar-null ensemble.
