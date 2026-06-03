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

## What Is Included

- `scripts/build_R41_degree_preserving_nulls.py`: full R41 analysis script.
- `scripts/build_R56_spatial_length_constrained_nulls.py`: full R56 analysis
  script. It requires cached road graphs with node coordinates.
- `scripts/replot_R56_spatial_null_figure.py`: public-data-only script that
  recreates the R56 figure from included derived source tables.
- `data/R41_degree_preserving_nulls/`: derived degree-null source tables.
- `data/R56_spatial_length_constrained_nulls/`: derived spatial-null source
  tables and report.
- `figures/Fig_R56_spatial_length_constrained_nulls.*`: exported figure.

## What Is Not Included

This package intentionally excludes raw OpenStreetMap downloads, cached graph
pickles, active manuscripts, cover letters, reviewer drafts, internal rounds,
logs and private author/funding material.

## Quick Start

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python scripts/replot_R56_spatial_null_figure.py
```

The full R56 null-model regeneration requires local graph caches or a
fresh OSMnx extraction workflow. The included derived tables are sufficient to
recreate the submitted R56 evidence figure.

## Boundary

The R56 null preserves degree sequence exactly and constrains edge chord-length
slots. It is not a full planar, block-preserving or road-geometry-preserving
null model.
