# Reproducible Runbook

## Environment

Use Python 3.11 or newer.

```bash
pip install -r requirements.txt
```

## Recreate R70 Fig. 1 Null-ladder Schematic

```bash
python scripts/fig1_R70_null_ladder_schematic.py
```

Expected outputs:

- `figures/Fig_R70_null_ladder_schematic.png`
- `figures/Fig_R70_null_ladder_schematic.svg`
- `figures/Fig_R70_null_ladder_schematic.pdf`
- `figures/Fig_R70_null_ladder_schematic.tiff`

## Recreate The Public R56 Figure

```bash
python scripts/replot_R56_spatial_null_figure.py
```

Expected output:

- `figures/Fig_R56_spatial_length_constrained_nulls_public.png`
- `figures/Fig_R56_spatial_length_constrained_nulls_public.svg`
- `figures/Fig_R56_spatial_length_constrained_nulls_public.pdf`

## Re-run R56 From Graph Caches

The full analysis script is included for transparency:

```bash
python scripts/build_R56_spatial_length_constrained_nulls.py --n-null 4 --n-perm 16 --n-bootstrap 300 --swap-fraction 0.25 --length-bins 12 --bin-tolerance 1 --max-attempt-factor 60
```

This requires cached OSMnx road graphs under `cache/private_graphs/` with node
longitude/latitude and edge data. Those caches are not redistributed because
they are raw third-party derived graph objects and may contain source-data
licensing or size concerns.

## Interpretation Check

Use `data/R56_spatial_length_constrained_nulls/spatial_length_constrained_nulls_report.md`
as the manuscript-safe boundary statement. Do not interpret R56 as a full
planar null or causal mechanism proof.

## Recreate R65 Public-Covariate Controls

R65 uses included R63/R64 derived inputs plus official public downloads from
JRC/GHSL and the World Bank WDI API.

```bash
python scripts/build_R65_public_covariates_and_controls.py
```

Expected outputs:

- `data/R65_public_covariates_and_controls/public_data_source_registry.csv`
- `data/R65_public_covariates_and_controls/ghsl_match_qc.csv`
- `data/R65_public_covariates_and_controls/wdi_country_controls.csv`
- `data/R65_public_covariates_and_controls/R65_integrated_public_covariates.csv`
- `data/R65_public_covariates_and_controls/R65_external_correlations.csv`
- `data/R65_public_covariates_and_controls/R65_model_comparison.csv`
- `data/R65_public_covariates_and_controls/R65_summary.json`
- `figures/Fig_R65_public_covariate_controls.png`
- `figures/Fig_R65_public_covariate_controls.svg`
- `figures/Fig_R65_public_covariate_controls.pdf`
- `figures/Fig_R65_public_covariate_controls.tiff`

The script writes GHSL raw zip downloads under
`data/R65_public_covariates_and_controls/raw/`. That directory is intentionally
ignored by git; redistribute only after checking the source terms and citation
requirements.

## Recreate R67 Geometry-null Sensitivity Figure

The public package includes the derived R67 replicate and summary tables. To
rebuild the R67 summaries and figure without raw graph caches:

```bash
python scripts/build_R67_geometry_null_sensitivity.py --skip-compute
```

Expected outputs:

- `data/R67_geometry_null_sensitivity/geometry_null_sensitivity_city_summary.csv`
- `data/R67_geometry_null_sensitivity/geometry_null_sensitivity_fraction_summary.csv`
- `data/R67_geometry_null_sensitivity/R67_summary.json`
- `figures/Fig_R67_geometry_null_sensitivity.png`
- `figures/Fig_R67_geometry_null_sensitivity.svg`
- `figures/Fig_R67_geometry_null_sensitivity.pdf`
- `figures/Fig_R67_geometry_null_sensitivity.tiff`

## Re-run R67 From Graph Caches

The full R67 geometry-null regeneration requires cached OSMnx road graph
objects with node longitude/latitude fields. Those graph caches are not
redistributed. If local caches are available in the expected project layout,
run:

```bash
python scripts/build_R67_geometry_null_sensitivity.py --force
```

Interpret R67 as a 21-city strict non-crossing sensitivity test. It is stronger
than a one-replicate pilot, but it is not a full 71-city planar-null ensemble
and does not prove a causal urban-design law.

## Recreate R68 Full-city Geometry-null Figure

The public package includes the derived R68 replicate, city-summary and
macro-region tables. To rebuild the R68 summaries and figure without raw graph
caches:

```bash
python scripts/build_R68_full71_geometry_null_ensemble.py --skip-compute
```

Expected outputs:

- `data/R68_full71_geometry_null_ensemble/full71_geometry_null_city_summary.csv`
- `data/R68_full71_geometry_null_ensemble/full71_geometry_null_macro_region_summary.csv`
- `data/R68_full71_geometry_null_ensemble/R68_summary.json`
- `figures/Fig_R68_full71_geometry_null_ensemble.png`
- `figures/Fig_R68_full71_geometry_null_ensemble.svg`
- `figures/Fig_R68_full71_geometry_null_ensemble.pdf`
- `figures/Fig_R68_full71_geometry_null_ensemble.tiff`

## Re-run R68 From Graph Caches

The full R68 geometry-null regeneration requires cached OSMnx road graph
objects with node longitude/latitude fields. Those graph caches are not
redistributed. If local caches are available in the expected project layout,
run:

```bash
python scripts/build_R68_full71_geometry_null_ensemble.py --force
```

Interpret R68 as full 71-city low-intensity strict geometry-null coverage with
one replicate per city. It answers the low-intensity coverage objection, but it
is not a high-intensity multi-replicate planar-null theorem and does not prove
a causal urban-design rule.

## Recreate R72 Matched-intensity Geometry-null Defense

The public package includes the derived R72 matched spatial-null replicates,
matched city summaries and geometry-null mobility summaries. To rebuild the
R72 summaries and Fig. 4 from included tables:

```bash
python scripts/build_R72_geometry_defense.py
```

Expected outputs:

- `data/R72_geometry_defense/matched_intensity_geometry_vs_spatial_city.csv`
- `data/R72_geometry_defense/matched_intensity_summary.csv`
- `data/R72_geometry_defense/geometry_null_mobility_summary.csv`
- `data/R72_geometry_defense/R72_summary.json`
- `figures/Fig_R72_geometry_null_defense.png`
- `figures/Fig_R72_geometry_null_defense.svg`
- `figures/Fig_R72_geometry_null_defense.pdf`
- `figures/Fig_R72_geometry_null_defense.tiff`

## Re-run R72 Matched Spatial Nulls From Graph Caches

The full R72 matched spatial-null recomputation requires cached OSMnx road
graph objects. Those graph caches are not redistributed. If local caches are
available in the expected project layout, run:

```bash
python scripts/build_R72_geometry_defense.py --force
```

Interpret R72 as a matched-intensity empirical defense on the 21-city R67
subset. It weakens the low-perturbation artifact objection, but it is not a
proof of full Markov-chain mixing or a high-intensity all-71 planar ensemble.
