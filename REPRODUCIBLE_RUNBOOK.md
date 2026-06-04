# Reproducible Runbook

## Environment

Use Python 3.11 or newer.

```bash
pip install -r requirements.txt
```

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
