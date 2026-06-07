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

R68 extends low-intensity strict non-crossing geometry-null coverage to the
full 71-city standardized-window matrix. The included derived tables record
71/71 passing one-replicate records, maximum degree drift of zero, mean
road-minus-spatial residual of 0.0965 and mean road-minus-geometry residual of
0.0014 (95% CI -0.0027 to 0.0056). R68 is full-city low-intensity empirical
coverage, not an analytical theorem or high-intensity multi-replicate planar
ensemble.

R72 adds a matched-intensity spatial-versus-geometry contrast.
On the 21-city sensitivity subset, spatial-scale and strict non-crossing geometry nulls
are compared at the same accepted-swap targets. Spatial-scale nulls retain
positive residuals at 0.005, 0.01 and 0.02 accepted swaps per edge, whereas
strict-geometry residuals remain near zero at the tested estimator resolution.
This separates edge scale from strict local geometry but does not prove full
Markov-chain mixing.

R73 adds a nested urban-form validation layer. It compares size/density,
degree-composition, edge-scale, embedded street-form and public-control model
families under five-fold and leave-region-out cross-validation. Embedded
street form remains the strongest descriptive mechanism-candidate signal, including for
the post-spatial residual; this is predictive evidence, not causal inference.

R75 hardens the submission source-data layer by adding explicit nested
validation aliases, geometry-null rejection-share summaries, mobility/fidelity
proxies and an edge-overlap availability table. The edge-overlap table records
that exact rewired-edge overlap cannot be reconstructed from the archived summaries
because final null edge sets were not stored; accepted-swap and mobility
metrics are reported as proxies only.

R76 rebuilds the submission Fig. 5 from the nested validation, public-control
comparison and high-order embedded-feature tables. The figure also includes a
claim-boundary panel stating that the 71-city geometry-null screen is
low-intensity and descriptive, not a causal urban-design rule or planar theorem.

R77 promotes the matched-intensity contrast to the head evidence figure and
adds two compact mechanism checks: geometry absorption against embedded
road-form metrics and a kappa-anchored one-variable relation for the 71-city
road-minus-CEBH gap. R78 recomputes observed-road thresholds for the 21-city
geometry subset on a local fine grid and shows that the matched spatial-minus-
geometry separation is unchanged when the shared observed threshold is
substituted.

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
- `scripts/build_R68_full71_geometry_null_ensemble.py`: R68 full-city
  low-intensity geometry-null script. With `--skip-compute`, it recreates R68
  summaries and figures from included derived tables.
- `scripts/build_R72_geometry_defense.py`: matched-intensity spatial-versus-
  geometry contrast on the 21-city sensitivity subset.
- `scripts/build_R73_urban_form_nested_cv.py`: nested urban-form model ladder
  and leave-region-out validation script.
- `scripts/build_R75_submission_hardening_tables.py`: source-data hardening
  script for nested-validation aliases, geometry mobility/rejection diagnostics
  and edge-overlap availability boundaries.
- `scripts/build_R76_fig5_nested_submission_figure.py`: submission Fig. 5
  builder that combines nested validation, public-control checks, high-order
  feature correlations and explicit claim-boundary notes.
- `scripts/build_R77_physics_takeaway.py`: matched-geometry head figure and
  kappa/geometry-absorption source-data builder.
- `scripts/build_R78_observed_road_fine_estimator_check.py`: 21-city
  observed-road fine-threshold estimator check and matched-intensity
  fine-observed substitution table.
- `scripts/build_R62_urban_form_mechanism.py`: R62 covariate and Fig.3 source
  workflow; full geometry-pilot reruns require local cached graph objects.
- `scripts/build_R72_geometry_defense.py`: R72 matched-intensity spatial vs
  strict-geometry contrast and Fig. 4 script. Without `--force`, it recreates
  summaries and figures from included derived tables; with `--force`, it reruns
  matched spatial nulls and requires local cached graph objects.
- `run_manifest.csv` and `run_manifest.json`: compact run-level provenance for
  the submission source-data bundle.
- `data/R41_degree_preserving_nulls/`: derived degree-null source tables.
- `data/R56_spatial_length_constrained_nulls/`: derived spatial-null source
  tables and report.
- `data/R61_nature_cities_scale_screen/`: derived 71-city standardized
  center-window threshold screen.
- `data/R62_urban_form_mechanism/`: derived 21-city geometry-subset registry
  and 71-city urban-form covariate table required by R67/R68 replotting.
- `data/R63_high_order_external_covariates/`: derived high-order 71-city input
  table used by R65.
- `data/R64_71city_spatial_null_screen/`: derived all-city spatial-null input
  table used by R65.
- `data/R65_public_covariates_and_controls/`: derived public-covariate match,
  model-comparison and source-registry tables. Raw GHSL zip packages are not
  redistributed.
- `data/R67_geometry_null_sensitivity/`: derived R67 replicate, city-summary,
  fraction-summary and report tables.
- `data/R68_full71_geometry_null_ensemble/`: derived R68 full-city
  geometry-null replicate, city-summary, macro-region and report tables.
- `data/R72_geometry_defense/`: derived R72 matched-intensity spatial-null
  replicates, geometry-vs-spatial city summaries, mobility summaries and report.
- `data/R73_urban_form_nested_cv/`: derived nested urban-form model summaries,
  city-level predictions, leave-region-out diagnostics and report.
- `data/R75_submission_hardening/`: derived submission-hardening aliases,
  geometry-null rejection summaries, mobility/fidelity proxy tables and
  figure-source-data tables.
- `data/R76_fig5_nested_submission/`: derived Fig. 5 source data and summary.
- `data/R77_physics_takeaway/`: derived matched-intensity head-figure source
  data, geometry-absorption table, kappa predictions and summary.
- `data/R78_observed_road_fine_estimator_check/`: derived 21-city fine
  observed-road estimator check and fine-observed substitution summaries.
- `source_data/`: submission-facing source-data bundle, column dictionary,
  claim-to-table map and run manifest. It contains derived project tables only.
- `figures/Fig_R56_spatial_length_constrained_nulls.*`: exported figure.
- `figures/Fig_R65_public_covariate_controls.*`: exported R65 public-control
  figure.
- `figures/Fig_R67_geometry_null_sensitivity.*`: exported R67 null-ladder and
  geometry-null sensitivity figure.
- `figures/Fig_R68_full71_geometry_null_ensemble.*`: exported R68 full-city
  geometry-null figure.
- `figures/Fig_R72_geometry_null_defense.*`: exported Fig. 4 R72
  matched-intensity geometry-null contrast figure.
- `figures/Fig_R73_urban_form_nested_cv.*`: exported supplementary nested
  urban-form validation figure.
- `figures/Fig_R76_fig5_nested_submission.*`: exported submission Fig. 5 with
  nested validation and public-control checks.
- `figures/Fig_R77_physics_takeaway.*`: exported matched-geometry head figure.
- `figures/Fig_R78_observed_road_fine_estimator_check.*`: exported observed-road
  fine-estimator sensitivity figure.

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
python scripts/build_R68_full71_geometry_null_ensemble.py --skip-compute
python scripts/build_R72_geometry_defense.py
python scripts/build_R75_submission_hardening_tables.py
python scripts/build_R76_fig5_nested_submission_figure.py
python scripts/build_R77_physics_takeaway.py
python scripts/build_R78_observed_road_fine_estimator_check.py
```

The full R56 null-model regeneration requires local graph caches or a
fresh OSMnx extraction workflow. The included derived tables are sufficient to
recreate the submitted R56 evidence figure. R65 can be recreated from the
included R63/R64 derived inputs plus official GHSL/WDI downloads.
R67 can be redrawn from included derived tables with `--skip-compute`; the full
geometry-null regeneration requires local cached OSMnx graph objects.
R68 can also be redrawn from included derived tables with `--skip-compute`; the
full 71-city geometry-null regeneration requires local cached OSMnx graph
objects and is computationally heavier.
R72 can be redrawn from included derived tables without `--force`; a matched
spatial-null recomputation requires local cached OSMnx graph objects.
R75, R76, R77 and R78 are derived-table packaging and plotting/checking steps; they do not require
raw third-party downloads or cached graph objects.

## Boundary

The R56 null preserves degree sequence exactly and constrains edge chord-length
slots. It is not a full planar, block-preserving or road-geometry-preserving
null model.

R67 adds a stricter non-crossing geometry-null sensitivity test for 21 cities.
It preserves degree sequence and rejects replacement edges that cross
non-adjacent existing segments, but it still does not preserve all road
semantics, planning history, directed traffic rules or block structure. It is
not a complete 71-city planar-null ensemble.

R68 adds full 71-city low-intensity geometry-null coverage with one replicate
per city. It answers the low-intensity coverage objection, but it is not a
high-intensity multi-replicate planar-null theorem and does not establish a
causal urban-design rule.

R72 adds a matched-intensity empirical contrast on the 21-city subset. It
shows that the spatial-scale null and strict-geometry null behave differently
under the same accepted-swap targets, but it does not prove full mixing or
isolate planarity alone from degree, length scale, angular and block structure.

R77/R78 add a head-figure synthesis, a kappa predictive relation and an
observed-road estimator check. They do not add a new null-model ensemble, do
not regenerate geometry-null fine curves and do not turn the urban-form screens
into causal inference.

