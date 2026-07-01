# 2026 Road-Network Percolation Reproducibility Package

This repository contains public, non-sensitive materials for reproducing the
derived evidence behind the road-network percolation null-model analysis
supporting the working manuscript:

**Why road networks leave random-graph percolation: testing a local
embedded-merging constraint**.

## R198 Release Scope

This release aligns the public reproducibility package with the R198
mechanism-first working version. R198 does not add a new raw-data download or a
new numerical rerun. It reorganizes the public evidence around a bounded
model-transfer claim:

1. observed road thresholds define a random-graph transfer problem;
2. degree sequence and edge scale are insufficient controls;
3. the implemented strict local non-crossing null bounds the remaining
   residual near estimator resolution;
4. threshold-scale, spectral, road-form and cross-domain checks define
   consequences and boundaries rather than a universal infrastructure law.

The self-contained R198 manuscript remains outside this public repository while
the paper is being revised. This repository publishes the code, derived source
tables, figure assets, manifests and data-source links needed for independent
inspection of the evidence package.

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

R81 extends the same strict non-crossing geometry-null design to a full
71-city higher-intensity ensemble. It includes 426/426 passing records at
accepted-swap fractions 0.01 and 0.02 with three replicates per city-fraction
pair. The mean road-minus-geometry residual is -0.0018 (95% CI -0.0043 to
0.0006), with maximum degree drift of zero and mean length-bin total-variation
distance of 1.8e-7. R81 answers the full-city high-intensity empirical ensemble
objection, but still does not prove Markov-chain mixing, a closed-form
planar-percolation theorem or a causal urban-design rule.

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
street form remains the strongest descriptive mechanism signal, including for
the post-spatial residual; this is predictive evidence, not causal inference.

R75 hardens the public source-data layer by adding explicit nested
validation aliases, geometry-null rejection-share summaries, mobility/fidelity
proxies and an edge-overlap availability table. The edge-overlap table records
that exact rewired-edge overlap cannot be reconstructed from the archived summaries
because final null edge sets were not stored; accepted-swap and mobility
metrics are reported as proxies only.

R76 rebuilds the nested-validation figure from the nested validation, public-control
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

R80 adds a geometry-null distance audit for the strict geometry null. It
summarizes residuals, null mobility, constraint pressure and archived
embedded-structure drift across R67/R68 geometry-null records. It supports a
measurable-perturbation interpretation while preserving the boundary that exact
rewired-edge overlap is not reconstructed from the archived R67/R68 summaries
and full mixing is not claimed.

R90 adds a fine-grid null-threshold and exact edge-overlap audit for the
21-city core subset. It regenerates matched spatial-scale and strict-geometry
null thresholds on local 0.005 p-grids at accepted-swap fractions 0.005 and
0.01, producing 84/84 passing records. The mean fine road-minus-spatial
residual is 0.0406, the mean fine road-minus-geometry residual is -0.0032 and
the mean fine spatial-minus-geometry separation is 0.0438 (95% CI 0.0371 to
0.0506). The regenerated strict-geometry nulls have mean exact edge Jaccard
0.9714 and mean LCC-curve L2 distance 0.0324. R90 is an estimator and
copy-artifact stress test, not a Markov-chain mixing theorem or causal
urban-design rule.

R95 adds a non-backtracking spectral threshold audit across the full 71-city
matrix. It computes a finite-graph message-passing proxy
pc_NB = 1/rho(B) and compares it with observed road thresholds, CEBH, the
71-city spatial null and the strict-geometry ensemble. The non-backtracking
proxy does not close the gap: its mean absolute error is 0.319 versus 0.217
for CEBH, and its mean improvement over CEBH is -0.102. A 160-versus-640
iteration stability audit gives mean absolute pc_NB change 0.000136 and
maximum change 0.000911. R95 is a negative spectral control and theory bridge,
not a replacement for the spatial/geometry null ladder or an analytical
theorem.

N99 adds two final stress tests. A compact transfer-error correction uses only
whitelisted road-form variables and excludes observed, null and spectral
threshold outcomes as predictors. The four-variable degree-form correction
reduces CEBH mean absolute threshold error from 0.2166 to 0.0150 under
leave-one-city-out validation and 0.0146 under leave-region-out validation,
and a 1000-run shuffled-label audit gives a permutation p-value proxy of
0.001 for both validation protocols. Coefficient signs are stable across
leave-one-city-out folds, with minimum sign consistency of 0.986.

The N99/R100 low-overlap strict-geometry audit covers 12 standardized city
windows at accepted-swap targets 0.04 and 0.08. It produces 24 usable records,
22/24 with exact edge Jaccard below 0.90, minimum Jaccard 0.783, mean Jaccard
0.848, zero degree drift, zero length-bin total-variation drift and near-zero
mean road-minus-geometry residual (-0.0056; 95% CI -0.0164 to 0.0051). These
are empirical stress tests, not causal laws or a full 71-city low-overlap
mixing theorem.

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
- `scripts/build_R81_full71_high_intensity_geometry_null_ensemble.py`: R81
  full-city high-intensity geometry-null script. With `--skip-compute`, it
  recreates R81 summaries and figures from included derived tables; a full
  rerun requires local cached OSMnx graph objects.
- `scripts/build_R72_geometry_defense.py`: matched-intensity spatial-versus-
  geometry contrast on the 21-city sensitivity subset.
- `scripts/build_R73_urban_form_nested_cv.py`: nested urban-form model ladder
  and leave-region-out validation script.
- `scripts/build_R75_submission_hardening_tables.py`: source-data hardening
  script for nested-validation aliases, geometry mobility/rejection diagnostics
  and edge-overlap availability boundaries.
- `scripts/build_R76_nested_validation_figure.py`: nested-validation
  figure builder that combines nested validation, public-control checks, high-order
  feature correlations and explicit claim-boundary notes.
- `scripts/build_R77_physics_takeaway.py`: matched-geometry head figure and
  kappa/geometry-absorption source-data builder.
- `scripts/build_R78_observed_road_fine_estimator_check.py`: 21-city
  observed-road fine-threshold estimator check and matched-intensity
  fine-observed substitution table.
- `scripts/build_R80_geometry_distance_audit.py`: geometry-null distance
  audit and public source-data builder.
- `scripts/build_R90_finegrid_edge_audit.py`: R90 fine-grid matched-null and
  exact edge-overlap audit. With `--skip-compute`, it recreates the R90
  summaries, report and figure from included derived tables; a full rerun
  requires local cached OSMnx graph objects.
- `scripts/build_R95_nonbacktracking_spectral_layer.py`: R95
  non-backtracking spectral threshold audit. With `--skip-compute`, it
  recreates the R95 decomposition, report and figure from included derived
  tables; a full rerun requires local cached OSMnx graph objects.
- `scripts/build_R95_nb_iteration_stability_audit.py`: R95 160-versus-640
  iteration-stability audit for the non-backtracking threshold proxy.
- `scripts/build_N99_prediction_correction_law.py`: N99 leakage-guarded
  compact transfer-error correction, shuffled-label permutation audit,
  coefficient-stability audit and source-figure builder.
- `scripts/build_N99_low_overlap_geometry_surrogate.py`: N99/R100 12-city
  low-overlap strict-geometry surrogate audit.
- `scripts/build_R100_main_fig4_geometry_null_contrast.py`: R100 main Fig. 4
  geometry-null contrast hardening builder.
- `scripts/build_R62_urban_form_mechanism.py`: R62 covariate and Fig.3 source
  workflow; full geometry-pilot reruns require local cached graph objects.
- `scripts/build_R72_geometry_defense.py`: R72 matched-intensity spatial vs
  strict-geometry contrast and Fig. 4 script. Without `--force`, it recreates
  summaries and figures from included derived tables; with `--force`, it reruns
  matched spatial nulls and requires local cached graph objects.
- `run_manifest.csv` and `run_manifest.json`: compact run-level provenance for
  the public source-data bundle.
- `data/R41_degree_preserving_nulls/`: derived degree-null source tables.
- `data/R56_spatial_length_constrained_nulls/`: derived spatial-null source
  tables and report.
- `data/R61_city_window_scale_screen/`: derived 71-city standardized
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
- `data/R81_full71_high_intensity_geometry_null_ensemble/`: derived R81
  high-intensity geometry-null replicate, city-summary, macro-region and report
  tables.
- `data/R72_geometry_defense/`: derived R72 matched-intensity spatial-null
  replicates, geometry-vs-spatial city summaries, mobility summaries and report.
- `data/R73_urban_form_nested_cv/`: derived nested urban-form model summaries,
  city-level predictions, leave-region-out diagnostics and report.
- `data/R75_submission_hardening/`: derived submission-hardening aliases,
  geometry-null rejection summaries, mobility/fidelity proxy tables and
  figure-source-data tables.
- `data/R76_nested_validation_figure/`: derived Fig. 5 source data and summary.
- `data/R77_physics_takeaway/`: derived matched-intensity head-figure source
  data, geometry-absorption table, kappa predictions and summary.
- `data/R78_observed_road_fine_estimator_check/`: derived 21-city fine
  observed-road estimator check and fine-observed substitution summaries.
- `data/R90_finegrid_edge_audit/`: derived R90 fine-grid matched-null
  thresholds, exact edge-overlap audit, pair summary, report and summary JSON.
- `data/R95_nonbacktracking_spectral_layer/`: derived R95 non-backtracking
  threshold proxies, CEBH-NB-geometry decomposition, localization metrics,
  iteration-stability audit, report and summary JSON.
- `data/N99_prediction_correction_law/`: derived N99 compact-correction
  predictions, model comparisons, feature-group comparison, coefficients,
  leave-region-out diagnostics, shuffled-label permutation baseline,
  coefficient-stability audit and report.
- `data/N99_low_overlap_geometry_surrogate/`: derived N99 low-overlap
  strict-geometry audit records, city summary, report and gate outputs.
- `data/R100_main_figure_hardening/`: derived Fig. 4 source data joining
  full-city, matched-intensity, fine-grid and low-overlap geometry-null checks.
- `source_data/`: public source-data bundle, column dictionary,
  claim-to-table map and run manifest. It contains derived project tables only.
- `figures/Fig_R56_spatial_length_constrained_nulls.*`: exported figure.
- `figures/Fig_R65_public_covariate_controls.*`: exported R65 public-control
  figure.
- `figures/Fig_R67_geometry_null_sensitivity.*`: exported R67 null-ladder and
  geometry-null sensitivity figure.
- `figures/Fig_R68_full71_geometry_null_ensemble.*`: exported R68 full-city
  geometry-null figure.
- `figures/Fig_R81_full71_high_intensity_geometry_null_ensemble.*`: exported
  R81 full-city high-intensity geometry-null figure.
- `figures/Fig_R72_geometry_null_defense.*`: exported Fig. 4 R72
  matched-intensity geometry-null contrast figure.
- `figures/Fig_R73_urban_form_nested_cv.*`: exported supplementary nested
  urban-form validation figure.
- `figures/Fig_R76_nested_validation.*`: exported nested-validation figure with
  nested validation and public-control checks.
- `figures/Fig_R77_physics_takeaway.*`: exported matched-geometry head figure.
- `figures/Fig_R78_observed_road_fine_estimator_check.*`: exported observed-road
  fine-estimator sensitivity figure.
- `figures/Fig_R90_finegrid_edge_audit.*`: exported fine-grid null-threshold
  and exact edge-overlap audit figure.
- `figures/Fig_R95_nonbacktracking_spectral_layer.*`: exported
  non-backtracking spectral negative-control figure.
- `figures/Fig_N99_compact_correction.*`: exported compact
  transfer-error correction figure.
- `figures/Fig_N99_low_overlap_geometry_surrogate.*`: exported low-overlap
  strict-geometry surrogate audit figure.
- `figures/Fig_R100_geometry_null_contrast.*`: exported main Fig. 4
  geometry-null contrast figure.
- `figures/R198_final_main/`: R198 final six-main-figure package as SVG, PDF
  and PNG, plus `Figure_Source_Map_R198.csv`. TIFF submission renders are not
  included to keep the public repository light.

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
python scripts/build_R76_nested_validation_figure.py
python scripts/build_R77_physics_takeaway.py
python scripts/build_R78_observed_road_fine_estimator_check.py
python scripts/build_R90_finegrid_edge_audit.py --skip-compute
python scripts/build_R95_nonbacktracking_spectral_layer.py --skip-compute
python scripts/build_R95_nb_iteration_stability_audit.py --short-iter 160
python scripts/build_N99_prediction_correction_law.py
python scripts/build_N99_low_overlap_geometry_surrogate.py --skip-compute --cities "Singapore,Dubai,Taipei,Seattle,Nairobi,Sydney,Hong Kong,Rio de Janeiro,Chicago,Montevideo,Dar es Salaam,Barcelona" --max-cities 12
python scripts/build_R100_main_fig4_geometry_null_contrast.py
```

The full R56 null-model regeneration requires local graph caches or a
fresh OSMnx extraction workflow. The included derived tables are sufficient to
recreate the R56 evidence figure. R65 can be recreated from the
included R63/R64 derived inputs plus official GHSL/WDI downloads.
R67 can be redrawn from included derived tables with `--skip-compute`; the full
geometry-null regeneration requires local cached OSMnx graph objects.
R68 can also be redrawn from included derived tables with `--skip-compute`; the
full 71-city geometry-null regeneration requires local cached OSMnx graph
objects and is computationally heavier.
R72 can be redrawn from included derived tables without `--force`; a matched
spatial-null recomputation requires local cached OSMnx graph objects.
R75, R76, R77 and R78 are derived-table packaging and plotting/checking steps; they do not require
raw third-party downloads or cached graph objects. R90 can be redrawn from included derived
tables with `--skip-compute`; full R90 regeneration requires local cached OSMnx graph objects.

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

R90 fills the null-threshold fine-grid and exact-overlap audit for the
regenerated 21-city subset. It should still be read as an empirical stress test,
not as a Markov-chain mixing proof or a causal urban-design rule.
