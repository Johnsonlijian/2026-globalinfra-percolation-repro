# R67 Geometry-null Sensitivity Hardening

## Purpose

R67 converts the R62 strict geometry-null pilot into a sensitivity-tested evidence layer. It keeps the same
21-city subset, preserves degree sequence, constrains edge-length bins, forbids new edge crossings, and tests
whether the observed CEBH-to-road percolation gap disappears when the geometry-null is replicated and pushed to
stronger feasible swap fractions.

## Run Contract

- Cities requested: 21
- Replicates requested per city/fraction: 3
- Base swap fractions: 0.005,0.01
- Extra high swap fraction: 0.02 when `n_edges <= 30,000`
- Threshold estimator: rank-order bond percolation, `n_perm=8`, `n_bootstrap=120`,
  `p_grid=0.1-0.9` in steps of 0.025
- Geometry constraints: non-crossing swaps, length bins=12, bin tolerance=1,
  neighbor search k=180, max attempt factor=900
- Null mobility: angular and dual-degree diagnostics are computed for all passing replicates; bridge,
  articulation, biconnected, and connected-component diagnostics are computed every
  3 replicate(s), marked by `full_connectivity_metrics`.

## Main Result

- Passing replicates: 162/162; failed replicates: 0
- Baseline swap fraction 0.005: mean road-minus-geometry residual =
  0.0056; mean remaining fraction after geometry =
  0.0223
- Highest completed fraction 0.02: mean road-minus-geometry residual =
  0.0000; mean remaining fraction after geometry =
  -0.0063
- Degree drift maximum = 0.0; mean edge-length-bin TVD = 0.0000

## Fraction Summary

|   swap_fraction |   n_cities_pass |   n_replicates_pass |   road_minus_geometry_mean |   road_minus_geometry_ci95_low |   road_minus_geometry_ci95_high |   remaining_fraction_after_geometry_mean |   accepted_swaps_per_edge_mean |   length_bin_tvd_mean |   degree_max_abs_drift_max |
|----------------:|----------------:|--------------------:|---------------------------:|-------------------------------:|--------------------------------:|-----------------------------------------:|-------------------------------:|----------------------:|---------------------------:|
|          0.0050 |         21.0000 |             63.0000 |                     0.0056 |                        -0.0024 |                          0.0136 |                                   0.0223 |                         0.0050 |                0.0000 |                     0.0000 |
|          0.0100 |         21.0000 |             63.0000 |                     0.0032 |                        -0.0039 |                          0.0103 |                                   0.0083 |                         0.0100 |                0.0000 |                     0.0000 |
|          0.0200 |         12.0000 |             36.0000 |                     0.0000 |                        -0.0090 |                          0.0090 |                                  -0.0063 |                         0.0200 |                0.0000 |                     0.0000 |

## Interpretation Boundary

This result is a sensitivity test, not a universal full-planar ensemble. R67 reports replicate-level stability,
stronger-swap feasibility, degree drift, length-bin drift, and structural mobility for the 21-city geometry subset.
However, it does not close the full 71-city high-intensity planar-null boundary. The safe claim is that the strict
geometry null absorbs part of the gap but leaves a measurable residual under the tested perturbation range.

## Outputs

- `data/R67_geometry_null_sensitivity/geometry_null_sensitivity_replicates.csv`
- `data/R67_geometry_null_sensitivity/geometry_null_sensitivity_city_summary.csv`
- `data/R67_geometry_null_sensitivity/geometry_null_sensitivity_fraction_summary.csv`
- `figures/Fig_R67_geometry_null_sensitivity.svg|pdf|png|tiff`

