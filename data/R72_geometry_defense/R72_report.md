# R72 Geometry-Null Defense

Generated: 2026-06-04 20:05:51

## Purpose

This analysis responds to a robustness objection about the full-city low-intensity geometry-null result:
the strict-geometry null might remain close to the original road graph because
the accepted-swap intensity is low. The round therefore compares spatial-scale
and strict-geometry nulls at matched swap-fraction targets on the 21-city
subset and reports mobility/fidelity diagnostics from the strict geometry
ensemble.

## Summary Metrics

```json
{
  "status": "pass",
  "round": "R72_geometry_defense",
  "matched_spatial_records": 63,
  "matched_city_records": 54,
  "swap_fractions": [
    0.005,
    0.01,
    0.02
  ],
  "baseline_fraction": 0.005,
  "baseline_n_cities": 21,
  "baseline_spatial_residual_mean": 0.05952380952380944,
  "baseline_geometry_residual_mean": 0.005555555555555543,
  "highest_fraction": 0.02,
  "highest_fraction_n_cities": 12,
  "highest_spatial_residual_mean": 0.08958333333333328,
  "highest_geometry_residual_mean": 8.673617379884035e-18,
  "r68_full71_geometry_residual_mean": 0.0014084507042253476,
  "r68_full71_geometry_residual_ci95_low": -0.0027487938167931316,
  "r68_full71_geometry_residual_ci95_high": 0.005565695225243827,
  "r68_full71_spatial_residual_mean": 0.09647887323943655,
  "geometry_acceptance_rate_mean": 0.0018958967823354616,
  "geometry_attempts_per_accepted_swap_mean": 547.2587301461137,
  "geometry_null_mobility_index_mean": 0.00691830721158937,
  "max_degree_drift": 0.0,
  "claim_boundary": "This analysis supports a matched-intensity empirical contrast on the 21-city subset, not full mixing or a theorem."
}
```

## Matched-Intensity Comparison

|   swap_fraction |   n_cities_matched |   road_minus_spatial_mean |   road_minus_spatial_ci95_low |   road_minus_spatial_ci95_high |   road_minus_geometry_mean |   road_minus_geometry_ci95_low |   road_minus_geometry_ci95_high |   mean_spatial_minus_geometry_residual |   positive_geometry_residual_share |   spatial_accepted_swaps_per_edge_mean |   geometry_accepted_swaps_per_edge_mean |   spatial_acceptance_rate_mean |   geometry_acceptance_rate_mean |   geometry_null_mobility_index_mean |   geometry_length_bin_tvd_mean |   geometry_degree_drift_max |
|----------------:|-------------------:|--------------------------:|------------------------------:|-------------------------------:|---------------------------:|-------------------------------:|--------------------------------:|---------------------------------------:|-----------------------------------:|---------------------------------------:|----------------------------------------:|-------------------------------:|--------------------------------:|------------------------------------:|-------------------------------:|----------------------------:|
|           0.005 |                 21 |                 0.0595238 |                     0.0491149 |                      0.0699327 |                0.00555556  |                    -0.00244082 |                       0.0135519 |                              0.0539683 |                           0.428571 |                             0.00499828 |                              0.00499828 |                      0.0271007 |                      0.00189129 |                          0.00321956 |                    0           |                           0 |
|           0.01  |                 21 |                 0.077381  |                     0.0672908 |                      0.0874711 |                0.0031746   |                    -0.00390702 |                       0.0102562 |                              0.0742063 |                           0.428571 |                             0.00999665 |                              0.00999665 |                      0.0277338 |                      0.00188003 |                          0.00611171 |                    1.19842e-06 |                           0 |
|           0.02  |                 12 |                 0.0895833 |                     0.0710339 |                      0.108133  |                8.67362e-18 |                    -0.0089912  |                       0.0089912 |                              0.0895833 |                           0.333333 |                             0.020001   |                              0.020001   |                      0.0271613 |                      0.00191637 |                          0.0114237  |                    0           |                           0 |

Interpretation: under the same accepted-swap targets, the spatial-scale null
retains a positive road-minus-null residual, whereas the strict non-crossing
geometry null remains near the observed road threshold at the tested estimator
resolution. This directly weakens the claim that the R68 result is only an
artifact of using a small perturbation.

## Mobility And Fidelity Diagnostics

|   swap_fraction |   n_replicates |   accepted_swaps_per_edge_mean |   acceptance_rate_mean |   attempts_per_accepted_swap |   topology_reject_share |   length_reject_share |   crossing_reject_share |   null_mobility_index_mean |   abs_delta_angular_entropy_mean |   abs_delta_straight_angle_share_mean |   abs_delta_right_angle_share_mean |   abs_delta_dual_endpoint_degree_cv_mean |   length_bin_tvd_mean |   degree_max_abs_drift_max |
|----------------:|---------------:|-------------------------------:|-----------------------:|-----------------------------:|------------------------:|----------------------:|------------------------:|---------------------------:|---------------------------------:|--------------------------------------:|-----------------------------------:|-----------------------------------------:|----------------------:|---------------------------:|
|           0.005 |             63 |                     0.00499828 |             0.00189129 |                      558.165 |               0.048108  |              0.917635 |               0.0342573 |                 0.00321956 |                       0.00690043 |                            0.00294481 |                         0.0045188  |                              9.0664e-05  |           0           |                          0 |
|           0.01  |             63 |                     0.00999665 |             0.00188003 |                      552.785 |               0.0482211 |              0.917501 |               0.0342778 |                 0.00611171 |                       0.0132613  |                            0.00578841 |                         0.00870286 |                              0.00014569  |           1.19842e-06 |                          0 |
|           0.02  |             36 |                     0.020001   |             0.00191637 |                      530.827 |               0.0502809 |              0.913814 |               0.0359047 |                 0.0114237  |                       0.0246701  |                            0.0111504  |                         0.0166328  |                              0.000191272 |           0           |                          0 |

The geometry null has low acceptance rates because most proposed moves violate
length or non-crossing constraints. That is expected for spatially embedded
road graphs and should be reported as a constraint, not hidden. The diagnostic
does not prove full mixing. It shows that accepted swaps occurred at the
registered targets, degree drift remained zero, edge-length bins were preserved
at reported precision and measured angular/dual mobility increased with swap
target.

## Submission-Safe Claim

At matched perturbation intensities in the 21-city sensitivity subset,
degree-preserving spatial-scale nulls leave a positive residual, whereas
strict non-crossing geometry nulls move close to the observed road thresholds
at the tested estimator resolution. Combined with the full 71-city low-intensity
screen, this supports local embedded-geometry constraints as a leading
mechanism candidate for the post-spatial residual.

## Still Not Claimed

- No full high-intensity multi-replicate geometry-null ensemble for all
  71 cities.
- No proof that the strict geometry null is fully mixed.
- No analytical planar theorem.
- No causal urban-design rule.
- No isolation of planarity alone from degree, length scale, angular and block
  structure.

