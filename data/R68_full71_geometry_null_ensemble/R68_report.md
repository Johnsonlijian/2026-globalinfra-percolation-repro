# R68 Full 71-city Geometry-null Ensemble

## Purpose

R68 pushes the strict non-crossing geometry-null layer from the 21-city R67 sensitivity subset to all 71 standardized
city windows. It targets one conservative low-intensity geometry-null replicate per city at accepted-swap fraction
0.005. The result is a full-coverage empirical null ensemble, not a high-intensity or analytical
planar-percolation theorem.

## Main Result

- City coverage: 71/71
- Replicate pass rate: 71/71
- Mean road-minus-geometry residual: 0.0014
  [-0.0027, 0.0056]
- Mean remaining fraction after geometry: 0.0042
- Mean road-minus-spatial residual before strict geometry: 0.0965
- Mean residual reduction versus spatial-scale null: 0.0951
- Maximum degree drift: 0.0
- Mean length-bin TVD: 0.000000
- Reused R67 same-algorithm records: 21

## Macro-region Summary

| macro_region   |   swap_fraction |   n_cities_total |   n_cities_pass |   road_minus_geometry_mean |   road_minus_geometry_ci95_low |   road_minus_geometry_ci95_high |   remaining_fraction_after_geometry_mean |   road_minus_spatial_mean |
|:---------------|----------------:|-----------------:|----------------:|---------------------------:|-------------------------------:|--------------------------------:|-----------------------------------------:|--------------------------:|
| Africa         |          0.0050 |               12 |              12 |                     0.0062 |                        -0.0044 |                          0.0169 |                                   0.0201 |                    0.1125 |
| Asia           |          0.0050 |               19 |              19 |                     0.0026 |                        -0.0079 |                          0.0132 |                                   0.0117 |                    0.1013 |
| Europe         |          0.0050 |               11 |              11 |                     0.0023 |                        -0.0057 |                          0.0102 |                                   0.0085 |                    0.0886 |
| Latin America  |          0.0050 |               10 |              10 |                     0.0025 |                        -0.0089 |                          0.0139 |                                   0.0111 |                    0.0975 |
| Middle East    |          0.0050 |                5 |               5 |                    -0.0050 |                        -0.0148 |                          0.0048 |                                  -0.0244 |                    0.0900 |
| North America  |          0.0050 |               10 |              10 |                     0.0000 |                        -0.0073 |                          0.0073 |                                  -0.0011 |                    0.0850 |
| Oceania        |          0.0050 |                4 |               4 |                    -0.0125 |                        -0.0266 |                          0.0016 |                                  -0.0593 |                    0.0812 |

## Largest Positive Geometry Residuals

| city          | country     | macro_region   |   road_minus_geometry_mean |   road_minus_spatial |   n_edges |
|:--------------|:------------|:---------------|---------------------------:|---------------------:|----------:|
| Hanoi         | Vietnam     | Asia           |                     0.0500 |               0.1500 |     39886 |
| Hong Kong     | China       | Asia           |                     0.0500 |               0.0750 |     13245 |
| Nairobi       | Kenya       | Africa         |                     0.0500 |               0.1500 |     28157 |
| Hyderabad     | India       | Asia           |                     0.0250 |               0.1250 |    132094 |
| Montevideo    | Uruguay     | Latin America  |                     0.0250 |               0.0750 |     22450 |
| Panama City   | Panama      | Latin America  |                     0.0250 |               0.1000 |     12302 |
| Lagos         | Nigeria     | Africa         |                     0.0250 |               0.1250 |     17952 |
| Sao Paulo     | Brazil      | Latin America  |                     0.0250 |               0.1000 |     77333 |
| Vancouver     | Canada      | North America  |                     0.0250 |               0.0750 |     22751 |
| Dar es Salaam | Tanzania    | Africa         |                     0.0250 |               0.1500 |     29826 |
| Warsaw        | Poland      | Europe         |                     0.0250 |               0.1000 |     24341 |
| Amsterdam     | Netherlands | Europe         |                     0.0250 |               0.1000 |     32887 |

## Manuscript-safe Interpretation

R68 upgrades the evidence from a 21-city subset to full 71-city low-intensity strict geometry-null coverage. This
directly answers the reviewer objection that the geometry-null result was only a small subset. The safe claim is that
strict local non-crossing geometry sharply reduces the residual left by edge-scale controls across the full city-window
matrix. The unsafe claim remains forbidden: R68 is not a high-intensity multi-replicate planar ensemble, not an
analytical proof and not a causal urban-design law.
