# R81 Full 71-city High-intensity Geometry-null Ensemble

## Purpose

R81 pushes the strict non-crossing geometry-null layer beyond the R68 full-city low-intensity screen. It targets all
71 standardized city windows at accepted-swap fraction(s) 0.01,0.02 with 3 replicate(s)
per fraction. The result is a full-coverage higher-intensity empirical null ensemble, not an analytical
planar-percolation theorem or causal urban-design law.

## Main Result

- City coverage: 71/71
- Replicate pass rate: 426/426
- Mean road-minus-geometry residual: -0.0018
  [-0.0043, 0.0006]
- Mean remaining fraction after geometry: -0.0117
- Mean road-minus-spatial residual before strict geometry: 0.0965
- Mean residual reduction versus spatial-scale null: 0.0983
- Maximum degree drift: 0.0
- Mean length-bin TVD: 0.000000
- Reused R67 same-algorithm records: 0

## Macro-region Summary

| macro_region   |   swap_fraction |   n_cities_total |   n_cities_pass |   road_minus_geometry_mean |   road_minus_geometry_ci95_low |   road_minus_geometry_ci95_high |   remaining_fraction_after_geometry_mean |   road_minus_spatial_mean |
|:---------------|----------------:|-----------------:|----------------:|---------------------------:|-------------------------------:|--------------------------------:|-----------------------------------------:|--------------------------:|
| Africa         |          0.0100 |               12 |              12 |                     0.0021 |                        -0.0074 |                          0.0115 |                                   0.0031 |                    0.1125 |
| Asia           |          0.0100 |               19 |              19 |                     0.0004 |                        -0.0071 |                          0.0080 |                                   0.0013 |                    0.1013 |
| Europe         |          0.0100 |               11 |              11 |                    -0.0015 |                        -0.0123 |                          0.0092 |                                  -0.0121 |                    0.0886 |
| Latin America  |          0.0100 |               10 |              10 |                    -0.0008 |                        -0.0087 |                          0.0070 |                                  -0.0051 |                    0.0975 |
| Middle East    |          0.0100 |                5 |               5 |                     0.0000 |                        -0.0089 |                          0.0089 |                                  -0.0027 |                    0.0900 |
| North America  |          0.0100 |               10 |              10 |                    -0.0058 |                        -0.0107 |                         -0.0009 |                                  -0.0309 |                    0.0850 |
| Oceania        |          0.0100 |                4 |               4 |                    -0.0167 |                        -0.0343 |                          0.0010 |                                  -0.0768 |                    0.0812 |
| Africa         |          0.0200 |               12 |              12 |                     0.0062 |                        -0.0044 |                          0.0169 |                                   0.0201 |                    0.1125 |
| Asia           |          0.0200 |               19 |              19 |                    -0.0022 |                        -0.0080 |                          0.0036 |                                  -0.0119 |                    0.1013 |
| Europe         |          0.0200 |               11 |              11 |                    -0.0023 |                        -0.0105 |                          0.0060 |                                  -0.0122 |                    0.0886 |
| Latin America  |          0.0200 |               10 |              10 |                     0.0025 |                        -0.0056 |                          0.0106 |                                   0.0106 |                    0.0975 |
| Middle East    |          0.0200 |                5 |               5 |                    -0.0050 |                        -0.0218 |                          0.0118 |                                  -0.0290 |                    0.0900 |
| North America  |          0.0200 |               10 |              10 |                    -0.0100 |                        -0.0163 |                         -0.0037 |                                  -0.0524 |                    0.0850 |
| Oceania        |          0.0200 |                4 |               4 |                    -0.0125 |                        -0.0207 |                         -0.0043 |                                  -0.0571 |                    0.0812 |

## Largest Positive Geometry Residuals

| city      | country              | macro_region   |   road_minus_geometry_mean |   road_minus_spatial |   n_edges |
|:----------|:---------------------|:---------------|---------------------------:|---------------------:|----------:|
| Nairobi   | Kenya                | Africa         |                     0.0500 |               0.1500 |     28157 |
| Hanoi     | Vietnam              | Asia           |                     0.0417 |               0.1500 |     39886 |
| Hong Kong | China                | Asia           |                     0.0417 |               0.0750 |     13245 |
| Nairobi   | Kenya                | Africa         |                     0.0417 |               0.1500 |     28157 |
| Hanoi     | Vietnam              | Asia           |                     0.0333 |               0.1500 |     39886 |
| Amsterdam | Netherlands          | Europe         |                     0.0333 |               0.1000 |     32887 |
| Dubai     | United Arab Emirates | Middle East    |                     0.0250 |               0.0750 |     22720 |
| Lagos     | Nigeria              | Africa         |                     0.0250 |               0.1250 |     17952 |
| Lisbon    | Portugal             | Europe         |                     0.0250 |               0.1250 |     36745 |
| Sao Paulo | Brazil               | Latin America  |                     0.0250 |               0.1000 |     77333 |
| Sao Paulo | Brazil               | Latin America  |                     0.0250 |               0.1000 |     77333 |
| Vienna    | Austria              | Europe         |                     0.0167 |               0.0750 |     26142 |

## Manuscript-safe Interpretation

R81 upgrades the evidence from low-intensity full-city coverage to a higher-intensity multi-replicate strict
geometry-null screen across the full 71-city matrix. The safe claim is that strict local non-crossing geometry sharply
reduces the residual left by edge-scale controls even when the constrained null is pushed harder than R68. The unsafe
claim remains forbidden: R81 is not a Markov-chain mixing proof, not an analytical proof and not a causal
urban-design law.
