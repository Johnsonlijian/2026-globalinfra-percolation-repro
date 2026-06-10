# R95/N95 Non-backtracking Spectral Layer

## Main result

- Passing city records: 71/71
- Mean CEBH absolute error: 0.2166
- Mean non-backtracking absolute error: 0.3190
- Mean NB improvement over CEBH: -0.1024
- Spearman pc_NB vs road threshold: 0.5898
- Spearman pc_CEBH vs road threshold: 0.9362

## Macro-region summary

| macro_region   |   n |   road_gap_cebh_mean |   road_minus_nb_mean |   nb_improvement_mean |
|:---------------|----:|---------------------:|---------------------:|----------------------:|
| Africa         |  12 |               0.2348 |               0.3567 |               -0.1219 |
| Asia           |  19 |               0.2192 |               0.3282 |               -0.1090 |
| Europe         |  11 |               0.2180 |               0.3201 |               -0.1021 |
| Latin America  |  10 |               0.2108 |               0.3029 |               -0.0921 |
| Middle East    |   5 |               0.2103 |               0.3264 |               -0.1162 |
| North America  |  10 |               0.1965 |               0.2655 |               -0.0691 |
| Oceania        |   4 |               0.2184 |               0.3235 |               -0.1051 |

## Claim boundary

The non-backtracking layer is a theory-bridge diagnostic motivated by
message-passing percolation thresholds. It tests whether graph-spectral
branching alone closes the road threshold gap. It does not replace the
constrained geometry-null evidence, prove full mixing, or establish a causal
urban-design rule.

## Iteration stability

Iteration-stability audit: 71/71 records pass; mean absolute pc_NB change from 160 to 640 iterations = 0.000136; maximum change = 0.000911.
