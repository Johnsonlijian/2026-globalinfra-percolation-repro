# N99 Compact Transfer-Error Correction Screen

## Purpose

This round tests a bounded empirical correction screen:

`p_c_pred = p_c_CEBH + f(edge scale, articulation exposure, biconnected structure, cycle density, angular order)`.

The correction uses only precomputed road-form and embedded-structure covariates.
It does not use observed thresholds, spatial-null thresholds, geometry-null
thresholds or non-backtracking thresholds as predictors.

## Headline results

- Cities: 71
- Primary model: degree_form_compact
- Primary feature count: 4
- CEBH MAE: 0.2166
- Compact correction LOO MAE: 0.0150
- Compact correction LOO improvement versus CEBH: 0.2016
- Compact correction leave-region-out MAE: 0.0146
- Compact correction leave-region-out gap R2: 0.4218
- Permutation baseline: observed LOO MAE is lower than 100.0% of shuffled-label runs
- Permutation baseline: observed leave-region-out MAE is lower than 100.0% of shuffled-label runs

## Model comparison

| model                                           |   n |   mae_gap |   rmse_gap |    r2_gap |   mae_pc |   rmse_pc |    r2_pc |   mae_improvement_vs_cebh |
|:------------------------------------------------|----:|----------:|-----------:|----------:|---------:|----------:|---------:|--------------------------:|
| CEBH                                            |  71 |    0.2166 |     0.2179 |  -79.0846 |   0.2166 |    0.2179 | -13.8763 |                    0.0000 |
| kappa-only LOO                                  |  71 |    0.0153 |     0.0192 |    0.3793 |   0.0153 |    0.0192 |   0.8847 |                    0.2012 |
| compact degree-form correction LOO              |  71 |    0.0150 |     0.0189 |    0.3968 |   0.0150 |    0.0189 |   0.8880 |                    0.2016 |
| compact degree-form correction 5-fold           |  71 |    0.0149 |     0.0186 |    0.4143 |   0.0149 |    0.0186 |   0.8912 |                    0.2017 |
| compact degree-form correction leave-region-out |  71 |    0.0146 |     0.0185 |    0.4218 |   0.0146 |    0.0185 |   0.8926 |                    0.2020 |
| full embedded correction LOO                    |  71 |    0.0174 |     0.0214 |    0.2300 |   0.0174 |    0.0214 |   0.8570 |                    0.1991 |
| full embedded correction leave-region-out       |  71 |    0.0154 |     0.0200 |    0.3274 |   0.0154 |    0.0200 |   0.8751 |                    0.2012 |
| non-backtracking spectral proxy                 |  71 |    0.3190 |     0.3226 | -174.4928 |   0.3190 |    0.3226 | -31.5992 |                   -0.1024 |

## Leave-region-out detail

| macro_region   |   n_test |   mae_gap |   rmse_gap |   r2_gap |   cebh_mae |
|:---------------|---------:|----------:|-----------:|---------:|-----------:|
| Africa         |       12 |    0.0164 |     0.0204 |   0.3121 |     0.2348 |
| Asia           |       19 |    0.0185 |     0.0223 |   0.3664 |     0.2192 |
| Europe         |       11 |    0.0165 |     0.0187 |   0.3509 |     0.2180 |
| Latin America  |       10 |    0.0075 |     0.0104 |   0.6194 |     0.2108 |
| Middle East    |        5 |    0.0165 |     0.0242 |  -1.9082 |     0.2103 |
| North America  |       10 |    0.0090 |     0.0111 |   0.0155 |     0.1965 |
| Oceania        |        4 |    0.0147 |     0.0148 |  -1.9627 |     0.2184 |

## Claim boundary

Cross-validated empirical transfer-error correction using only precomputed embedded road-form covariates. It is not a causal urban-design rule, not a universal law and not an analytical theorem.
