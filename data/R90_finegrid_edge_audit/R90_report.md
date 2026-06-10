# R90 Fine-grid and Edge-overlap Audit

## Main result

- Passing null-threshold records: 84/84
- Cities covered: 21
- Mean fine road-minus-spatial residual: 0.0406
- Mean fine road-minus-geometry residual: -0.0032
- Mean fine spatial-minus-geometry separation: 0.0438
- Mean exact geometry edge Jaccard: 0.9714
- Mean geometry LCC-curve L2 distance: 0.0324

## Matched pair summary

|   swap_fraction |   n_pairs |   fine_road_minus_spatial_mean |   fine_road_minus_geometry_mean |   fine_spatial_minus_geometry_mean |   geometry_edge_jaccard_mean |
|----------------:|----------:|-------------------------------:|--------------------------------:|-----------------------------------:|-----------------------------:|
|          0.0050 |   21.0000 |                         0.0376 |                         -0.0048 |                             0.0424 |                       0.9806 |
|          0.0100 |   21.0000 |                         0.0436 |                         -0.0017 |                             0.0452 |                       0.9622 |

## Claim boundary

R90 addresses two reviewer concerns: whether the near-zero geometry residual is a coarse-grid artifact, and whether the
strict geometry null is merely an unmeasured copy of the observed road graph. It regenerates nulls and records exact
edge-overlap metrics for the 21-city geometry subset. It does not prove full Markov-chain mixing, identify a causal
urban-design rule or create a closed-form planar-percolation theorem.
