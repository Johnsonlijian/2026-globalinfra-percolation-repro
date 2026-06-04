# R56 spatial-length constrained null models

## Purpose

R56 adds a real spatial-scale constraint to the R41 degree-preserving null. Each null replicate preserves the degree sequence exactly and uses double-edge swaps that are accepted only when new edge chord lengths remain in the original length slots within the configured bin tolerance.

This is an empirical spatial null, not a theorem-level planar null. It tests whether adding spatial edge-scale information materially narrows the gap between configuration-like degree nulls and observed road percolation thresholds.

## Aggregate result

- Cities: 8.
- Mean observed road gap: 0.2096.
- Mean R41 degree-null gap: 0.0255.
- Mean R56 degree + spatial-scale null gap: 0.0694.
- Mean increment over degree-only null: 0.0438.
- Mean remaining road gap after spatial null: 66.8%.
- Mean spatial-null fraction of road gap: 33.2%.
- Mean accepted swaps per edge: 0.250.
- Mean acceptance rate: 2.790%.
- Mean chord-length bin total-variation distance: 0.0793.
- Runtime: 5.9 min.

## Manuscript-safe interpretation

The safe claim is that degree sequence alone is insufficient and that adding a spatial edge-scale constraint shifts the null threshold toward the observed road threshold, but does not in itself license a universal law, a causal mechanism, or a strict planar-null equivalence. The residual after this null should be described as the part not reproduced by degree sequence plus the tested spatial scale constraint.

## City summary

| City | road gap | degree null gap | spatial null gap | remaining after spatial | swaps/edge | length TVD | status |
|---|---:|---:|---:|---:|---:|---:|---|
| Berlin | 0.2100 | 0.0260 | 0.0675 | 67.9% | 0.250 | 0.0791 | pass |
| Buenos Aires | 0.1751 | 0.0198 | 0.0600 | 65.7% | 0.250 | 0.0793 | pass |
| Chicago | 0.2166 | 0.0248 | 0.0525 | 75.8% | 0.250 | 0.0793 | pass |
| Madrid | 0.2009 | 0.0247 | 0.0775 | 61.4% | 0.250 | 0.0793 | pass |
| Rotterdam | 0.2126 | 0.0317 | 0.0825 | 61.2% | 0.250 | 0.0794 | pass |
| Stockholm | 0.2264 | 0.0319 | 0.0875 | 61.4% | 0.250 | 0.0793 | pass |
| Sydney | 0.2095 | 0.0190 | 0.0625 | 70.2% | 0.250 | 0.0792 | pass |
| Toronto | 0.2256 | 0.0265 | 0.0650 | 71.2% | 0.250 | 0.0791 | pass |

## Boundary

- Uses cached live-OSMnx road graphs for the eight cities with complete node coordinates.
- The spatial constraint is based on node-to-node chord length because rewired null edges have no road geometry.
- The algorithm preserves degree exactly; it preserves the empirical length-bin distribution approximately.
- The result should not be phrased as a full planar constrained null or as proof of a mechanism.

