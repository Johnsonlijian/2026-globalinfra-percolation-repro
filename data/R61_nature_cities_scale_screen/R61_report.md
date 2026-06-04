# R61 Nature Cities Scale Screen

Generated: 2026-06-03 19:58:26

## Purpose

R61 tests whether the CEBH road-threshold gap survives a larger, more globally
distributed sample when every city is analysed through the same OSMnx
centre-window extraction rule. This is a scale-expansion screen, not a
replacement for the R56/R60 city-boundary and geometry-null evidence.

## Method

- Extraction: OSMnx live/cache drive-road graph from the geocoded city centre.
- Window: 10,000 m radius, bounding-box distance, edge truncation retained.
- Threshold: R56-family rank-order bond percolation.
- Monte Carlo: 6 rank-order permutations; 80 bootstrap resamples.
- Grid: p=0.100 to 0.900, step=0.025.

## Results

- Attempted city windows: 71
- Passed city windows: 71
- Failed/skipped windows: 0
- Passed macro-regions: 7 (Africa, Asia, Europe, Latin America, Middle East, North America, Oceania)
- Total passed-window nodes: 2,492,958
- Total passed-window edges: 3,557,249
- Mean road gap: 0.217
- Median road gap: 0.212
- Positive-gap fraction: 1.000

Top gap city windows:

- Nairobi, Kenya: gap=0.282, nodes=22,960, edges=28,157
- Bangkok, Thailand: gap=0.281, nodes=63,366, edges=71,694
- Dar es Salaam, Tanzania: gap=0.267, nodes=22,312, edges=29,826
- Hyderabad, India: gap=0.262, nodes=98,461, edges=132,094
- Hanoi, Vietnam: gap=0.261, nodes=29,320, edges=39,886
- London, United Kingdom: gap=0.255, nodes=48,353, edges=62,602
- Lagos, Nigeria: gap=0.255, nodes=13,226, edges=17,952
- Lisbon, Portugal: gap=0.254, nodes=26,686, edges=36,745

Regional pass counts:

```json
{
  "Africa": 12,
  "Asia": 19,
  "Europe": 11,
  "Latin America": 10,
  "Middle East": 5,
  "North America": 10,
  "Oceania": 4
}
```

## Nature Cities Target-Fit Reading

Current R61 scale status: **credible scale layer, still needing formal nulls and urban mechanism analysis**.

The strengthened message is that the phenomenon is not confined to the original
eight city-boundary cases. However, the Nature Cities-level argument still
requires a larger and more explicitly urban-science layer: at least 30-60
method-equivalent city windows, covariates for urban form and development
history, and formal null-model replication on a representative subset. The
current R61 output should therefore support manuscript triage and figure
planning, while the main claim remains bounded to "road networks across sampled
cities show a persistent positive gap beyond degree-only theory."

## Failed Windows

- None.

## Outputs

- `data/R61_nature_cities_scale_screen/central_window_screen_results.csv`
- `data/R61_nature_cities_scale_screen/R61_summary.json`
- `figures/Fig_R61_nature_cities_scale_screen.pdf`
- `figures/Fig_R61_nature_cities_scale_screen.svg`
- `figures/Fig_R61_nature_cities_scale_screen.png`
- `figures/Fig_R61_nature_cities_scale_screen.tiff`

