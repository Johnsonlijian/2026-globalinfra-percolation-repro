# Finite-Network Percolation Screening

Public code, provenance records and selected derived data for graph-based
infrastructure screening. This repository is the citable reproducibility
record accompanying the revised manuscript **"Unsafe-side decision errors in
infrastructure robustness screening: finite-network controls and external
city-holdout transfer"** submitted to *Reliability Engineering & System
Safety*.

| Item | Value |
|---|---|
| Release tag | `v1.1.1-r215` |
| GitHub repository | <https://github.com/Johnsonlijian/2026-globalinfra-percolation-repro> |
| Zenodo concept DOI | [10.5281/zenodo.21102470](https://doi.org/10.5281/zenodo.21102470) |
| Release archive | The version DOI is assigned by Zenodo to the tagged release; use the concept DOI above for the latest version. |
| License | MIT for code; see `DATASETS_AND_LINKS.csv` for third-party source terms |

The tagged release is the authoritative public code-and-derived-data version
for this manuscript. It contains only reproducibility materials and selected
outputs that can be redistributed with the paper.

## What Runs Without Private Data

With Python 3.12 and the dependencies in requirements.txt:

```text
python scripts/check.py
python scripts/analyze.py --data-root . --output generated --associations
python scripts/external_split.py --data-root . --output generated
python scripts/retention_sensitivity.py --data-root . --output generated
python scripts/local_difference_sensitivity.py --data-root . --output generated
python scripts/make_core_figures.py --data-root . --output generated
python scripts/make_length_control_figure.py --data-root . --output generated
python scripts/make_paper_assets.py --data-root . --output generated
python scripts/audit_package.py --data-root . --output generated
```

The table/figure commands use only included scalar derived inputs and a fixed
produced map layer. They neither load source graphs nor access the network.
Figure 1 is rebuilt from fig1_response.csv plus the actual produced vector
map; no raw OSM node/edge-coordinate table is distributed. All six figures
are exported as PDF, SVG and PNG. Figure 2b retains its upper-left legend.

The core estimator has an unaccelerated fallback; the additional
component-observable diagnostic requires Numba, included in requirements.txt.
The unaccelerated fallback changes execution speed, not the equations or
random-number stream. Arial is used when installed; otherwise the bundled
Matplotlib DejaVu Sans is used. Font fallback can change layout, not data.

## Included Evidence

- 91 primary simple graphs: 71 roads, 10 power cases in six source families,
  six rail networks, and four original water benchmarks.
- 284 ordinary and 142 connectivity-repaired degree-null realizations, each
  estimated using the declared 256-order, exact-endpoint protocol.
- 21 external road cities, held out from the 71-city reference calibration.
  A retrospective provenance stratification distinguishes 19 primary cities
  from Stockholm and Cape Town, both used in earlier project work.
- Three additional water benchmarks, making seven water sources in total.
  C-Town's endpoint estimate belongs to this additional scope.
- 312 fixed-count stress-test records and 36 clean length-control stages.
  Every included length-control stage is a simple graph with zero self-loops.
- Conditional local-control difference ranges, with bootstrap peak-count
  distributions permitting their regeneration without raw graphs.
- Comparable whole-graph rail crossing summaries, road exposure summaries,
  and supplementary scalar window/RTS checks.
- 272 freshly computed order/switch-budget records, 91 alternative-response
  records and a separate mapped SciGRID model diagnostic. Radial subsets
  inside existing windows do not measure outward-window bias.

DATA_MANIFEST.json binds the included inputs by SHA256, fields and row counts.
data/source_registry.csv records 148 author-held snapshot contracts: 91
primary, 21 external, and 36 clean control stages. Those graph bytes are not
included. Source links and rights boundaries are in DATASETS_AND_LINKS.csv.

## What Requires Author-Held Snapshots

rerun_topology.py validates node count, edge count, graph simplicity, canonical
topology hash, snapshot byte hash, edge ordering where recorded, and settings.
It can then rerun the core estimator on primary/external snapshots or saved
clean-control stages. It also contains the extracted degree-swap and
connectivity-repair helpers for the reference-road null protocol.

run_scenarios.py provides the static targeted algorithm. Geographic scenarios
additionally require the author's verified coordinates and a hash/CRS/node-order
contract. Missing water geography is not replaced by schematic coordinates.

The archive does **not** regenerate OSM acquisition, rail/road polyline
measurement, clean-control swap trajectories, water source-model conversion,
or inner-window extraction from the internet. These data-preparation stages
require their source snapshots and corresponding provenance. Reestimating a
saved clean stage is not regenerating the stage's swap trajectory.
Additional-water and RTS/window results can be plotted and tabulated from the
included derived data; their original source adapters are not claimed to be
packaged. A new web query is not a byte-identical snapshot replay.

## Interpretation

All 19 primary and both prior-use external errors lie within the unchanged
reference-only allowance. Retention is 70.4% for the primary cohort and 70.3%
for all 21 cities. These are measures over the uniform target interval [.15,.55],
not a fraction of cities, a reliability probability, or zero deployment risk.
It includes retained acceptances and rejections and depends on the target
band width. Local-control terminal difference ranges all include zero;
their consistent negative point direction is not a resolved decrease in
every chain at the recorded numerical budget.
Power inference uses six families rather than ten independent systems.
Crossing associations and paired-city long/short intervals remain exploratory.
Water abstract planarity is not verified geographic embedding or a measured
zero crossing density.

The protocol/ directory preserves timestamped external-cohort acquisition and
calibration records made before the response calculation; it is not a public
preregistration.
The 19/2 provenance stratification and constant-offset comparisons are post
hoc. Shared and predictor-specific error allowances are reported separately.

See REPRODUCIBLE_RUNBOOK.md for commands, settings, tests and limitations.
No manuscript, author-response document, literature PDF, internal round log,
private path, credential, raw graph, or original benchmark model is included.
