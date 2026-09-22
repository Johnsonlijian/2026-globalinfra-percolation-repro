# Finite-Network Percolation Screening

Code, configuration, derived tables and generated figures for graph-based
infrastructure robustness screening.

**This repository is the public, citable record of the reproducibility package
for the revised manuscript "Unsafe-side decision errors in infrastructure
robustness screening: finite-network controls and external city-holdout
transfer" (Reliability Engineering & System Safety).**

| Item | Value |
|---|---|
| Latest release tag | `v1.1.0-r214` (22 September 2026) |
| Version DOI (cite this release) | [10.5281/zenodo.22886920](https://doi.org/10.5281/zenodo.22886920) |
| Concept DOI (all versions) | [10.5281/zenodo.21102470](https://doi.org/10.5281/zenodo.21102470) |
| Zenodo record | <https://zenodo.org/records/22886920> |
| Release page | <https://github.com/Johnsonlijian/2026-globalinfra-percolation-repro/releases/tag/v1.1.0-r214> |
| Earlier releases | `v1.0.2-r209` ([10.5281/zenodo.21102471](https://doi.org/10.5281/zenodo.21102471), R198 version DOI) and earlier tags |
| License | MIT for code; see `DATASETS_AND_LINKS.csv` for third-party source terms |

This release is archived on Zenodo under the version DOI above and can be cited
directly. To cite the evolving package as a whole, use the concept DOI. Cite
whichever DOI the manuscript states, so that the manuscript and the archive
stay aligned.

## What Runs Without Private Data

With Python 3.12 and the dependencies in `requirements.txt`:

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

The table and figure commands use only the included scalar derived inputs and a
fixed produced map layer. They neither load source graphs nor access the network.
Figure 1 is rebuilt from `data/fig1_response.csv` plus the produced vector map;
no raw OpenStreetMap node or edge-coordinate table is distributed. All six
figures are exported as PDF, SVG and PNG.

The core estimator has an unaccelerated fallback. The component-observable
diagnostic additionally requires Numba, which is listed in `requirements.txt`.
The fallback changes execution speed, not the equations or the random-number
stream. Arial is used when installed; otherwise the bundled Matplotlib
DejaVu Sans is used. Font fallback can change layout, not data.

## Included Evidence

- 91 primary simple graphs: 71 roads, 10 power cases in six source families, six
  rail networks and four original water benchmarks.
- 284 ordinary and 142 connectivity-repaired degree-null realizations, each
  estimated with the declared 256-order, exact-endpoint protocol.
- 21 acquired external road cities held out from the 71-city reference
  calibration. A retrospective provenance stratification distinguishes 19
  primary cities from two cities used in earlier project work, which are
  disclosed and reported separately.
- Three additional water benchmarks, making seven water sources in total.
- Fixed-count stress-test records and 36 clean length-control stages, each a
  simple graph with zero self-loops.
- Conditional local-control difference ranges with the bootstrap peak-count
  distributions that permit their regeneration without raw graphs.
- Comparable whole-graph rail crossing summaries, road exposure summaries and
  supplementary scalar window and synthetic-power checks.
- Regenerated order- and switch-budget control records, alternative
  response-definition records and a mapped transmission-model diagnostic.

`DATA_MANIFEST.json` binds the included inputs by SHA-256, fields and row counts.
`data/source_registry.csv` records the author-held snapshot contracts. Those
graph bytes are not included. Source links and rights boundaries are in
`DATASETS_AND_LINKS.csv`.

## What Requires Author-Held Snapshots

`scripts/rerun_topology.py` validates node count, edge count, graph simplicity,
canonical topology hash, snapshot byte hash, edge ordering where recorded, and
settings. It can then rerun the core estimator on primary or external snapshots
or on saved clean-control stages.

`scripts/run_scenarios.py` provides the static targeted-removal algorithm.
Geographic scenarios additionally require verified coordinates and a hash, CRS
and node-order contract. Missing water geography is not replaced by schematic
coordinates.

The archive does not regenerate source acquisition, rail or road polyline
measurement, clean-control swap trajectories, water source-model conversion, or
inner-window extraction from the internet. Those stages require their source
snapshots and provenance. Re-estimating a saved clean stage is not regenerating
that stage's swap trajectory. A new web query is not a byte-identical snapshot
replay.

## Interpretation

Errors of the external cities and of the two previously used cities lie within
the unchanged reference-only allowance; retention is measured over the uniform
target interval [0.15, 0.55] and is not a fraction of cities, a reliability
probability or zero deployment risk. It includes retained acceptances and
rejections and depends on the target-band width. Local-control terminal
difference ranges include zero; their consistent negative point direction is not
a resolved decrease in every chain at the recorded numerical budget. Power
inference uses six families rather than ten independent systems. Crossing
associations and paired-city long/short intervals remain exploratory. Water
abstract planarity is not verified geographic embedding or a measured zero
crossing density.

The `protocol/` directory preserves timestamped external-cohort acquisition and
calibration records made before the response calculation; it is not a public
preregistration. The provenance stratification and the constant-offset
comparison are post hoc.

See `REPRODUCIBLE_RUNBOOK.md` for commands, settings, tests and limitations, and
`RELEASE_NOTES_R214.md` for what changed in this release. No manuscript,
author-response document, literature PDF, internal round log, private path,
credential, raw graph or original benchmark model is included.
