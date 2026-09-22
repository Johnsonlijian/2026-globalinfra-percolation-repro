# Reproducible Runbook

## Environment

Use Python 3.12.10 (tested). Install requirements.txt in a separate environment.
NumPy 2.3.5, SciPy 1.16.3, pandas 2.3.3, NetworkX 3.6.1, Matplotlib 3.10.9,
statsmodels 0.14.6, Shapely 2.1.2, PyProj 3.7.2 and PyMuPDF 1.28.0 were used
for the local checks. Numba 0.67.0 and h5py 3.16.0 support the additional
component-observable and mapped-power diagnostics.
Changing graph/runtime versions requires rechecking graph and edge-order
hashes before claiming bytewise replay.

All plotting and table scripts accept --data-root, --config and --output.
Paths inside config/reproduction.json are relative to --data-root, not to
an author's workstation. Default configuration and data
are resolved relative to the installed scripts, so another working directory
is supported. No script performs downloads.

## Self-Contained Table And Figure Regeneration

Run from the extracted attachment root:

```text
python -B scripts/check.py
python -B scripts/analyze.py --data-root . --output generated --associations
python -B scripts/external_split.py --data-root . --output generated
python -B scripts/retention_sensitivity.py --data-root . --output generated
python -B scripts/local_difference_sensitivity.py --data-root . --output generated
python -B scripts/make_core_figures.py --data-root . --output generated
python -B scripts/make_length_control_figure.py --data-root . --output generated
python -B scripts/make_paper_assets.py --data-root . --output generated
python -B scripts/audit_package.py --data-root . --output generated
```

analyze.py recomputes instance losses, family-weighted summaries, reference-only
allowance, leave-region-out decisions, matched null contrasts, external
decisions, asymmetric loss curves, scenario aggregation and, with
--associations, Freedman-Lane/paired-city inference. Supplied summaries are
used as regression expectations, not substituted for these calculations.
The tests compare regenerated tables to their included values.

external_split.py reproduces the retrospective 19-city primary cohort,
the two prior-use cases and the full 21-city roster. It also reports the
constant-offset comparator with a shared numerical allowance and with its
own 69th ordered reference error. No main-rule allowance is refitted.

retention_sensitivity.py keeps the reference-calibrated allowance unchanged
and reproduces retained acceptances, retained rejections and target-band
sensitivities. Only [0.15,0.55] is the locked primary band; the full, lower
and upper bands are post hoc weighting checks, not new validation cohorts.

local_difference_sensitivity.py reproduces conditional stage-minus-baseline
percentile ranges from the included bootstrap peak-count distributions.
It uses the product of independent stage distributions, not paired edge
orders or additional Monte Carlo simulations.

Figure scripts default to the bound included tables in data/. To plot the
freshly regenerated summaries, use config/regenerated_figures.json after
analyze.py. The numerical values are regression-checked in either route.

Outputs: generated/*.csv/json, generated/tables/*.tex, generated/facts.tex,
and generated/figures/fig1 through fig6 as PDF/SVG/PNG. TeX files contain
numeric bindings and preamble row macros (MainErrorRows and MainValidationRows),
not manuscript prose. Load the table files in the document preamble and use
their macros inside the corresponding tabular environment with booktabs.
Figure 1 composes an
authorized fixed map layer with a newly drawn response/decision plot.
The map is not claimed to be reconstructed from raw OSM geometry here.

## Snapshot Validation Without Monte Carlo

Place the author-held snapshots under a separate directory, retaining the
relative snapshot_name values from data/source_registry.csv.

```text
python -B scripts/rerun_topology.py --data-root $PRIMARY_SNAPSHOTS --cohort reference --validate-only --output generated/snapshot_checks
python -B scripts/rerun_topology.py --data-root $EXTERNAL_SNAPSHOTS --cohort external --allow-trusted-pickle --validate-only --output generated/external_checks
python -B scripts/rerun_topology.py --data-root $CLEAN_STAGES --cohort length_control --validate-only --output generated/stage_checks
```

The pickle switch is explicit because loading pickle can execute code. Use
only a trusted author-held source. There is no pickle or NPZ in the archive.
A missing snapshot is reported as required input, never silently downloaded.

Remove --validate-only to run the declared estimator. A single-instance run
can use --instance Net3 or --limit 1. Use a new output directory; existing
estimate files are not overwritten. A graph validation report is not an MC
rerun. For reference roads, --kind degree or --kind connected reconstructs
the specified null realization, verifies its hash before MC, and uses its
recorded seed and 256-order setting.

The helper preserves historical iteration/edge-order behavior. Set iteration,
NetworkX changes, different normalized node labels or a fresh snapshot can
produce a different realization even when coarse topology statistics agree;
the hash check must fail rather than conceal that divergence.

## Fixed-Count Scenarios

```text
python -B scripts/run_scenarios.py --data-root $PRIMARY_SNAPSHOTS --instance Net3 --output $SCENARIO_OUTPUT
```

This runs three static approximate edge-betweenness rankings, not adaptive
recomputed targeting. Optional --coordinates and --coordinate-contract add
12 localized orders for verified road/rail geography only. The private
coordinate NPZ must contain xy in frozen integer-node order. Its separate
JSON contract states input_hash, coordinate_sha256, source_crs=EPSG:4326 and
node_order=frozen_integer_labels. Coordinates are projected to local AEQD
metres. Ranking uses edge endpoint-chord midpoints, not polyline hazard
intersections. No coordinates or coordinate table are included.

## Numerical Contract

- Canonical response: binomial conversion of fixed-count LCC trajectories.
  Kernel tails are truncated at 1e-14 and their total mass is checked.
- Diagnostic: first maximum of the mean analytic derivative on p=0:.0025:1.
  This is a finite-graph diagnostic, not an infinite-network critical point.
- The p=1 derivative is computed exactly from bridges and applied to every
  order before the peak/bootstrap calculation. The resulting order bootstrap
  is conditional on this fixed graph and exact endpoint treatment.
- Primary/external estimates: 8192 orders, 1000 bootstrap replicates.
  Nulls: 256 orders; clean saved stages: 512 orders. All use step .0025.
- Calibration: reference-only ceil((71+1)*.95)-th absolute error;
  q=0.061536786614714734. External cities never enter calibration.
- City/family bootstrap: 10000 draws, seed 20260905.
- Association permutations: 10000 Freedman-Lane draws; eight tests form one
  BH/Holm family. Long/short uses paired city resampling, not within-city swaps.
- Power means use six source families. Scenario replication does not enlarge
  the independent-city sample size.
- Random versus scenario AUC uses fixed retained-edge fraction, not one
  canonical curve versus one fixed-count curve. Random uses 401 sampled
  points; other scenarios integrate all counts. The absolute quadrature
  comparison bound is .00125; the saved scenario difference was <=.000431593.
- Primary 91 point estimates are interior. C-Town pc=1 in additional-water
  scope is retained. Do not erase endpoints or overstate interval coverage.
- Primary rail exposure uses whole-graph proper nonadjacent polyline crossing
  pairs without the 15-degree cutoff. LCC/15-degree outputs are sensitivities.
  Water planarity is not assigned numerical geographic crossing density.

## Scope And Release

Additional diagnostic reruns require matching author-held graph snapshots:

```text
python -B scripts/control_budget_verified.py --data data --graphs $REFERENCE_SNAPSHOTS --output $CONTROL_OUTPUT
python -B scripts/response_definition_verified.py --data data --graphs $REFERENCE_SNAPSHOTS --output $RESPONSE_OUTPUT
python -B scripts/power_scigrid_check.py --network $SCIGRID_NETWORK --output $SCIGRID_OUTPUT
```

The first command generates fresh degree-preserving controls for 24 cities
at five nested order budgets and eight cities at four successful-switch
budgets. It saves private rewired graphs outside the attachment. These are
budget diagnostics, not a uniform-ensemble or mixing certificate. The second
command tests the component-size bookkeeping against NetworkX at every
occupation step of small graphs before calculating the 91-graph diagnostic.
The third uses one UTM32N endpoint chord per simple edge; it does not measure
line-route crossings or utility operating reliability. Its source file is
identified in DATASETS_AND_LINKS.csv. These computations are distinct from
self-contained regeneration of derived tables and figures.

All supplied empirical files are derived scalar tables, source identity
metadata, or explicitly authorized produced artwork. Excluded: original
source files, raw snapshots, geographic edge lists, literature PDFs,
manuscripts/replies, credentials and internal logs.
No raw source is relicensed by this attachment.

Supplementary_Code.zip is the authoritative code and derived-data version
accompanying the article. This release contains only the reproducibility
materials and selected derived outputs distributed with the paper.
