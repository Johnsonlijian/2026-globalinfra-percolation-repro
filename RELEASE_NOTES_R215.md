# Release notes: v1.1.1-r215

Date: 26 September 2026

This tagged public release accompanies the R1 revision of the manuscript
"Unsafe-side decision errors in infrastructure robustness screening:
finite-network controls and external city-holdout transfer" submitted to
*Reliability Engineering & System Safety*.

## Scope

- Reproducible code, configuration, source registries, derived tables and
  publication figures for the revised analysis.
- The finite-network estimator with exact endpoint treatment, finite
  degree-preserving controls, the locked reference allowance and external
  city-holdout evaluation.
- Response-definition checks, order/switch-budget ladders, window and failure
  process sensitivities, and the figure/table builders.

## Reproducibility boundary

The included scalar derived outputs regenerate the reported decision tables and
figures without downloading data. Raw OpenStreetMap extracts, power, rail and
water source files, private graph snapshots, credentials, manuscripts,
reviewer responses and internal round records are not included. Their source
locations, versions and access conditions are documented in
`DATASETS_AND_LINKS.csv` and the source registry.

## Verification

- `python -B scripts/check.py` passed.
- `PYTHONPATH=scripts python -B -m pytest -q` passed with 20 tests.
- The runbook replay regenerated the decision tables and six vector figures.
- `scripts/audit_package.py` passed with stable input hashes and no embedded
  raster images in the audited figure PDFs.
