"""Reconstruct q-independent local difference ranges from sparse bootstrap counts.

Only included derived CSVs are used. Omitted grid bins have zero count. Counts
retain the empirical bootstrap distributions, not raw graphs or edge orders.
"""

from __future__ import annotations

import json

import numpy as np
from common import Context, parser

PROBABILITIES = np.array([0.025, 0.975])


def product_quantiles(stage_p, stage_counts, baseline_p, baseline_counts):
    """Linear empirical quantiles without expanding the count-weighted product."""
    arrays = []
    for values, counts in ((stage_p, stage_counts), (baseline_p, baseline_counts)):
        values, counts = np.asarray(values, dtype=float), np.asarray(counts)
        if (
            values.ndim != 1 or not values.size or counts.shape != values.shape
            or not np.isfinite(values).all() or not np.isfinite(counts).all()
            or np.any((values < 0) | (values > 1))
            or np.any(counts <= 0) or np.any(counts != counts.astype(np.int64))
        ):
            raise ValueError("Expected finite peak support with positive integer counts")
        arrays.append((values, counts.astype(np.int64)))
    (stage_p, stage_counts), (baseline_p, baseline_counts) = arrays
    values = np.subtract.outer(stage_p, baseline_p).ravel()
    weights = np.multiply.outer(stage_counts, baseline_counts).ravel()
    order = np.argsort(values, kind="stable")
    values, cumulative = values[order], np.cumsum(weights[order], dtype=np.int64)
    ranks = PROBABILITIES * (int(cumulative[-1]) - 1)
    lower, upper = np.floor(ranks).astype(np.int64), np.ceil(ranks).astype(np.int64)
    # The first cumulative count exceeding a zero-based rank contains that value.
    left = values[np.searchsorted(cumulative, lower, side="right")]
    right = values[np.searchsorted(cumulative, upper, side="right")]
    return left + (ranks - lower) * (right - left)


def self_test():
    examples = (
        ([0, 1], [1, 1], [0, 1], [1, 1]),
        ([0, 1], [1, 1], [0, 0.25, 0.75], [1, 1, 1]),
        ([0, 1], [1, 3], [0, 0.5, 1], [2, 1, 1]),
        ([1], [10], [0.5, 1], [1, 9]),
    )
    for stage, n_stage, baseline, n_baseline in examples:
        expanded = np.subtract.outer(np.repeat(stage, n_stage), np.repeat(baseline, n_baseline))
        expected = np.quantile(expanded, PROBABILITIES, method="linear")
        np.testing.assert_allclose(
            product_quantiles(stage, n_stage, baseline, n_baseline), expected, rtol=0, atol=1e-15
        )
        np.testing.assert_allclose(
            product_quantiles(stage[::-1], n_stage[::-1], baseline[::-1], n_baseline[::-1]),
            expected, rtol=0, atol=1e-15,
        )
    np.testing.assert_allclose(product_quantiles([0, 1], [1, 1], [0, 1], [1, 1]), [-0.925, 0.925])
    for counts in ([0], [-1], [1.5], [np.nan]):
        try:
            product_quantiles([0], counts, [1], [1])
        except ValueError:
            continue
        raise AssertionError("Invalid frequency counts were accepted")
    return {"expanded_product_cases": len(examples), "permutation_checks": len(examples),
            "unpaired_distribution_check": True, "invalid_count_checks": 4}


def main(context):
    checks = self_test()
    if context.config["inputs"]["local_difference_ranges"] == "generated/local_difference_ranges.csv":
        raise ValueError("Use config/reproduction.json: regression must check the included CSV, not its generated output")
    counts = context.csv("local_bootstrap_peak_counts")
    reference = context.csv("local_difference_ranges")
    keys = ["graph", "seed", "stage_index"]
    if len(reference) != 36 or reference.duplicated(keys).any() or counts.duplicated(keys + ["grid_index"]).any():
        raise ValueError("Expected 36 unique stages and unique sparse grid bins")
    groups = dict(tuple(counts.groupby(keys, sort=True)))
    if set(groups) != set(reference[keys].itertuples(index=False, name=None)):
        raise ValueError("Count distributions and reference stages differ")
    grid = np.linspace(0, 1, 401)
    for key, group in groups.items():
        indices = group.grid_index.to_numpy()
        if (
            not np.isfinite(indices).all() or np.any(indices != indices.astype(int))
            or np.any((indices < 0) | (indices > 400))
            or group["count"].sum() != 1000 or not group["count"].gt(0).all()
            or not group["count"].eq(group["count"].astype(int)).all()
        ):
            raise ValueError(f"Invalid sparse distribution: {key}")
        np.testing.assert_allclose(group.p, grid[indices.astype(int)], rtol=0, atol=1e-15)
        stage_seed = key[1] + (7 if key[2] == 0 else 17 + key[2] * 1009)
        if not group.estimator_seed.eq(stage_seed).all():
            raise ValueError(f"Unexpected independent-stage estimator seed: {key}")
    regenerated = reference.copy()
    for index, row in reference.iterrows():
        key = (row.graph, row.seed, row.stage_index)
        stage, baseline = groups[key], groups[(row.graph, row.seed, 0)]
        if row.stage_index == 0:
            low, high = 0.0, 0.0
        else:
            low, high = product_quantiles(stage.p, stage["count"], baseline.p, baseline["count"])
        regenerated.loc[index, ["difference_low", "difference_high"]] = [low, high]
        regenerated.loc[index, "contains_zero"] = bool(low <= 0 <= high)
    fields = ["difference_low", "difference_high"]
    np.testing.assert_allclose(regenerated[fields], reference[fields], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(regenerated.contains_zero, reference.contains_zero)
    if not reference.calibration_q_used.eq(False).all():
        raise ValueError("These numerical ranges must not use calibration q")
    context.write_csv("local_difference_ranges.csv", regenerated)
    context.write_json("local_difference_sensitivity_checks.json", {
        "status": "PASS_COUNT_PRODUCT_REGRESSION", "stages": 36,
        "identity_baselines": 9, "nonbaseline_products": 27,
        "sparse_count_rows": len(counts), "counts_per_stage": 1000,
        "product_weight_per_nonbaseline_stage": 1_000_000,
        "quantiles": PROBABILITIES.tolist(), "quantile_method": "linear",
        "max_absolute_regression_error": float(np.max(np.abs((regenerated[fields] - reference[fields]).to_numpy()))),
        "calibration_q_used": False, "new_edge_orders": 0, "new_bootstrap_resamples": 0,
        "baseline_convention": "Same estimator minus itself: exactly [0, 0].",
        "uncertainty_scope": "Pointwise numerical uncertainty conditional on realized stage graphs, not population or simultaneous coverage.",
        "synthetic_checks": checks,
    })
    context.finish(__file__)
    print("Rebuilt 36 local difference ranges from sparse bootstrap counts; included-range regression passed.")


if __name__ == "__main__":
    command = parser(__doc__)
    command.add_argument("--self-test", action="store_true", help="Run only synthetic checks; write nothing")
    args = command.parse_args()
    if args.self_test:
        print(json.dumps({"status": "PASS", "checks": self_test()}, indent=2))
    else:
        main(Context(args))
