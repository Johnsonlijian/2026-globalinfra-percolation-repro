"""Split the acquired external cohort and add a reference-only shift comparator.

Two additions to the locked external evaluation, both computed from derived
outputs without recalculating any response curve:

1. Stockholm and Cape Town have documented earlier project use. A retrospective
   provenance audit separates these two cases from the 19-city primary cohort.
   All 21 acquired cases remain reported under the unchanged locked rule.
2. A post hoc constant-offset comparator: the mean CEBH gap of the reference
   roads is added to every prediction. It is calibrated on the reference cohort
   only, is not prespecified, and exists to show whether the mean-degree
   reference is the only way to remove the level bias.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from common import Context, parser
from screening import allowance, bootstrap_mean, evaluate_allowance, integrated_errors

SENSITIVITY_CASES = ("Cape Town", "Stockholm")
SEED = 20260905
LOW, HIGH = 0.15, 0.55
PAIRED_MAE_NOTE = (
    "Paired city-bootstrap difference of external mean absolute error, "
    "shift comparator minus mean-degree reference; 10000 draws."
)


def cohort_summary(frame: pd.DataFrame, q: float, label: str) -> dict:
    """Locked-rule outcomes for one cohort under the unchanged allowance."""
    predicted = 1 - frame.pc_dimension.to_numpy()
    reference = 1 - frame.pc.to_numpy()
    error = np.abs(frame.pc.to_numpy() - frame.pc_dimension.to_numpy())
    mean, low, high = bootstrap_mean(error)
    unsafe, conservative = integrated_errors(predicted, reference, LOW, HIGH)
    decisions = evaluate_allowance(predicted, reference, q, LOW, HIGH)
    covered = error <= q
    interval = binomtest(int(covered.sum()), len(frame)).proportion_ci()
    retained_acceptance = float(
        np.maximum(
            0,
            np.minimum(HIGH, np.minimum(1 - frame.pc_dimension - q, 1 - frame.pc)) - LOW,
        ).mean()
        / (HIGH - LOW)
    )
    retained_rejection = float(
        np.maximum(
            0,
            HIGH - np.maximum(LOW, np.maximum(1 - frame.pc_dimension + q, 1 - frame.pc)),
        ).mean()
        / (HIGH - LOW)
    )
    # The escalation interval is [1 - h - q, 1 - h + q] in target-loss coordinates.
    # It lies wholly inside the band when retention equals 1 - 2q/(b - a).
    loss_index = 1 - frame.pc_dimension.to_numpy()
    interior = int(((loss_index - q >= LOW) & (loss_index + q <= HIGH)).sum())
    return {
        "cohort": label,
        "n": int(len(frame)),
        "q": q,
        "mae": mean,
        "mae_low": low,
        "mae_high": high,
        "max_abs_error": float(error.max()),
        "unsafe": float(unsafe.mean()),
        "conservative": float(conservative.mean()),
        "covered": int(covered.sum()),
        "coverage": float(covered.mean()),
        "coverage_binomial_low": float(interval.low),
        "coverage_binomial_high": float(interval.high),
        "retained": float(decisions.retained.mean()),
        "retained_low": float(bootstrap_mean(decisions.retained.to_numpy())[1]),
        "retained_high": float(bootstrap_mean(decisions.retained.to_numpy())[2]),
        "escalated": float(decisions.escalated.mean()),
        "unsafe_retained": float(decisions.unsafe_retained.mean()),
        "safe_retained": float(decisions.safe_retained.mean()),
        "retained_acceptance": retained_acceptance,
        "retained_rejection": retained_rejection,
        "interior_escalation_intervals": interior,
        "interval_note": (
            "Descriptive binomial interval under a working independent-trial "
            "assumption; the quota cohort is not a deployment population."
        ),
    }


def predictor_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """All four predictors on one cohort, for the split sensitivity check."""
    rows = []
    for name, column in (
        ("CEBH", "pc_cebh"),
        ("2D", "pc_dimension"),
        ("junction", "pc_junction_anchor"),
        ("nonbacktracking", "pc_nonbacktracking"),
    ):
        error = np.abs(frame.pc - frame[column]).to_numpy()
        mean, low, high = bootstrap_mean(error)
        unsafe, conservative = integrated_errors(
            1 - frame[column].to_numpy(), 1 - frame.pc.to_numpy(), LOW, HIGH
        )
        rows.append(
            {
                "predictor": name,
                "n": int(len(frame)),
                "mae": mean,
                "mae_low": low,
                "mae_high": high,
                "unsafe": float(unsafe.mean()),
                "unsafe_low": float(bootstrap_mean(unsafe)[1]),
                "unsafe_high": float(bootstrap_mean(unsafe)[2]),
                "conservative": float(conservative.mean()),
                "conservative_low": float(bootstrap_mean(conservative)[1]),
                "conservative_high": float(bootstrap_mean(conservative)[2]),
            }
        )
    return pd.DataFrame(rows)


def shift_comparator(road: pd.DataFrame, external: pd.DataFrame, q: float) -> tuple[pd.DataFrame, dict]:
    """Reference-only constant offset added to the degree-moment prediction."""
    offset = float((road.pc - road.pc_cebh).mean())
    rows = []
    for label, frame in (("reference", road), ("external_primary", external)):
        prediction = frame.pc_cebh.to_numpy() + offset
        truth = frame.pc.to_numpy()
        error = np.abs(truth - prediction)
        mean, low, high = bootstrap_mean(error)
        unsafe, _ = integrated_errors(1 - prediction, 1 - truth, LOW, HIGH)
        decisions = evaluate_allowance(1 - prediction, 1 - truth, q, LOW, HIGH)
        retained = float(decisions.retained.mean())
        rows.append(
            {
                "cohort": label,
                "predictor": "CEBH+reference_offset",
                "n": int(len(frame)),
                "offset": offset,
                "q": q,
                "mae": mean,
                "mae_low": low,
                "mae_high": high,
                "covered": int((error <= q).sum()),
                "coverage": float((error <= q).mean()),
                "unsafe": float(unsafe.mean()),
                "retained": retained,
                "unsafe_retained": float(decisions.unsafe_retained.mean()),
                "safe_retained": float(decisions.safe_retained.mean()),
                "conditional_unsafe": float(
                    decisions.unsafe_retained.sum() / decisions.retained.sum()
                ),
            }
        )
    shifted_error = np.abs(
        external.pc.to_numpy() - (external.pc_cebh.to_numpy() + offset)
    )
    reference_error = np.abs(external.pc.to_numpy() - external.pc_dimension.to_numpy())
    difference = shifted_error - reference_error
    draws = np.random.default_rng(20260905).integers(
        0, len(difference), (10000, len(difference))
    )
    means = difference[draws].mean(axis=1)
    summary = {
        "offset": offset,
        "q": q,
        "paired_external_mae_difference_shift_minus_2d": float(difference.mean()),
        "paired_external_mae_difference_low": float(np.quantile(means, 0.025)),
        "paired_external_mae_difference_high": float(np.quantile(means, 0.975)),
        "paired_difference_note": PAIRED_MAE_NOTE,
        "interpretation": (
            "Post hoc reference-only comparator fitted on the 71 reference roads; "
            "not prespecified, not a refitted primary predictor, and not an "
            "alternative screening rule."
        ),
    }
    return pd.DataFrame(rows), summary


def main(context):
    q = context.json("reference_calibration")["q95_rank"]
    external = context.csv("external")
    reference = context.csv("observed")
    road = reference[reference.domain.eq("road")].copy()
    if len(external) != 21 or not set(SENSITIVITY_CASES) <= set(external.instance):
        raise ValueError("Expected all 21 acquired cities and both prior-use cases")
    primary = external[~external.instance.isin(SENSITIVITY_CASES)].copy()
    sensitivity = external[external.instance.isin(SENSITIVITY_CASES)].copy()

    split_summary = pd.DataFrame(
        [
            cohort_summary(primary, q, "external_primary19"),
            cohort_summary(sensitivity, q, "external_sensitivity2"),
            cohort_summary(external, q, "external_acquired21"),
        ]
    )
    primary_predictors = predictor_metrics(primary)
    acquired_predictors = predictor_metrics(external)
    shift_table, shift_summary = shift_comparator(road, primary, q)
    split_note = {
        "sensitivity_cases": list(SENSITIVITY_CASES),
        "primary_n": int(len(primary)),
        "acquired_n": int(len(external)),
        "locked_allowance": q,
        "allowance_recomputed": False,
        "cohort_role": (
            "All 21 acquired cities were evaluated under the locked rule. The reported "
            "primary cohort contains 19 cities with no earlier use identified in the "
            "project's manuscript history. Stockholm was in the earlier eight-city "
            "analysis; Cape Town was an earlier Geofabrik diagnostic scope check. "
            "This provenance stratification was made after outcome calculation, not "
            "pre-declared. Both prior-use cases and the full roster remain reported. "
            "No exclusion is based on prediction error; predictor and allowance are unchanged."
        ),
    }
    for name, frame in (
        ("external_cohort_split_summary.csv", split_summary),
        ("external_primary_predictors.csv", primary_predictors),
        ("external_predictors.csv", acquired_predictors),
        ("external_shift_comparator.csv", shift_table),
    ):
        context.write_csv(name, frame)
        frame.to_csv(context.root / "data" / name, index=False)
    for name, value in (
        ("external_shift_summary.json", shift_summary),
        ("external_split_note.json", split_note),
    ):
        context.write_json(name, value)
        (context.root / "data" / name).write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    shift_q = allowance(np.abs(road.pc - (road.pc_cebh + shift_summary["offset"])))
    ranked, ranked_summary = shift_comparator(road, primary, shift_q)
    ranked["allowance_basis"] = "69th ordered reference error for shifted predictor"
    context.write_csv("external_shift_equal_rank.csv", ranked)
    ranked.to_csv(context.root / "data/external_shift_equal_rank.csv", index=False)
    context.finish(__file__)
    print(split_summary.to_string(index=False))
    print()
    print(shift_table.to_string(index=False))


if __name__ == "__main__":
    main(Context(parser(__doc__).parse_args()))
