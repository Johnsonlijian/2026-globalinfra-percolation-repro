"""Recalculate band-dependent retention without refitting the locked allowance."""

import numpy as np
import pandas as pd

from common import Context, parser


def band_measures(predicted_loss, reference_loss, q, a, b):
    if not 0 <= a < b <= 1 or q < 0:
        raise ValueError("Invalid target band or allowance")
    width = b - a
    accept = np.maximum(np.minimum(b, predicted_loss - q) - a, 0) / width
    reject = np.maximum(b - np.maximum(a, predicted_loss + q), 0) / width
    unsafe = np.maximum(
        np.minimum(b, predicted_loss - q) - np.maximum(a, reference_loss), 0
    ) / width
    conservative = np.maximum(
        np.minimum(b, reference_loss) - np.maximum(a, predicted_loss + q), 0
    ) / width
    retained = accept + reject
    return {
        "retained": float(retained.mean()),
        "retained_accept": float(accept.mean()),
        "retained_reject": float(reject.mean()),
        "conditional_disagreement": float((unsafe + conservative).sum() / retained.sum())
        if retained.sum() else None,
        "interior_escalation_intervals": int(
            ((predicted_loss - q >= a) & (predicted_loss + q <= b)).sum()
        ),
    }


def main(context):
    external = context.csv("external")
    q = context.json("reference_calibration")["q95_rank"]
    assert len(external) == 21
    predicted = 1 - external.pc_dimension.to_numpy()
    reference = 1 - external.pc.to_numpy()
    primary = tuple(context.config["settings"]["target_band"])
    bands = [("locked_primary", *primary), ("post_hoc_full", 0, 1),
             ("post_hoc_lower", .15, .35), ("post_hoc_upper", .35, .55)]
    rows = [
        {"scope": name, "a": a, "b": b, "q_unchanged": q,
         **band_measures(predicted, reference, q, a, b)}
        for name, a, b in bands
    ]
    for row in rows:
        assert 0 <= row["retained"] <= 1
        assert np.isclose(row["retained_accept"] + row["retained_reject"], row["retained"])
    assert rows[0]["interior_escalation_intervals"] == 19
    assert np.isclose(rows[0]["retained"], context.json("external_summary")["retained"])
    assert all(row["conditional_disagreement"] == 0 for row in rows)
    synthetic = band_measures(np.array([.5]), np.array([.5]), .1, 0, 1)
    assert np.isclose(synthetic["retained"], .8)
    assert synthetic["conditional_disagreement"] == 0
    assert np.isclose(band_measures(np.array([.5]), np.array([.2]), .1, 0, 1)
                      ["conditional_disagreement"], .25)
    context.write_csv("retention_band_sensitivity.csv", pd.DataFrame(rows))
    context.finish(__file__)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main(Context(parser(__doc__).parse_args()))
