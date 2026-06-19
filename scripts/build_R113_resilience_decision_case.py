"""R113 (gate G2): a named resilience-decision flip from the baseline choice.

R109 showed the degree-moment (CEBH) benchmark overstates the tolerable random-loss
fraction f_c = 1 - p_c in all 81 networks. This round turns that screening result
into one concrete, named planning decision, to make the consequence actionable
rather than statistical.

Decision model (standard topological robustness proxy; stated caveats apply):
an operator sets a robustness target phi* - the network must retain a giant
connected component after random loss of a fraction phi* of links. The tolerable
loss estimated from the degree-moment benchmark is f_c^CEBH = 1 - p_c^CEBH; the
geometry-aware baseline gives f_c = 1 - 2/<k> (and the directly observed value is
f_c^obs = 1 - p_c^obs). A DECISION FLIP occurs when the degree-moment benchmark
clears a network (f_c^CEBH >= phi*) that the geometry-aware/observed baseline does
not (f_c^obs < phi*): the operator would wrongly conclude "no reinforcement needed".

This is a screening implication on the standard proxy, NOT a validation against
external outage records, and it concerns absolute safety margins, not the relative
fragility ranking (which the degree-moment formula still preserves; see R109).

Uses the registered R109 population table only; no third-party raw data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
POP = ROOT / "data" / "R109_resilience_screening" / "R109_resilience_population.csv"
OUT = ROOT / "data" / "R113_resilience_decision_case"
ROUND = ROOT / "rounds" / "R113_resilience_decision_case"

TARGETS = [0.30, 0.40, 0.50]
HEADLINE_PHI = 0.40


def flips_at(d, phi):
    cebh_pass = d["fc_cebh"] >= phi
    true_fail = d["fc_obs"] < phi
    derived_fail = d["fc_derived"] < phi
    flip = cebh_pass & true_fail
    flip_vs_derived = cebh_pass & derived_fail
    return {
        "phi_star": phi,
        "cebh_clears": int(cebh_pass.sum()),
        "flip_vs_observed": int(flip.sum()),
        "flip_vs_derived": int(flip_vs_derived.sum()),
        "flip_share_of_population": float(flip.mean()),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ROUND.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(POP).copy()
    d["margin_cebh"] = d["fc_cebh"] - HEADLINE_PHI       # operator's apparent margin (degree moments)
    d["margin_true"] = d["fc_obs"] - HEADLINE_PHI         # actual margin (observed)
    d["margin_derived"] = d["fc_derived"] - HEADLINE_PHI  # geometry-aware margin
    d["flip"] = (d["fc_cebh"] >= HEADLINE_PHI) & (d["fc_obs"] < HEADLINE_PHI)

    sweep = [flips_at(d, p) for p in TARGETS]

    # Named example: a flipped network with a large, clear sign change, preferring a
    # recognizable case and a sizeable margin reversal.
    flipped = d[d["flip"]].copy()
    flipped["reversal"] = flipped["margin_cebh"] - flipped["margin_true"]
    flipped = flipped.sort_values("reversal", ascending=False)
    example = flipped.iloc[0].to_dict() if len(flipped) else None

    summary = {
        "decision_model": (
            "Target phi*: keep a giant component after random loss of a fraction phi* of links. "
            "Degree-moment estimate f_c^CEBH=1-p_c^CEBH vs geometry-aware f_c=1-2/<k> and observed "
            "f_c^obs=1-p_c^obs. Flip = CEBH clears (f_c^CEBH>=phi*) but observed fails (f_c^obs<phi*)."
        ),
        "headline_phi_star": HEADLINE_PHI,
        "n_population": int(len(d)),
        "sweep": sweep,
        "flip_at_headline": int(d["flip"].sum()),
        "flip_share_at_headline": float(d["flip"].mean()),
        "caveats": (
            "Screening implication on the standard topological robustness proxy, not an external "
            "outage validation; concerns absolute safety margins, not the relative fragility ranking "
            "(degree moments preserve ranking, Spearman ~0.84, see R109)."
        ),
    }
    if example is not None:
        summary["named_example"] = {
            "network": example["city"], "domain": example["domain"],
            "mean_degree": round(float(example["mean_degree"]), 3),
            "pc_cebh": round(float(example["pc_cebh"]), 3),
            "pc_observed": round(float(example["pc_obs"]), 3),
            "fc_cebh": round(float(example["fc_cebh"]), 3),
            "fc_observed": round(float(example["fc_obs"]), 3),
            "fc_derived": round(float(example["fc_derived"]), 3),
            "apparent_margin_degree_moments": round(float(example["margin_cebh"]), 3),
            "actual_margin_observed": round(float(example["margin_true"]), 3),
            "geometry_aware_margin": round(float(example["margin_derived"]), 3),
            "reading": (
                "At phi*=%.2f the degree-moment benchmark shows a +%.2f safety margin (apparently passes, "
                "no reinforcement), whereas the observed threshold gives a %.2f margin (fails the target) "
                "and the geometry-aware baseline 2/<k> gives %.2f - flipping the decision from "
                "'no action' to 'reinforce'."
            ) % (HEADLINE_PHI, example["margin_cebh"], example["margin_true"], example["margin_derived"]),
        }

    d_out = d[["city", "domain", "mean_degree", "pc_obs", "pc_cebh", "fc_obs", "fc_cebh",
               "fc_derived", "margin_cebh", "margin_true", "margin_derived", "flip"]]
    d_out.to_csv(OUT / "R113_decision_margins.csv", index=False)
    (OUT / "R113_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


    print("[R113] decision target phi*=%.2f on %d networks" % (HEADLINE_PHI, len(d)), flush=True)
    for s in sweep:
        print("   phi*=%.2f: CEBH clears %d, flips vs observed %d (%.0f%%), vs derived %d" % (
            s["phi_star"], s["cebh_clears"], s["flip_vs_observed"],
            100 * s["flip_share_of_population"], s["flip_vs_derived"]), flush=True)
    if example is not None:
        e = summary["named_example"]
        print("[R113] named example: %s (%s) <k>=%.2f  fc_CEBH=%.2f vs fc_obs=%.2f -> %s" % (
            e["network"], e["domain"], e["mean_degree"], e["fc_cebh"], e["fc_observed"],
            "DECISION FLIP at phi*=%.2f" % HEADLINE_PHI), flush=True)


if __name__ == "__main__":
    main()
