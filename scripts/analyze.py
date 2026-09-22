"""Regenerate tables from included derived per-instance data; no raw data or MC."""

from __future__ import annotations

import numpy as np
import pandas as pd
from canonical_percolation import strict_flips
from common import Context, parser
from scipy.stats import binomtest
from screening import (
    PREDICTORS,
    allowance,
    bootstrap_mean,
    bootstrap_pair,
    coefficient_pair,
    evaluate_allowance,
    freedman_lane,
    integrated_errors,
)
from statsmodels.stats.multitest import multipletests


def run(context, associations=False):
    settings = context.config["settings"]
    low, high = settings["target_band"]
    if (low, high) != (0.15, 0.55):
        raise ValueError("This reproduction configuration binds the declared target band")
    seed = settings["bootstrap_seed"]
    ref, ext = context.csv("observed"), context.csv("external")
    if len(ref) != 91 or len(ext) != 21:
        raise ValueError("Cohort counts differ")
    details, metrics = [], []
    targets = np.linspace(low, high, 17)
    for domain, group in ref.groupby("domain"):
        for name, column in PREDICTORS.items():
            pred, truth = 1 - group[column].to_numpy(), 1 - group.pc.to_numpy()
            u, c = integrated_errors(pred, truth, low, high)
            ru, _ = integrated_errors(pred, 1 - group.pc_low.to_numpy(), low, high)
            _, rc = integrated_errors(pred, 1 - group.pc_high.to_numpy(), low, high)
            gu, gc = strict_flips(pred, truth, targets)
            fu, fc = integrated_errors(pred, truth, 0, 1)
            frame = pd.DataFrame(
                {
                    "instance": group.instance.to_numpy(),
                    "system_id": group.system_id.to_numpy(),
                    "domain": domain,
                    "predictor": name,
                    "absolute_error": abs(group.pc - group[column]).to_numpy(),
                    "signed_error": (group.pc - group[column]).to_numpy(),
                    "unsafe_integrated": u,
                    "safe_integrated": c,
                    "flip_integrated": u + c,
                    "unsafe_interval_separated": ru,
                    "safe_interval_separated": rc,
                    "grid_unsafe": gu.mean(axis=1),
                    "grid_safe": gc.mean(axis=1),
                    "full_band_unsafe": fu,
                    "full_band_safe": fc,
                }
            )
            columns = frame.select_dtypes(include="number").columns
            units = (
                frame.groupby("system_id")[columns].mean() if domain == "power" else frame[columns]
            )
            row = {
                "domain": domain,
                "predictor": name,
                "n_instances": len(group),
                "n_inference_units": len(units),
            }
            for col in columns:
                row[col], row[col + "_low"], row[col + "_high"] = bootstrap_mean(
                    units[col], seed=seed
                )
            details.append(frame)
            metrics.append(row)
    metrics = pd.DataFrame(metrics)
    context.write_csv("domain_metrics.csv", metrics)
    context.write_csv("decision_instances.csv", pd.concat(details))
    roads = ref[ref.domain == "road"].merge(
        context.csv("road_regions"), on="instance", validate="one_to_one"
    )
    q = allowance(abs(roads.pc - roads.pc_dimension).to_numpy())
    if not np.isclose(q, context.json("reference_calibration")["q95_rank"], atol=1e-14, rtol=0):
        raise ValueError("Reference-only calibration no longer matches")
    loo = []
    for region in sorted(roads.macro_region.unique()):
        train, held = roads[roads.macro_region != region], roads[roads.macro_region == region]
        rq = allowance(abs(train.pc - train.pc_dimension).to_numpy())
        part = evaluate_allowance(1 - held.pc_dimension, 1 - held.pc, rq, low, high)
        part["instance"] = held.instance.to_numpy()
        part["macro_region"], part["reference_q"] = region, rq
        part["error_covered"] = abs(held.pc - held.pc_dimension).to_numpy() <= rq
        loo.append(part)
    context.write_csv("leave_region_out.csv", pd.concat(loo))
    null_summaries = []
    for kind in ("degree", "connected"):
        raw = context.csv("null_" + kind)
        counts = raw.groupby("instance", as_index=False).agg(
            pc_null=("pc", "mean"), endpoint_count=("point_endpoint", "sum"), n_null=("pc", "size")
        )
        paired = roads.merge(counts, on="instance", validate="one_to_one")
        paired["null_gap"] = paired.pc_null - paired.pc_cebh
        paired["organization_residual"] = paired.pc - paired.pc_null
        context.write_csv(kind + "_paired.csv", paired)
        row = {"kind": kind, "n_cities": len(paired), "n_nulls": int(paired.n_null.sum())}
        for column in ("null_gap", "organization_residual"):
            row[column], row[column + "_low"], row[column + "_high"] = bootstrap_mean(
                paired[column], seed=seed
            )
        null_summaries.append(row)
    context.write_csv("null_summary.csv", pd.DataFrame(null_summaries))
    predictors = []
    for name, column in PREDICTORS.items():
        u, c = integrated_errors(1 - ext[column], 1 - ext.pc, low, high)
        row = {"predictor": name, "n": len(ext)}
        for metric, values in (
            ("mae", abs(ext.pc - ext[column])),
            ("unsafe", u),
            ("conservative", c),
        ):
            row[metric], row[metric + "_low"], row[metric + "_high"] = bootstrap_mean(
                values, seed=seed
            )
        predictors.append(row)
    context.write_csv("external_predictors.csv", pd.DataFrame(predictors))
    decisions = evaluate_allowance(1 - ext.pc_dimension, 1 - ext.pc, q, low, high)
    decisions["instance"], decisions["macro_region"] = ext.instance, ext.macro_region
    decisions["absolute_error"] = abs(ext.pc - ext.pc_dimension)
    decisions["covered"] = decisions.absolute_error <= q
    context.write_csv("external_decisions.csv", decisions)
    coverage_ci = binomtest(int(decisions.covered.sum()), len(ext)).proportion_ci()
    external_summary = {
        "n": len(ext),
        "q": q,
        "covered": int(decisions.covered.sum()),
        "coverage": float(decisions.covered.mean()),
        "retained": float(decisions.retained.mean()),
        "escalated": float(decisions.escalated.mean()),
        "unsafe_retained": float(decisions.unsafe_retained.mean()),
        "safe_retained": float(decisions.safe_retained.mean()),
        "coverage_binomial_low": float(coverage_ci.low),
        "coverage_binomial_high": float(coverage_ci.high),
        "interpretation": "Retained measure over the uniform target band, not cities or zero deployment risk.",
    }
    context.write_json("external_summary.json", external_summary)
    power = metrics[metrics.domain == "power"].set_index("predictor")
    lam = (power.loc["2D", "safe_integrated"] - power.loc["junction", "safe_integrated"]) / (
        power.loc["junction", "unsafe_integrated"] - power.loc["2D", "unsafe_integrated"]
    )
    context.write_json(
        "loss_tradeoff.json",
        {
            "power_junction_2d_break_even_unsafe_cost": float(lam),
            "loss_definition": "lambda*unsafe+conservative; six-family means; illustrative uniform target band",
        },
    )
    losses = []
    for cost in (1, 2, 5, 10):
        for row in metrics.to_dict("records"):
            losses.append(
                {
                    "domain": row["domain"],
                    "predictor": row["predictor"],
                    "lambda": cost,
                    "loss": cost * row["unsafe_integrated"] + row["safe_integrated"],
                }
            )
    context.write_csv("asymmetric_losses.csv", pd.DataFrame(losses))
    scenarios = context.csv("scenarios")
    context.write_csv(
        "scenario_instances.csv",
        scenarios.groupby(["domain", "instance", "system_id", "scenario"], as_index=False).agg(
            auc_change=("auc_change", "median"),
            endpoint_count=("point_endpoint", "sum"),
            scenario_auc=("auc_fractional", "median"),
            random_auc=("random_fixed_count_auc", "first"),
        ),
    )
    if associations:
        geometry = context.csv("road_geometry")
        tests = []
        for exposure in ("chi_poly", "chi_annotation_supported"):
            for controls in (
                [],
                ["mean_degree"],
                ["mean_degree", "log_edges"],
                ["mean_degree", "log_edges", "log_gdp"],
            ):
                frame = geometry.dropna(subset=[exposure, "road_gap_fine", *controls])
                rho, p = freedman_lane(
                    frame[exposure].to_numpy(),
                    frame.road_gap_fine.to_numpy(),
                    frame[controls].to_numpy(),
                    212 + len(tests) * 1009,
                )
                tests.append(
                    {
                        "exposure": exposure,
                        "controls": ";".join(controls) or "none",
                        "n": len(frame),
                        "rho": rho,
                        "p": p,
                        "fully_adjusted": len(controls) == 3,
                    }
                )
        table = pd.DataFrame(tests)
        table["q_bh"] = multipletests(table.p, method="fdr_bh")[1]
        table["p_holm"] = multipletests(table.p, method="holm")[1]
        context.write_csv("association_tests.csv", table)
        complete = geometry.dropna(
            subset=["chi_long", "chi_short", "road_gap_fine", "mean_degree", "log_edges", "log_gdp"]
        )
        observed = coefficient_pair(complete)
        draws = bootstrap_pair(complete, 10000, seed)
        context.write_json(
            "long_short.json",
            {
                "n": len(complete),
                "rho_long": observed[0],
                "rho_short": observed[1],
                "difference": observed[2],
                "intervals": {
                    k: [float(v) for v in np.quantile(draws[k], [0.025, 0.975])]
                    for k in ("rho_long", "rho_short", "difference")
                },
                "inference": "Paired city bootstrap; no within-city label swapping",
            },
        )
    context.finish(__file__)
    print(
        f"Regenerated decision/null tables: q={q:.12f}, external coverage={int(decisions.covered.sum())}/21, retained={decisions.retained.mean():.10f}"
    )


if __name__ == "__main__":
    cli = parser(__doc__)
    cli.add_argument("--associations", action="store_true")
    args = cli.parse_args()
    run(Context(args), args.associations)
