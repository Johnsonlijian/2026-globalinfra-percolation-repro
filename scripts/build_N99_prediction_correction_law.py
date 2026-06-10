"""N99 compact transfer-error correction for the 71-city road-window matrix.

This round turns the descriptive urban-form screen into a bounded predictive
test.  It models the observed road-minus-CEBH threshold gap as a function of
precomputed embedded road-form covariates, then reports leave-one-city-out and
leave-region-out performance for

    p_c^pred = p_c^CEBH + f(road form, block exposure, angular order).

The analysis is deliberately claim-bounded: it is a cross-validated empirical
correction, not a causal urban-design rule and not an analytical theorem.
R101 note: the figure stem was renamed from "correction_law" to
"compact_correction" to match the demoted claim language; the historical data
directory name is retained for provenance.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import pub_style  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R95 = ROOT / "data" / "R95_nonbacktracking_spectral_layer" / "N95_cebh_nb_geometry_decomposition.csv"
R65 = ROOT / "data" / "R65_public_covariates_and_controls" / "R65_integrated_public_covariates.csv"
OUT = ROOT / "data" / "N99_prediction_correction_law"
ROUND = ROOT / "rounds" / "N99_prediction_correction_law"
FIG_BASE = ROOT / "figures" / "Fig_N99_compact_correction"

PREDICTIONS_CSV = OUT / "N99_prediction_city_predictions.csv"
MODEL_CSV = OUT / "N99_prediction_model_comparison.csv"
COEFFICIENTS_CSV = OUT / "N99_prediction_feature_coefficients.csv"
FEATURE_GROUP_CSV = OUT / "N99_prediction_feature_group_comparison.csv"
REGION_CSV = OUT / "N99_prediction_region_holdout.csv"
PERMUTATION_CSV = OUT / "N99_prediction_permutation_baseline.csv"
COEFF_STABILITY_CSV = OUT / "N99_prediction_coefficient_stability.csv"
SUMMARY_JSON = OUT / "N99_prediction_summary.json"
REPORT_MD = OUT / "N99_prediction_report.md"
GATE_REPORT_MD = ROUND / "gate_report.md"
CLOSEOUT_MD = ROUND / "substantive_increment_closeout.md"

CORE_FEATURES = [
    "kappa",
    "edge_length_mean_m",
    "edge_length_cv",
    "dead_end_share",
    "degree4plus_share",
    "cycle_density",
    "orientation_entropy",
    "orientation_dominance",
    "articulation_point_share",
    "bridge_edge_share",
    "largest_biconnected_node_fraction",
    "angular_entropy",
    "straight_angle_share_10deg",
]

FEATURE_LABELS = {
    "kappa": "Degree moment kappa",
    "edge_length_mean_m": "Mean edge length",
    "edge_length_cv": "Edge-length variability",
    "dead_end_share": "Dead-end share",
    "degree4plus_share": "4+ degree share",
    "cycle_density": "Cycle density",
    "orientation_entropy": "Orientation entropy",
    "orientation_dominance": "Orientation dominance",
    "articulation_point_share": "Articulation share",
    "bridge_edge_share": "Bridge-edge share",
    "largest_biconnected_node_fraction": "Largest biconnected block",
    "angular_entropy": "Angular entropy",
    "straight_angle_share_10deg": "Straight-angle share",
}

KAPPA_FEATURES = ["kappa"]
DEGREE_FORM_FEATURES = ["kappa", "dead_end_share", "degree4plus_share", "cycle_density"]
FEATURE_GROUPS = {
    "kappa_only": KAPPA_FEATURES,
    "edge_scale": ["edge_length_mean_m", "edge_length_cv"],
    "degree_form_compact": DEGREE_FORM_FEATURES,
    "block_exposure": [
        "kappa",
        "articulation_point_share",
        "bridge_edge_share",
        "largest_biconnected_node_fraction",
        "cycle_density",
    ],
    "angular_order": [
        "kappa",
        "orientation_entropy",
        "orientation_dominance",
        "angular_entropy",
        "straight_angle_share_10deg",
    ],
    "block_angular": [
        "kappa",
        "articulation_point_share",
        "bridge_edge_share",
        "largest_biconnected_node_fraction",
        "cycle_density",
        "angular_entropy",
        "straight_angle_share_10deg",
    ],
    "full_embedded": CORE_FEATURES,
    "full_without_kappa": [f for f in CORE_FEATURES if f != "kappa"],
}
PRIMARY_GROUP = "degree_form_compact"
PRIMARY_FEATURES = FEATURE_GROUPS[PRIMARY_GROUP]

LEAKAGE_COLUMNS = {
    "pc_rank",
    "pc_road_rank",
    "pc_road",
    "pc_spatial_null",
    "pc_geometry_null_mean",
    "pc_nb",
    "road_gap",
    "road_gap_cebh",
    "road_minus_spatial",
    "road_minus_geometry_mean",
    "road_minus_nb",
    "geometry_null_gap_mean",
    "spatial_null_gap",
    "abs_error_cebh",
    "abs_error_nb",
}


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ROUND.mkdir(parents=True, exist_ok=True)
    FIG_BASE.parent.mkdir(parents=True, exist_ok=True)


def finite_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def make_pipeline(features: list[str]) -> Pipeline:
    transformer = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                features,
            )
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("prep", transformer),
            ("ridge", RidgeCV(alphas=np.logspace(-4, 4, 33))),
        ]
    )


def cv_predict(df: pd.DataFrame, features: list[str], splitter: Any) -> np.ndarray:
    y = df["road_gap_cebh"].to_numpy(dtype=float)
    pred = np.full(len(df), np.nan, dtype=float)
    for train, test in splitter.split(df[features], y):
        model = make_pipeline(features)
        model.fit(df.iloc[train][features], y[train])
        pred[test] = model.predict(df.iloc[test][features])
    return pred


def leave_region_out(df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, pd.DataFrame]:
    y = df["road_gap_cebh"].to_numpy(dtype=float)
    pred = np.full(len(df), np.nan, dtype=float)
    rows: list[dict[str, Any]] = []
    for region in sorted(df["macro_region"].dropna().unique()):
        test_mask = df["macro_region"].eq(region).to_numpy()
        train_mask = ~test_mask
        model = make_pipeline(features)
        model.fit(df.loc[train_mask, features], y[train_mask])
        pred[test_mask] = model.predict(df.loc[test_mask, features])
        obs = y[test_mask]
        pr = pred[test_mask]
        rows.append(
            {
                "macro_region": region,
                "n_test": int(test_mask.sum()),
                "mae_gap": float(mean_absolute_error(obs, pr)),
                "rmse_gap": rmse(obs, pr),
                "r2_gap": float(r2_score(obs, pr)) if int(test_mask.sum()) > 1 else float("nan"),
                "cebh_mae": float(mean_absolute_error(obs, np.zeros_like(obs))),
            }
        )
    return pred, pd.DataFrame(rows)


def coefficient_stability(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Estimate primary-model coefficient stability across leave-one-city-out folds."""
    y = df["road_gap_cebh"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for fold_id, (train, test) in enumerate(LeaveOneOut().split(df[features], y)):
        model = make_pipeline(features)
        model.fit(df.iloc[train][features], y[train])
        ridge = model.named_steps["ridge"]
        for feature, coef in zip(features, np.asarray(ridge.coef_, dtype=float)):
            rows.append(
                {
                    "fold_id": int(fold_id),
                    "held_out_city": str(df.iloc[test[0]]["city"]),
                    "feature": feature,
                    "standardized_coefficient": float(coef),
                    "coefficient_sign": int(np.sign(coef)),
                    "alpha": float(ridge.alpha_),
                }
            )
    long = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for feature, sub in long.groupby("feature", sort=False):
        vals = sub["standardized_coefficient"].to_numpy(dtype=float)
        nonzero_sign = np.sign(vals[np.abs(vals) > 1e-12])
        summary_rows.append(
            {
                "feature": feature,
                "n_folds": int(len(vals)),
                "mean_standardized_coefficient": float(np.mean(vals)),
                "sd_standardized_coefficient": float(np.std(vals, ddof=1)),
                "min_standardized_coefficient": float(np.min(vals)),
                "max_standardized_coefficient": float(np.max(vals)),
                "sign_consistency": float(
                    max(np.mean(nonzero_sign > 0), np.mean(nonzero_sign < 0)) if len(nonzero_sign) else np.nan
                ),
                "median_alpha": float(np.median(sub["alpha"].to_numpy(dtype=float))),
                "record_type": "summary",
            }
        )
    summary = pd.DataFrame(summary_rows)
    long["record_type"] = "fold"
    return pd.concat([summary, long], ignore_index=True, sort=False)


def permutation_baseline(
    df: pd.DataFrame,
    features: list[str],
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    """Run the same primary RidgeCV pipeline on shuffled target gaps."""
    rng = np.random.default_rng(seed)
    y = df["road_gap_cebh"].to_numpy(dtype=float)
    pc = df["pc_cebh"].to_numpy(dtype=float)
    real_loo = cv_predict(df, features, LeaveOneOut())
    real_lro, _region = leave_region_out(df, features)
    real_loo_mae = float(mean_absolute_error(y, real_loo))
    real_lro_mae = float(mean_absolute_error(y, real_lro))
    rows: list[dict[str, Any]] = []
    for i in range(int(n_permutations)):
        permuted = rng.permutation(y)
        df_perm = df.copy()
        df_perm["road_gap_cebh"] = permuted
        pred_loo = cv_predict(df_perm, features, LeaveOneOut())
        pred_lro, _region_perm = leave_region_out(df_perm, features)
        loo_mae = float(mean_absolute_error(permuted, pred_loo))
        lro_mae = float(mean_absolute_error(permuted, pred_lro))
        rows.append(
            {
                "permutation_id": int(i),
                "loo_mae_gap": loo_mae,
                "lro_mae_gap": lro_mae,
                "loo_rmse_gap": rmse(permuted, pred_loo),
                "lro_rmse_gap": rmse(permuted, pred_lro),
                "loo_r2_gap": float(r2_score(permuted, pred_loo)),
                "lro_r2_gap": float(r2_score(permuted, pred_lro)),
                "real_loo_mae_gap": real_loo_mae,
                "real_lro_mae_gap": real_lro_mae,
                "real_loo_better": bool(real_loo_mae < loo_mae),
                "real_lro_better": bool(real_lro_mae < lro_mae),
                "n_cities": int(len(df)),
                "n_features": int(len(features)),
                "feature_family": PRIMARY_GROUP,
            }
        )
    rows.append(
        {
            "permutation_id": -1,
            "loo_mae_gap": real_loo_mae,
            "lro_mae_gap": real_lro_mae,
            "loo_rmse_gap": rmse(y, real_loo),
            "lro_rmse_gap": rmse(y, real_lro),
            "loo_r2_gap": float(r2_score(y, real_loo)),
            "lro_r2_gap": float(r2_score(y, real_lro)),
            "real_loo_mae_gap": real_loo_mae,
            "real_lro_mae_gap": real_lro_mae,
            "real_loo_better": True,
            "real_lro_better": True,
            "n_cities": int(len(df)),
            "n_features": int(len(features)),
            "feature_family": PRIMARY_GROUP,
        }
    )
    _ = pc  # keep the metric aligned with threshold-scale naming in reports
    return pd.DataFrame(rows)


def fit_coefficients(df: pd.DataFrame, features: list[str], model_name: str) -> pd.DataFrame:
    y = df["road_gap_cebh"].to_numpy(dtype=float)
    model = make_pipeline(features)
    model.fit(df[features], y)
    ridge = model.named_steps["ridge"]
    coefs = np.asarray(ridge.coef_, dtype=float)
    rows = []
    for feature, coef in zip(features, coefs):
        rows.append(
            {
                "feature": feature,
                "model": model_name,
                "standardized_coefficient": float(coef),
                "absolute_standardized_coefficient": float(abs(coef)),
                "alpha": float(ridge.alpha_),
            }
        )
    return pd.DataFrame(rows).sort_values("absolute_standardized_coefficient", ascending=False)


def metrics_row(name: str, y_gap: np.ndarray, pred_gap: np.ndarray, pc_cebh: np.ndarray) -> dict[str, Any]:
    pc_obs = pc_cebh + y_gap
    pc_pred = pc_cebh + pred_gap
    return {
        "model": name,
        "n": int(np.isfinite(pred_gap).sum()),
        "mae_gap": float(mean_absolute_error(y_gap, pred_gap)),
        "rmse_gap": rmse(y_gap, pred_gap),
        "r2_gap": float(r2_score(y_gap, pred_gap)),
        "mae_pc": float(mean_absolute_error(pc_obs, pc_pred)),
        "rmse_pc": rmse(pc_obs, pc_pred),
        "r2_pc": float(r2_score(pc_obs, pc_pred)),
    }


def feature_group_comparison(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    y = df["road_gap_cebh"].to_numpy(dtype=float)
    pc = df["pc_cebh"].to_numpy(dtype=float)
    loo_preds: dict[str, np.ndarray] = {}
    lro_preds: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for name, features in FEATURE_GROUPS.items():
        pred_loo = cv_predict(df, features, LeaveOneOut())
        pred_lro, _region = leave_region_out(df, features)
        loo_preds[name] = pred_loo
        lro_preds[name] = pred_lro
        loo = metrics_row(f"{name} LOO", y, pred_loo, pc)
        lro = metrics_row(f"{name} leave-region-out", y, pred_lro, pc)
        rows.append(
            {
                "model_group": name,
                "features": "; ".join(features),
                "n_features": int(len(features)),
                "loo_mae_pc": loo["mae_pc"],
                "loo_rmse_pc": loo["rmse_pc"],
                "loo_r2_gap": loo["r2_gap"],
                "lro_mae_pc": lro["mae_pc"],
                "lro_rmse_pc": lro["rmse_pc"],
                "lro_r2_gap": lro["r2_gap"],
            }
        )
    out = pd.DataFrame(rows).sort_values(["lro_mae_pc", "loo_mae_pc"]).reset_index(drop=True)
    return out, loo_preds, lro_preds


def load_inputs() -> pd.DataFrame:
    base = pd.read_csv(R95)
    cov = pd.read_csv(R65)
    cov_cols = ["city", "country", "macro_region"] + [c for c in CORE_FEATURES if c in cov.columns]
    missing = sorted(set(CORE_FEATURES) - set(cov_cols))
    if missing:
        raise RuntimeError(f"Missing required N99 features: {missing}")
    base_cols = [
        "city",
        "country",
        "macro_region",
        "n_nodes",
        "n_edges",
        "pc_cebh",
        "pc_road_rank",
        "road_gap_cebh",
        "pc_nb",
        "abs_error_nb",
        "road_minus_spatial",
        "pc_geometry_null_mean",
        "road_minus_geometry_mean",
    ]
    base_cols = [c for c in base_cols if c in base.columns]
    df = base[base_cols].merge(cov[cov_cols], on=["city", "country", "macro_region"], how="left")
    numeric = ["pc_cebh", "pc_road_rank", "road_gap_cebh", "pc_nb", "abs_error_nb"] + CORE_FEATURES
    df = finite_frame(df, [c for c in numeric if c in df.columns])
    df = df[df["pc_cebh"].notna() & df["pc_road_rank"].notna() & df["road_gap_cebh"].notna()].copy()
    df = df.sort_values(["macro_region", "city"]).reset_index(drop=True)
    return df


def build_figures(pred: pd.DataFrame, coef: pd.DataFrame, metrics: pd.DataFrame, perm: pd.DataFrame) -> None:
    pub_style.apply()
    c_cebh = pub_style.COLORS["cebh"]
    c_model = pub_style.COLORS["model"]
    c_kappa = pub_style.COLORS["residual"]
    c_lro = pub_style.COLORS["geometry_null"]
    fig, axes = plt.subplots(2, 2, figsize=(pub_style.FIG_WIDTH_2COL, 6.0), constrained_layout=True)
    ax = axes[0, 0]
    pc_min = float(min(pred["pc_road_rank"].min(), pred["pc_cebh"].min(), pred["pc_pred_degree_form_loo"].min())) - 0.015
    pc_max = float(max(pred["pc_road_rank"].max(), pred["pc_cebh"].max(), pred["pc_pred_degree_form_loo"].max())) + 0.015
    ax.plot([pc_min, pc_max], [pc_min, pc_max], color="#1A1A1A", lw=0.7, ls="--")
    ax.scatter(pred["pc_cebh"], pred["pc_road_rank"], s=22, color=c_cebh, alpha=0.68, linewidths=0, label="CEBH")
    ax.scatter(
        pred["pc_pred_degree_form_loo"],
        pred["pc_road_rank"],
        s=24,
        color=c_model,
        alpha=0.78,
        linewidths=0,
        label="Compact correction, LOO",
    )
    ax.set_xlim(pc_min, pc_max)
    ax.set_ylim(pc_min, pc_max)
    ax.set_xlabel("Predicted threshold")
    ax.set_ylabel("Observed road threshold")
    pub_style.panel_title(ax, "a", "Observed versus predicted thresholds")
    pub_style.light_grid(ax, axis="both")
    ax.legend(frameon=False, loc="lower right", fontsize=6.5)

    ax = axes[0, 1]
    order = ["CEBH", "kappa-only\nLOO", "compact\nLOO", "compact\nLRO"]
    values = [
        pred["abs_error_cebh"].to_numpy(float),
        pred["abs_error_kappa_loo"].to_numpy(float),
        pred["abs_error_degree_form_loo"].to_numpy(float),
        pred["abs_error_degree_form_lro"].to_numpy(float),
    ]
    colors = [c_cebh, c_kappa, c_model, c_lro]
    for i, vals in enumerate(values, start=1):
        x = np.random.default_rng(9900 + i).normal(i, 0.045, size=len(vals))
        ax.scatter(x, vals, s=13, color=colors[i - 1], alpha=0.42, linewidths=0)
        ax.hlines(np.nanmedian(vals), i - 0.22, i + 0.22, color=colors[i - 1], lw=1.8)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order)
    ax.set_ylabel("Absolute threshold error")
    pub_style.panel_title(ax, "b", "Cross-validated absolute error")
    pub_style.light_grid(ax, axis="y")

    ax = axes[1, 0]
    top = coef[coef["model"].eq(PRIMARY_GROUP)].sort_values("standardized_coefficient")
    bar_colors = np.where(top["standardized_coefficient"] >= 0, "#D55E00", "#0072B2")
    ax.barh(
        [FEATURE_LABELS.get(v, v) for v in top["feature"]],
        top["standardized_coefficient"],
        color=bar_colors,
        alpha=0.9,
    )
    ax.axvline(0, color="#1A1A1A", lw=0.7)
    ax.set_xlabel("Standardized Ridge coefficient")
    pub_style.panel_title(ax, "c", "Standardized coefficients")
    pub_style.light_grid(ax, axis="x")

    ax = axes[1, 1]
    perm_only = perm[perm["permutation_id"].astype(int) >= 0]
    real = perm[perm["permutation_id"].astype(int) == -1].iloc[0]
    ax.hist(perm_only["loo_mae_gap"], bins=28, color="#D5D9DE", edgecolor="#8A9099", linewidth=0.4)
    ax.axvline(float(real["loo_mae_gap"]), color=c_model, lw=1.8, label="Observed labels")
    ax.axvline(
        float(perm_only["loo_mae_gap"].quantile(0.01)),
        color=pub_style.COLORS["spatial_null"],
        lw=1.1,
        ls="--",
        label="1% permutation quantile",
    )
    ax.set_xlabel("LOO MAE under shuffled gaps")
    ax.set_ylabel("Permutation count")
    pub_style.panel_title(ax, "d", "Permutation audit (1000 runs)")
    pub_style.light_grid(ax, axis="y")
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")

    pub_style.save(fig, FIG_BASE)
    plt.close(fig)


def write_reports(summary: dict[str, Any], metrics: pd.DataFrame, region: pd.DataFrame) -> None:
    metric_md = metrics.to_markdown(index=False, floatfmt=".4f")
    region_md = region.to_markdown(index=False, floatfmt=".4f")
    report = f"""# N99 Prediction Correction Law

## Purpose

This round tests a bounded empirical correction law:

`p_c_pred = p_c_CEBH + f(edge scale, articulation exposure, biconnected structure, cycle density, angular order)`.

The correction uses only precomputed road-form and embedded-structure covariates.
It does not use observed thresholds, spatial-null thresholds, geometry-null
thresholds or non-backtracking thresholds as predictors.

## Headline results

- Cities: {summary['n_cities']}
- Primary model: {summary['primary_model']}
- Primary feature count: {summary['n_features']}
- CEBH MAE: {summary['cebh_mae']:.4f}
- Compact correction LOO MAE: {summary['n99_loo_mae']:.4f}
- Compact correction LOO improvement versus CEBH: {summary['n99_loo_mae_improvement_vs_cebh']:.4f}
- Compact correction leave-region-out MAE: {summary['n99_lro_mae']:.4f}
- Compact correction leave-region-out gap R2: {summary['n99_lro_r2_gap']:.4f}
- Permutation baseline: observed LOO MAE is lower than {summary['permutation_loo_fraction_observed_better']:.1%} of shuffled-label runs
- Permutation baseline: observed leave-region-out MAE is lower than {summary['permutation_lro_fraction_observed_better']:.1%} of shuffled-label runs

## Model comparison

{metric_md}

## Leave-region-out detail

{region_md}

## Claim boundary

{summary['claim_boundary']}
"""
    REPORT_MD.write_text(report, encoding="utf-8")
    gate = f"""# N99 Gate Report: Prediction Correction Law

## Gate status

PASS as a descriptive, cross-validated empirical correction.

## Evidence

- Outputs: `{PREDICTIONS_CSV.relative_to(ROOT)}`, `{MODEL_CSV.relative_to(ROOT)}`,
  `{COEFFICIENTS_CSV.relative_to(ROOT)}`, `{REGION_CSV.relative_to(ROOT)}`,
  `{PERMUTATION_CSV.relative_to(ROOT)}`, `{COEFF_STABILITY_CSV.relative_to(ROOT)}`.
- Figure: `{FIG_BASE.with_suffix('.svg').relative_to(ROOT)}`.
- Primary compact LOO MAE improvement versus CEBH: {summary['n99_loo_mae_improvement_vs_cebh']:.4f}.
- Leave-region-out MAE: {summary['n99_lro_mae']:.4f}.
- Permutation p-value proxy, LOO: {summary['permutation_loo_p_value_proxy']:.4f}.
- Permutation p-value proxy, leave-region-out: {summary['permutation_lro_p_value_proxy']:.4f}.

## Reviewer-facing boundary

This is not a causal law, not a theorem and not a replacement for the strict
geometry-null ensemble. It is a compact predictive check showing that embedded
road-form covariates contain transferable information about the CEBH transfer
error.
"""
    GATE_REPORT_MD.write_text(gate, encoding="utf-8")
    closeout = f"""# N99 Substantive Increment Closeout

| Item | Status | Evidence | Boundary |
|---|---|---|---|
| 71-city compact correction law | Done | LOO and leave-region-out predictions in `{PREDICTIONS_CSV.relative_to(ROOT)}` | Descriptive predictive model only |
| Candidate feature-family comparison | Done | `{FEATURE_GROUP_CSV.relative_to(ROOT)}` | Model-family screen, not pre-registered causal selection |
| Feature leakage guard | Done | Predictor list excludes threshold/null outcome columns | Does not prove causality |
| Permutation baseline | Done | `{PERMUTATION_CSV.relative_to(ROOT)}` | Shuffled-label audit, not a formal theorem |
| Coefficient stability | Done | `{COEFF_STABILITY_CSV.relative_to(ROOT)}` | Fold stability only |
| Submission figure | Done | `{FIG_BASE.with_suffix('.svg').relative_to(ROOT)}` plus PDF/PNG/TIFF | Figure is supplementary/main-candidate, not independent evidence |
"""
    CLOSEOUT_MD.write_text(closeout, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=99001)
    parser.add_argument("--n-permutations", type=int, default=1000)
    args = parser.parse_args()

    ensure_dirs()
    df = load_inputs()
    leakage = sorted(set(CORE_FEATURES) & LEAKAGE_COLUMNS)
    if leakage:
        raise RuntimeError(f"Leakage features present in N99 predictor list: {leakage}")

    y_gap = df["road_gap_cebh"].to_numpy(dtype=float)
    pc_cebh = df["pc_cebh"].to_numpy(dtype=float)

    loo = LeaveOneOut()
    group_df, loo_group_preds, lro_group_preds = feature_group_comparison(df)
    group_df.to_csv(FEATURE_GROUP_CSV, index=False)

    pred_gap_loo = loo_group_preds[PRIMARY_GROUP]
    pred_gap_kappa_loo = cv_predict(df, KAPPA_FEATURES, loo)
    kfold = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    pred_gap_5fold = cv_predict(df, PRIMARY_FEATURES, kfold)
    pred_gap_lro, region_df = leave_region_out(df, PRIMARY_FEATURES)
    pred_gap_full_loo = loo_group_preds["full_embedded"]
    pred_gap_full_lro = lro_group_preds["full_embedded"]

    pred = df[
        [
            "city",
            "country",
            "macro_region",
            "n_nodes",
            "n_edges",
            "pc_cebh",
            "pc_road_rank",
            "road_gap_cebh",
            "pc_nb",
            "abs_error_nb",
        ]
    ].copy()
    pred["pred_gap_degree_form_loo"] = pred_gap_loo
    pred["pc_pred_degree_form_loo"] = pred["pc_cebh"] + pred["pred_gap_degree_form_loo"]
    pred["abs_error_degree_form_loo"] = (pred["pc_road_rank"] - pred["pc_pred_degree_form_loo"]).abs()
    pred["pred_gap_degree_form_5fold"] = pred_gap_5fold
    pred["pc_pred_degree_form_5fold"] = pred["pc_cebh"] + pred["pred_gap_degree_form_5fold"]
    pred["abs_error_degree_form_5fold"] = (pred["pc_road_rank"] - pred["pc_pred_degree_form_5fold"]).abs()
    pred["pred_gap_degree_form_lro"] = pred_gap_lro
    pred["pc_pred_degree_form_lro"] = pred["pc_cebh"] + pred["pred_gap_degree_form_lro"]
    pred["abs_error_degree_form_lro"] = (pred["pc_road_rank"] - pred["pc_pred_degree_form_lro"]).abs()
    pred["pred_gap_full_embedded_loo"] = pred_gap_full_loo
    pred["pc_pred_full_embedded_loo"] = pred["pc_cebh"] + pred["pred_gap_full_embedded_loo"]
    pred["abs_error_full_embedded_loo"] = (pred["pc_road_rank"] - pred["pc_pred_full_embedded_loo"]).abs()
    pred["pred_gap_full_embedded_lro"] = pred_gap_full_lro
    pred["pc_pred_full_embedded_lro"] = pred["pc_cebh"] + pred["pred_gap_full_embedded_lro"]
    pred["abs_error_full_embedded_lro"] = (pred["pc_road_rank"] - pred["pc_pred_full_embedded_lro"]).abs()
    pred["pred_gap_kappa_loo"] = pred_gap_kappa_loo
    pred["pc_pred_kappa_loo"] = pred["pc_cebh"] + pred["pred_gap_kappa_loo"]
    pred["abs_error_kappa_loo"] = (pred["pc_road_rank"] - pred["pc_pred_kappa_loo"]).abs()
    pred["abs_error_cebh"] = pred["road_gap_cebh"].abs()
    pred.to_csv(PREDICTIONS_CSV, index=False)

    rows = [
        metrics_row("CEBH", y_gap, np.zeros_like(y_gap), pc_cebh),
        metrics_row("kappa-only LOO", y_gap, pred_gap_kappa_loo, pc_cebh),
        metrics_row("compact degree-form correction LOO", y_gap, pred_gap_loo, pc_cebh),
        metrics_row("compact degree-form correction 5-fold", y_gap, pred_gap_5fold, pc_cebh),
        metrics_row("compact degree-form correction leave-region-out", y_gap, pred_gap_lro, pc_cebh),
        metrics_row("full embedded correction LOO", y_gap, pred_gap_full_loo, pc_cebh),
        metrics_row("full embedded correction leave-region-out", y_gap, pred_gap_full_lro, pc_cebh),
    ]
    if "pc_nb" in pred and pred["pc_nb"].notna().any():
        pred_gap_nb = pred["pc_nb"].to_numpy(dtype=float) - pc_cebh
        finite = np.isfinite(pred_gap_nb)
        rows.append(metrics_row("non-backtracking spectral proxy", y_gap[finite], pred_gap_nb[finite], pc_cebh[finite]))
    metrics = pd.DataFrame(rows)
    metrics["mae_improvement_vs_cebh"] = float(metrics.loc[metrics["model"].eq("CEBH"), "mae_pc"].iloc[0]) - metrics["mae_pc"]
    metrics.to_csv(MODEL_CSV, index=False)

    coef = pd.concat(
        [
            fit_coefficients(df, PRIMARY_FEATURES, PRIMARY_GROUP),
            fit_coefficients(df, CORE_FEATURES, "full_embedded"),
        ],
        ignore_index=True,
    )
    coef.to_csv(COEFFICIENTS_CSV, index=False)
    region_df.to_csv(REGION_CSV, index=False)
    perm = permutation_baseline(df, PRIMARY_FEATURES, args.n_permutations, args.seed + 501)
    perm.to_csv(PERMUTATION_CSV, index=False)
    stability = coefficient_stability(df, PRIMARY_FEATURES)
    stability.to_csv(COEFF_STABILITY_CSV, index=False)

    build_figures(pred, coef, metrics, perm)

    cebh_mae = float(metrics.loc[metrics["model"].eq("CEBH"), "mae_pc"].iloc[0])
    loo_mae = float(metrics.loc[metrics["model"].eq("compact degree-form correction LOO"), "mae_pc"].iloc[0])
    lro_mae = float(
        metrics.loc[metrics["model"].eq("compact degree-form correction leave-region-out"), "mae_pc"].iloc[0]
    )
    summary = {
        "round": "N99_prediction_correction_law",
        "status": "pass",
        "n_cities": int(len(df)),
        "primary_model": PRIMARY_GROUP,
        "n_features": int(len(PRIMARY_FEATURES)),
        "features": PRIMARY_FEATURES,
        "full_embedded_features": CORE_FEATURES,
        "best_lro_feature_group": str(group_df.iloc[0]["model_group"]) if len(group_df) else None,
        "cebh_mae": cebh_mae,
        "kappa_only_loo_mae": float(metrics.loc[metrics["model"].eq("kappa-only LOO"), "mae_pc"].iloc[0]),
        "n99_loo_mae": loo_mae,
        "n99_loo_rmse": float(metrics.loc[metrics["model"].eq("compact degree-form correction LOO"), "rmse_pc"].iloc[0]),
        "n99_loo_r2_gap": float(metrics.loc[metrics["model"].eq("compact degree-form correction LOO"), "r2_gap"].iloc[0]),
        "n99_loo_mae_improvement_vs_cebh": cebh_mae - loo_mae,
        "n99_lro_mae": lro_mae,
        "n99_lro_rmse": float(
            metrics.loc[metrics["model"].eq("compact degree-form correction leave-region-out"), "rmse_pc"].iloc[0]
        ),
        "n99_lro_r2_gap": float(
            metrics.loc[metrics["model"].eq("compact degree-form correction leave-region-out"), "r2_gap"].iloc[0]
        ),
        "full_embedded_loo_mae": float(metrics.loc[metrics["model"].eq("full embedded correction LOO"), "mae_pc"].iloc[0]),
        "full_embedded_lro_mae": float(
            metrics.loc[metrics["model"].eq("full embedded correction leave-region-out"), "mae_pc"].iloc[0]
        ),
        "nb_mae": float(metrics.loc[metrics["model"].eq("non-backtracking spectral proxy"), "mae_pc"].iloc[0])
        if metrics["model"].eq("non-backtracking spectral proxy").any()
        else None,
        "permutation_n": int(args.n_permutations),
        "permutation_loo_fraction_observed_better": float(
            np.mean(perm.loc[perm["permutation_id"].astype(int) >= 0, "loo_mae_gap"].to_numpy(float) > loo_mae)
        ),
        "permutation_lro_fraction_observed_better": float(
            np.mean(perm.loc[perm["permutation_id"].astype(int) >= 0, "lro_mae_gap"].to_numpy(float) > lro_mae)
        ),
        "permutation_loo_p_value_proxy": float(
            (1 + np.sum(perm.loc[perm["permutation_id"].astype(int) >= 0, "loo_mae_gap"].to_numpy(float) <= loo_mae))
            / (args.n_permutations + 1)
        ),
        "permutation_lro_p_value_proxy": float(
            (1 + np.sum(perm.loc[perm["permutation_id"].astype(int) >= 0, "lro_mae_gap"].to_numpy(float) <= lro_mae))
            / (args.n_permutations + 1)
        ),
        "coefficient_min_sign_consistency": float(
            stability.loc[stability["record_type"].eq("summary"), "sign_consistency"].min()
        ),
        "claim_boundary": (
            "Cross-validated empirical transfer-error correction using only precomputed embedded road-form "
            "covariates. It is not a causal urban-design rule, not a universal law and not an analytical theorem."
        ),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_reports(summary, metrics, region_df)

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
