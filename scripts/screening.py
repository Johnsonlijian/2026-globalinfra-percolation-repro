"Analytic screening metrics and city/family uncertainty helpers."

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

PREDICTORS = {
    "CEBH": "pc_cebh",
    "2D": "pc_dimension",
    "junction": "pc_junction_anchor",
    "nonbacktracking": "pc_nonbacktracking",
}


def integrated_errors(pred_fc, true_fc, low=0.15, high=0.55):
    pred_fc, true_fc = np.asarray(pred_fc), np.asarray(true_fc)
    unsafe = np.maximum(0, np.minimum(high, pred_fc) - np.maximum(low, true_fc)) / (high - low)
    safe = np.maximum(0, np.minimum(high, true_fc) - np.maximum(low, pred_fc)) / (high - low)
    return unsafe, safe


def bootstrap_mean(values, seed=20260905, count=10000):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (count, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def allowance(errors, alpha=0.05):
    rank = int(np.ceil((len(errors) + 1) * (1 - alpha)))
    return float(np.sort(errors)[rank - 1]) if rank <= len(errors) else float("inf")


def evaluate_allowance(pred_fc, true_fc, q, low=0.15, high=0.55):
    pred_fc, true_fc = np.asarray(pred_fc), np.asarray(true_fc)
    width = high - low
    escalated = np.maximum(0, np.minimum(high, pred_fc + q) - np.maximum(low, pred_fc - q)) / width
    unsafe = np.maximum(0, np.minimum(high, pred_fc - q) - np.maximum(low, true_fc)) / width
    safe = np.maximum(0, np.minimum(high, true_fc) - np.maximum(low, pred_fc + q)) / width
    return pd.DataFrame(
        {
            "escalated": escalated,
            "unsafe_retained": unsafe,
            "safe_retained": safe,
            "retained": 1 - escalated,
        }
    )


def ranked(values: np.ndarray) -> np.ndarray:
    return rankdata(np.asarray(values, dtype=float), method="average")


def residualize(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    y = ranked(values)
    ranked_controls = np.column_stack(
        [ranked(controls[:, idx]) for idx in range(controls.shape[1])]
    )
    design = np.column_stack([np.ones(len(y)), ranked_controls])
    return y - design @ np.linalg.lstsq(design, y, rcond=None)[0]


def partial_spearman(x: np.ndarray, y: np.ndarray, controls: np.ndarray) -> float:
    rx = residualize(x, controls)
    ry = residualize(y, controls)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def coefficient_pair(frame: pd.DataFrame) -> tuple[float, float, float]:
    common = frame[["mean_degree", "log_edges", "log_gdp"]].to_numpy(dtype=float)
    long_controls = np.column_stack([frame["chi_short"].to_numpy(dtype=float), common])
    short_controls = np.column_stack([frame["chi_long"].to_numpy(dtype=float), common])
    outcome = frame["road_gap_fine"].to_numpy(dtype=float)
    rho_long = partial_spearman(frame["chi_long"].to_numpy(dtype=float), outcome, long_controls)
    rho_short = partial_spearman(frame["chi_short"].to_numpy(dtype=float), outcome, short_controls)
    return rho_long, rho_short, rho_long - rho_short


def bootstrap_pair(frame: pd.DataFrame, draws: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for draw in range(draws):
        sample = frame.iloc[rng.integers(0, len(frame), size=len(frame))]
        long_rho, short_rho, difference = coefficient_pair(sample)
        rows.append(
            {"draw": draw, "rho_long": long_rho, "rho_short": short_rho, "difference": difference}
        )
    return pd.DataFrame(rows)


def freedman_lane(exposure, outcome, controls, seed):
    n = len(outcome)
    x, y = rankdata(exposure), rankdata(outcome)
    design = np.column_stack(
        [np.ones(n)] + [rankdata(controls[:, i]) for i in range(controls.shape[1])]
    )
    residual = np.eye(n) - design @ np.linalg.pinv(design)
    rx, ry = residual @ x, residual @ y
    observed = float(np.dot(rx, ry) / np.linalg.norm(rx) / np.linalg.norm(ry))
    rng = np.random.default_rng(seed)
    permuted = np.column_stack([ry[rng.permutation(n)] for _ in range(10000)])
    # Freedman-Lane: permute reduced-model residuals, then re-project controls.
    simulated = residual @ permuted
    values = rx @ simulated / np.linalg.norm(rx) / np.linalg.norm(simulated, axis=0)
    return observed, float((1 + np.sum(abs(values) >= abs(observed))) / 10001)
