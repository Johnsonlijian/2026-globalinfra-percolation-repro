"""R95 iteration-stability audit for the non-backtracking threshold proxy."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_R56_spatial_length_constrained_nulls as r56  # noqa: E402
import build_R95_nonbacktracking_spectral_layer as r95  # noqa: E402

OUT_CSV = r95.OUT / "R95_nb_iteration_stability_audit.csv"
OUT_JSON = r95.OUT / "R95_nb_iteration_stability_summary.json"


def mean_ci95(values: pd.Series) -> tuple[float, float, float]:
    arr = values.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    if len(arr) == 1:
        return mean, mean, mean
    se = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short-iter", type=int, default=160)
    parser.add_argument("--seed", type=int, default=9501)
    args = parser.parse_args()

    r95.OUT.mkdir(parents=True, exist_ok=True)
    long_df = pd.read_csv(r95.THRESHOLDS_CSV)
    long_df = long_df[long_df["status"].eq("pass")][["city", "pc_nb", "rho_nb", "nb_iterations", "nb_converged"]].rename(
        columns={
            "pc_nb": "pc_nb_long",
            "rho_nb": "rho_nb_long",
            "nb_iterations": "long_iterations",
            "nb_converged": "long_converged",
        }
    )
    rows = pd.read_csv(r95.R61)
    rows = rows[rows["status"].eq("pass")].sort_values(["n_edges", "city"]).reset_index(drop=True)

    records: list[dict[str, object]] = []
    for idx, row in rows.iterrows():
        city = str(row["city"])
        t0 = time.time()
        print(f"[R95-stability] {idx + 1}/{len(rows)} {city}", flush=True)
        try:
            graph = r56.load_graph(r95.as_project_path(row["graph_cache_file"]))
            _H, edges, degrees, _lon, _lat = r56.to_simple_spatial_graph(graph)
            nb = r95.nonbacktracking_power(edges, len(degrees), args.seed + 1009 * (idx + 1), args.short_iter, 1e-8)
            rho = float(nb["rho_nb"])
            pc = float(1.0 / rho) if np.isfinite(rho) and rho > 0 else float("nan")
            records.append(
                {
                    "city": city,
                    "country": row["country"],
                    "macro_region": row["macro_region"],
                    "n_edges": int(len(edges)),
                    "short_iterations": int(args.short_iter),
                    "rho_nb_short": rho,
                    "pc_nb_short": pc,
                    "short_converged": bool(nb["converged"]),
                    "runtime_seconds": float(time.time() - t0),
                    "status": "pass",
                }
            )
        except Exception as exc:  # noqa: BLE001
            records.append(
                {
                    "city": city,
                    "country": row.get("country", ""),
                    "macro_region": row.get("macro_region", ""),
                    "short_iterations": int(args.short_iter),
                    "status": "fail",
                    "notes": repr(exc),
                    "runtime_seconds": float(time.time() - t0),
                }
            )

    short_df = pd.DataFrame(records)
    audit = short_df.merge(long_df, on="city", how="left")
    audit["pc_nb_abs_delta_short_vs_long"] = (audit["pc_nb_short"] - audit["pc_nb_long"]).abs()
    audit["rho_nb_abs_delta_short_vs_long"] = (audit["rho_nb_short"] - audit["rho_nb_long"]).abs()
    audit.to_csv(OUT_CSV, index=False)

    mean_delta, lo_delta, hi_delta = mean_ci95(audit["pc_nb_abs_delta_short_vs_long"])
    summary = {
        "status": "pass" if int(audit["status"].eq("pass").sum()) == len(audit) else "partial",
        "n_records_total": int(len(audit)),
        "n_records_pass": int(audit["status"].eq("pass").sum()),
        "short_iterations": int(args.short_iter),
        "long_iterations": int(audit["long_iterations"].dropna().max()) if "long_iterations" in audit else 0,
        "mean_abs_pc_delta_short_vs_long": mean_delta,
        "ci95_abs_pc_delta_low": lo_delta,
        "ci95_abs_pc_delta_high": hi_delta,
        "median_abs_pc_delta_short_vs_long": float(audit["pc_nb_abs_delta_short_vs_long"].median()),
        "max_abs_pc_delta_short_vs_long": float(audit["pc_nb_abs_delta_short_vs_long"].max()),
        "n_long_converged": int(audit["long_converged"].sum()),
        "claim_boundary": "Iteration-stability diagnostic for the R95 non-backtracking spectral proxy.",
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
