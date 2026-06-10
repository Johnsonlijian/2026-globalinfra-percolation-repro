"""R77 physics-takeaway figure and source data.

This round promotes the matched-intensity geometry contrast from a defensive
diagnostic to the main physical result: at the same accepted-swap intensity,
the spatial-scale null and the strict non-crossing geometry null separate.

The analysis uses existing derived tables only. It does not regenerate null
edge sets, compute true edge-set Jaccard distance, or claim causal inference.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut, cross_val_predict

import pub_style


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "R77_physics_takeaway"
ROUND_DIR = ROOT / "rounds" / "R77_physics_takeaway"
FIG_BASE = ROOT / "figures" / "Fig_R77_physics_takeaway"
TARGETS = [
    ROOT / "submission" / "communications_physics" / "target_submission",
    ROOT / "submission" / "npj_complexity" / "target_submission",
]

MATCHED_SUMMARY = ROOT / "data" / "R72_geometry_defense" / "matched_intensity_summary.csv"
MATCHED_CITY = ROOT / "data" / "R72_geometry_defense" / "matched_intensity_geometry_vs_spatial_city.csv"
R68_CITY = ROOT / "data" / "R68_full71_geometry_null_ensemble" / "full71_geometry_null_city_summary.csv"
R63 = ROOT / "data" / "R63_high_order_external_covariates" / "integrated_high_order_external_covariates.csv"

SELECTED_GEOMETRY_FEATURES = [
    "bridge_edge_share",
    "articulation_point_share",
    "largest_biconnected_node_fraction",
    "dual_degree_mean",
    "angular_entropy",
    "straight_angle_share_10deg",
    "cycle_density",
    "dead_end_share",
    "kappa",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_style() -> None:
    pub_style.apply()


def _segments_cross(p1, p2, q1, q2) -> bool:
    """Strict crossing test for the schematic; shared endpoints do not count."""
    if len({p1, p2, q1, q2}) < 4:
        return False

    def orient(a, b, c) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = orient(q1, q2, p1), orient(q1, q2, p2)
    d3, d4 = orient(p1, p2, q1), orient(p1, p2, q2)
    return d1 * d2 < 0 and d3 * d4 < 0


def _toy_street_patch(seed: int = 12):
    """Small jittered street patch used only as an explanatory schematic."""
    rng = np.random.default_rng(seed)
    nx_, ny_ = 6, 4
    pos = {}
    for j in range(ny_):
        for i in range(nx_):
            pos[(i, j)] = (i + rng.uniform(-0.16, 0.16), j + rng.uniform(-0.16, 0.16))
    edges = []
    for j in range(ny_):
        for i in range(nx_):
            if i + 1 < nx_ and rng.random() > 0.08:
                edges.append(((i, j), (i + 1, j)))
            if j + 1 < ny_ and rng.random() > 0.14:
                edges.append(((i, j), (i, j + 1)))
    return pos, edges


def _rewire_schematic(pos, edges, seed, n_target, length_tol=None, non_crossing=False):
    """Degree-preserving double-edge swaps mirroring the real null constraints."""
    rng = np.random.default_rng(seed)
    edges = [tuple(e) for e in edges]
    edge_set = set(map(frozenset, edges))

    def chord(e) -> float:
        (x1, y1), (x2, y2) = pos[e[0]], pos[e[1]]
        return float(np.hypot(x2 - x1, y2 - y1))

    accepted, attempts = 0, 0
    while accepted < n_target and attempts < 6000:
        attempts += 1
        i, j = rng.choice(len(edges), size=2, replace=False)
        (a, b), (c, d) = edges[i], edges[j]
        if len({a, b, c, d}) < 4:
            continue
        new1, new2 = (a, c), (b, d)
        if frozenset(new1) in edge_set or frozenset(new2) in edge_set:
            continue
        if length_tol is not None:
            if abs(chord(new1) - chord(edges[i])) > length_tol or abs(chord(new2) - chord(edges[j])) > length_tol:
                continue
        if non_crossing:
            others = [edges[k] for k in range(len(edges)) if k not in (i, j)]
            blocked = _segments_cross(pos[new1[0]], pos[new1[1]], pos[new2[0]], pos[new2[1]])
            for u, v in () if blocked else (new1, new2):
                for p, q in others:
                    if _segments_cross(pos[u], pos[v], pos[p], pos[q]):
                        blocked = True
                        break
                if blocked:
                    break
            if blocked:
                continue
        edge_set.discard(frozenset(edges[i]))
        edge_set.discard(frozenset(edges[j]))
        edges[i], edges[j] = new1, new2
        edge_set.add(frozenset(new1))
        edge_set.add(frozenset(new2))
        accepted += 1
    return edges


def _draw_glyph(ax, pos, edges, base_edges, x0, highlight, title, subtitle) -> None:
    base_set = set(map(frozenset, base_edges))
    for u, v in edges:
        changed = frozenset((u, v)) not in base_set
        ax.plot(
            [pos[u][0] + x0, pos[v][0] + x0],
            [pos[u][1], pos[v][1]],
            lw=1.25 if changed else 0.85,
            color=highlight if changed else "#C2C7CC",
            zorder=3 if changed else 2,
            solid_capstyle="round",
        )
    xs = [p[0] + x0 for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.scatter(xs, ys, s=5.0, color="#3B3F44", zorder=4, linewidths=0)
    cx = x0 + 2.5
    ax.text(cx, -0.75, title, ha="center", va="top", fontsize=6.8, fontweight="bold", color="#1A1A1A")
    ax.text(cx, -1.32, subtitle, ha="center", va="top", fontsize=6.0, color=pub_style.COLORS["annot"])


def draw_null_ladder_schematic(ax) -> None:
    """Panel a: explanatory schematic of the constrained null-model ladder."""
    pos, base = _toy_street_patch()
    observed = list(base)
    degree_null = _rewire_schematic(pos, base, seed=21, n_target=6)
    spatial_null = _rewire_schematic(pos, base, seed=33, n_target=7, length_tol=0.34)
    geometry_null = _rewire_schematic(pos, base, seed=46, n_target=6, length_tol=0.34, non_crossing=True)

    gap = 2.1
    width = 5.0
    glyphs = [
        (observed, pub_style.COLORS["observed"], "Observed road graph", "finite, spatially embedded"),
        (degree_null, "#6E6E6E", "Degree null", "degree sequence preserved"),
        (spatial_null, pub_style.COLORS["spatial_null"], "Spatial-scale null", "+ chord-length bins"),
        (geometry_null, pub_style.COLORS["geometry_null"], "Strict geometry null", "+ non-crossing chords"),
    ]
    arrow_labels = ["degree-preserving\nrewiring", "+ length bins", "+ non-crossing"]
    for k, (edges, color, title, subtitle) in enumerate(glyphs):
        x0 = k * (width + gap)
        _draw_glyph(ax, pos, edges, base, x0, color, title, subtitle)
        if k < len(glyphs) - 1:
            x_a0 = x0 + width + 0.25
            x_a1 = x0 + width + gap - 0.45
            ax.annotate(
                "",
                xy=(x_a1, 1.5),
                xytext=(x_a0, 1.5),
                arrowprops={"arrowstyle": "-|>", "color": "#4D4D4D", "lw": 0.9},
            )
            ax.text(
                (x_a0 + x_a1) / 2,
                1.78,
                arrow_labels[k],
                ha="center",
                va="bottom",
                fontsize=5.8,
                color=pub_style.COLORS["annot"],
            )
    ax.set_xlim(-0.6, 3 * (width + gap) + width + 0.4)
    ax.set_ylim(-1.95, 3.75)
    ax.set_aspect("equal")
    ax.axis("off")


def label_feature(name: str) -> str:
    return (
        name.replace("_10deg", "")
        .replace("_share", " share")
        .replace("_fraction", " fraction")
        .replace("_", " ")
        .replace("largest biconnected node", "largest biconnected")
        .replace("dual degree", "dual-degree")
    )


def build_absorption_table(r68: pd.DataFrame, cov: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["city", "country", "macro_region", *SELECTED_GEOMETRY_FEATURES]
    merged = r68.merge(cov[cols], on=["city", "country", "macro_region"], how="left")
    merged["geometry_absorption"] = merged["road_minus_spatial"] - merged["road_minus_geometry_mean"]
    merged["geometry_absorption_fraction_of_road_gap"] = merged["geometry_absorption"] / merged["road_gap"]
    rows: list[dict[str, object]] = []
    for feature in SELECTED_GEOMETRY_FEATURES:
        sub = merged[[feature, "geometry_absorption", "geometry_absorption_fraction_of_road_gap"]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        if len(sub) < 20 or sub[feature].nunique() < 3:
            continue
        sp = spearmanr(sub[feature], sub["geometry_absorption"])
        pr = pearsonr(sub[feature], sub["geometry_absorption"])
        rows.append(
            {
                "feature": feature,
                "feature_label": label_feature(feature),
                "n": int(len(sub)),
                "spearman_rho": float(sp.statistic),
                "spearman_p": float(sp.pvalue),
                "pearson_r": float(pr.statistic),
                "pearson_p": float(pr.pvalue),
                "abs_spearman_rho": float(abs(sp.statistic)),
            }
        )
    corr = pd.DataFrame(rows).sort_values("abs_spearman_rho", ascending=False)
    return merged, corr


def build_kappa_prediction(cov: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    work = cov[["city", "country", "macro_region", "kappa", "road_gap"]].replace([np.inf, -np.inf], np.nan).dropna()
    X = work[["kappa"]].to_numpy(float)
    y = work["road_gap"].to_numpy(float)
    groups = work["macro_region"].astype(str).to_numpy()
    cv5 = KFold(n_splits=5, shuffle=True, random_state=20260605)
    logo = LeaveOneGroupOut()
    pred_cv5 = cross_val_predict(LinearRegression(), X, y, cv=cv5)
    pred_lro = cross_val_predict(LinearRegression(), X, y, cv=logo.split(X, y, groups))
    model = LinearRegression().fit(X, y)
    pred_fit = model.predict(X)
    sp = spearmanr(work["kappa"], work["road_gap"])
    pr = pearsonr(work["kappa"], work["road_gap"])
    out = work.copy()
    out["predicted_cv5"] = pred_cv5
    out["predicted_leave_region_out"] = pred_lro
    out["predicted_fit"] = pred_fit
    metrics = {
        "n": float(len(work)),
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "in_sample_r2": float(r2_score(y, pred_fit)),
        "cv5_r2": float(r2_score(y, pred_cv5)),
        "leave_region_out_r2": float(r2_score(y, pred_lro)),
        "cv5_mae": float(mean_absolute_error(y, pred_cv5)),
        "leave_region_out_mae": float(mean_absolute_error(y, pred_lro)),
        "spearman_rho": float(sp.statistic),
        "spearman_p": float(sp.pvalue),
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
    }
    return out, metrics


def make_figure(
    matched_summary: pd.DataFrame,
    matched_city: pd.DataFrame,
    absorption: pd.DataFrame,
    absorption_corr: pd.DataFrame,
    kappa_pred: pd.DataFrame,
    kappa_metrics: dict[str, float],
) -> None:
    set_style()
    colors = {
        "spatial": pub_style.COLORS["spatial_null"],
        "geometry": pub_style.COLORS["geometry_null"],
        "kappa": "#7C7C7C",
        "absorb": pub_style.COLORS["accent"],
        "neutral": pub_style.COLORS["observed"],
        "light": pub_style.COLORS["grid"],
    }
    fig = plt.figure(figsize=(pub_style.FIG_WIDTH_2COL, 7.9), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.62, 0.92, 1.02])
    ax_s = fig.add_subplot(gs[0, :])
    ax_a = fig.add_subplot(gs[1, :])
    ax_b = fig.add_subplot(gs[2, 0])
    ax_c = fig.add_subplot(gs[2, 1])

    draw_null_ladder_schematic(ax_s)
    pub_style.panel_title(ax_s, "a", "Constrained null-model ladder (schematic)")

    fractions = matched_summary["swap_fraction"].to_numpy()
    for null_name, color, mean_col, lo_col, hi_col, label in [
        (
            "spatial",
            colors["spatial"],
            "road_minus_spatial_mean",
            "road_minus_spatial_ci95_low",
            "road_minus_spatial_ci95_high",
            "Spatial-scale null",
        ),
        (
            "geometry",
            colors["geometry"],
            "road_minus_geometry_mean",
            "road_minus_geometry_ci95_low",
            "road_minus_geometry_ci95_high",
            "Strict geometry null",
        ),
    ]:
        y = matched_summary[mean_col]
        yerr = np.vstack([y - matched_summary[lo_col], matched_summary[hi_col] - y])
        ax_a.errorbar(fractions, y, yerr=yerr, marker="o", lw=1.7, capsize=2.5, color=color, label=label)
    rng = np.random.default_rng(77)
    for f in fractions:
        sub = matched_city[np.isclose(matched_city["swap_fraction"], f)]
        xj = f * np.exp(rng.normal(0, 0.035, size=len(sub)))
        ax_a.scatter(xj, sub["road_minus_spatial"], s=9, color=colors["spatial"], alpha=0.25, linewidth=0)
        ax_a.scatter(xj, sub["road_minus_geometry_mean"], s=9, color=colors["geometry"], alpha=0.25, linewidth=0)
    ax_a.axhline(0, color=colors["neutral"], lw=0.75)
    ax_a.set_xscale("log")
    ax_a.xaxis.set_major_locator(FixedLocator(fractions))
    ax_a.xaxis.set_major_formatter(FixedFormatter([f"{x:g}" for x in fractions]))
    ax_a.xaxis.set_minor_formatter(NullFormatter())
    ax_a.tick_params(axis="x", which="minor", bottom=False)
    ax_a.set_xlabel("Matched accepted-swap target")
    ax_a.set_ylabel("Road minus null threshold")
    pub_style.panel_title(ax_a, "b", "Matched-intensity null contrast (21 cities)")
    pub_style.light_grid(ax_a, axis="y")
    ax_a.legend(fontsize=6.3, loc="upper left")

    top = absorption_corr.iloc[0]
    feature = str(top["feature"])
    sub = absorption[[feature, "geometry_absorption", "macro_region"]].dropna()
    ax_b.scatter(
        sub[feature],
        sub["geometry_absorption"],
        s=22,
        color=colors["absorb"],
        alpha=0.78,
        edgecolor="white",
        linewidth=0.25,
    )
    if len(sub) > 2:
        xline = np.linspace(sub[feature].min(), sub[feature].max(), 80)
        coef = np.polyfit(sub[feature], sub["geometry_absorption"], 1)
        ax_b.plot(xline, coef[0] * xline + coef[1], color=colors["neutral"], lw=0.9)
    ax_b.set_xlabel(label_feature(feature).capitalize())
    ax_b.set_ylabel("Spatial residual absorbed by geometry")
    pub_style.panel_title(ax_b, "c", "Geometry absorption versus road form")
    pub_style.annot(ax_b, 0.03, 0.96, rf"Spearman $\rho$ = {top['spearman_rho']:.2f}; $n$ = {int(top['n'])}")
    pub_style.light_grid(ax_b, axis="both")

    ax_c.scatter(
        kappa_pred["kappa"],
        kappa_pred["road_gap"],
        s=22,
        color=colors["kappa"],
        alpha=0.78,
        edgecolor="white",
        linewidth=0.25,
        label="Observed",
    )
    order = np.argsort(kappa_pred["kappa"].to_numpy())
    ax_c.plot(
        kappa_pred["kappa"].to_numpy()[order],
        kappa_pred["predicted_fit"].to_numpy()[order],
        color=colors["neutral"],
        lw=1.0,
        label="Linear fit",
    )
    ax_c.scatter(
        kappa_pred["kappa"],
        kappa_pred["predicted_leave_region_out"],
        s=13,
        color="none",
        edgecolor=pub_style.COLORS["model"],
        linewidth=0.7,
        label="Held-region prediction",
    )
    ax_c.set_xlabel(r"$\kappa=\langle k^2\rangle/\langle k\rangle$")
    ax_c.set_ylabel("Road minus CEBH threshold")
    pub_style.panel_title(ax_c, "d", "Road gap versus degree-moment ratio")
    pub_style.annot(
        ax_c,
        0.03,
        0.96,
        rf"Spearman $\rho$ = {kappa_metrics['spearman_rho']:.2f}; LRO $R^2$ = {kappa_metrics['leave_region_out_r2']:.2f}; $n$ = {int(kappa_metrics['n'])}",
    )
    pub_style.light_grid(ax_c, axis="both")
    ax_c.legend(fontsize=6.0, loc="lower left")

    pub_style.save(fig, FIG_BASE)
    plt.close(fig)


def update_source_data_manifest(target: Path, files: list[tuple[Path, str]]) -> None:
    sd = target / "source_data"
    manifest = sd / "Supplementary_Data_1_source_data_manifest.csv"
    df = pd.read_csv(manifest)
    for src, copied_file in files:
        if src.suffix == ".csv":
            data = pd.read_csv(src)
            n_rows = int(data.shape[0])
            n_columns_or_keys = int(data.shape[1])
        else:
            payload = json.loads(src.read_text(encoding="utf-8"))
            n_rows = 1
            n_columns_or_keys = len(payload) if isinstance(payload, dict) else 1
        entry = {
            "round": "R77",
            "source_role": "physics takeaway and matched-intensity main-result source data",
            "source_path": f"data/R77_physics_takeaway/{src.name}",
            "copied_file": copied_file,
            "format": src.suffix.lstrip("."),
            "n_rows": n_rows,
            "n_columns_or_keys": n_columns_or_keys,
            "sha256": sha256_file(src),
            "redistribution_boundary": "derived project table; raw OSM/Geofabrik archives excluded",
        }
        df = df[df["copied_file"] != copied_file]
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    df.to_csv(manifest, index=False)

    claim_map = sd / "Supplementary_Data_3_claim_to_table_map.csv"
    cm = pd.read_csv(claim_map)
    new_claims = pd.DataFrame(
        [
            {
                "claim": "Matched-intensity spatial and strict-geometry nulls diverge at the same accepted-swap targets",
                "supporting_source_data_files": "R77_matched_intensity_main_result_source_data.csv",
                "manuscript_location": "Main text Results 2.1 and Fig. 1",
            },
            {
                "claim": "Geometry absorption is associated with embedded road-form and block-structure metrics",
                "supporting_source_data_files": "R77_geometry_absorption_correlations.csv; R77_geometry_absorption_city_table.csv",
                "manuscript_location": "Main text Results 2.4 and Fig. 1",
            },
            {
                "claim": "Kappa provides a theory-anchored one-variable predictive relation for the road-minus-CEBH gap",
                "supporting_source_data_files": "R77_kappa_predictive_relation.csv; R77_kappa_predictive_metrics.json",
                "manuscript_location": "Main text Results 2.4 and Fig. 1",
            },
        ]
    )
    cm = cm[~cm["claim"].isin(new_claims["claim"])]
    pd.concat([cm, new_claims], ignore_index=True).to_csv(claim_map, index=False)


def copy_to_targets() -> None:
    figure_files = [FIG_BASE.with_suffix(ext) for ext in [".svg", ".pdf", ".png", ".tiff"]]
    source_specs = [
        (OUT / "R77_matched_intensity_main_result_source_data.csv", "figures/R77_matched_intensity_main_result_source_data.csv"),
        (OUT / "R77_geometry_absorption_city_table.csv", "tables/R77_geometry_absorption_city_table.csv"),
        (OUT / "R77_geometry_absorption_correlations.csv", "tables/R77_geometry_absorption_correlations.csv"),
        (OUT / "R77_kappa_predictive_relation.csv", "tables/R77_kappa_predictive_relation.csv"),
        (OUT / "R77_kappa_predictive_metrics.json", "tables/R77_kappa_predictive_metrics.json"),
    ]
    for target in TARGETS:
        fig_dir = target / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        for ext_file in figure_files:
            shutil.copy2(ext_file, fig_dir / f"Fig1_matched_geometry_partition{ext_file.suffix}")
        # Move old schematic out of upload-facing figure sequence if present.
        legacy = fig_dir / "legacy"
        for old in fig_dir.glob("Fig1_null_ladder_schematic.*"):
            legacy.mkdir(exist_ok=True)
            shutil.move(str(old), str(legacy / old.name))
        for src, copied in source_specs:
            dst = target / "source_data" / copied
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        update_source_data_manifest(target, source_specs)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ROUND_DIR.mkdir(parents=True, exist_ok=True)
    matched_summary = pd.read_csv(MATCHED_SUMMARY)
    matched_city = pd.read_csv(MATCHED_CITY)
    r68 = pd.read_csv(R68_CITY)
    cov = pd.read_csv(R63)

    absorption, absorption_corr = build_absorption_table(r68, cov)
    kappa_pred, kappa_metrics = build_kappa_prediction(cov)

    matched_summary.to_csv(OUT / "R77_matched_intensity_main_result_source_data.csv", index=False)
    absorption.to_csv(OUT / "R77_geometry_absorption_city_table.csv", index=False)
    absorption_corr.to_csv(OUT / "R77_geometry_absorption_correlations.csv", index=False)
    kappa_pred.to_csv(OUT / "R77_kappa_predictive_relation.csv", index=False)
    (OUT / "R77_kappa_predictive_metrics.json").write_text(json.dumps(kappa_metrics, indent=2), encoding="utf-8")

    make_figure(matched_summary, matched_city, absorption, absorption_corr, kappa_pred, kappa_metrics)
    copy_to_targets()

    payload = {
        "status": "pass",
        "round": "R77_physics_takeaway",
        "matched_intensity_spatial_residuals": matched_summary["road_minus_spatial_mean"].round(6).tolist(),
        "matched_intensity_geometry_residuals": matched_summary["road_minus_geometry_mean"].round(6).tolist(),
        "top_absorption_feature": absorption_corr.iloc[0].to_dict(),
        "kappa_metrics": kappa_metrics,
        "claim_boundary": (
            "R77 promotes the matched-intensity contrast and a kappa-anchored "
            "predictive relation. It remains empirical, descriptive and model-conditional."
        ),
    }
    (OUT / "R77_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (ROUND_DIR / "gate_report.md").write_text(
        "# R77 Physics Takeaway Gate Report\n\n"
        "| Gate | Status | Evidence |\n"
        "|---|---|---|\n"
        "| Matched-intensity main-result source data | Pass | `data/R77_physics_takeaway/R77_matched_intensity_main_result_source_data.csv` |\n"
        "| Geometry absorption correlations | Pass | `data/R77_physics_takeaway/R77_geometry_absorption_correlations.csv` |\n"
        "| Kappa predictive relation | Pass | `data/R77_physics_takeaway/R77_kappa_predictive_relation.csv` |\n"
        "| Fig. 1 replacement generated | Pass | `figures/Fig_R77_physics_takeaway.*` copied to target packages as `Fig1_matched_geometry_partition.*` |\n"
        "| Claim boundary | Pass | Empirical, descriptive, no theorem, no causal design rule, no true edge-Jaccard. |\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
