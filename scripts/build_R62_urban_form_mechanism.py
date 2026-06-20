"""R62 urban-form covariates and stratified geometry-null pilots.

R61 showed that the CEBH road-threshold gap persists across 71 standardized
city windows. R62 adds a city-mechanism layer: road-form covariates are
extracted for all 71 windows, then a stratified subset is passed through a
strict non-crossing degree-preserving geometry-null pilot.

The outputs are descriptive and mechanism-targeting. They are not a causal
urban-form model, not a full 71-city geometry-null ensemble, and not a Nature
Cities-ready urban policy claim by themselves.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_R56_spatial_length_constrained_nulls as r56  # noqa: E402
import build_R60_expansion_geometry_nulls as r60  # noqa: E402
import pub_style  # noqa: E402


OUT = ROOT / "data" / "R62_urban_form_mechanism"
ROUND_DIR = ROOT / "rounds" / "R62_urban_form_mechanism"
FIG_BASE = ROOT / "figures" / "Fig_R62_urban_form_mechanism"
R61_RESULTS = ROOT / "data" / "R61_city_window_scale_screen" / "central_window_screen_results.csv"

REGION_ORDER = ["Europe", "Middle East", "Africa", "Latin America", "Asia", "North America", "Oceania"]
REGION_COLORS = pub_style.REGION_COLORS

ARTERIAL = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
}
COLLECTOR = {"tertiary", "tertiary_link", "unclassified"}
LOCAL = {"residential", "living_street", "service", "road"}


@dataclass
class CovariateRow:
    city: str
    country: str
    macro_region: str
    latitude: float
    longitude: float
    window_radius_m: int
    graph_cache_file: str
    bbox_area_km2: float
    n_nodes: int
    n_edges: int
    pc_cebh: float
    pc_rank: float
    road_gap: float
    node_density_per_km2: float
    edge_density_per_km2: float
    chord_length_density_km_per_km2: float
    edge_length_mean_m: float
    edge_length_median_m: float
    edge_length_iqr_m: float
    edge_length_cv: float
    short_edge_share_lt40m: float
    long_edge_share_gt150m: float
    orientation_entropy: float
    orientation_dominance: float
    orientation_fourfold_order: float
    orthogonal_alignment_share_10deg: float
    dead_end_share: float
    degree2_share: float
    degree3_share: float
    degree4plus_share: float
    mean_degree: float
    kappa: float
    degree_gini: float
    connected_components: int
    cyclomatic_number: int
    cycle_density: float
    arterial_length_share: float
    collector_length_share: float
    local_length_share: float
    motorway_trunk_length_share: float
    residential_length_share: float
    oneway_length_share: float
    bridge_tunnel_length_share: float
    maxspeed_coverage_share: float
    lanes_coverage_share: float
    edge_circuity_mean: float
    edge_circuity_median: float


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ROUND_DIR.mkdir(parents=True, exist_ok=True)
    FIG_BASE.parent.mkdir(parents=True, exist_ok=True)


def load_graph(path: Path) -> Any:
    with path.open("rb") as fh:
        return pickle.load(fh)


def as_path(value: str) -> Path:
    p = Path(str(value))
    return p if p.is_absolute() else ROOT / p


def degree_gini(degrees: np.ndarray) -> float:
    if len(degrees) == 0:
        return float("nan")
    x = np.sort(degrees.astype(float))
    total = float(np.sum(x))
    if total <= 0:
        return 0.0
    n = len(x)
    idx = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.sum(idx * x)) / (n * total) - (n + 1.0) / n)


def orientation_metrics(edges: np.ndarray, lon: np.ndarray, lat: np.ndarray, n_bins: int = 18) -> dict[str, float]:
    if len(edges) == 0:
        return {
            "orientation_entropy": float("nan"),
            "orientation_dominance": float("nan"),
            "orientation_fourfold_order": float("nan"),
            "orthogonal_alignment_share_10deg": float("nan"),
        }
    mean_lat = float(np.nanmean(lat))
    scale_x = 111_320.0 * math.cos(math.radians(mean_lat))
    scale_y = 110_540.0
    dx = (lon[edges[:, 1]] - lon[edges[:, 0]]) * scale_x
    dy = (lat[edges[:, 1]] - lat[edges[:, 0]]) * scale_y
    theta = np.mod(np.arctan2(dy, dx), math.pi)
    counts, _ = np.histogram(theta, bins=np.linspace(0.0, math.pi, n_bins + 1))
    probs = counts.astype(float) / max(1.0, float(np.sum(counts)))
    nz = probs[probs > 0]
    entropy = -float(np.sum(nz * np.log(nz))) / math.log(n_bins)
    dominance = float(np.max(probs))
    fourfold = float(abs(np.mean(np.exp(4j * theta))))
    d_orth = np.minimum.reduce([theta, np.abs(theta - math.pi / 2.0), np.abs(theta - math.pi)])
    orth_share = float(np.mean(d_orth <= math.radians(10.0)))
    return {
        "orientation_entropy": entropy,
        "orientation_dominance": dominance,
        "orientation_fourfold_order": fourfold,
        "orthogonal_alignment_share_10deg": orth_share,
    }


def highway_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if v is not None]
    return [str(value)]


def hierarchy_label(highway: Any) -> str:
    vals = set(highway_values(highway))
    if vals & ARTERIAL:
        return "arterial"
    if vals & COLLECTOR:
        return "collector"
    if vals & LOCAL:
        return "local"
    if vals:
        return "other"
    return "unknown"


def raw_edge_metrics(G: Any) -> dict[str, float]:
    length_total = 0.0
    length_by_class = {"arterial": 0.0, "collector": 0.0, "local": 0.0, "other": 0.0, "unknown": 0.0}
    motorway_trunk = 0.0
    residential = 0.0
    oneway = 0.0
    bridge_tunnel = 0.0
    maxspeed_len = 0.0
    lanes_len = 0.0
    circuity: list[float] = []
    for u, v, data in G.edges(data=True):
        length = data.get("length", np.nan)
        try:
            length = float(length)
        except (TypeError, ValueError):
            length = float("nan")
        if not math.isfinite(length) or length <= 0:
            continue
        length_total += length
        label = hierarchy_label(data.get("highway"))
        length_by_class[label] += length
        vals = set(highway_values(data.get("highway")))
        if vals & {"motorway", "motorway_link", "trunk", "trunk_link"}:
            motorway_trunk += length
        if "residential" in vals:
            residential += length
        if bool(data.get("oneway")):
            oneway += length
        if data.get("bridge") is not None or data.get("tunnel") is not None:
            bridge_tunnel += length
        if data.get("maxspeed") is not None:
            maxspeed_len += length
        if data.get("lanes") is not None:
            lanes_len += length
        try:
            lon1 = float(G.nodes[u].get("x"))
            lat1 = float(G.nodes[u].get("y"))
            lon2 = float(G.nodes[v].get("x"))
            lat2 = float(G.nodes[v].get("y"))
            chord = r56.chord_length_m(lon1, lat1, lon2, lat2)
            if chord > 1.0:
                ratio = length / chord
                if math.isfinite(ratio) and 0.5 <= ratio <= 20.0:
                    circuity.append(ratio)
        except Exception:
            pass
    denom = max(length_total, 1e-9)
    return {
        "arterial_length_share": float(length_by_class["arterial"] / denom),
        "collector_length_share": float(length_by_class["collector"] / denom),
        "local_length_share": float(length_by_class["local"] / denom),
        "motorway_trunk_length_share": float(motorway_trunk / denom),
        "residential_length_share": float(residential / denom),
        "oneway_length_share": float(oneway / denom),
        "bridge_tunnel_length_share": float(bridge_tunnel / denom),
        "maxspeed_coverage_share": float(maxspeed_len / denom),
        "lanes_coverage_share": float(lanes_len / denom),
        "edge_circuity_mean": float(np.nanmean(circuity)) if circuity else float("nan"),
        "edge_circuity_median": float(np.nanmedian(circuity)) if circuity else float("nan"),
    }


def covariates_for_row(row: pd.Series) -> CovariateRow:
    import networkx as nx

    graph_path = as_path(str(row["graph_cache_file"]))
    G = load_graph(graph_path)
    H, edges, degrees, lon, lat = r56.to_simple_spatial_graph(G)
    lengths = r56.chord_lengths_for_edges(edges, lon, lat) if len(edges) else np.array([], dtype=float)
    finite_lengths = lengths[np.isfinite(lengths) & (lengths > 0)]
    radius_m = int(row["window_radius_m"])
    bbox_area_km2 = float((2.0 * radius_m / 1000.0) ** 2)
    n = int(H.number_of_nodes())
    m = int(H.number_of_edges())
    comps = int(nx.number_connected_components(H)) if n else 0
    cyclomatic = int(max(0, m - n + comps))
    k_mean = float(np.mean(degrees)) if len(degrees) else float("nan")
    kappa = float(np.mean(degrees.astype(float) ** 2) / k_mean) if k_mean and math.isfinite(k_mean) else float("nan")
    om = orientation_metrics(edges, lon, lat)
    rem = raw_edge_metrics(G)
    del G, H
    gc.collect()

    return CovariateRow(
        city=str(row["city"]),
        country=str(row["country"]),
        macro_region=str(row["macro_region"]),
        latitude=float(row["latitude"]),
        longitude=float(row["longitude"]),
        window_radius_m=radius_m,
        graph_cache_file=str(row["graph_cache_file"]),
        bbox_area_km2=bbox_area_km2,
        n_nodes=n,
        n_edges=m,
        pc_cebh=float(row["pc_cebh"]),
        pc_rank=float(row["pc_rank"]),
        road_gap=float(row["road_gap"]),
        node_density_per_km2=float(n / bbox_area_km2),
        edge_density_per_km2=float(m / bbox_area_km2),
        chord_length_density_km_per_km2=float(np.sum(finite_lengths) / 1000.0 / bbox_area_km2) if len(finite_lengths) else float("nan"),
        edge_length_mean_m=float(np.nanmean(finite_lengths)) if len(finite_lengths) else float("nan"),
        edge_length_median_m=float(np.nanmedian(finite_lengths)) if len(finite_lengths) else float("nan"),
        edge_length_iqr_m=float(np.nanpercentile(finite_lengths, 75) - np.nanpercentile(finite_lengths, 25)) if len(finite_lengths) else float("nan"),
        edge_length_cv=float(np.nanstd(finite_lengths) / np.nanmean(finite_lengths)) if len(finite_lengths) and np.nanmean(finite_lengths) else float("nan"),
        short_edge_share_lt40m=float(np.mean(finite_lengths < 40.0)) if len(finite_lengths) else float("nan"),
        long_edge_share_gt150m=float(np.mean(finite_lengths > 150.0)) if len(finite_lengths) else float("nan"),
        orientation_entropy=om["orientation_entropy"],
        orientation_dominance=om["orientation_dominance"],
        orientation_fourfold_order=om["orientation_fourfold_order"],
        orthogonal_alignment_share_10deg=om["orthogonal_alignment_share_10deg"],
        dead_end_share=float(np.mean(degrees == 1)) if len(degrees) else float("nan"),
        degree2_share=float(np.mean(degrees == 2)) if len(degrees) else float("nan"),
        degree3_share=float(np.mean(degrees == 3)) if len(degrees) else float("nan"),
        degree4plus_share=float(np.mean(degrees >= 4)) if len(degrees) else float("nan"),
        mean_degree=k_mean,
        kappa=kappa,
        degree_gini=degree_gini(degrees),
        connected_components=comps,
        cyclomatic_number=cyclomatic,
        cycle_density=float(cyclomatic / max(1, n)),
        arterial_length_share=rem["arterial_length_share"],
        collector_length_share=rem["collector_length_share"],
        local_length_share=rem["local_length_share"],
        motorway_trunk_length_share=rem["motorway_trunk_length_share"],
        residential_length_share=rem["residential_length_share"],
        oneway_length_share=rem["oneway_length_share"],
        bridge_tunnel_length_share=rem["bridge_tunnel_length_share"],
        maxspeed_coverage_share=rem["maxspeed_coverage_share"],
        lanes_coverage_share=rem["lanes_coverage_share"],
        edge_circuity_mean=rem["edge_circuity_mean"],
        edge_circuity_median=rem["edge_circuity_median"],
    )


PREDICTORS = [
    "node_density_per_km2",
    "chord_length_density_km_per_km2",
    "edge_length_median_m",
    "edge_length_iqr_m",
    "edge_length_cv",
    "orientation_entropy",
    "orientation_dominance",
    "orientation_fourfold_order",
    "orthogonal_alignment_share_10deg",
    "dead_end_share",
    "degree2_share",
    "degree3_share",
    "degree4plus_share",
    "mean_degree",
    "kappa",
    "degree_gini",
    "cycle_density",
    "arterial_length_share",
    "collector_length_share",
    "local_length_share",
    "motorway_trunk_length_share",
    "residential_length_share",
    "oneway_length_share",
    "bridge_tunnel_length_share",
    "maxspeed_coverage_share",
    "lanes_coverage_share",
    "edge_circuity_median",
]


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    y = pd.to_numeric(df["road_gap"], errors="coerce").to_numpy(dtype=float)
    for col in PREDICTORS:
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if int(mask.sum()) < 8 or np.nanstd(x[mask]) <= 0:
            continue
        sp = spearmanr(x[mask], y[mask])
        pr = pearsonr(x[mask], y[mask])
        records.append(
            {
                "covariate": col,
                "n": int(mask.sum()),
                "spearman_rho": float(sp.statistic),
                "spearman_p": float(sp.pvalue),
                "pearson_r": float(pr.statistic),
                "pearson_p": float(pr.pvalue),
                "abs_spearman_rho": float(abs(sp.statistic)),
            }
        )
    out = pd.DataFrame(records)
    return out.sort_values("abs_spearman_rho", ascending=False).reset_index(drop=True)


def zscore_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = df[cols].apply(pd.to_numeric, errors="coerce").copy()
    for col in cols:
        mu = float(X[col].mean())
        sd = float(X[col].std(ddof=0))
        if not math.isfinite(sd) or sd == 0:
            X[col] = 0.0
        else:
            X[col] = (X[col].fillna(mu) - mu) / sd
    return X


def predictive_screen(df: pd.DataFrame) -> dict[str, Any]:
    model_cols = [
        "node_density_per_km2",
        "edge_length_median_m",
        "orientation_fourfold_order",
        "dead_end_share",
        "degree4plus_share",
        "cycle_density",
        "arterial_length_share",
        "oneway_length_share",
        "edge_circuity_median",
    ]
    X = zscore_frame(df, model_cols).to_numpy(dtype=float)
    y = pd.to_numeric(df["road_gap"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(y)
    X = X[mask]
    y = y[mask]
    try:
        from sklearn.linear_model import RidgeCV
        from sklearn.model_selection import KFold, cross_val_score

        alphas = np.logspace(-3, 3, 31)
        model = RidgeCV(alphas=alphas).fit(X, y)
        cv = KFold(n_splits=5, shuffle=True, random_state=6201)
        scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
        coefs = dict(zip(model_cols, [float(v) for v in model.coef_]))
        return {
            "model": "standardized RidgeCV",
            "n": int(len(y)),
            "predictors": model_cols,
            "alpha": float(model.alpha_),
            "in_sample_r2": float(model.score(X, y)),
            "cv_r2_mean": float(np.mean(scores)),
            "cv_r2_sd": float(np.std(scores, ddof=1)),
            "coefficients": coefs,
            "interpretation": "descriptive predictive screen only; not a causal model",
        }
    except Exception as exc:
        X1 = np.column_stack([np.ones(len(X)), X])
        beta = np.linalg.lstsq(X1, y, rcond=None)[0]
        pred = X1 @ beta
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        coefs = dict(zip(model_cols, [float(v) for v in beta[1:]]))
        return {
            "model": "standardized OLS fallback",
            "n": int(len(y)),
            "predictors": model_cols,
            "in_sample_r2": float(1.0 - ss_res / ss_tot) if ss_tot else float("nan"),
            "coefficients": coefs,
            "warning": f"sklearn RidgeCV unavailable or failed: {type(exc).__name__}: {exc}",
            "interpretation": "descriptive predictive screen only; not a causal model",
        }


def pca_scores(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = zscore_frame(df, cols).to_numpy(dtype=float)
    _u, _s, vt = np.linalg.svd(X, full_matrices=False)
    scores = X @ vt[:2].T
    out = df[["city", "country", "macro_region", "road_gap", "n_edges"]].copy()
    out["morph_pc1"] = scores[:, 0]
    out["morph_pc2"] = scores[:, 1]
    return out


def select_geometry_subset(df: pd.DataFrame, max_cities: int, max_edges: int) -> pd.DataFrame:
    morph_cols = [
        "road_gap",
        "node_density_per_km2",
        "edge_length_median_m",
        "orientation_fourfold_order",
        "dead_end_share",
        "arterial_length_share",
        "cycle_density",
    ]
    z = zscore_frame(df, morph_cols)
    work = df.copy()
    work["_morph_norm"] = np.sqrt((z**2).sum(axis=1))
    work["_edge_penalty"] = np.log10(work["n_edges"].clip(lower=1)) / 10.0
    selected: list[pd.Series] = []
    selected_cities: set[str] = set()
    for region in REGION_ORDER:
        group = work[(work["macro_region"] == region) & (work["n_edges"] <= max_edges)].copy()
        if group.empty:
            group = work[work["macro_region"] == region].copy()
        if group.empty:
            continue
        med = group[morph_cols].apply(pd.to_numeric, errors="coerce").median()
        dist = []
        for idx, row in group.iterrows():
            vec = pd.to_numeric(row[morph_cols], errors="coerce")
            scale = group[morph_cols].apply(pd.to_numeric, errors="coerce").std(ddof=0).replace(0, 1)
            d = float(np.sqrt((((vec - med) / scale).fillna(0) ** 2).sum()))
            d += float(row["_edge_penalty"])
            dist.append((d, idx))
        idx = min(dist)[1]
        selected.append(work.loc[idx])
        selected_cities.add(str(work.loc[idx, "city"]))
    if len(selected) < max_cities:
        extras = work[(work["n_edges"] <= max_edges) & (~work["city"].isin(selected_cities))].copy()
        extremes = []
        for col in ["road_gap", "orientation_fourfold_order", "arterial_length_share", "dead_end_share"]:
            extremes.extend(extras.nlargest(1, col).index.tolist())
            extremes.extend(extras.nsmallest(1, col).index.tolist())
        for idx in extremes:
            if len(selected) >= max_cities:
                break
            city = str(work.loc[idx, "city"])
            if city not in selected_cities:
                selected.append(work.loc[idx])
                selected_cities.add(city)
    if len(selected) < max_cities:
        eligible = work[(work["n_edges"] <= max_edges) & (~work["city"].isin(selected_cities))].copy()
        if not eligible.empty:
            feature_cols = morph_cols + [
                "orientation_entropy",
                "degree4plus_share",
                "collector_length_share",
                "oneway_length_share",
            ]
            z_all = zscore_frame(work, feature_cols)
            while len(selected) < max_cities:
                remaining = eligible[~eligible["city"].isin(selected_cities)]
                if remaining.empty:
                    break
                selected_idx = [int(row.name) for row in selected]
                best_idx = None
                best_score = -float("inf")
                for idx, row in remaining.iterrows():
                    vec = z_all.loc[idx].to_numpy(dtype=float)
                    selected_mat = z_all.loc[selected_idx].to_numpy(dtype=float)
                    d = float(np.min(np.sqrt(((selected_mat - vec) ** 2).sum(axis=1))))
                    d -= float(row["_edge_penalty"])
                    if d > best_score:
                        best_score = d
                        best_idx = idx
                if best_idx is None:
                    break
                selected.append(work.loc[best_idx])
                selected_cities.add(str(work.loc[best_idx, "city"]))
    out = pd.DataFrame(selected).drop(columns=[c for c in ["_morph_norm", "_edge_penalty"] if c in work.columns])
    out["selection_reason"] = "one_per_macro_region_or_morphology_extreme"
    return out.head(max_cities).reset_index(drop=True)


def run_geometry_nulls(selected: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rep_path = OUT / "geometry_null_pilot_replicates.csv"
    summary_path = OUT / "geometry_null_pilot_summary.csv"
    if rep_path.exists() and summary_path.exists() and not args.force_geometry:
        return pd.read_csv(rep_path), pd.read_csv(summary_path)

    rep_records: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []
    p_grid = np.arange(args.geom_p_min, args.geom_p_max + args.geom_p_step / 2.0, args.geom_p_step)
    p_grid = np.unique(np.round(p_grid, 6))

    for city_idx, row in selected.iterrows():
        city = str(row["city"])
        graph_path = as_path(str(row["graph_cache_file"]))
        print(f"[geometry-null] {city}", flush=True)
        G = load_graph(graph_path)
        _H, edges, degrees, lon, lat = r56.to_simple_spatial_graph(G)
        del G
        gc.collect()
        pc_cebh = float(row["pc_cebh"])
        pc_road = float(row["pc_rank"])
        road_gap = float(row["road_gap"])
        city_reps: list[dict[str, Any]] = []
        for rep in range(args.geom_reps):
            seed = args.seed + 1009 * (city_idx + 1) + rep
            t0 = time.time()
            try:
                rewired, diag = r60.geometry_preserving_rewire(
                    edges,
                    degrees,
                    lon,
                    lat,
                    seed=seed,
                    length_bins=args.geom_length_bins,
                    bin_tolerance=args.geom_bin_tolerance,
                    swap_fraction=args.geom_swap_fraction,
                    max_attempt_factor=args.geom_max_attempt_factor,
                    neighbor_k=args.geom_neighbor_k,
                    rebuild_interval=args.geom_rebuild_interval,
                )
                samples = r56.rank_order_samples(
                    len(degrees),
                    rewired[:, 0].astype(np.int64),
                    rewired[:, 1].astype(np.int64),
                    p_grid,
                    args.geom_n_perm,
                    seed + 50_000,
                )
                pc_null, ci_low, ci_high, boundary_hit = r56.bootstrap_pc_ci(
                    p_grid, samples, args.geom_n_bootstrap, seed + 80_000
                )
                geom_gap = float(pc_null - pc_cebh)
                record = {
                    "city": city,
                    "country": row["country"],
                    "macro_region": row["macro_region"],
                    "replicate": rep,
                    "n_nodes": int(row["n_nodes"]),
                    "n_edges": int(row["n_edges"]),
                    "pc_cebh": pc_cebh,
                    "pc_road": pc_road,
                    "road_gap": road_gap,
                    "pc_geometry_null": float(pc_null),
                    "pc_geometry_null_ci_low": float(ci_low),
                    "pc_geometry_null_ci_high": float(ci_high),
                    "geometry_null_gap": geom_gap,
                    "road_minus_geometry": float(pc_road - pc_null),
                    "threshold_boundary_hit": bool(boundary_hit),
                    "runtime_seconds": float(time.time() - t0),
                    **diag,
                    "status": "pass",
                    "notes": "strict non-crossing degree-preserving geometry-null pilot",
                }
            except Exception as exc:
                record = {
                    "city": city,
                    "country": row["country"],
                    "macro_region": row["macro_region"],
                    "replicate": rep,
                    "n_nodes": int(row["n_nodes"]),
                    "n_edges": int(row["n_edges"]),
                    "pc_cebh": pc_cebh,
                    "pc_road": pc_road,
                    "road_gap": road_gap,
                    "pc_geometry_null": float("nan"),
                    "geometry_null_gap": float("nan"),
                    "road_minus_geometry": float("nan"),
                    "threshold_boundary_hit": True,
                    "runtime_seconds": float(time.time() - t0),
                    "accepted_swaps": 0,
                    "attempted_swaps": 0,
                    "acceptance_rate": 0.0,
                    "accepted_swaps_per_edge": 0.0,
                    "degree_max_abs_drift": -1,
                    "length_bin_tvd": float("nan"),
                    "status": "fail",
                    "notes": f"{type(exc).__name__}: {exc}",
                }
            rep_records.append(record)
            city_reps.append(record)
            pd.DataFrame(rep_records).to_csv(rep_path, index=False)
        city_df = pd.DataFrame(city_reps)
        passed = city_df[city_df["status"] == "pass"]
        summary_records.append(
            {
                "city": city,
                "country": row["country"],
                "macro_region": row["macro_region"],
                "n_nodes": int(row["n_nodes"]),
                "n_edges": int(row["n_edges"]),
                "pc_cebh": pc_cebh,
                "pc_road": pc_road,
                "road_gap": road_gap,
                "pc_geometry_null_mean": float(passed["pc_geometry_null"].mean()) if len(passed) else float("nan"),
                "geometry_null_gap_mean": float(passed["geometry_null_gap"].mean()) if len(passed) else float("nan"),
                "road_minus_geometry_mean": float(passed["road_minus_geometry"].mean()) if len(passed) else float("nan"),
                "accepted_swaps_per_edge_mean": float(passed["accepted_swaps_per_edge"].mean()) if len(passed) else float("nan"),
                "acceptance_rate_mean": float(passed["acceptance_rate"].mean()) if len(passed) else float("nan"),
                "length_bin_tvd_mean": float(passed["length_bin_tvd"].mean()) if len(passed) else float("nan"),
                "degree_max_abs_drift_max": int(passed["degree_max_abs_drift"].max()) if len(passed) else -1,
                "n_replicates_pass": int(len(passed)),
                "n_replicates_total": int(len(city_df)),
            }
        )
        pd.DataFrame(summary_records).to_csv(summary_path, index=False)
        gc.collect()
    return pd.DataFrame(rep_records), pd.DataFrame(summary_records)


def set_plot_style() -> None:
    pub_style.apply()


def make_figure(cov: pd.DataFrame, corr: pd.DataFrame, pca: pd.DataFrame, selected: pd.DataFrame, geom: pd.DataFrame) -> None:
    set_plot_style()
    fig = plt.figure(figsize=(pub_style.FIG_WIDTH_2COL, 6.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0])

    ax_a = fig.add_subplot(gs[0, 0])
    for region, group in cov.groupby("macro_region"):
        ax_a.scatter(
            group["orientation_fourfold_order"],
            group["road_gap"],
            s=18 + np.log10(group["n_edges"].clip(lower=1)) * 13,
            color=REGION_COLORS.get(str(region), REGION_COLORS["Other"]),
            edgecolor="#222222",
            linewidth=0.35,
            alpha=0.82,
            label=str(region),
        )
    ax_a.set_xlabel("Fourfold orientation order")
    ax_a.set_ylabel("Road-minus-CEBH gap")
    pub_style.panel_title(ax_a, "a", "Urban-form screen (71 windows)")
    pub_style.light_grid(ax_a, axis="both")
    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
        ncol=7,
        fontsize=6.0,
        handletextpad=0.25,
        columnspacing=0.9,
        frameon=False,
    )

    ax_b = fig.add_subplot(gs[0, 1])
    top = corr.head(10).copy().iloc[::-1]
    colors = [
        pub_style.COLORS["geometry_null"] if v < 0 else pub_style.COLORS["spatial_null"]
        for v in top["spearman_rho"]
    ]
    labels = [c.replace("_", " ") for c in top["covariate"]]
    ax_b.barh(np.arange(len(top)), top["spearman_rho"], color=colors, alpha=0.88)
    ax_b.axvline(0, color="#222222", lw=0.8)
    ax_b.set_yticks(np.arange(len(top)))
    ax_b.set_yticklabels(labels, fontsize=6)
    ax_b.set_xlabel(r"Spearman $\rho$ with road-minus-CEBH gap")
    pub_style.panel_title(ax_b, "b", "Road-form correlations")
    pub_style.light_grid(ax_b, axis="x")

    ax_c = fig.add_subplot(gs[1, 0])
    selected_cities = set(selected["city"].astype(str))
    for region, group in pca.groupby("macro_region"):
        ax_c.scatter(
            group["morph_pc1"],
            group["morph_pc2"],
            s=32,
            color=REGION_COLORS.get(str(region), REGION_COLORS["Other"]),
            edgecolor="#222222",
            linewidth=0.25,
            alpha=0.58,
        )
    chosen = pca[pca["city"].isin(selected_cities)]
    ax_c.scatter(
        chosen["morph_pc1"],
        chosen["morph_pc2"],
        s=88,
        facecolor="none",
        edgecolor="#111111",
        linewidth=1.2,
        label="geometry-null subset",
    )
    # Label only morphology-space extremes to avoid overlapping text.
    chosen_sorted = chosen.assign(_r=np.hypot(chosen["morph_pc1"], chosen["morph_pc2"])).sort_values(
        "_r", ascending=False
    )
    label_offsets = {
        "Dubai": (-0.95, -0.40),
        "Rio de Janeiro": (-1.35, -0.30),
        "Buenos Aires": (0.12, 0.22),
    }
    for _, row in chosen_sorted.head(7).iterrows():
        dx, dy = label_offsets.get(str(row["city"]), (0.10, 0.16))
        ax_c.text(
            row["morph_pc1"] + dx,
            row["morph_pc2"] + dy,
            row["city"],
            fontsize=5.8,
            color=pub_style.COLORS["annot"],
        )
    ax_c.set_xlabel("Morphology PC1")
    ax_c.set_ylabel("Morphology PC2")
    pub_style.panel_title(ax_c, "c", "Stratified geometry-null subset")
    pub_style.light_grid(ax_c, axis="both")
    ax_c.legend(loc="lower left", fontsize=6.0)

    ax_d = fig.add_subplot(gs[1, 1])
    if len(geom):
        g = geom.sort_values("road_gap").reset_index(drop=True)
        x = np.arange(len(g))
        ax_d.plot(
            x,
            g["road_gap"],
            "o-",
            color=pub_style.COLORS["observed"],
            lw=1.0,
            ms=3.6,
            label="Observed window gap",
        )
        ax_d.plot(
            x,
            g["geometry_null_gap_mean"],
            "o-",
            color=pub_style.COLORS["geometry_null"],
            lw=1.0,
            ms=3.6,
            label="Geometry-null gap",
        )
        for i, row in g.iterrows():
            ax_d.vlines(i, row["geometry_null_gap_mean"], row["road_gap"], color="#C7C7C7", lw=0.7)
        ax_d.set_xticks(x)
        ax_d.set_xticklabels(g["city"], rotation=90, fontsize=5.8)
        ax_d.set_ylabel("Gap relative to CEBH")
        ax_d.legend(loc="upper left", fontsize=6.0)
    else:
        ax_d.text(0.5, 0.5, "Geometry-null pilot not run", ha="center", va="center", transform=ax_d.transAxes)
        ax_d.set_xticks([])
        ax_d.set_yticks([])
    pub_style.panel_title(ax_d, "d", "Cross-region geometry-null pilot")
    pub_style.light_grid(ax_d, axis="y")

    pub_style.save(fig, FIG_BASE)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, cols: list[str], max_rows: int = 8) -> str:
    rows = []
    for _, row in df.head(max_rows).iterrows():
        rows.append("| " + " | ".join(str(row[c]) if not isinstance(row[c], float) else f"{row[c]:.4g}" for c in cols) + " |")
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    return "\n".join([header, sep] + rows)


def write_report(
    cov: pd.DataFrame,
    corr: pd.DataFrame,
    model_summary: dict[str, Any],
    selected: pd.DataFrame,
    geom: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    geom_pass = geom[geom["n_replicates_pass"] > 0] if len(geom) else geom
    if len(geom_pass):
        geom_mean = float(geom_pass["road_minus_geometry_mean"].mean())
        geom_range = (float(geom_pass["road_minus_geometry_mean"].min()), float(geom_pass["road_minus_geometry_mean"].max()))
        geom_sentence = (
            f"The geometry-null pilot passed {len(geom_pass)} cities. Mean road-minus-geometry residual is "
            f"{geom_mean:.3f}, with city range {geom_range[0]:.3f} to {geom_range[1]:.3f}."
        )
    else:
        geom_sentence = "The geometry-null pilot was not completed in this run."

    report = f"""# R62 Urban-Form Mechanism Layer

Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Purpose

R62 upgrades the R61 scale result from a pure graph-threshold screen toward an
urban-form mechanism layer. It extracts morphology and road-hierarchy
covariates for all 71 standardized city windows and runs a stratified
cross-region strict non-crossing geometry-null pilot.

## All-City Urban-Form Covariates

- Cities: {len(cov)}
- Macro-regions: {cov['macro_region'].nunique()}
- Mean road gap: {cov['road_gap'].mean():.3f}
- Mean node density: {cov['node_density_per_km2'].mean():.1f} nodes per km2
- Mean chord-length density: {cov['chord_length_density_km_per_km2'].mean():.2f} km per km2
- Mean arterial length share: {cov['arterial_length_share'].mean():.3f}
- Mean fourfold orientation order: {cov['orientation_fourfold_order'].mean():.3f}

Top absolute Spearman associations with the road gap:

{markdown_table(corr, ['covariate', 'n', 'spearman_rho', 'spearman_p'], max_rows=10)}

Predictive screen:

```json
{json.dumps(model_summary, indent=2)}
```

## Stratified Geometry-Null Pilot

Subset selection rule: one representative city per macro-region where possible,
with morphology extremes added only if `max_geom_cities` exceeds the region
count. Max edge count target: {args.geom_max_edges:,}.

Selected cities:

{markdown_table(selected, ['city', 'country', 'macro_region', 'n_nodes', 'n_edges', 'road_gap'], max_rows=20)}

{geom_sentence}

## Interpretation

R62 supports a stronger Nature Cities-facing mechanism statement than R61 alone:
the CEBH transfer gap is associated with measurable urban road-form variables,
including edge scale, hierarchy, orientation order, degree composition and
cycle density. The geometry-null pilot tests whether strict non-crossing
embedding can move degree-preserving null thresholds toward the observed
city-window thresholds across regions.

This is still not a causal urban-form theory. The covariate screen is
associational, the geometry-null pilot is a stratified subset, and all policy
or planning implications must remain bounded until population, accessibility,
road-class completeness and land-use covariates are added.

## Outputs

- `data/R62_urban_form_mechanism/urban_form_covariates.csv`
- `data/R62_urban_form_mechanism/urban_form_correlations.csv`
- `data/R62_urban_form_mechanism/urban_form_model_summary.json`
- `data/R62_urban_form_mechanism/geometry_null_subset_registry.csv`
- `data/R62_urban_form_mechanism/geometry_null_pilot_summary.csv`
- `data/R62_urban_form_mechanism/geometry_null_pilot_replicates.csv`
- `figures/Fig_R62_urban_form_mechanism.pdf`
- `figures/Fig_R62_urban_form_mechanism.svg`
- `figures/Fig_R62_urban_form_mechanism.png`
- `figures/Fig_R62_urban_form_mechanism.tiff`
"""
    (OUT / "R62_report.md").write_text(report, encoding="utf-8")
    (ROUND_DIR / "gate_report.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=6201)
    parser.add_argument("--force-covariates", action="store_true")
    parser.add_argument("--force-geometry", action="store_true")
    parser.add_argument("--skip-geometry", action="store_true")
    parser.add_argument("--max-geom-cities", type=int, default=7)
    parser.add_argument("--geom-max-edges", type=int, default=45_000)
    parser.add_argument("--geom-reps", type=int, default=1)
    parser.add_argument("--geom-swap-fraction", type=float, default=0.005)
    parser.add_argument("--geom-max-attempt-factor", type=int, default=900)
    parser.add_argument("--geom-neighbor-k", type=int, default=180)
    parser.add_argument("--geom-rebuild-interval", type=int, default=40)
    parser.add_argument("--geom-length-bins", type=int, default=12)
    parser.add_argument("--geom-bin-tolerance", type=int, default=1)
    parser.add_argument("--geom-n-perm", type=int, default=8)
    parser.add_argument("--geom-n-bootstrap", type=int, default=120)
    parser.add_argument("--geom-p-min", type=float, default=0.10)
    parser.add_argument("--geom-p-max", type=float, default=0.90)
    parser.add_argument("--geom-p-step", type=float, default=0.025)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    cov_path = OUT / "urban_form_covariates.csv"
    if cov_path.exists() and not args.force_covariates:
        cov = pd.read_csv(cov_path)
    else:
        r61 = pd.read_csv(R61_RESULTS)
        r61 = r61[r61["status"] == "pass"].reset_index(drop=True)
        rows = []
        for _, row in r61.iterrows():
            print(f"[covariates] {row['city']}", flush=True)
            rows.append(asdict(covariates_for_row(row)))
            pd.DataFrame(rows).to_csv(cov_path, index=False)
        cov = pd.DataFrame(rows)
    if "graph_cache_file" not in cov.columns:
        r61_paths = pd.read_csv(R61_RESULTS)[["city", "country", "graph_cache_file"]]
        cov = cov.merge(r61_paths, on=["city", "country"], how="left")
        cov.to_csv(cov_path, index=False)

    corr = compute_correlations(cov)
    corr.to_csv(OUT / "urban_form_correlations.csv", index=False)
    model_summary = predictive_screen(cov)
    (OUT / "urban_form_model_summary.json").write_text(json.dumps(model_summary, indent=2), encoding="utf-8")

    pca_cols = [
        "node_density_per_km2",
        "edge_length_median_m",
        "orientation_fourfold_order",
        "dead_end_share",
        "degree4plus_share",
        "cycle_density",
        "arterial_length_share",
        "oneway_length_share",
    ]
    pca = pca_scores(cov, pca_cols)
    pca.to_csv(OUT / "urban_form_pca_scores.csv", index=False)

    selected = select_geometry_subset(cov, args.max_geom_cities, args.geom_max_edges)
    selected.to_csv(OUT / "geometry_null_subset_registry.csv", index=False)

    if args.skip_geometry:
        geom_reps = pd.DataFrame()
        geom_summary = pd.DataFrame()
    else:
        geom_reps, geom_summary = run_geometry_nulls(selected, args)

    make_figure(cov, corr, pca, selected, geom_summary)
    write_report(cov, corr, model_summary, selected, geom_summary, args)

    final = {
        "n_cities": int(len(cov)),
        "n_regions": int(cov["macro_region"].nunique()),
        "mean_gap": float(cov["road_gap"].mean()),
        "top_spearman_covariate": str(corr.iloc[0]["covariate"]) if len(corr) else "",
        "top_spearman_rho": float(corr.iloc[0]["spearman_rho"]) if len(corr) else float("nan"),
        "geometry_pilot_cities": int(len(geom_summary)) if len(geom_summary) else 0,
        "geometry_pilot_passed_cities": int((geom_summary["n_replicates_pass"] > 0).sum()) if len(geom_summary) else 0,
        "geometry_road_minus_null_mean": float(geom_summary["road_minus_geometry_mean"].mean()) if len(geom_summary) else float("nan"),
        "model": model_summary,
    }
    (OUT / "R62_summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
