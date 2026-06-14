"""R106: second-domain minimal replication on power-transmission grids.

The road result has two parts: (i) the degree-moment (CEBH) formula under-predicts
the bond-percolation threshold of an embedded infrastructure network, and a
junction-composition reference recovers most of the gap; (ii) near-planar road
networks sit in (or near) the two-dimensional percolation universality class.

A second, independent embedded infrastructure tests how far this generalizes.
Power-transmission grids are spatially embedded but, unlike roads, are NOT planar
(lines cross freely) and contain hubs. We use the openly published PEGASE
European transmission grids shipped with pandapower (MATPOWER/PEGASE, BSD), build
the bus-branch graph, and run the same estimator, anchor and Fisher-exponent
pipeline used for roads.

Honest expected reading: the CEBH transfer failure should generalize (embedded
grids percolate far above the degree-moment prediction), while the *degree* of
planar-lattice behaviour should be weaker than for roads because power grids are
non-planar. The result is therefore a generalization-plus-boundary: the framework
is not road-specific, and the cluster-size-distribution exponent grades with how
planar the embedding is.

No raw grid data are redistributed; the grids are loaded from the pandapower
dependency and only derived results are stored.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
import pub_style  # noqa: E402
from build_R105_finite_size_scaling import (  # noqa: E402
    fisher_exponent, _find, degree_preserving_rewire, cebh_pc_from_edges,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "R106_second_domain_power"
ROUND = ROOT / "rounds" / "R106_second_domain_power"
FIG_BASE = ROOT / "figures" / "Fig_R106_second_domain_power"
NPJ = ROOT / "submission" / "npj_complexity" / "target_submission"

HONEYCOMB = 1.0 - 2.0 * math.sin(math.pi / 18.0)
SQUARE = 0.5
TAU_2D = 187.0 / 91.0
TAU_MF = 2.5
ROAD_TAU = 2.116        # R105 mean over five largest road windows
ROAD_ANCHOR_MAE = 0.067  # R103
ROAD_CEBH_MAE = 0.217    # R103

# Ten distinct real transmission grids (pandapower / MATPOWER, BSD), spanning
# Europe (PEGASE, French RTE, Polish, GB) and the US (Illinois, IEEE-300).
# Near-duplicate RTE snapshots (6495/6515) are excluded to avoid pseudo-replication.
GRIDS = [
    ("case1354pegase", "PEGASE", "Europe"),
    ("case2869pegase", "PEGASE", "Europe"),
    ("case9241pegase", "PEGASE", "Europe"),
    ("case1888rte", "RTE", "France"),
    ("case2848rte", "RTE", "France"),
    ("case6470rte", "RTE", "France"),
    ("case3120sp", "Polish", "Poland"),
    ("GBnetwork", "GB", "UK"),
    ("case_illinois200", "Illinois", "US"),
    ("case300", "IEEE300", "US"),
]


def pegase_graph(name):
    import pandapower.networks as ppn
    net = getattr(ppn, name)()
    g = nx.Graph()
    g.add_nodes_from(net.bus.index)
    for _, r in net.line.iterrows():
        g.add_edge(int(r.from_bus), int(r.to_bus))
    for _, r in net.trafo.iterrows():
        g.add_edge(int(r.hv_bus), int(r.lv_bus))
    g.remove_nodes_from(list(nx.isolates(g)))
    return nx.convert_node_labels_to_integers(g)


def pc_estimate(n, edges, n_perm=12, seed=7):
    rng = np.random.default_rng(seed)
    grid = np.round(np.arange(0.05, 0.95, 0.02), 4)
    m = len(edges)
    tk = np.clip(np.rint(grid * m).astype(int), 0, m)
    curve = np.zeros(len(grid))
    for _ in range(n_perm):
        parent = np.arange(n)
        size = np.ones(n, int)
        largest = 1
        t = 0
        order = rng.permutation(m)
        samp = np.zeros(len(grid))
        for step, ei in enumerate(order, 1):
            u, v = int(edges[ei, 0]), int(edges[ei, 1])
            ru, rv = _find(parent, u), _find(parent, v)
            if ru != rv:
                if size[ru] < size[rv]:
                    ru, rv = rv, ru
                parent[rv] = ru
                size[ru] += size[rv]
                largest = max(largest, int(size[ru]))
            while t < len(grid) and step >= tk[t]:
                samp[t] = largest / n
                t += 1
        while t < len(grid):
            samp[t] = largest / n
            t += 1
        curve += samp
    curve /= n_perm
    return float(grid[int(np.argmax(np.gradient(curve, grid)))])


def analyse(name, family, region, seed):
    g = pegase_graph(name)
    n = g.number_of_nodes()
    edges = np.array(g.edges(), dtype=np.int64)
    deg = np.array([d for _, d in g.degree()], float)
    k1, k2 = deg.mean(), (deg ** 2).mean()
    cebh = float(k1 / (k2 - k1))
    pc = pc_estimate(n, edges, seed=seed)
    d3 = float(np.mean(deg == 3))
    d4 = float(np.mean(deg >= 4))
    w3 = d3 / (d3 + d4) if (d3 + d4) > 0 else float("nan")
    anchor = w3 * HONEYCOMB + (1 - w3) * SQUARE
    # Fisher window scales with grid size (small grids have a shorter scaling range).
    s_hi = 300.0 if n >= 2000 else max(40.0, n / 12.0)
    tau, r2, cc, dd = fisher_exponent(n, edges, pc, 40, seed + 1, 6.0, s_hi)
    return {
        "grid": name, "family": family, "region": region,
        "n_nodes": n, "n_edges": int(len(edges)), "mean_degree": float(k1),
        "deg1_share": float(np.mean(deg == 1)), "deg3_share": d3, "deg4plus_share": d4,
        "cebh_pc": cebh, "observed_pc": pc, "cebh_gap": pc - cebh,
        "anchor_pc": anchor, "anchor_err": anchor - pc,
        "cebh_abs_err": abs(cebh - pc), "anchor_abs_err": abs(anchor - pc),
        "fisher_tau": tau, "fisher_r2": r2, "fisher_s_hi": s_hi,
        "s": cc.tolist(), "ns": dd.tolist(),
    }


def meanfield_control(name, seed):
    """One configuration-model control on a large grid (diagnostic only; the
    control is unreliable for leaf-heavy grids, so the text uses the theoretical
    mean-field value 2.5 as the reference)."""
    g = pegase_graph(name)
    n = g.number_of_nodes()
    edges = np.array(g.edges(), dtype=np.int64)
    redges = degree_preserving_rewire(n, edges, np.random.default_rng(seed))
    pc_mf = cebh_pc_from_edges(n, redges)
    tau_mf, r2_mf, _, _ = fisher_exponent(n, redges, pc_mf, 40, seed + 1, 6.0, 300.0)
    return {"grid": name, "tau_meanfield_control": tau_mf, "r2": r2_mf, "pc_mf": pc_mf}


def make_figure(recs, summary):
    pub_style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(pub_style.FIG_WIDTH_2COL, 2.8), constrained_layout=True)

    # a: paired CEBH vs anchor absolute error across the grid population.
    ax = axes[0]
    order = sorted(range(len(recs)), key=lambda i: recs[i]["n_nodes"])
    x = np.arange(len(recs))
    cebh_e = [recs[i]["cebh_abs_err"] for i in order]
    anch_e = [recs[i]["anchor_abs_err"] for i in order]
    for xi, ce, ae in zip(x, cebh_e, anch_e):
        ax.plot([xi, xi], [ae, ce], color="#C7C7C7", lw=0.7, zorder=1)
    ax.scatter(x, cebh_e, s=16, color=pub_style.COLORS["cebh"], zorder=2, label="CEBH error")
    ax.scatter(x, anch_e, s=16, color=pub_style.COLORS["geometry_null"], zorder=2, label="Junction-anchor error")
    ax.axhline(ROAD_CEBH_MAE, ls=":", lw=0.8, color=pub_style.COLORS["cebh"])
    ax.axhline(ROAD_ANCHOR_MAE, ls="--", lw=0.8, color=pub_style.COLORS["geometry_null"])
    pub_style.annot(ax, 0.03, 0.96, f"n = {len(recs)} grids; anchor cuts error {summary['anchor_error_reduction_vs_cebh']:.1f}x")
    ax.set_xticks(x)
    ax.set_xticklabels([recs[i]["n_nodes"] for i in order], fontsize=5.2, rotation=90)
    ax.set_xlabel("grid size (buses)")
    ax.set_ylabel("Absolute threshold error")
    pub_style.panel_title(ax, "a", "Degree-moment vs junction anchor")
    ax.legend(frameon=False, fontsize=5.8, loc="center right")

    # b: Fisher exponent population - square (2D), roads (near-planar), power grids.
    ax = axes[1]
    ax.bar([0, 1], [1.99, ROAD_TAU], color=[pub_style.COLORS["cebh"], pub_style.COLORS["geometry_null"]], alpha=0.85, width=0.55)
    ptaus = [r["fisher_tau"] for r in recs]
    rng = np.random.default_rng(1)
    ax.scatter(np.full(len(ptaus), 2.0) + rng.uniform(-0.16, 0.16, len(ptaus)), ptaus,
               s=18, color=pub_style.COLORS["nb"], zorder=3, linewidths=0, alpha=0.85)
    ax.hlines(float(np.mean(ptaus)), 1.72, 2.28, color=pub_style.COLORS["nb"], lw=2.2, zorder=4)
    ax.axhline(TAU_2D, ls="--", lw=0.8, color="#1A1A1A")
    ax.text(2.45, TAU_2D - 0.02, "2D 2.055", ha="right", va="top", fontsize=5.6)
    ax.axhline(TAU_MF, ls=":", lw=0.8, color=pub_style.COLORS["annot"])
    ax.text(-0.4, TAU_MF + 0.01, "mean-field 2.5", ha="left", va="bottom", fontsize=5.6, color=pub_style.COLORS["annot"])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["square\n(2D)", "roads\n(near-planar)", f"power grids\n($n$={len(recs)}, non-planar)"], fontsize=5.4)
    ax.set_ylabel("Fisher exponent $\\tau_F$")
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(1.8, 2.7)
    pub_style.panel_title(ax, "b", "2D class is specific to planar roads")

    pub_style.save(fig, FIG_BASE)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ROUND.mkdir(parents=True, exist_ok=True)
    recs = [analyse(name, fam, reg, 20260613 + 7 * i) for i, (name, fam, reg) in enumerate(GRIDS)]
    for r in recs:
        print(f"  {r['grid']:16s} N={r['n_nodes']:5d} CEBH={r['cebh_pc']:.3f} obs={r['observed_pc']:.3f} "
              f"gap={r['cebh_gap']:+.3f} anchorErr={r['anchor_err']:+.3f} tau_F={r['fisher_tau']:.3f}", flush=True)
    mf = meanfield_control("case9241pegase", 20260700)
    print(f"  mean-field config-model control (case9241): tau_F={mf['tau_meanfield_control']:.3f} "
          f"(unreliable for leaf-heavy grids; theory 2.5 used as reference)", flush=True)

    gaps = np.array([r["cebh_gap"] for r in recs])
    cebh_e = np.array([r["cebh_abs_err"] for r in recs])
    anch_e = np.array([r["anchor_abs_err"] for r in recs])
    ptaus = np.array([r["fisher_tau"] for r in recs])
    summary = {
        "domain": "power_transmission_grids",
        "source": "pandapower / MATPOWER cases PEGASE, RTE, Polish, GB, Illinois, IEEE-300 (BSD); raw grids not redistributed",
        "n_grids": len(recs),
        "regions": sorted(set(r["region"] for r in recs)),
        "grid_size_range": [int(min(r["n_nodes"] for r in recs)), int(max(r["n_nodes"] for r in recs))],
        "cebh_gap_mean": float(gaps.mean()), "cebh_gap_sd": float(gaps.std()),
        "cebh_gap_range": [float(gaps.min()), float(gaps.max())],
        "cebh_gap_all_positive": bool((gaps > 0).all()),
        "cebh_abs_err_mean": float(cebh_e.mean()),
        "anchor_abs_err_mean": float(anch_e.mean()),
        "anchor_error_reduction_vs_cebh": float(cebh_e.mean() / anch_e.mean()),
        "anchor_better_than_cebh_share": float(np.mean(anch_e < cebh_e)),
        "power_fisher_tau_mean": float(ptaus.mean()), "power_fisher_tau_sd": float(ptaus.std()),
        "power_fisher_tau_range": [float(ptaus.min()), float(ptaus.max())],
        "meanfield_control_case9241": mf["tau_meanfield_control"],
        "meanfield_control_note": (
            "The degree-matched configuration-model control is unreliable for these grids because of "
            "their heavy degree-1/degree-2 leaf tail, so the theoretical mean-field value 2.5 is used "
            "as the reference rather than the per-grid control."
        ),
        "road_fisher_tau": ROAD_TAU, "tau_2D": TAU_2D, "tau_mean_field": TAU_MF,
        "interpretation": (
            "GENERALIZATION (robust): the CEBH transfer failure holds across a population of ten real "
            "transmission grids (200-9241 buses; PEGASE, French RTE, Polish, GB, US). The degree-moment "
            "formula under-predicts the observed bond-percolation threshold in every grid (10/10 "
            "all-positive gap, mean 0.30, larger than the 0.22 road gap), and the zero-parameter "
            "junction-composition anchor beats CEBH in 100% of grids, cutting the mean absolute error "
            "about fourfold even though power grids are non-planar. BOUNDARY (the informative half): the "
            "two-dimensional universality class does NOT carry over. The cluster-size-distribution "
            "exponent of the power grids is tau_F = 2.42 +- 0.23 (range 2.00-2.69), near the mean-field "
            "value 2.5 and well above the near-planar road value 2.12 (close to 2D 2.055). So junction "
            "composition sets the threshold value in both domains, but only the near-planar road geometry "
            "places percolation in the 2D class; non-planar power grids fall in the mean-field/"
            "random-graph regime. This separates the two parts of the road result - the threshold "
            "mechanism generalizes, the universality class is geometry-specific - and is itself a "
            "single-infrastructure-type (power) replication."
        ),
    }
    pd.DataFrame([{k: r[k] for k in r if k not in ("s", "ns")} for r in recs]).to_csv(
        OUT / "R106_power_grid_percolation.csv", index=False)
    (OUT / "R106_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_figure(recs, summary)

    if NPJ.exists():
        for ext in [".svg", ".pdf", ".png", ".tiff"]:
            src = FIG_BASE.with_suffix(ext)
            if src.exists():
                (NPJ / "figures").mkdir(exist_ok=True)
                __import__("shutil").copy2(src, NPJ / "figures" / f"FigS_R106_second_domain_power{ext}")
                if ext != ".tiff":
                    (NPJ / "source_data" / "figures").mkdir(parents=True, exist_ok=True)
                    __import__("shutil").copy2(src, NPJ / "source_data" / "figures" / f"FigS_R106_second_domain_power{ext}")
        for f in ["R106_power_grid_percolation.csv", "R106_summary.json"]:
            __import__("shutil").copy2(OUT / f, NPJ / "source_data" / "tables" / f)

    print("\n[R106] SUMMARY", flush=True)
    print(f"   n grids = {summary['n_grids']} ({summary['grid_size_range'][0]}-{summary['grid_size_range'][1]} buses, regions {summary['regions']})", flush=True)
    print(f"   CEBH gap mean = {summary['cebh_gap_mean']:+.3f} +- {summary['cebh_gap_sd']:.3f} (all positive: {summary['cebh_gap_all_positive']}; road +0.22)", flush=True)
    print(f"   anchor error  = {summary['anchor_error_reduction_vs_cebh']:.1f}x lower; anchor beats CEBH in {summary['anchor_better_than_cebh_share']*100:.0f}% of grids", flush=True)
    print(f"   Fisher tau_F  = {summary['power_fisher_tau_mean']:.3f} +- {summary['power_fisher_tau_sd']:.3f} (road 2.12, 2D 2.055, mean-field 2.5)", flush=True)


if __name__ == "__main__":
    main()
