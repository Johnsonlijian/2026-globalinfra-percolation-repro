"""R206-C: build the RESS figure set (Figs 1, 2, 4, 5, 6) from source data.

Fig 3 (null ladder) reuses the existing R201 mechanism-package assets; all other
figures are generated here directly from the R205/R206 outputs and public tables.
Outputs: outputs/figures_R207/Fig{1,2,4,5,6}_R207.{png,pdf,svg}
"""

from __future__ import annotations

import pickle
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

ROOT = Path(__file__).resolve().parents[2]
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
TABLES = ROOT / "source_data" / "tables"
R205O = ROOT / "data" / "R205_measured_geometry_dose_response"
R206O = ROOT / "data" / "R206_R208_ress_reframe"
CACHE = ROOT / "cache" / "R61_osmnx_city_windows"
FIGD = R206O / "figures_R207"
FIGD.mkdir(parents=True, exist_ok=True)

try:
    import pub_style
    pub_style.apply()
    W2 = getattr(pub_style, "FIG_WIDTH_2COL", 7.2)
    C = getattr(pub_style, "COLORS", {})
except Exception:
    pub_style = None
    W2, C = 7.2, {}

BLUE = C.get("observed", "#1f5fa8")
GREEN = C.get("geometry_null", "#2a9d8f")
RED = "#c1121f"  # unsafe-side accent; pub_style's "cebh" is grey, keep red explicit
PURPLE = C.get("accent", "#7a5195")
GREY = "#9a9a9a"


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGD / f"{name}.{ext}", dpi=450, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"[C] {name} written", flush=True)


def panel_title(ax, letter, text):
    if pub_style:
        pub_style.panel_title(ax, letter, text)
    else:
        ax.set_title(f"{letter}  {text}", fontsize=7, loc="left")


# ---------------- Fig 1 ----------------

def _city_segments(pkl, max_edges=None):
    from shapely.geometry import LineString
    from shapely.strtree import STRtree
    with open(CACHE / pkl, "rb") as f:
        G = pickle.load(f)
    nodes_xy = {nid: (d["x"], d["y"]) for nid, d in G.nodes(data=True)}
    seen = {}
    for u, v, data in G.edges(data=True):
        if u == v:
            continue
        key = (u, v) if u <= v else (v, u)
        if key in seen:
            continue
        geom = data.get("geometry") or LineString([nodes_xy[u], nodes_xy[v]])
        seen[key] = geom
    keys = list(seen.keys())
    geoms = [seen[k] for k in keys]
    tree = STRtree(geoms)
    qi, ti = tree.query(geoms, predicate="crosses")
    crossing = set()
    for a, b in zip(qi.tolist(), ti.tolist()):
        if a >= b:
            continue
        ka, kb = keys[a], keys[b]
        if ka[0] in kb or ka[1] in kb:
            continue
        crossing.add(a)
        crossing.add(b)
    segs, cross_segs = [], []
    for i, g in enumerate(geoms):
        xy = np.asarray(g.coords)
        (cross_segs if i in crossing else segs).append(xy)
    return segs, cross_segs


def fig1():
    fig = plt.figure(figsize=(W2, 2.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 1.15, 0.95, 1.05])
    for j, (pkl, label, chi) in enumerate([
            ("dar_es_salaam_tanzania__drive_10000m.pkl", "Dar es Salaam", 0.0010),
            ("hong_kong_china__drive_10000m.pkl", "Hong Kong", 0.152)]):
        ax = fig.add_subplot(gs[0, j])
        segs, cross = _city_segments(pkl)
        ax.add_collection(LineCollection(segs, colors="#c9c9c9", linewidths=0.12))
        if cross:
            ax.add_collection(LineCollection(cross, colors=RED, linewidths=0.35))
        ax.autoscale()
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel(f"{label}\n$\\chi$ = {chi:g}", fontsize=6.5)
        panel_title(ax, "ab"[j], "at-grade grid" if j == 0 else "elevated motorways")

    # c: chi definition schematic
    ax = fig.add_subplot(gs[0, 2])
    label_box = dict(facecolor="white", edgecolor="none", alpha=0.92, pad=1.0)
    ax.plot([0.08, 0.92], [0.36, 0.50], color="#555555", lw=1.4)
    ax.plot([0.30, 0.82], [0.78, 0.34], color=RED, lw=1.4)
    ax.scatter([0.61], [0.52], s=26, facecolor="none", edgecolor=RED, lw=1.1, zorder=5)
    ax.annotate("projected crossing,\nno shared junction", xy=(0.61, 0.52), xytext=(0.49, 0.23),
                fontsize=5.4, ha="left", bbox=label_box,
                arrowprops=dict(arrowstyle="-", color="#555555", lw=0.5, shrinkA=1, shrinkB=2))
    ax.plot([0.08, 0.58], [0.73, 0.62], color="#555555", lw=1.4)
    ax.plot([0.58, 0.92], [0.62, 0.80], color="#555555", lw=1.4)
    ax.scatter([0.58], [0.62], s=14, color="#555555", zorder=5)
    ax.annotate("junction:\nnot counted", xy=(0.58, 0.62), xytext=(0.30, 0.85),
                fontsize=5.4, ha="left", bbox=label_box,
                arrowprops=dict(arrowstyle="-", color="#555555", lw=0.5, shrinkA=1, shrinkB=2))
    ax.text(0.02, 0.02, r"$\chi=\dfrac{\#\,\mathrm{crossing\ pairs}}{\#\,\mathrm{edges}}$"
            "\nmedian city: 100% of crossings\ntouch bridge/tunnel tags",
            fontsize=5.6, va="bottom", bbox=label_box)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    panel_title(ax, "c", "Measured exposure $\\chi$")

    # d: screening problem
    ax = fig.add_subplot(gs[0, 3])
    r61 = pd.read_csv(TABLES / "R61_central_window_screen_results.csv")
    fc_obs = 1 - r61["pc_rank"]
    fc_cebh = 1 - r61["pc_cebh"]
    ax.scatter(fc_obs, fc_cebh, s=6, c=RED, alpha=0.65, lw=0, label="degree-moment")
    fc_dim = 1 - 2 / r61["mean_degree"]
    ax.scatter(fc_obs, fc_dim, s=6, c=GREEN, alpha=0.65, lw=0, label=r"$2/\langle k\rangle$ baseline")
    lim = [0.0, 0.75]
    ax.plot(lim, lim, color="#333333", lw=0.7, ls="--")
    ax.axhline(0.4, color=GREY, lw=0.5, ls=":")
    ax.axvline(0.4, color=GREY, lw=0.5, ls=":")
    ax.text(0.03, 0.43, r"target $\phi^*$", fontsize=5, color="#666666",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.8))
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r"observed tolerable loss $f_c$")
    ax.set_ylabel(r"predicted $f_c$")
    ax.legend(frameon=False, fontsize=5, loc="lower right", handletextpad=0.4)
    panel_title(ax, "d", "Unsafe-side level bias")
    save(fig, "Fig1_R207_problem_and_exposure")


# ---------------- Fig 2 ----------------

def fig2():
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.3), constrained_layout=True)
    r61 = pd.read_csv(TABLES / "R61_central_window_screen_results.csv")
    p106 = pd.read_csv(TABLES / "R106_power_grid_percolation.csv")

    ax = axes[0]
    ax.scatter(1 - r61.pc_rank, 1 - r61.pc_cebh, s=7, c=RED, alpha=0.6, lw=0, label="roads: degree-moment")
    ax.scatter(1 - r61.pc_rank, 1 - 2 / r61.mean_degree, s=7, c=GREEN, alpha=0.6, lw=0,
               label=r"roads: $2/\langle k\rangle$")
    ax.scatter(1 - p106.observed_pc, 1 - p106.cebh_pc, s=16, c=RED, marker="^", alpha=0.85, lw=0,
               label="power: degree-moment")
    ax.scatter(1 - p106.observed_pc, 1 - p106.anchor_pc, s=16, c=PURPLE, marker="^", alpha=0.85, lw=0,
               label="power: junction anchor")
    lim = [0.0, 0.85]
    ax.plot(lim, lim, color="#333333", lw=0.7, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(r"observed $f_c$")
    ax.set_ylabel(r"predicted $f_c$")
    ax.legend(frameon=False, fontsize=4.6, loc="upper left", handletextpad=0.4)
    panel_title(ax, "a", "Predicted vs observed tolerance")

    ax = axes[1]
    sweep = pd.read_csv(R206O / "R206_A3_flip_threshold_sweep.csv")
    styles = {("road", "cebh"): (RED, "-", "roads: degree-moment"),
              ("road", "derived"): (GREEN, "-", r"roads: $2/\langle k\rangle$"),
              ("power", "cebh"): (RED, ":", "power: degree-moment"),
              ("power", "derived"): (GREEN, ":", r"power: $2/\langle k\rangle$ (safe-side)")}
    for (dom, pred), (col, ls, lab) in styles.items():
        sub = sweep[(sweep.domain == dom) & (sweep.predictor == pred)].sort_values("threshold")
        ax.plot(sub.threshold, sub.flip_rate, ls, color=col, lw=1.1, label=lab)
    # junction-anchor sweep for power, computed from the R106 table directly
    fc_obs_p = 1 - p106.observed_pc
    fc_anch_p = 1 - p106.anchor_pc
    ts = np.round(np.arange(0.10, 0.601, 0.025), 3)
    fr = [float((((fc_anch_p - t) * (fc_obs_p - t)) < 0).mean()) for t in ts]
    ax.plot(ts, fr, ":", color=PURPLE, lw=1.1, label="power: junction anchor (unsafe-side)")
    ax.axvline(0.40, color=GREY, lw=0.5, ls=":")
    ax.annotate(r"reference $\phi^*$", xy=(0.40, 0.04), xytext=(0.43, 0.16),
                fontsize=5, va="center", ha="left", color="#666666",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.8),
                arrowprops=dict(arrowstyle="-", color="#777777", lw=0.45,
                                shrinkA=1, shrinkB=1))
    ax.set_xlabel(r"tolerable-loss target $\phi^*$")
    ax.set_ylabel("pass/fail flip rate")
    ax.set_ylim(-0.03, 1.28)
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.94,
              fontsize=4.2, loc="upper right", handletextpad=0.45,
              borderpad=0.25, labelspacing=0.18)
    panel_title(ax, "b", "Flips across the target sweep")

    ax = axes[2]
    d = pd.read_csv(TABLES / "R113_decision_margins.csv")
    err_c = (d.fc_cebh - d.fc_obs).abs()
    err_d = (d.fc_derived - d.fc_obs).abs()
    bp = ax.boxplot([err_c, err_d], tick_labels=["degree-\nmoment", "geometric\nbaseline"],
                    widths=0.5, patch_artist=True, showfliers=False)
    for patch, col in zip(bp["boxes"], [RED, GREEN]):
        patch.set_facecolor(col)
        patch.set_alpha(0.5)
    for med in bp["medians"]:
        med.set_color("#222222")
    ax.set_ylabel(r"margin error $|f_c^{\rm pred}-f_c^{\rm obs}|$")
    ax.text(0.97, 0.95, "means: 0.227 vs 0.035", transform=ax.transAxes,
            fontsize=5.4, ha="right", va="top")
    panel_title(ax, "c", "Margin error (81 networks)")
    save(fig, "Fig2_R207_decision_consequences")


# ---------------- Fig 4 ----------------

def fig4():
    s1 = pd.read_csv(R205O / "R205_S1_crossing_density_71cities.csv")
    s1 = s1[s1.status == "ok"]
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.3), constrained_layout=True)

    ax = axes[0]
    d = s1.sort_values("chi_poly").reset_index(drop=True)
    ax.scatter(np.arange(len(d)), d.chi_poly, s=8, c=BLUE, alpha=0.8, lw=0)
    ax.set_yscale("log")
    for name in ("Hong Kong", "Taipei", "Dar es Salaam", "Kampala"):
        row = d[d.city == name]
        if len(row):
            i = int(row.index[0])
            ax.annotate(name, (i, row.chi_poly.iloc[0]), fontsize=5,
                        xytext=(-3, 4), textcoords="offset points",
                        ha="right" if i > 35 else "left")
    ax.set_xlabel("city rank")
    ax.set_ylabel(r"measured $\chi$")
    panel_title(ax, "a", "Two decades of exposure")

    ax = axes[1]
    ax.scatter(s1.chi_poly, s1.road_gap, s=8, c=BLUE, alpha=0.75, lw=0)
    x = np.log(s1.chi_poly)
    b1, b0 = np.polyfit(x, s1.road_gap, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(np.exp(xs), b0 + b1 * xs, color=RED, lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\chi$ (log)")
    ax.set_ylabel(r"gap $p_c^{\rm obs}-p_c^{\rm CEBH}$")
    ax.text(0.03, 0.05, "Spearman $-0.40$ ($p=6\\times10^{-4}$)\npartial $-0.28$ ($p=0.017$)",
            transform=ax.transAxes, fontsize=5.4, va="bottom")
    panel_title(ax, "b", "Exposure organizes the gap")

    ax = axes[2]
    sub = s1.dropna(subset=["spatial_resid_f0.01", "geom_resid_f0.01"])
    ax.scatter(sub.chi_poly, sub["spatial_resid_f0.01"], s=8, c=RED, alpha=0.8, lw=0,
               label="crossing-permissive null")
    ax.scatter(sub.chi_poly, sub["geom_resid_f0.01"], s=8, c=GREEN, alpha=0.8, lw=0,
               label="strict non-crossing null")
    x = np.log(sub.chi_poly)
    b1, b0 = np.polyfit(x, sub["spatial_resid_f0.01"], 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(np.exp(xs), b0 + b1 * xs, color=RED, lw=1)
    ax.axhline(0, color="#888888", lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\chi$ (log)")
    ax.set_ylabel("road $-$ null threshold")
    ax.text(0.03, 0.44, "$\\rho=-0.54$ ($p<10^{-5}$)\npartial $-0.32$ ($p=0.006$)",
            transform=ax.transAxes, fontsize=5.4, va="top", color=RED)
    ax.text(0.03, 0.16, "strict null: flat\n($\\rho=-0.11$, $p=0.38$)",
            transform=ax.transAxes, fontsize=5.4, va="top", color=GREEN)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.92,
              fontsize=4.5, loc="center left", bbox_to_anchor=(1.01, 0.56),
              handletextpad=0.35, borderaxespad=0.0)
    panel_title(ax, "c", "Null error structured by $\\chi$")
    save(fig, "Fig4_R207_measured_validity")


# ---------------- Fig 5 ----------------

def fig5():
    a1 = pd.read_csv(R206O / "R206_A1_dial_fine_grid.csv")
    s2 = pd.read_csv(R205O / "R205_S2_matched_dial.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W2 * 0.78, 2.4), constrained_layout=True)

    ax = axes[0]
    palette = {"delaunay20k": GREY, "bengaluru_india": BLUE, "osaka_japan": PURPLE, "tehran_iran": GREEN}
    try:
        a5 = pd.read_csv(R206O / "R208_A5_dial_breadth.csv")
        for g, chain in a5.groupby("graph"):
            chain = chain.sort_values("chi_chord")
            ax.plot(chain.chi_chord.clip(lower=2e-3), chain.pc, "-", lw=0.6,
                    color="#b7b7b7", alpha=0.8, zorder=1)
    except Exception:
        pass
    for g, sub in a1.groupby("graph"):
        for seed, chain in sub.groupby("seed"):
            chain = chain.sort_values("chi_chord")
            ax.plot(chain.chi_chord.clip(lower=2e-3), chain.pc, "o-", ms=2.2, lw=0.8,
                    color=palette.get(g, "#333333"), alpha=0.8,
                    label=g.replace("_", " ") if seed == sub.seed.min() else None)
            ax.fill_between(chain.chi_chord.clip(lower=2e-3), chain.pc_ci_lo, chain.pc_ci_hi,
                            color=palette.get(g, "#333333"), alpha=0.10, lw=0)
    try:
        a8 = pd.read_csv(R206O / "R208_A8_midrange_dial.csv")
        for g, chain in a8.groupby("graph"):
            chain = chain.sort_values("chi_chord")
            ax.plot(chain.chi_chord.clip(lower=2e-3), chain.pc, "s--", ms=2.4, lw=0.8,
                    mfc="none", color=palette.get(g, "#333333"), alpha=0.9, zorder=3)
    except Exception:
        pass
    ax.plot([], [], "-", lw=0.6, color="#b7b7b7", label="8 further cities (1 chain)")
    ax.plot([], [], "s--", ms=2.4, mfc="none", color="#555555", label="mid-range arm (5-15x chords)")
    ax.set_xscale("log")
    ax.set_xlabel(r"measured $\chi$ (chord space, log)")
    ax.set_ylabel(r"$p_c$ (fine grid, 95% CI)")
    ax.set_ylim(0.16, 0.88)
    ax.legend(frameon=False, fontsize=4.4, loc="center left", handletextpad=0.4)
    ax.text(0.03, 0.012, "1-3-crossing injection, degree +\nedge count preserved: largest\nshift 0.028 across 12 graphs",
            transform=ax.transAxes, fontsize=4.9, va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.8))
    panel_title(ax, "a", "Local/mid-range crossings: $p_c$ bounded")

    ax = axes[1]
    palette.update({"cairo_egypt": "#e07b39", "gabriel20k": "#5c5c5c"})
    cross = s2[s2.arm == "cross"]
    for g, sub in cross.groupby("graph"):
        sub = sub.sort_values("chi_chord")
        base = a1[(a1.graph == g) & (a1.chi_added == 0)]
        if len(base):
            x0 = max(base.chi_chord.mean(), 2e-3)
            xs = np.concatenate([[x0], sub.chi_chord.values])
            ys = np.concatenate([[base.pc.mean()], sub.pc.values])
        else:
            xs, ys = sub.chi_chord.values, sub.pc.values
        ax.plot(xs, ys, "s--", ms=2.2, lw=0.8, color=palette.get(g, "#333333"),
                alpha=0.85, label=g.replace("_", " "))
    ax.set_xscale("log")
    ax.set_xlabel(r"measured $\chi$ (chord space, log)")
    ax.set_ylabel(r"$p_c$")
    ax.set_ylim(0.20, 0.78)
    ax.axvspan(2e-3, 0.193, color=GREEN, alpha=0.08, lw=0)
    ax.text(0.02, 0.03, "shaded: observed city range", transform=ax.transAxes, fontsize=5.2, va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.5))
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.94,
              fontsize=4.7, loc="center left", bbox_to_anchor=(1.02, 0.63),
              handletextpad=0.35, borderaxespad=0.0, borderpad=0.25,
              labelspacing=0.18)
    panel_title(ax, "b", "Long-range shortcuts collapse $p_c$")
    save(fig, "Fig5_R207_controlled_dials")


# ---------------- Fig 6 ----------------

def fig6():
    fig, axes = plt.subplots(1, 2, figsize=(W2 * 0.82, 2.4), constrained_layout=True)
    ax = axes[0]
    domains = ["roads (71)", "water (4)", "rail (6)", "power (10)"]
    cebh = [0.217, 0.276, 0.240, 0.303]
    dim = [0.022, 0.028, 0.050, 0.132]
    anch = [0.067, np.nan, np.nan, 0.082]
    xpos = np.arange(len(domains))
    w = 0.27
    ax.bar(xpos - w, cebh, w, color=RED, alpha=0.75, label="degree-moment")
    ax.bar(xpos, dim, w, color=GREEN, alpha=0.75, label=r"$2/\langle k\rangle$")
    ax.bar(xpos + w, anch, w, color=PURPLE, alpha=0.75, label="junction anchor")
    ax.set_xticks(xpos)
    ax.set_xticklabels(domains, fontsize=5.6)
    ax.set_ylabel("threshold MAE")
    ax.legend(frameon=False, fontsize=5, loc="upper left", handletextpad=0.4)
    panel_title(ax, "a", "Zero-parameter accuracy by domain")

    ax = axes[1]
    entries = [
        ("water", 0.0, 1.96, GREEN, "planar end ($\\chi\\approx0$)"),
        ("roads", 0.021, 2.05, BLUE, "median city"),
        ("roads: Hong Kong", 0.152, 2.05, BLUE, "highest measured"),
        ("rail (validated)", 0.106, np.nan, PURPLE, "local-type crossings"),
        ("power", np.nan, 2.6, RED, "$\\chi$ unmeasured\n(no coordinates)"),
    ]
    for i, (name, chi, deff, col, note) in enumerate(entries):
        y = len(entries) - i
        if not np.isnan(chi):
            ax.scatter([max(chi, 1.2e-3)], [y], s=30, color=col, zorder=5)
        else:
            ax.scatter([0.4], [y], s=30, facecolor="none", edgecolor=col, zorder=5)
        label = name + ("" if np.isnan(deff) else f"  ($d_{{\\rm eff}}$ {deff:g})")
        ax.text(1.5e-3 if np.isnan(chi) else max(chi, 1.2e-3), y + 0.22, label, fontsize=5.6, ha="left")
        ax.text(1.5e-3 if np.isnan(chi) else max(chi, 1.2e-3), y - 0.34, note, fontsize=4.8,
                ha="left", color="#666666")
    ax.axvspan(1e-3, 0.193, color=GREEN, alpha=0.06, lw=0)
    ax.text(0.080, 0.28, "planar-baseline validity domain\n(local-type exposure)",
            fontsize=5.0, color="#2a6f5f", ha="center", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=0.6))
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 0.7)
    ax.set_ylim(0.2, len(entries) + 0.8)
    ax.set_yticks([])
    ax.set_xlabel(r"measured crossing exposure $\chi$ (log)")
    panel_title(ax, "b", "Cross-domain validity map")
    save(fig, "Fig6_R207_validity_map")


for fn in (fig1, fig2, fig4, fig5, fig6):
    try:
        fn()
    except Exception:
        print(f"[C] {fn.__name__} FAILED:\n{traceback.format_exc(limit=5)}", flush=True)
print("[C] DONE", flush=True)
