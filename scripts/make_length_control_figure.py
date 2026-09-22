"""Plot the verified clean length control from included derived stage summaries."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from common import Context, parser
from matplotlib import font_manager
from matplotlib.lines import Line2D

CITIES = [
    ("bengaluru_india", "Bengaluru", "#2b61a0", [111, 122, 133]),
    ("osaka_japan", "Osaka", "#137b7b", [211, 222, 233]),
    ("tehran_iran", "Tehran", "#b33c45", [311, 322, 333]),
]


def main(context):
    frame = context.csv("length_control")
    if (
        len(frame) != 36
        or not frame.selfloops.eq(0).all()
        or not frame.invariants_pass.all()
        or not frame.n_orders.eq(512).all()
    ):
        raise ValueError("Only the verified 36-stage simple-graph control is eligible")
    keys = ["graph", "seed", "stage_index"]
    differences = context.csv("local_difference_ranges").set_index(keys, verify_integrity=True)
    original = frame.set_index(keys, verify_integrity=True)
    if len(differences) != 36 or set(differences.index) != set(original.index):
        raise ValueError("Difference ranges must match all 36 original stages")
    differences = differences.loc[original.index]
    for field in ("pc", "pc_shift", "chi_chord", "chi_added"):
        np.testing.assert_allclose(differences[field], original[field], rtol=0, atol=1e-12)
    if (
        not differences.calibration_q_used.eq(False).all()
        or not differences.quantile_method.eq("linear").all()
        or not differences.quantile_low.eq(0.025).all()
        or not differences.quantile_high.eq(0.975).all()
        or not np.isfinite(differences[["difference_low", "difference_high"]]).all().all()
        or not differences.difference_low.le(differences.difference_high).all()
    ):
        raise ValueError("Invalid numerical difference-range definition")
    baseline = differences.loc[differences.index.get_level_values("stage_index") == 0]
    nonbaseline = differences.loc[differences.index.get_level_values("stage_index") != 0]
    if (
        not baseline.distribution.eq("baseline_identity").all()
        or not baseline[["difference_low", "difference_high", "product_combinations"]].eq(0).all().all()
        or not nonbaseline.distribution.eq("empirical_independent_stage_product").all()
        or not nonbaseline.product_combinations.eq(1_000_000).all()
        or not nonbaseline.stage_estimator_seed.ne(nonbaseline.baseline_estimator_seed).all()
    ):
        raise ValueError("Expected baseline identities and independent-stage product ranges")
    font = "Arial"
    try:
        font_manager.findfont(font, fallback_to_default=False)
    except ValueError:
        font = "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": font,
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "mathtext.fontset": "custom",
            "mathtext.rm": font,
            "mathtext.it": font + ":italic",
            "mathtext.bf": font + ":bold",
            "svg.hashsalt": "clean-local-control",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2))
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.18, top=0.76, wspace=0.43)
    markers = ["o", "s", "^"]
    for graph, _, color, seeds in CITIES:
        for replicate, seed in enumerate(seeds):
            chain = frame[(frame.graph == graph) & (frame.seed == seed)].sort_values("stage_index")
            if chain.stage_index.tolist() != [0, 1, 2, 3]:
                raise ValueError("Incomplete control chain")
            np.testing.assert_allclose(chain.pc_shift, chain.pc - chain.pc.iloc[0], atol=1e-12)
            for ax, field in zip(axes, ("pc_shift", "ks_length_statistic"), strict=True):
                ax.plot(
                    chain.chi_chord,
                    chain[field],
                    color=color,
                    linewidth=0.85,
                    marker=markers[replicate],
                    markersize=3.6,
                    markeredgewidth=0.8,
                    markerfacecolor="white",
                    alpha=0.92,
                )
            terminal = differences.loc[(graph, seed, 3)]
            low, high = terminal.difference_low, terminal.difference_high
            bars = axes[0].vlines(
                chain.chi_chord.iloc[-1], low, high, color=color, linewidth=0.7, alpha=0.65, zorder=1
            )
            bars.set_gid(f"terminal-range-{graph}-{seed}")
            caps, = axes[0].plot(
                [chain.chi_chord.iloc[-1]] * 2, [low, high], color=color, linestyle="none",
                marker="_", markersize=4, markeredgewidth=0.7, alpha=0.65, zorder=1,
            )
            caps.set_gid(f"terminal-caps-{graph}-{seed}")
    axes[0].axhline(0, color="#8a8a8a", linewidth=0.6, linestyle=(0, (3, 3)), zorder=0)
    axes[0].set_ylabel(r"Threshold change, $\Delta p_\star$", labelpad=4)
    axes[1].set_ylabel("Continuous edge-length KS", labelpad=4)
    axes[0].set_title("Response-peak change", loc="left", pad=8)
    axes[1].set_title("Within-bin length drift", loc="left", pad=8)
    for ax, letter in zip(axes, ("a", "b"), strict=True):
        ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontweight="bold", fontsize=11)
    margin = 0.045 * (frame.chi_chord.max() - frame.chi_chord.min())
    for ax in axes:
        ax.set_xlim(frame.chi_chord.min() - margin, frame.chi_chord.max() + margin)
        ax.set_xlabel(r"Achieved chord crossing density, $\chi$", labelpad=5)
        ax.tick_params(direction="out", length=3, pad=3)
        ax.grid(False)
    axes[0].margins(y=0.13)
    if not axes[0].get_ylim()[0] < 0 < axes[0].get_ylim()[1]:
        raise ValueError("The zero reference must remain inside panel (a)")
    axes[1].set_ylim(-0.0004, frame.ks_length_statistic.max() * 1.12)
    for ax in axes:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        ax.set_xticks([tick for tick in ax.get_xticks() if xmin <= tick <= xmax])
        ax.set_yticks([tick for tick in ax.get_yticks() if ymin <= tick <= ymax])
    cities = [Line2D([], [], color=color, lw=1.25, label=city) for _, city, color, _ in CITIES]
    seeds = [
        Line2D(
            [],
            [],
            color="#4b4b4b",
            marker=marker,
            ls="none",
            ms=3.8,
            mfc="white",
            mew=0.8,
            label=f"Replicate {i + 1}",
        )
        for i, marker in enumerate(markers)
    ]
    fig.legend(
        handles=cities,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.55, 1.01),
        frameon=False,
        columnspacing=2.0,
        handlelength=1.8,
    )
    fig.legend(
        handles=seeds,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.925),
        frameon=False,
        columnspacing=1.5,
        handlelength=1.4,
    )
    directory = context.output / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(directory / f"fig4_local_control.{suffix}", dpi=300, facecolor="white")
    plt.close(fig)
    context.finish(__file__)
    print(
        "Regenerated Fig4 from 36 unchanged stage summaries and nine terminal product-difference ranges; "
        "no private curve caches or raw edge/length recalculation are used."
    )


if __name__ == "__main__":
    main(Context(parser(__doc__).parse_args()))
