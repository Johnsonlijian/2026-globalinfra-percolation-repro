"""Build numeric bindings, main tables and transfer figures from checked outputs."""

import matplotlib
import numpy as np
from common import Context, parser
from matplotlib import font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

CTX = None
FIG = MAN = None
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.5,
        "savefig.dpi": 220,
    }
)
BLUE, TEAL, RED, AMBER, GRAY = "#2b61a0", "#137b7b", "#b33c45", "#b9901a", "#9c9c9c"
READS = {}


def read_csv(rel):
    return CTX.csv(rel)


def read_json(rel):
    return CTX.json(rel)


def save(fig, stem):
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(FIG / f"{stem}.{suffix}", bbox_inches="tight", pad_inches=0.045)
    plt.close(fig)


def panel(ax, letter, title):
    ax.set_title(title, loc="left", pad=10)
    ax.text(-0.14, 1.065, letter, transform=ax.transAxes, weight="bold", fontsize=12)


def main(context):
    global CTX, FIG, MAN
    CTX = context
    FIG, MAN = context.output / "figures", context.output
    try:
        font_manager.findfont("Arial", fallback_to_default=False)
    except ValueError:
        plt.rcParams["font.family"] = "DejaVu Sans"
    FIG.mkdir(parents=True, exist_ok=True)
    (MAN / "tables").mkdir(exist_ok=True)
    ref = read_csv("observed")
    roads = ref[ref.domain.eq("road")].copy()
    ext = read_csv("external")
    split = read_json("external_split_note")
    cases = split["sensitivity_cases"]
    ext_primary = ext[~ext.instance.isin(cases)].copy()
    ext_sensitivity = ext[ext.instance.isin(cases)].copy()
    if len(ext_primary) != 19 or len(ext_sensitivity) != 2:
        raise ValueError("External cohort split changed")
    met = read_csv("domain_metrics")
    null = read_csv("null_summary").set_index("kind")
    ep = read_csv("external_primary_predictors").set_index("predictor")
    ep_all = read_csv("external_predictors").set_index("predictor")
    splits = read_csv("external_cohort_split_summary").set_index("cohort")
    primary = splits.loc["external_primary19"]
    shift = read_csv("external_shift_comparator").set_index("cohort")
    shift_summary = read_json("external_shift_summary")
    summ = read_json("external_summary")
    loss = read_json("loss_tradeoff")
    q = summ["q"]
    for value in (splits.loc["external_sensitivity2", "q"], shift.loc["reference", "q"]):
        if abs(float(value) - q) > 1e-14:
            raise ValueError("Split outputs do not share the locked allowance")
    road_cebh = met.query("domain == 'road' and predictor == 'CEBH'").iloc[0]
    facts = {
        "WaterAdditionalN": "3",
        "RoadMAE": f"{road_cebh.absolute_error:.3f}",
        "LockedMargin": f"{q:.4f}",
        "ExternalRetained": f"{100 * primary['retained']:.1f}",
        "ExternalEscalated": f"{100 * primary['escalated']:.1f}",
        "ExternalCEBHMAE": f"{ep.loc['CEBH', 'mae']:.3f}",
        "ExternalTwoDMAE": f"{ep.loc['2D', 'mae']:.3f}",
        "ExternalAcquiredCEBHMAE": f"{float(ep_all.loc['CEBH', 'mae']):.3f}",
        "ExternalCEBHUnsafePrimary": f"{100 * float(ep.loc['CEBH', 'unsafe']):.1f}",
        "ExternalTwoDUnsafePrimary": f"{100 * float(ep.loc['2D', 'unsafe']):.1f}",
        "ExternalTwoDConservativePrimary": f"{100 * float(ep.loc['2D', 'conservative']):.1f}",
        "ExternalPrimaryRetainAccept": f"{100 * float(primary['retained_acceptance']):.1f}",
        "ExternalPrimaryRetainReject": f"{100 * float(primary['retained_rejection']):.1f}",
        "ExternalPrimaryInterior": f"{int(primary['interior_escalation_intervals'])}",
        "ExternalCaseCEBHMAE": f"{abs(float(ext_sensitivity.pc.iloc[0]) - float(ext_sensitivity.pc_cebh.iloc[0])):.3f}",
        "ExternalCovered": f"{int(primary['covered'])}/19",
        "ExternalCaseMAE": f"{float(splits.loc['external_sensitivity2', 'mae']):.3f}",
        "ExternalCaseRetained": f"{100 * float(splits.loc['external_sensitivity2', 'retained']):.1f}",
        "ExternalAllCovered": f"{int(splits.loc['external_acquired21', 'covered'])}/21",
        "ExternalAllRetained": f"{100 * float(splits.loc['external_acquired21', 'retained']):.1f}",
        "ShiftOffset": f"{float(shift.loc['reference', 'offset']):.3f}",
        "ShiftExternalMAE": f"{float(shift.loc['external_primary', 'mae']):.3f}",
        "ShiftExternalCovered": f"{int(shift.loc['external_primary', 'covered'])}/19",
        "ShiftPairedDifference": f"{shift_summary['paired_external_mae_difference_shift_minus_2d']:.4f}",
        "ShiftPairedLow": f"{shift_summary['paired_external_mae_difference_low']:.4f}",
        "ShiftPairedHigh": f"{shift_summary['paired_external_mae_difference_high']:.4f}",
        "ShiftConditionalUnsafe": f"{100 * float(shift.loc['external_primary', 'conditional_unsafe']):.2f}",
        "PowerBreakEven": f"{loss['power_junction_2d_break_even_unsafe_cost']:.2f}",
    }
    for dom in ("road", "power", "rail", "water"):
        for pred, key in (
            ("CEBH", "CEBH"),
            ("2D", "TwoD"),
            ("junction", "Junction"),
            ("nonbacktracking", "NB"),
        ):
            row = met[(met.domain == dom) & (met.predictor == pred)].iloc[0]
            stem = dom.title() + key
            for field, tag, scale, fmt in (
                ("absolute_error", "MAE", 1, ".3f"),
                ("unsafe_integrated", "Unsafe", 100, ".1f"),
                ("safe_integrated", "Conservative", 100, ".1f"),
            ):
                facts[stem + tag] = format(row[field] * scale, fmt)
    for kind in ("degree", "connected"):
        for field, key in (("null_gap", "Gap"), ("organization_residual", "Residual")):
            facts[kind.title() + key] = f"{null.loc[kind, field]:.3f}"
    (MAN / "facts.tex").write_text(
        "\n".join(r"\newcommand{\%s}{%s}" % (k, v) for k, v in facts.items()) + "\n",
        encoding="ascii",
    )
    names = {
        "CEBH": "Degree moment",
        "2D": r"$2/\langle k\rangle$",
        "junction": "Junction",
        "nonbacktracking": "Leading NB",
    }
    rows = []
    for dom in ("road", "power", "rail"):
        for pred in names:
            v = met[(met.domain == dom) & (met.predictor == pred)].iloc[0]
            rows.append(
                f"{dom.title()} & {names[pred]} & {v.absolute_error:.3f} "
                f"({v.absolute_error_low:.3f}--{v.absolute_error_high:.3f}) & "
                f"{100 * v.unsafe_integrated:.1f} & {100 * v.safe_integrated:.1f} " + r"\\"
            )
        if dom != "rail":
            rows.append(r"\addlinespace")
    (MAN / "tables/errors.tex").write_text(
        r"\newcommand{\MainErrorRows}{%" + "\n" + "\n".join(rows) + "\n}\n",
        encoding="ascii",
    )
    vals = []
    for pred in ("CEBH", "2D"):
        row = ep.loc[pred]
        vals.append(
            f"{names[pred]} & {row.mae:.3f} & {100 * row.unsafe:.1f} & "
            f"{100 * row.conservative:.1f} & 100.0 " + r"\\"
        )
    vals.append(
        r"$2/\langle k\rangle$ with escalation & -- & 0.0 & 0.0 & "
        + f"{100 * primary['retained']:.1f} "
        + r"\\"
    )
    (MAN / "tables/validation.tex").write_text(
        r"\newcommand{\MainValidationRows}{%" + "\n" + "\n".join(vals) + "\n}\n",
        encoding="ascii",
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.1), gridspec_kw={"width_ratios": [1, 1.02]})
    fig.subplots_adjust(wspace=0.4)
    ax = axes[0]
    for data, name, col, ls in (
        (abs(roads.pc - roads.pc_dimension), "Reference, n = 71", BLUE, "-"),
        (abs(ext_primary.pc - ext_primary.pc_dimension), "Primary cities, n = 19", TEAL, "--"),
    ):
        arr = np.sort(data.to_numpy())
        ax.step(
            np.r_[0, arr, 0.12],
            np.r_[0, np.arange(1, len(arr) + 1) / len(arr), 1],
            where="post",
            label=name,
            color=col,
            ls=ls,
        )
    for j, row in enumerate(ext_sensitivity.itertuples()):
        case_error = abs(row.pc-row.pc_dimension)
        ax.scatter([case_error], [0.035], color=AMBER, marker='|', s=110,
                   label='Prior-use (2)' if j == 0 else None, zorder=4)
    ax.axvline(q, color=RED, ls=":", lw=1.5)
    ax.text(q + 0.0022, 0.48, f"Locked q = {q:.4f}", rotation=90, color=RED,
            va="bottom", ha="left", fontsize=9)
    ax.set(
        xlabel=r"Absolute error of $2/\langle k\rangle$",
        ylabel="Fraction of cities",
        xlim=(0, 0.12),
        ylim=(0, 1.05),
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    handles, labels = ax.get_legend_handles_labels()
    labels = [label.replace('Reference, n = 71', 'Reference (71)').replace(
        'Primary cities, n = 19', 'Primary (19)') for label in labels]
    ax.legend(handles, labels, frameon=False, loc="lower right", fontsize=8.6,
              handlelength=1.2, handletextpad=0.6, bbox_to_anchor=(1.05, 0))
    panel(ax, "a", "Fixed allowance on external cities")
    ax = axes[1]
    unsafe = [ep.loc["CEBH", "unsafe"], ep.loc["2D", "unsafe"], primary["unsafe_retained"]]
    conservative = [
        ep.loc["CEBH", "conservative"],
        ep.loc["2D", "conservative"],
        primary["safe_retained"],
    ]
    escalated = [0, 0, primary["escalated"]]
    agree = 1 - np.array(unsafe) - np.array(conservative) - np.array(escalated)
    left = np.zeros(3)
    for values, label, col in (
        (agree, "Retained agreement", TEAL),
        (unsafe, "Unsafe disagreement", RED),
        (conservative, "Conservative disagreement", AMBER),
        (escalated, "Escalated", GRAY),
    ):
        ax.barh(
            np.arange(3),
            values,
            left=left,
            color=col,
            label=label,
            height=0.54,
            edgecolor="white",
            linewidth=0.4,
        )
        left += values
    ax.set(
        yticks=np.arange(3),
        yticklabels=["Degree\nmoment", r"$2/\langle k\rangle$", "With\nescalation"],
        xlim=(0, 1),
        xlabel="Fraction of target decisions",
    )
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
        loc="upper center",
        ncol=2,
        fontsize=9,
    )
    panel(ax, "b", "Decision allocation, external cohort")
    save(fig, "fig5_validation")

    geometry = read_csv("road_geometry")
    rail = read_csv("rail_geometry")
    rts = read_json("rts")
    scenarios = read_csv("scenario_instances")
    roadgeo = geometry.merge(
        roads[["instance", "pc", "pc_dimension"]],
        left_on="city",
        right_on="instance",
        validate="one_to_one",
    )
    railgeo = rail.merge(
        ref[ref.domain.eq("rail")][["instance", "pc", "pc_dimension"]],
        left_on="city",
        right_on="instance",
        validate="one_to_one",
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.35))
    fig.subplots_adjust(wspace=0.43)
    ax = axes[0]
    ax.scatter(
        roadgeo.chi_poly,
        abs(roadgeo.pc - roadgeo.pc_dimension),
        s=21,
        color=BLUE,
        alpha=0.65,
        label="Roads, polylines (71)",
    )
    ax.scatter(
        railgeo.chi_whole_poly_projected,
        abs(railgeo.pc - railgeo.pc_dimension),
        s=36,
        marker="^",
        color=TEAL,
        label="Rail, polylines* (6)",
    )
    chi = rts.get("chi_chord", rts.get("chi"))
    if chi is None:
        raise KeyError("RTS chord exposure missing; inspect source schema")
    ax.scatter(
        [chi],
        [abs(rts["pc"] - rts["pc_dimension_anchor"])],
        s=55,
        marker="D",
        facecolors="none",
        edgecolors=RED,
        label="RTS-GMLC, chords (1)",
    )
    scigrid = read_csv('scigrid')
    full = scigrid[scigrid.scope.eq('germany_full')].iloc[0]
    ax.scatter([full.chi_chord], [abs(full.pc-full.pc_dimension)], s=48,
               marker='s', facecolors='none', edgecolors=AMBER,
               label='SciGRID, chords (1)')
    ax.set(
        xlabel=r"Map-view crossing pairs per edge, $\chi$",
        ylabel=r"Absolute error of $2/\langle k\rangle$",
        xlim=(-0.012, 0.47),
        ylim=(-0.004, 0.17),
    )
    ax.legend(frameon=False, loc="upper left", fontsize=8.3, handletextpad=0.3)
    fig.text(
        0.08,
        -0.07,
        "Not on the crossing axis: 10 power cases and 7 water benchmarks lack verified geography.",
        fontsize=8.5,
        color="#555555",
    )
    panel(ax, "a", "Crossings and prediction error")
    ax = axes[1]
    domains = ("road", "power", "rail", "water")
    for i, dom in enumerate(domains):
        for sc, offset, col, marker in (
            ("targeted", -0.12, RED, "o"),
            ("localized", 0.12, TEAL, "s"),
        ):
            # Scenario labels are required to be explicit in the derived file.
            subset = scenarios[(scenarios.domain == dom) & (scenarios.scenario.str.contains(sc))]
            if subset.empty:
                continue
            ys = subset.auc_change.to_numpy()
            x = i + offset + np.linspace(-0.055, 0.055, len(ys))
            ax.scatter(x, ys, color=col, s=18, marker=marker, alpha=0.6)
            ax.plot(
                [i + offset - 0.075, i + offset + 0.075], [np.median(ys)] * 2, color=col, lw=2.4
            )
    ax.axhline(0, color="#777777", lw=0.8)
    ax.set(
        xticks=range(4),
        xticklabels=["Road", "Power", "Rail", "Water"],
        ylabel="Change in connectivity-curve AUC",
        xlim=(-0.55, 3.55),
    )
    for col, marker, label in ((RED, "o", "Static targeted"), (TEAL, "s", "Localized")):
        ax.scatter([], [], color=col, marker=marker, label=label, s=22)
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    panel(ax, "b", "Dependence on removal scenario")
    save(fig, "fig6_geometry")
    manifest = {
        "inputs_sha256": context.reads,
        "figures": ["fig5_validation", "fig6_geometry"],
        "fact_macros": facts,
        "notes": [
            "Figure 5 metrics averaged by city over a uniform target band; escalation is not agreement.",
            "Figure 5a marks two prior-use cities identified by a retrospective provenance audit; allowance unchanged.",
            "Figure 6 rail polyline geometry falls back to endpoint chords when no geometry is stored.",
            "Power/water geographic scenarios excluded when source geography is unverified.",
            "No generated data or AI-generated raster assets.",
        ],
    }
    context.write_json("main_asset_bindings.json", manifest)
    context.finish(__file__)
    print("Built facts, two tables, figures 5 and 6.")


if __name__ == "__main__":
    main(Context(parser(__doc__).parse_args()))
