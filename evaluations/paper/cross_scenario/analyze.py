"""Aggregate cross-scenario CSVs → two LaTeX tables + violins.

Linear ``main()`` — every step here is one call into ``_common``.
Reads top-to-bottom as a recipe; no function-in-function nesting.

Inputs (from :mod:`evaluations.paper.cross_scenario.run`)::

    evaluations/results/cross_scenario/eval/<cell>/compare_<scenario>.csv
    evaluations/results/cross_scenario/eval_finetune/<cell>/compare_<scenario>.csv
    evaluations/results/cross_scenario/eval_leader_only/<cell>/compare_<scenario>.csv
    evaluations/results/cross_scenario/selection.json

Outputs::

    evaluations/reports/tables/*.tex, *.html
    evaluations/reports/figures/*.pdf, *.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from evaluations._common.csv_loader import load_compare_csv
from evaluations._common.latex import bold_max, fmt_pm, fmt_rate, row, write_tabular
from evaluations._common.stats import mean_std
from evaluations._common.paths import CS_RESULTS, FIGURES, TABLES
from evaluations.paper.cross_scenario.selection import apply_selection

# The three eval kinds under results/cross_scenario/, and the method label
# their rows get (None keeps the CSV's own ``mode``):
#   eval              unguided / value-only / value-and-ood / uniform
#   eval_finetune     the cell's fine-tuned leader, run unguided
#   eval_leader_only  alpha=0: the planner solves the leader objective only
EVAL_KINDS: dict[str, str | None] = {
    "eval": None,
    "eval_finetune": "finetune-baseline",
    "eval_leader_only": "leader-only",
}

# Mode column order in every table. ``finetune-baseline`` is the
# zero-deploy-cost RL alternative to CEM: the leader was warm-started
# with PPO on L_norm + α·F_norm using a Monte-Carlo follower rollout
# for the terminal reward, evaluated unguided. See
# training/leader_follower_finetune.py.
# ``leader-only`` is the reviewer-requested alpha=0 baseline: same CEM
# planner as PAVE but with alpha=0 zeroing the follower value term, so
# the planner optimises the leader objective in isolation.
MODES = ("unguided", "value-only", "value-and-ood", "uniform",
         "leader-only", "finetune-baseline")
MODE_LABELS = {
    "unguided": r"Naive",
    "value-only": r"PAVE (w/o OOD)",
    "value-and-ood": r"PAVE",
    "uniform": r"PAVE (w/o $\pi_L$)",
    "leader-only": r"Leader-only ($\alpha=0$)",
    "finetune-baseline": r"FT",
}
# One color per method, consistent across the whole combined violin figure.
MODE_COLORS = {
    "unguided": "#7f7f7f",       # gray
    "value-only": "#1f77b4",     # blue
    "value-and-ood": "#2ca02c",  # green
    "uniform": "#ff7f0e",        # orange
    "leader-only": "#8c564b",    # brown
    "finetune-baseline": "#9467bd",  # purple
}

# Per-scenario success-event column name (scenarios disagree on what
# "follower success" means at the env level — see SUCCESS_INFO_KEY on
# each env class).
FOLLOWER_SUCCESS_COL = {
    "mobile_arm": "follower_dropped",
    "obstacle_arena": "follower_reached_goal",
    "warehouse": "follower_on_goal",
    "warehouse_pickup_blocked": "follower_on_goal",
    "warehouse_pickup_unseen": "follower_on_goal",
    "household": "follower_reached_goal",
}
LEADER_SUCCESS_COL = {
    # Composite: leader chose to terminate AND ended up within ½ sigma of
    # the desired stand-off radius (info["leader_success"] from
    # leader_env.py). "leader_terminated" was a stale name — that column
    # never existed in mobile_arm result rows.
    "mobile_arm": "leader_success",
    "obstacle_arena": "leader_hit_wall",
    "warehouse": "leader_dropped_object",
    "warehouse_pickup_blocked": "leader_dropped_object",
    "warehouse_pickup_unseen": "leader_dropped_object",
    "household": "leader_picked_up_dirt",
}


def load_results(results_dir: Path) -> pd.DataFrame:
    """Every eval CSV under ``results_dir`` as one long frame with a ``cell``
    column; rows of the finetune / leader-only evals relabelled per
    :data:`EVAL_KINDS`. Rows are concatenated cell by cell, kinds in
    :data:`EVAL_KINDS` order, so aggregates are order-stable."""
    cells = sorted({p.name for kind in EVAL_KINDS
                    if (results_dir / kind).is_dir()
                    for p in (results_dir / kind).iterdir() if p.is_dir()})
    frames: list[pd.DataFrame] = []
    for cell in cells:
        for kind, label in EVAL_KINDS.items():
            for csv in sorted((results_dir / kind / cell).glob("compare_*.csv")):
                d = load_compare_csv(csv)
                d["cell"] = cell
                if label is not None:
                    d["mode"] = label
                frames.append(d)
    if not frames:
        raise FileNotFoundError(f"No compare_*.csv under {results_dir}/{{{','.join(EVAL_KINDS)}}}")
    return pd.concat(frames, ignore_index=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="evaluations.paper.cross_scenario.analyze",
        description="Aggregate cross-scenario CSVs into LaTeX tables.",
    )
    p.add_argument("--results-dir", type=Path, default=CS_RESULTS,
                   help="Results root from run.py (default: %(default)s).")
    p.add_argument("--allow-missing-selection", action="store_true",
                   help="Analyse a results dir without SWEEP.json-declared "
                        "competence gates. Never use for paper tables.")
    p.add_argument("--tables-dir", type=Path, default=TABLES)
    p.add_argument("--figures-dir", type=Path, default=FIGURES)
    return p.parse_args(argv)


def _rate_cells(means: Sequence[float], stds: Sequence[float]) -> list[str]:
    """``mean ± std`` cells (across-seed), no bolding."""
    return [
        (f"{fmt_rate(m)}\\,$\\pm$\\,{fmt_rate(s)}" if np.isfinite(m) else "--")
        for m, s in zip(means, stds)
    ]


def _build_success_rows(
    df: pd.DataFrame,
    scenarios: Sequence[str],
    modes: Sequence[str],
) -> tuple[list[str], list[tuple]]:
    """Build the S_L / S_F / joint sub-rows for a success-rate table.

    Returns ``(rows, pave_vs_naive_stats)``. ``pave_vs_naive_stats`` is
    populated only when both ``unguided`` and ``value-and-ood`` are in
    ``modes`` — used by the ``pave_vs_naive.tex`` summary block.
    """
    rows: list[str] = []
    stats: list[tuple] = []
    naive_idx = modes.index("unguided") if "unguided" in modes else None
    pave_idx = modes.index("value-and-ood") if "value-and-ood" in modes else None
    for scenario in scenarios:
        scen_df = df[df["scenario"] == scenario]
        sl_col = LEADER_SUCCESS_COL.get(scenario)
        sf_col = FOLLOWER_SUCCESS_COL.get(scenario)
        if sf_col is None or sf_col not in scen_df.columns:
            continue
        sl_m, sl_s, sf_m, sf_s = [], [], [], []
        joint_m, joint_s = [], []
        have_sl = bool(sl_col) and sl_col in scen_df.columns
        for mode in modes:
            mode_df = scen_df[scen_df["mode"] == mode]
            if mode_df.empty:
                for lst in (sl_m, sl_s, sf_m, sf_s, joint_m, joint_s):
                    lst.append(float("nan"))
                continue
            sf_by_cell = mode_df.groupby("cell")[sf_col].mean()
            m, s = mean_std(sf_by_cell.tolist())
            sf_m.append(m); sf_s.append(s)
            if have_sl:
                sl_by_cell = mode_df.groupby("cell")[sl_col].mean()
                m, s = mean_std(sl_by_cell.tolist())
                sl_m.append(m); sl_s.append(s)
                # True joint: per-episode conjunction (both leader and
                # follower succeeded on the same episode), averaged per
                # seed. Multiplying per-seed S_L * S_F would assume
                # independence, which is wrong -- leader and follower
                # success are correlated within an episode.
                joint_events = (
                    mode_df[sl_col].astype(float)
                    * mode_df[sf_col].astype(float)
                )
                joint_by_cell = joint_events.groupby(mode_df["cell"]).mean()
                jm, js = mean_std(joint_by_cell.tolist())
                joint_m.append(jm); joint_s.append(js)
            else:
                sl_m.append(float("nan")); sl_s.append(float("nan"))
                joint_m.append(float("nan")); joint_s.append(float("nan"))
        scen_name = scenario.replace("_", r"\_")
        rows.append(row([f"\\scenario{{{scen_name}}}", "$S_L=$",
                         *_rate_cells(sl_m, sl_s)]))
        rows.append(row(["", "$S_F=$", *_rate_cells(sf_m, sf_s)]))
        rows.append(row(["", "$S_{L\\wedge F}=$",
                         *bold_max(_rate_cells(joint_m, joint_s), joint_m)]))
        rows.append("\\midrule")
        if naive_idx is not None and pave_idx is not None:
            stats.append((
                scenario,
                sf_m[naive_idx], sf_m[pave_idx],
                joint_m[naive_idx], joint_m[pave_idx],
            ))
    if rows and rows[-1] == "\\midrule":
        rows.pop()  # no rule directly above \bottomrule
    return rows, stats


def _success_stats(
    df: pd.DataFrame,
    scenarios: Sequence[str],
    modes: Sequence[str],
) -> dict[str, dict[str, tuple[float, float, float, float]]]:
    """Return ``{scenario: {mode: (sl_m, sl_s, sf_m, sf_s)}}`` (rates in [0, 1]).

    Aggregation matches ``_build_success_rows``: per-cell means first,
    then mean±std across cells (seeds).
    """
    out: dict[str, dict[str, tuple[float, float, float, float]]] = {}
    nan4 = (float("nan"),) * 4
    for scenario in scenarios:
        scen_df = df[df["scenario"] == scenario]
        sl_col = LEADER_SUCCESS_COL.get(scenario)
        sf_col = FOLLOWER_SUCCESS_COL.get(scenario)
        if sf_col is None or sf_col not in scen_df.columns:
            continue
        have_sl = bool(sl_col) and sl_col in scen_df.columns
        out[scenario] = {}
        for mode in modes:
            mode_df = scen_df[scen_df["mode"] == mode]
            if mode_df.empty:
                out[scenario][mode] = nan4
                continue
            sf_m, sf_s = mean_std(
                mode_df.groupby("cell")[sf_col].mean().tolist()
            )
            if have_sl:
                sl_m, sl_s = mean_std(
                    mode_df.groupby("cell")[sl_col].mean().tolist()
                )
            else:
                sl_m, sl_s = float("nan"), float("nan")
            out[scenario][mode] = (sl_m, sl_s, sf_m, sf_s)
    return out


_BAR_CELL_W = 1.9      # per-cell plot area, inches (S_L or S_F cell)
_BAR_CELL_H = 1.15
_BAR_LEFT_IN = 2.10    # scenario row labels (longest: "warehouse")
_BAR_RIGHT_IN = 0.20
_BAR_TOP_IN = 0.55     # $S_L$ / $S_F$ column titles
_BAR_BOTTOM_IN = 0.90  # method legend row
_BAR_WSPACE = 0.20     # fraction of cell_w
_BAR_HSPACE = 0.35     # fraction of cell_h
_BAR_TITLE_FS = 14     # $S_L$ / $S_F$ column titles
_BAR_YLABEL_FS = 12    # scenario row labels
_BAR_YTICK_FS = 10     # y-axis tick labels (0, 25, 50, 75, 100)
_BAR_LEGEND_FS = 11    # method legend


def _plot_success_bars(
    df: pd.DataFrame,
    scenarios: Sequence[str],
    modes: Sequence[str],
    mode_labels: dict[str, str],
    mode_colors: dict[str, str],
    out_pdf: Path,
    out_png: Path,
) -> None:
    """(scenario × {S_L, S_F}) grid with one bar per method inside each cell.

    Column layout is fixed at 2 (S_L on the left, S_F on the right) — this
    makes leader vs follower success rates directly comparable across
    methods within the same scenario row. Cell dimensions and margins are
    locked in absolute inches so subsets (fewer scenarios / fewer methods)
    produce figures with pixel-identical cell sizing — safe to lay out
    next to each other in a presentation without rescaling. Bars are
    coloured by method, matching the violin palette; a single legend row
    at the figure bottom names them.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    stats = _success_stats(df, scenarios, modes)
    metric_cols = (("sl", "$S_L$"), ("sf", "$S_F$"))
    n_rows, n_cols = len(scenarios), len(metric_cols)

    plot_w = _BAR_CELL_W * (n_cols + max(n_cols - 1, 0) * _BAR_WSPACE)
    plot_h = _BAR_CELL_H * (n_rows + max(n_rows - 1, 0) * _BAR_HSPACE)
    fig_w = _BAR_LEFT_IN + plot_w + _BAR_RIGHT_IN
    fig_h = _BAR_BOTTOM_IN + plot_h + _BAR_TOP_IN
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False, sharey=True,
    )

    xs = list(range(len(modes)))
    colors = [mode_colors[m] for m in modes]
    for r, scenario in enumerate(scenarios):
        for c, (metric_key, metric_label) in enumerate(metric_cols):
            ax = axes[r][c]
            heights = []
            for mode in modes:
                sl_m, _, sf_m, _ = stats.get(scenario, {}).get(
                    mode, (float("nan"),) * 4)
                v = sl_m if metric_key == "sl" else sf_m
                heights.append(100.0 * v if np.isfinite(v) else 0.0)
            ax.bar(
                xs, heights, width=0.78,
                color=colors,
                edgecolor="#212121", linewidth=0.6,
            )
            ax.set_xticks([])
            ax.set_xlim(-0.6, len(modes) - 0.4)
            ax.set_ylim(0, 108)
            ax.tick_params(axis="y", labelsize=_BAR_YTICK_FS)
            ax.grid(True, axis="y", alpha=0.25)
            if r == 0:
                ax.set_title(metric_label, fontsize=_BAR_TITLE_FS)
            if c == 0:
                ax.set_ylabel(scenario.replace("_", "\n"),
                              fontsize=_BAR_YLABEL_FS, rotation=0,
                              labelpad=44, va="center", ha="right")

    handles = [Patch(facecolor=mode_colors[m], edgecolor="#212121",
                     label=mode_labels[m]) for m in modes]
    # Legend centre placed inside the reserved bottom margin, so no
    # cropping is needed to keep cells at their absolute size.
    fig.legend(
        handles=handles, loc="lower center",
        bbox_to_anchor=(0.5, 0.32 / fig_h),
        ncol=len(modes), frameon=False, fontsize=_BAR_LEGEND_FS,
    )
    fig.subplots_adjust(
        left=_BAR_LEFT_IN / fig_w,
        right=1 - _BAR_RIGHT_IN / fig_w,
        top=1 - _BAR_TOP_IN / fig_h,
        bottom=_BAR_BOTTOM_IN / fig_h,
        wspace=_BAR_WSPACE, hspace=_BAR_HSPACE,
    )
    fig.savefig(out_pdf, metadata={"CreationDate": None})
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# Absolute-inch layout constants for the locked-cell violin variant.
# Kept parallel to the ``_BAR_*`` constants so subset violins line up
# with the bar plots visually in a presentation.
_VIO_CELL_W = 2.3      # per-scenario column width
_VIO_CELL_H = 2.3      # per-metric row height
_VIO_LEFT_IN = 1.05    # y-tick labels + $R_L$/$R_F$ ylabel
_VIO_RIGHT_IN = 0.25
_VIO_TOP_IN = 0.95     # method legend row + scenario column titles
_VIO_BOTTOM_IN = 0.35  # y-tick labels of bottom row
_VIO_WSPACE = 0.22     # fraction of cell_w (holds per-column y-ticks)
_VIO_HSPACE = 0.35     # fraction of cell_h


def _plot_violins(
    df: pd.DataFrame,
    scenarios: Sequence[str],
    modes: Sequence[str],
    mode_labels: dict[str, str],
    palette: dict[str, str],
    out_pdf: Path,
    out_png: Path,
    *,
    lock_cells: bool = False,
) -> None:
    """Combined cross-scenario violin figure.

    rows = (R_L, R_F); columns = scenarios; each cell = the method
    violins, one color per method. ``density_norm="width"`` matches the
    "same max width per violin" rendering; ``cut=0`` keeps the violin
    within the observed data range.

    When ``lock_cells=True`` the per-cell width/height is fixed in
    absolute inches (see the ``_VIO_*`` constants) and margins are set
    with ``subplots_adjust`` instead of ``tight_layout`` — subsets that
    change the scenario count then produce figures whose cells are
    pixel-identical, safe to place next to each other in a slide.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Patch

    METRICS = [("leader_return", r"$R_L$"), ("follower_return", r"$R_F$")]
    n_rows, n_cols = len(METRICS), len(scenarios)

    if lock_cells:
        plot_w = _VIO_CELL_W * (n_cols + max(n_cols - 1, 0) * _VIO_WSPACE)
        plot_h = _VIO_CELL_H * (n_rows + max(n_rows - 1, 0) * _VIO_HSPACE)
        fig_w = _VIO_LEFT_IN + plot_w + _VIO_RIGHT_IN
        fig_h = _VIO_BOTTOM_IN + plot_h + _VIO_TOP_IN
    else:
        fig_w = 2.3 * n_cols
        fig_h = 5.4
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False,
    )
    for r, (col, rlabel) in enumerate(METRICS):
        for c, scenario in enumerate(scenarios):
            ax = axes[r][c]
            scen_df = df[(df["scenario"] == scenario) & df["mode"].isin(modes)]
            if scen_df.empty:
                ax.set_visible(False)
                continue
            sns.violinplot(
                data=scen_df, x="mode", y=col, order=list(modes),
                hue="mode", palette=palette, hue_order=list(modes),
                inner=None, density_norm="width", cut=0,
                linewidth=0.4, ax=ax, legend=False,
            )
            for i, mode in enumerate(modes):
                v = scen_df.loc[scen_df["mode"] == mode, col]
                if v.size:
                    ax.scatter([i], [float(v.mean())], color="white",
                               s=7, zorder=5, edgecolor="black", linewidths=0.4)
            ax.set_xlabel("")
            ax.set_ylabel(rlabel if c == 0 else "",
                          fontsize=16, rotation=0, labelpad=22, va="center")
            ax.set_xticks([])
            ax.tick_params(axis="y", labelsize=10)
            ax.grid(True, axis="y", alpha=0.25)
            if r == 0:
                ax.set_title(scenario.replace("_", "\n"), fontsize=11)
    handles = [Patch(facecolor=palette[m], label=mode_labels[m]) for m in modes]

    if lock_cells:
        # Legend inside the reserved top margin, so no cropping needed.
        legend_y_frac = 1 - 0.20 / fig_h
        fig.legend(handles=handles, loc="upper center",
                   bbox_to_anchor=(0.5, legend_y_frac),
                   ncol=len(modes), frameon=False, fontsize=12)
        fig.subplots_adjust(
            left=_VIO_LEFT_IN / fig_w,
            right=1 - _VIO_RIGHT_IN / fig_w,
            top=1 - _VIO_TOP_IN / fig_h,
            bottom=_VIO_BOTTOM_IN / fig_h,
            wspace=_VIO_WSPACE, hspace=_VIO_HSPACE,
        )
        fig.savefig(out_pdf, metadata={"CreationDate": None})
        fig.savefig(out_png, dpi=150)
    else:
        fig.legend(handles=handles, loc="upper center", ncol=len(modes),
                   frameon=False, fontsize=12, bbox_to_anchor=(0.5, 1.02))
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(out_pdf, bbox_inches="tight", metadata={"CreationDate": None})
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    results_dir = args.results_dir
    print(f"Reading: {results_dir}")
    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # 1. Load all CSVs into one long DataFrame, with cell-name column.
    # ---------------------------------------------------------------------
    df = load_results(results_dir)
    # ``household`` is excluded from the cross-scenario success-rate
    # comparison: the cells in the canonical 2026-05-19 sweep were trained
    # and evaluated on the *pre-aid-window* household env, which has since
    # evolved (post_pickup_steps=50 + centered-half follower goal). The
    # baselines and any new fine-tune would therefore be measured on
    # different envs, making the comparison invalid. Household lives on as
    # a hardware-deployment example, not a method-comparison row.
    _EXCLUDE_FROM_ANALYSIS = {"household"}
    df = df[~df["scenario"].isin(_EXCLUDE_FROM_ANALYSIS)].reset_index(drop=True)
    if df.empty:
        print(f"All scenarios filtered out by exclusion list "
              f"{sorted(_EXCLUDE_FROM_ANALYSIS)}", file=sys.stderr)
        return 1

    # Per-scenario competence filter, driven by the gate manifests the
    # results dir declares in SWEEP.json. A declared-but-missing manifest
    # raises rather than letting rejected seeds into the tables.
    df = apply_selection(df, results_dir,
                         allow_missing=args.allow_missing_selection,
                         verbose=True)

    scenarios = sorted(df["scenario"].unique())
    print(f"  scenarios: {scenarios}")
    print(f"  rows:      {len(df)}")
    print(f"  cells:     {sorted(df['cell'].unique())}")

    # ---------------------------------------------------------------------
    # 2. success_rates.tex — S_L / S_F per (scenario, mode) ± seed std.
    # ---------------------------------------------------------------------
    success_rows, pave_vs_naive = _build_success_rows(df, scenarios, MODES)

    write_tabular(
        tables_dir / "success_rates.tex",
        column_spec="ll" + "c" * len(MODES),
        header=row(["Scenario", "", *(MODE_LABELS[m] for m in MODES)]),
        rows=success_rows,
    )
    print(f"  wrote {tables_dir / 'success_rates.tex'}")

    # ---------------------------------------------------------------------
    # 2b. PAVE - Naive improvement summary (S_F + joint).
    #     Stdout pretty-print AND `pave_vs_naive.tex` with newcommand-style
    #     definitions for direct \input{} in the paper body.
    # ---------------------------------------------------------------------
    if pave_vs_naive:
        # Per-scenario deltas in percentage points (raw means are in [0, 1]).
        rows_sum = []
        for scen, n_sf, p_sf, n_j, p_j in pave_vs_naive:
            rows_sum.append({
                "scenario":   scen,
                "naive_sf":   100.0 * n_sf,
                "pave_sf":    100.0 * p_sf,
                "delta_sf":   100.0 * (p_sf - n_sf),
                "naive_joint": 100.0 * n_j,
                "pave_joint":  100.0 * p_j,
                "delta_joint": 100.0 * (p_j - n_j),
            })
        deltas_sf    = np.array([r["delta_sf"]    for r in rows_sum])
        deltas_joint = np.array([r["delta_joint"] for r in rows_sum])
        argmax_sf    = int(np.argmax(deltas_sf))
        argmax_joint = int(np.argmax(deltas_joint))

        print()
        print("  PAVE - Naive improvement (percentage points)")
        print(f"  {'scenario':28s} {'ΔS_F':>9s}  {'ΔJoint':>9s}")
        print("  " + "-" * 50)
        for r in rows_sum:
            print(f"  {r['scenario']:28s} {r['delta_sf']:>+8.2f}   {r['delta_joint']:>+8.2f}")
        print("  " + "-" * 50)
        print(f"  {'max':28s} {deltas_sf.max():>+8.2f}   {deltas_joint.max():>+8.2f}")
        print(f"  {'max on':28s} {rows_sum[argmax_sf]['scenario']:>9s}  "
              f"{rows_sum[argmax_joint]['scenario']:>9s}")
        print(f"  {'mean':28s} {deltas_sf.mean():>+8.2f}   {deltas_joint.mean():>+8.2f}")
        print(f"  {'median':28s} {np.median(deltas_sf):>+8.2f}   "
              f"{np.median(deltas_joint):>+8.2f}")

        # Paper-friendly tex include: newcommand definitions for inline use.
        tex_lines = [
            "% Auto-generated by evaluations/paper/cross_scenario/analyze.py.",
            "% PAVE - Naive improvement across cross-scenario evaluation, in",
            "% absolute percentage points. \\paveNaiveSFMax  / \\paveNaiveSFMean",
            "% are for follower success rate (S_F);  \\paveNaiveJointMax /",
            "% \\paveNaiveJointMean are for the true joint success rate",
            "% (fraction of episodes with both leader and follower succeeding).",
            f"\\newcommand{{\\paveNaiveSFMax}}{{{deltas_sf.max():.1f}}}",
            f"\\newcommand{{\\paveNaiveSFMean}}{{{deltas_sf.mean():.1f}}}",
            f"\\newcommand{{\\paveNaiveSFMaxScenario}}{{\\scenario{{{rows_sum[argmax_sf]['scenario'].replace('_', r'\_')}}}}}",
            f"\\newcommand{{\\paveNaiveJointMax}}{{{deltas_joint.max():.1f}}}",
            f"\\newcommand{{\\paveNaiveJointMean}}{{{deltas_joint.mean():.1f}}}",
            f"\\newcommand{{\\paveNaiveJointMaxScenario}}{{\\scenario{{{rows_sum[argmax_joint]['scenario'].replace('_', r'\_')}}}}}",
        ]
        tex_path = tables_dir / "pave_vs_naive.tex"
        tex_path.write_text("\n".join(tex_lines) + "\n")
        print(f"  wrote {tex_path}")


    # ---------------------------------------------------------------------
    # 3. unnormalized_returns.tex — R_L / R_F / total mean ± std.
    # ---------------------------------------------------------------------
    return_rows: list[str] = []
    for scenario in scenarios:
        scen_df = df[df["scenario"] == scenario]
        cells = []
        for mode in MODES:
            mode_df = scen_df[scen_df["mode"] == mode]
            if mode_df.empty:
                cells.append("--")
                continue
            lr_m, lr_s = mean_std(mode_df["leader_return"].tolist())
            fr_m, fr_s = mean_std(mode_df["follower_return"].tolist())
            total = mode_df["leader_return"] + mode_df["follower_return"]
            tot_m, tot_s = mean_std(total.tolist())
            cells.append(
                f"{fmt_pm(lr_m, lr_s)} / {fmt_pm(fr_m, fr_s)} / {fmt_pm(tot_m, tot_s)}"
            )
        return_rows.append(
            row([f"\\scenario{{{scenario.replace('_', r'\_')}}}", *cells])
        )

    write_tabular(
        tables_dir / "unnormalized_returns.tex",
        column_spec="l" + "c" * len(MODES),
        header=row(["Scenario", *(MODE_LABELS[m] for m in MODES)]),
        rows=return_rows,
    )
    print(f"  wrote {tables_dir / 'unnormalized_returns.tex'}")

    # ---------------------------------------------------------------------
    # 4. Combined cross-scenario violin figure (seaborn).
    # ---------------------------------------------------------------------
    palette_all = {m: MODE_COLORS[m] for m in MODES}
    pdf_path = figures_dir / "cross_scenario_violins.pdf"
    png_path = figures_dir / "cross_scenario_violins.png"
    _plot_violins(df, scenarios, MODES, MODE_LABELS, palette_all,
                  pdf_path, png_path)
    print(f"  wrote {pdf_path}")
    print(f"  wrote {png_path}")

    # ---------------------------------------------------------------------
    # 5. Additional subset outputs (kept alongside the main tables above).
    # ---------------------------------------------------------------------
    #   5a. Naive + PAVE variants × warehouse_pickup_blocked only.
    _PICKUP_BLOCKED_MODES = ("unguided", "value-only", "value-and-ood")
    if "warehouse_pickup_blocked" in scenarios:
        rows_pb, _ = _build_success_rows(
            df, ["warehouse_pickup_blocked"], _PICKUP_BLOCKED_MODES,
        )
        write_tabular(
            tables_dir / "success_rates_pickup_blocked.tex",
            column_spec="ll" + "c" * len(_PICKUP_BLOCKED_MODES),
            header=row(["Scenario", "",
                        *(MODE_LABELS[m] for m in _PICKUP_BLOCKED_MODES)]),
            rows=rows_pb,
        )
        print(f"  wrote {tables_dir / 'success_rates_pickup_blocked.tex'}")

    #   5b. All scenarios × PAVE vs PAVE-w/o-π_L only.
    _PAVE_UNIFORM_MODES = ("value-and-ood", "uniform")
    rows_pu, _ = _build_success_rows(df, scenarios, _PAVE_UNIFORM_MODES)
    write_tabular(
        tables_dir / "success_rates_pave_vs_uniform.tex",
        column_spec="ll" + "c" * len(_PAVE_UNIFORM_MODES),
        header=row(["Scenario", "",
                    *(MODE_LABELS[m] for m in _PAVE_UNIFORM_MODES)]),
        rows=rows_pu,
    )
    print(f"  wrote {tables_dir / 'success_rates_pave_vs_uniform.tex'}")

    #   5c. All scenarios × PAVE vs finetune-baseline only.
    _PAVE_FT_MODES = ("value-and-ood", "finetune-baseline")
    rows_ft, _ = _build_success_rows(df, scenarios, _PAVE_FT_MODES)
    write_tabular(
        tables_dir / "success_rates_pave_vs_ft.tex",
        column_spec="ll" + "c" * len(_PAVE_FT_MODES),
        header=row(["Scenario", "",
                    *(MODE_LABELS[m] for m in _PAVE_FT_MODES)]),
        rows=rows_ft,
    )
    print(f"  wrote {tables_dir / 'success_rates_pave_vs_ft.tex'}")

    #   5d. Success-rate bars: (scenario × method) grid, 2 bars (S_L, S_F).
    #        Includes a bar sibling for each of the 3 subset tables above.
    bar_variants = [
        ("success_rates_bars", scenarios, MODES),
    ]
    if "warehouse_pickup_blocked" in scenarios:
        bar_variants.append((
            "success_rates_bars_pickup_blocked",
            ["warehouse_pickup_blocked"], _PICKUP_BLOCKED_MODES,
        ))
    bar_variants.append((
        "success_rates_bars_pave_vs_uniform", scenarios, _PAVE_UNIFORM_MODES,
    ))
    bar_variants.append((
        "success_rates_bars_pave_vs_ft", scenarios, _PAVE_FT_MODES,
    ))
    for stem, scen_subset, mode_subset in bar_variants:
        pdf_bars = figures_dir / f"{stem}.pdf"
        png_bars = figures_dir / f"{stem}.png"
        _plot_success_bars(df, scen_subset, mode_subset,
                           MODE_LABELS, MODE_COLORS, pdf_bars, png_bars)
        print(f"  wrote {pdf_bars}")
        print(f"  wrote {png_bars}")

    #   5e. Naive + PAVE violins only.
    _NAIVE_PAVE_MODES = ("unguided", "value-and-ood")
    palette_np = {m: MODE_COLORS[m] for m in _NAIVE_PAVE_MODES}
    pdf_np = figures_dir / "cross_scenario_violins_naive_pave.pdf"
    png_np = figures_dir / "cross_scenario_violins_naive_pave.png"
    _plot_violins(df, scenarios, _NAIVE_PAVE_MODES,
                  MODE_LABELS, palette_np, pdf_np, png_np)
    print(f"  wrote {pdf_np}")
    print(f"  wrote {png_np}")

    #   5f. Naive + PAVE violins, excluding obstacle_arena.
    #        Paired with 5g so the two can be shown side by side on a slide
    #        (locked absolute cell size — see ``_VIO_*`` constants).
    scenarios_no_oa = [s for s in scenarios if s != "obstacle_arena"]
    if scenarios_no_oa and len(scenarios_no_oa) < len(scenarios):
        pdf_no_oa = figures_dir / "cross_scenario_violins_naive_pave_no_oa.pdf"
        png_no_oa = figures_dir / "cross_scenario_violins_naive_pave_no_oa.png"
        _plot_violins(df, scenarios_no_oa, _NAIVE_PAVE_MODES,
                      MODE_LABELS, palette_np, pdf_no_oa, png_no_oa,
                      lock_cells=True)
        print(f"  wrote {pdf_no_oa}")
        print(f"  wrote {png_no_oa}")

    #   5g. Naive + PAVE violins, obstacle_arena only. Cell dimensions
    #        match 5f exactly (same ``lock_cells=True`` layout).
    if "obstacle_arena" in scenarios:
        pdf_oa = figures_dir / "obstacle_arena_violins_naive_pave.pdf"
        png_oa = figures_dir / "obstacle_arena_violins_naive_pave.png"
        _plot_violins(df, ["obstacle_arena"], _NAIVE_PAVE_MODES,
                      MODE_LABELS, palette_np, pdf_oa, png_oa,
                      lock_cells=True)
        print(f"  wrote {pdf_oa}")
        print(f"  wrote {png_oa}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
