"""Alpha sweep results -> table, summary CSV and figure.

Success rates exactly as in the main table (:mod:`evaluations.paper.
cross_scenario.analyze`): per cell, the fraction of episodes in which the
leader / follower succeeded, and the joint rate as the per-episode
conjunction; then mean and std across the sweep's cells. Each alpha is one
column.

Inputs::

    evaluations/results/alpha_sweep/alpha_<a>/<cell>/compare_<scenario>.csv

Outputs::

    evaluations/reports/tables/alpha_sweep.tex (+ .html)   tabular only
    evaluations/reports/tables/alpha_sweep_summary.csv     scenario, alpha, metric, mean, std, n_cells
    evaluations/reports/figures/alpha_sweep.pdf (+ .png)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from evaluations._common.csv_loader import load_compare_csv  # noqa: E402
from evaluations._common.latex import row, write_tabular  # noqa: E402
from evaluations._common.paths import AS_RESULTS, FIGURES, TABLES  # noqa: E402
from evaluations._common.stats import mean_std  # noqa: E402
from evaluations.paper.cross_scenario.analyze import (  # noqa: E402
    FOLLOWER_SUCCESS_COL,
    LEADER_SUCCESS_COL,
    _build_success_rows,
)

METRICS = (("S_L", "Leader success"), ("S_F", "Follower success"),
           ("S_LF", "Joint success"))


def alpha_label(alpha: float) -> str:
    return f"alpha_{alpha:g}"


def load(results_dir: Path) -> pd.DataFrame:
    """All sweep CSVs, ``mode`` set to the alpha label, ``alpha`` added."""
    frames = []
    for adir in sorted(results_dir.glob("alpha_*"), key=lambda p: float(p.name[6:])):
        alpha = float(adir.name[6:])
        for cell_dir in sorted(p for p in adir.iterdir() if p.is_dir()):
            for csv in sorted(cell_dir.glob("compare_*.csv")):
                d = load_compare_csv(csv)
                d = d[d["mode"] == "value-and-ood"].copy()
                d["cell"] = cell_dir.name
                d["alpha"] = alpha
                d["mode"] = alpha_label(alpha)
                frames.append(d)
    if not frames:
        raise FileNotFoundError(f"no alpha_*/<cell>/compare_*.csv under {results_dir}")
    return pd.concat(frames, ignore_index=True)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (scenario, alpha, metric): mean and std across cells."""
    out = []
    for (scenario, alpha), g in df.groupby(["scenario", "alpha"], sort=True):
        sl = g[LEADER_SUCCESS_COL[scenario]].astype(float)
        sf = g[FOLLOWER_SUCCESS_COL[scenario]].astype(float)
        per_cell = {
            "S_L": sl.groupby(g["cell"]).mean(),
            "S_F": sf.groupby(g["cell"]).mean(),
            "S_LF": (sl * sf).groupby(g["cell"]).mean(),
        }
        for metric, values in per_cell.items():
            m, s = mean_std(values.tolist())
            out.append({"scenario": scenario, "alpha": alpha, "metric": metric,
                        "mean": m, "std": s, "n_cells": int(values.size)})
    return pd.DataFrame(out)


def plot(summ: pd.DataFrame, pdf: Path, png: Path) -> None:
    alphas = sorted(summ["alpha"].unique())
    x = range(len(alphas))
    scenarios = sorted(summ["scenario"].unique())
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 3.4),
                             sharey=True)
    for ax, (metric, title) in zip(axes, METRICS):
        for i, scenario in enumerate(scenarios):
            s = summ[(summ["scenario"] == scenario) & (summ["metric"] == metric)]
            s = s.set_index("alpha").loc[alphas]
            ax.errorbar(list(x), 100 * s["mean"], yerr=100 * s["std"], marker="o",
                        capsize=3, color=f"C{i}", label=scenario.replace("_", " "))
        ax.set_xticks(list(x), [f"{a:g}" for a in alphas])
        ax.set_xlabel(r"$\alpha$")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("success rate (%)")
    axes[0].set_ylim(0, 102)
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf, metadata={"CreationDate": None})
    fig.savefig(png, dpi=150)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="evaluations.paper.alpha_sweep.analyze",
                                description=__doc__.splitlines()[0])
    p.add_argument("--results-dir", type=Path, default=AS_RESULTS)
    p.add_argument("--tables-dir", type=Path, default=TABLES)
    p.add_argument("--figures-dir", type=Path, default=FIGURES)
    args = p.parse_args(argv)

    df = load(args.results_dir)
    alphas = sorted(df["alpha"].unique())
    scenarios = sorted(df["scenario"].unique())
    cells = df.groupby("scenario")["cell"].unique()
    for scenario in scenarios:
        print(f"  {scenario}: cells {sorted(cells[scenario])}")
    missing = [(s, a) for s in scenarios for a in alphas
               if df[(df["scenario"] == s) & (df["alpha"] == a)]["cell"].nunique()
               != len(cells[s])]
    if missing:
        raise SystemExit(f"incomplete sweep, (scenario, alpha) with missing cells: {missing}")

    modes = [alpha_label(a) for a in alphas]
    rows, _ = _build_success_rows(df, scenarios, modes)
    write_tabular(args.tables_dir / "alpha_sweep.tex",
                  column_spec="ll" + "c" * len(modes),
                  header=row(["Scenario", "", *(f"$\\alpha={a:g}$" for a in alphas)]),
                  rows=rows)
    summ = summary(df)
    summ.to_csv(args.tables_dir / "alpha_sweep_summary.csv", index=False,
                float_format="%.6f")
    plot(summ, args.figures_dir / "alpha_sweep.pdf", args.figures_dir / "alpha_sweep.png")
    print(f"  wrote {args.tables_dir / 'alpha_sweep.tex'}, alpha_sweep_summary.csv, "
          f"{args.figures_dir / 'alpha_sweep.pdf'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
