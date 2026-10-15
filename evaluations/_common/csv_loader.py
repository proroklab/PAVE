"""CSV loaders for evaluation results.

One ``critic-sharing eval`` invocation writes one
``compare_<scenario>.csv`` (plus a few figures). Each row is one
(episode, mode) pair with the fields the eval framework chose
(``leader_return``, ``follower_return``, ``mode``, ``episode``, plus
scenario-specific success flags).

This module exposes three loaders with increasing scope:

  * :func:`load_compare_csv`   — one CSV file → DataFrame
  * :func:`load_run_dir`       — one results directory (all CSVs in it)
                                 → DataFrame with ``scenario`` column
  * :func:`gather_grid`        — recursively glob a grid of run dirs
                                 → DataFrame with extra grid-coord columns

All loaders are pure — they never write, never invoke subprocesses,
never look at anything outside the path you pass them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def load_compare_csv(path: Path) -> pd.DataFrame:
    """Read a single ``compare_<scenario>.csv`` written by ``eval``.

    Adds a ``scenario`` column derived from the filename suffix so
    multi-scenario merges work.
    """
    path = Path(path)
    df = pd.read_csv(path)
    if "scenario" not in df.columns:
        scenario = path.stem.replace("compare_", "", 1)
        df["scenario"] = scenario
    return df


def load_run_dir(run_dir: Path) -> pd.DataFrame:
    """Load every ``compare_*.csv`` inside a single eval-output directory.

    The directory layout matches what the CLI writes:
    ``<run_dir>/compare_<scenario>.csv`` (plus optional figures /
    episodes/ subdirs which are ignored).
    """
    run_dir = Path(run_dir)
    csvs = sorted(run_dir.glob("compare_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No compare_*.csv under {run_dir}")
    return pd.concat([load_compare_csv(p) for p in csvs], ignore_index=True)


def gather_grid(
    results_root: Path,
    *,
    scenarios: Optional[Iterable[str]] = None,
    seeds: Optional[Iterable[int]] = None,
    extra_label: str = "cell",
) -> pd.DataFrame:
    """Walk ``results_root`` looking for ``<cell_dir>/compare_<scenario>.csv``.

    A "cell" is whatever subdirectory the run.py orchestrator created
    for one cross-product slot (e.g.
    ``2026-05-19_120000_seed42/warehouse``). The directory name is
    preserved in the ``extra_label`` column.

    Optional filters narrow the union: ``scenarios=("warehouse",)``
    keeps only that scenario's CSVs; ``seeds=(0, 1, 2)`` requires a
    ``_seed<N>`` somewhere in the cell-dir name.
    """
    results_root = Path(results_root)
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(results_root.rglob("compare_*.csv")):
        scen = csv_path.stem.replace("compare_", "", 1)
        if scenarios is not None and scen not in set(scenarios):
            continue
        cell_label = csv_path.parent.relative_to(results_root).as_posix()
        if seeds is not None:
            seed_tags = {f"seed{s}" for s in seeds}
            if not any(tag in cell_label for tag in seed_tags):
                continue
        df = load_compare_csv(csv_path)
        df[extra_label] = cell_label
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


__all__ = ["gather_grid", "load_compare_csv", "load_run_dir"]
