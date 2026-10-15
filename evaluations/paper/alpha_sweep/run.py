"""Alpha sweep: PAVE at several alphas, first cells of every paper scenario.

For each scenario the first ``N_CELLS`` cells of ``selection.json`` (in
selection order) are evaluated with PAVE alone (``--modes value-and-ood``)
at every alpha in ``ALPHAS``, with exactly the setup of the main eval: the
cell's pool checkpoints and OOD detector, the cell's eval seed (so every
alpha sees the same episodes), 300 episodes and the scenario's fixed eval
condition. alpha=0 is the same command as the main table's Leader-only
column and alpha=4 the same as its PAVE column::

    evaluations/results/alpha_sweep/alpha_<a>/<cell>/compare_<scenario>.csv

Existing outputs are skipped; ``--force`` archives and recomputes them
(see :mod:`evaluations.paper.cross_scenario.run`). Needs the OOD detectors
in ``results/cross_scenario/ood/`` (``run.py --steps ood``).

    python -m evaluations.paper.alpha_sweep.run [--workers 8] [--no-figures]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Sequence

from evaluations._common.cli import critic_sharing_cli, rel
from evaluations._common.paths import AS_RESULTS, CS_POOL, CS_RESULTS
from evaluations._common.seeding import SeedManager
from evaluations.paper.cross_scenario.run import (
    _SCENARIO_EVAL_FIXED_OBSTACLE,
    SCENARIOS,
    cell_index,
    cell_scenario,
    environment_record,
    run_step,
    supersede,
)
from evaluations.paper.cross_scenario.selection import cell_checkpoints, load_selection

ALPHAS = (0.0, 1.0, 2.0, 4.0, 8.0)
N_CELLS = 3
MASTER_SEED = 42


def alpha_dir(alpha: float) -> str:
    return f"alpha_{alpha:g}"


def sweep_cells(results_dir: Path = CS_RESULTS, n_cells: int = N_CELLS,
                scenarios: Sequence[str] = SCENARIOS) -> list[str]:
    """The first ``n_cells`` selected cells of every scenario."""
    sel = load_selection(results_dir)["scenarios"]
    return [c["cell"] for s in scenarios for c in sel[s]["selected"][:n_cells]]


def eval_alpha(cell: str, alpha: float, *, out: Path, workers: int,
               n_episodes: int, no_figures: bool, dry_run: bool) -> int:
    scenario, idx = cell_scenario(cell), cell_index(cell)
    ckpt = cell_checkpoints(cell, CS_RESULTS, CS_POOL)
    ood = CS_RESULTS / "ood" / cell / "follower_ood.pt"
    if not ood.is_file() and not dry_run:
        raise FileNotFoundError(f"{ood} missing; run cross_scenario run.py --steps ood")
    seed = SeedManager(master=MASTER_SEED).cell_seed("cross_scenario", scenario, idx, "eval")
    cmd = [critic_sharing_cli(), "eval", "--scenario", scenario,
           "--leader-model", rel(ckpt["leader"]), "--follower-model", rel(ckpt["follower"]),
           "--vae", rel(ood), "--seed", str(seed), "--episodes", str(n_episodes),
           "--workers", str(workers), "--output-dir", rel(out),
           "--modes", "value-and-ood", "--alpha", f"{alpha:g}", "--no-trajectory"]
    fixed = _SCENARIO_EVAL_FIXED_OBSTACLE.get(scenario)
    if fixed is not None:
        cmd += ["--fixed-obstacle", fixed]
    if no_figures:
        cmd.append("--no-figures")
    return run_step(cmd, out / "eval.log", f"{cell} alpha={alpha:g}", dry_run)


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(prog="evaluations.paper.alpha_sweep.run",
                                description=__doc__.splitlines()[0])
    p.add_argument("--alphas", default=",".join(f"{a:g}" for a in ALPHAS))
    p.add_argument("--cells", default=None, help="Default: first N_CELLS per scenario.")
    p.add_argument("--n-episodes", type=int, default=300)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    cells = ([c.strip() for c in args.cells.split(",") if c.strip()]
             if args.cells else sweep_cells())
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    if not args.dry_run:
        (AS_RESULTS / "runs").mkdir(parents=True, exist_ok=True)
        (AS_RESULTS / "runs" / f"{stamp}.json").write_text(json.dumps(
            environment_record([sys.argv[0], *argv]), indent=1) + "\n")

    failures = []
    for cell in cells:
        for alpha in alphas:
            out = AS_RESULTS / alpha_dir(alpha) / cell
            if any(out.glob("compare_*.csv")):
                if not args.force:
                    print(f"  [{cell} alpha={alpha:g}] exists, skipped")
                    continue
                supersede([out], stamp, args.dry_run)
            if not args.dry_run:
                out.mkdir(parents=True, exist_ok=True)
            if eval_alpha(cell, alpha, out=out, workers=args.workers,
                          n_episodes=args.n_episodes, no_figures=args.no_figures,
                          dry_run=args.dry_run) != 0:
                failures.append(f"{cell} alpha={alpha:g}")
    if failures:
        print(f"\n{len(failures)} failed: {failures}")
        return 1
    print("\nDone. Report: python -m evaluations.paper.alpha_sweep.analyze")
    return 0


if __name__ == "__main__":
    sys.exit(main())
