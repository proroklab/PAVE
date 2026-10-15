"""Cross-scenario pipeline: checkpoints in the pool -> eval results.

Reproduction (the default) works on the cells ``selection.json`` selects
and only reads the pool::

    pool/cross_scenario/candidates/<scenario>/seed<i>/{leader,follower}.zip
    pool/cross_scenario/finetune/<cell>/leader_finetune.zip
        |
        |  ood               critic-sharing train ood   (follower rollouts -> kNN)
        v
    results/cross_scenario/ood/<cell>/follower_ood.pt
        |  eval              4 modes: unguided, value-only, value-and-ood, uniform
        |  eval_finetune     the cell's fine-tuned leader, unguided
        |  eval_leader_only  alpha=0: the planner optimises the leader objective only
        v
    results/cross_scenario/<step>/<cell>/compare_<scenario>.csv, eval.log

A step whose output already exists is skipped, so a run can start from any
intermediate. ``--force`` recomputes it: the previous output is first moved
to ``archive/cross_scenario/superseded/<timestamp>/`` (never deleted).

Training writes policies into a NEW pool (``--pool-dir``); it never
touches the published one. Retraining is not bit-reproducible; it gives a
sweep from the same distribution::

    train_follower / train_leader   -> <pool>/candidates/<scenario>/seed<i>/
    (then: python -m evaluations.paper.cross_scenario.selection --pool-dir <pool> ...)
    finetune                        -> <pool>/finetune/<cell>/

Every sub-process seed derives from the master seed via :class:`SeedManager`
as ``cell_seed("cross_scenario", scenario, <index>, <role>)``. Each
invocation writes an environment record (GPU, driver, CUDA, packages, git)
to ``<results>/runs/<timestamp>.json``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from evaluations._common.cli import critic_sharing_cli, rel
from evaluations._common.paths import CS_ARCHIVE, CS_POOL, CS_RESULTS, REPO
from evaluations._common.seeding import SeedManager
from evaluations.paper.cross_scenario.selection import (
    RULES,
    cell_checkpoints,
    load_selection,
)

SCENARIOS = tuple(RULES)  # the five paper scenarios

EVAL_STEPS = ("eval", "eval_finetune", "eval_leader_only")
REPRODUCE_STEPS = ("ood",) + EVAL_STEPS
TRAIN_STEPS = ("train_follower", "train_leader", "finetune")

# Per-scenario EVAL CONDITION that every method column must share.
# obstacle_arena's canonical sweep pins the obstacle at the gap -- the hard
# case the paper reports, which makes the follower route through the gap.
# Applied automatically so no eval can silently land on another
# distribution. Scenarios absent here use their env default.
_SCENARIO_EVAL_FIXED_OBSTACLE: dict[str, str] = {
    "obstacle_arena": "gap-center",
}


def cell_index(cell: str) -> str:
    return cell.rsplit("_seed", 1)[1]


def cell_scenario(cell: str) -> str:
    return cell.rsplit("_seed", 1)[0]


# ---------------------------------------------------------------------------
# Running sub-processes
# ---------------------------------------------------------------------------

def run_step(cmd: list[str], log_path: Path, label: str, dry_run: bool) -> int:
    """Run one sub-process; stdout+stderr go to ``log_path``."""
    print(f"  [{label}] " + " ".join(cmd))
    if dry_run:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as logf:
        rc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                            cwd=REPO).returncode
    if rc != 0:
        print(f"    FAILED ({label}) rc={rc}; see {log_path}")
    return int(rc)


def supersede(paths: list[Path], stamp: str, dry_run: bool) -> None:
    """Move existing outputs to the archive before they are recomputed."""
    for p in paths:
        if not p.exists():
            continue
        dst = CS_ARCHIVE / "superseded" / stamp / p.relative_to(REPO / "evaluations")
        print(f"  superseded: {p} -> {dst}")
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(dst))


# ---------------------------------------------------------------------------
# Reproduction steps (pool -> results)
# ---------------------------------------------------------------------------

def step_outputs(step: str, cell: str, results_dir: Path) -> Path:
    """The directory a reproduction step writes for ``cell``."""
    return results_dir / step / cell


def reproduce_cell(cell: str, steps: Sequence[str], *, pool_dir: Path,
                   results_dir: Path, seed_mgr: SeedManager, n_episodes: int,
                   workers: int, force: bool, stamp: str, dry_run: bool,
                   no_figures: bool = False) -> int:
    scenario, idx = cell_scenario(cell), cell_index(cell)
    ckpt = cell_checkpoints(cell, results_dir, pool_dir)
    leader, follower = ckpt["leader"], ckpt["follower"]
    leader_ft = pool_dir / "finetune" / cell / "leader_finetune.zip"
    ood = results_dir / "ood" / cell / "follower_ood.pt"
    seed = lambda role: str(seed_mgr.cell_seed("cross_scenario", scenario, idx, role))  # noqa: E731
    fixed = _SCENARIO_EVAL_FIXED_OBSTACLE.get(scenario)
    cli = critic_sharing_cli()
    print(f"\n==== {cell} ====")

    for step in steps:
        out = step_outputs(step, cell, results_dir)
        done = (ood.is_file() if step == "ood"
                else any(out.glob("compare_*.csv")))
        if done and not force:
            print(f"  [{step}] exists, skipped: {out}")
            continue
        if done:
            supersede([out], stamp, dry_run)
        if not dry_run:
            out.mkdir(parents=True, exist_ok=True)
        if step == "ood":
            cmd = [cli, "train", "ood", "--scenario", scenario,
                   "--follower-model", rel(follower), "--seed", seed("ood"),
                   "--output", rel(ood)]
            log = out / "train_ood.log"
        else:
            cmd = [cli, "eval", "--scenario", scenario,
                   "--leader-model", rel(leader_ft if step == "eval_finetune" else leader),
                   "--follower-model", rel(follower), "--vae", rel(ood),
                   "--seed", seed("eval"), "--episodes", str(n_episodes),
                   "--workers", str(workers), "--output-dir", rel(out)]
            if step == "eval":
                cmd += ["--modes", "unguided,value-only,value-and-ood,uniform",
                        "--no-trajectory"]
            elif step == "eval_finetune":
                cmd += ["--modes", "unguided", "--no-figures"]
            else:  # eval_leader_only
                cmd += ["--modes", "value-and-ood", "--alpha", "0",
                        "--no-trajectory"]
            if fixed is not None:
                cmd += ["--fixed-obstacle", fixed]
            if no_figures and "--no-figures" not in cmd:
                cmd.append("--no-figures")
            log = out / "eval.log"
        if run_step(cmd, log, step, dry_run) != 0:
            return 1
    return 0


# ---------------------------------------------------------------------------
# Training steps (-> a new pool)
# ---------------------------------------------------------------------------

def train_cell(scenario: str, idx: int, steps: Sequence[str], *, pool_dir: Path,
               results_dir: Path, seed_mgr: SeedManager, total_timesteps: int | None,
               finetune_alpha: float, finetune_timesteps: int, dry_run: bool) -> int:
    s_idx = str(idx)
    seed = lambda role: str(seed_mgr.cell_seed("cross_scenario", scenario, s_idx, role))  # noqa: E731
    cand = pool_dir / "candidates" / scenario / f"seed{idx}"
    cli = critic_sharing_cli()
    print(f"\n==== train {scenario} seed{idx} ====")
    for role in ("follower", "leader"):
        if f"train_{role}" not in steps:
            continue
        out = cand / f"{role}.zip"
        if out.is_file():
            print(f"  [train_{role}] exists, skipped: {out}")
            continue
        cmd = [cli, "train", role, "--scenario", scenario, "--seed", seed(role),
               "--save-path", rel(out)]
        if total_timesteps is not None:
            cmd += ["--total-timesteps", str(total_timesteps)]
        if run_step(cmd, cand / f"train_{role}.log", f"train_{role}", dry_run) != 0:
            return 1
    if "finetune" in steps:
        cell = f"{scenario}_seed{idx}"
        out = pool_dir / "finetune" / cell / "leader_finetune.zip"
        if out.is_file():
            print(f"  [finetune] exists, skipped: {out}")
            return 0
        ckpt = cell_checkpoints(cell, results_dir, pool_dir)
        cmd = [cli, "finetune-leader", "--scenario", scenario,
               "--leader-model", rel(ckpt["leader"]),
               "--follower-model", rel(ckpt["follower"]),
               "--alpha", str(finetune_alpha),
               "--total-timesteps", str(finetune_timesteps),
               "--seed", seed("finetune"), "--save-path", rel(out),
               "--wandb-name", f"finetune-{cell}"]
        if run_step(cmd, out.parent / "train_finetune.log", "finetune", dry_run) != 0:
            return 1
    return 0


# ---------------------------------------------------------------------------
# Environment record
# ---------------------------------------------------------------------------

def environment_record(argv: Sequence[str]) -> dict:
    def sh(cmd: list[str]) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True,
                                  cwd=REPO).stdout.strip()
        except OSError:
            return ""
    rec: dict = {
        "argv": list(argv),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": sh(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(sh(["git", "status", "--porcelain", "--untracked-files=no"])),
        "nvidia_smi": sh(["nvidia-smi", "--query-gpu=name,driver_version",
                          "--format=csv,noheader"]),
    }
    try:
        import torch
        rec["torch"] = torch.__version__
        rec["cuda"] = torch.version.cuda
        rec["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        pass
    from importlib.metadata import distributions
    rec["packages"] = sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions())
    return rec


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="evaluations.paper.cross_scenario.run",
                                description=__doc__.splitlines()[0])
    p.add_argument("--steps", default=",".join(REPRODUCE_STEPS),
                   help=f"Comma-separated, run in this order: "
                        f"{', '.join(TRAIN_STEPS + REPRODUCE_STEPS)}.")
    p.add_argument("--pool-dir", type=Path, default=CS_POOL)
    p.add_argument("--results-dir", type=Path, default=CS_RESULTS)
    p.add_argument("--scenarios", default=",".join(SCENARIOS))
    p.add_argument("--cells", default=None,
                   help="Comma-separated cell names; default: every selected cell.")
    p.add_argument("--seed-indices", default="0,1,2,3,4,5",
                   help="Training only: indices to train per scenario.")
    p.add_argument("--seed", type=int, default=42, help="Master seed.")
    p.add_argument("--n-episodes", type=int, default=300)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--force", action="store_true",
                   help="Recompute existing outputs (old ones are archived).")
    p.add_argument("--total-timesteps", type=int, default=None,
                   help="Training only: override timesteps (smoke tests).")
    p.add_argument("--finetune-alpha", type=float, default=1.0)
    p.add_argument("--finetune-timesteps", type=int, default=1_500_000)
    p.add_argument("--no-figures", action="store_true",
                   help="Skip the evals' diagnostic plots (CSVs are unaffected).")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    steps = [s for s in TRAIN_STEPS + REPRODUCE_STEPS
             if s in {x.strip() for x in args.steps.split(",") if x.strip()}]
    unknown = {x.strip() for x in args.steps.split(",") if x.strip()} - set(steps)
    if unknown:
        raise SystemExit(f"unknown step(s) {sorted(unknown)}")
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    train = [s for s in steps if s in TRAIN_STEPS]
    if train and args.pool_dir.resolve() == CS_POOL.resolve():
        raise SystemExit("Training writes policies; it never touches the published "
                         f"pool {CS_POOL}. Pass --pool-dir <new dir>.")
    seed_mgr = SeedManager(master=args.seed)
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    if not args.dry_run:
        runs = args.results_dir / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        (runs / f"{stamp}.json").write_text(json.dumps(
            environment_record([sys.argv[0], *argv]), indent=1) + "\n")

    failures: list[str] = []
    if any(s in ("train_follower", "train_leader") for s in steps):
        for scenario in scenarios:
            for idx in (int(i) for i in args.seed_indices.split(",") if i.strip()):
                if train_cell(scenario, idx, steps, pool_dir=args.pool_dir,
                              results_dir=args.results_dir, seed_mgr=seed_mgr,
                              total_timesteps=args.total_timesteps,
                              finetune_alpha=args.finetune_alpha,
                              finetune_timesteps=args.finetune_timesteps,
                              dry_run=args.dry_run) != 0:
                    failures.append(f"{scenario}_seed{idx}")

    selection = None
    def cells() -> list[str]:
        nonlocal selection
        if args.cells:
            return [c.strip() for c in args.cells.split(",") if c.strip()]
        selection = selection or load_selection(args.results_dir)
        return [c["cell"] for s in scenarios
                for c in selection["scenarios"][s]["selected"]]

    if "finetune" in steps:
        for cell in cells():
            if train_cell(cell_scenario(cell), int(cell_index(cell)), ["finetune"],
                          pool_dir=args.pool_dir, results_dir=args.results_dir,
                          seed_mgr=seed_mgr, total_timesteps=None,
                          finetune_alpha=args.finetune_alpha,
                          finetune_timesteps=args.finetune_timesteps,
                          dry_run=args.dry_run) != 0:
                failures.append(cell)

    repro = [s for s in steps if s in REPRODUCE_STEPS]
    if repro:
        for cell in cells():
            if reproduce_cell(cell, repro, pool_dir=args.pool_dir,
                              results_dir=args.results_dir, seed_mgr=seed_mgr,
                              n_episodes=args.n_episodes, workers=args.workers,
                              force=args.force, stamp=stamp, dry_run=args.dry_run,
                              no_figures=args.no_figures) != 0:
                failures.append(cell)

    if failures:
        print(f"\n{len(failures)} failed: {failures}")
        return 1
    print("\nDone. Reports: python -m evaluations.paper.cross_scenario.analyze")
    return 0


if __name__ == "__main__":
    sys.exit(main())
