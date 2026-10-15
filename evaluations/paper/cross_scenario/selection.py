"""Which cells enter the paper tables: all trained policies -> competent -> selected.

Scores every candidate policy in ``evaluations/pool/cross_scenario/candidates/`` with one protocol,
keeps the competent ones and selects the first ``N_SELECT`` in training-index
order. The result is written to ``evaluations/results/cross_scenario/selection.json``, the single
record of which cells enter the paper tables.

Each candidate is scored with ``critic-sharing competence``: the policy alone,
deterministic, in its training env, ``N_EPISODES`` episodes with reset seeds
``train_seed + 1_000_000 + k`` (``train_seed`` = the seed ``run.py`` assigns
that (scenario, index, role) under master ``MASTER_SEED``); the mobile_arm
follower only on spawns within the arm's reach. The protocol is defined once,
in ``critic_sharing.training.competence``, and shared with the warehouse
trainer's convergence gate. Competent means success rate >= ``THRESHOLD``.

Selection rules:

  * ``by_index``: candidate ``i`` is competent if both its leader and its
    follower are; the first ``N_SELECT`` competent indices become cells
    ``<scenario>_seed<i>``.
  * ``shared_follower_pool``: the pickup variants train their follower in
    the same env, so their followers form one pool (ordered blocked seeds,
    then unseen seeds). Cell ``k`` pairs the ``k``-th competent leader of the
    scenario with the ``k``-th competent follower of the shared pool.

Readers (``analyze.py`` and the reports) call :func:`apply_selection` or
:func:`selected_cells`; they need only ``selection.json``, no checkpoints. A
missing record, a scenario without an entry or a selected cell without
results raises instead of silently letting other seeds into a table.

Usage::

    python -m evaluations.paper.cross_scenario.selection            # score + write
    python -m evaluations.paper.cross_scenario.selection --check    # re-score + verify
    python -m evaluations.paper.cross_scenario.selection --summary  # print the record
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import pandas as pd

from evaluations._common.cli import rel, run_cli
from evaluations._common.paths import CS_POOL, CS_RESULTS

THRESHOLD = 0.70
N_SELECT = 6
N_EPISODES = 100
MASTER_SEED = 42

PICKUP_POOL = ("warehouse_pickup_blocked", "warehouse_pickup_unseen")
RULES: dict[str, dict] = {
    "warehouse": {"rule": "by_index"},
    "warehouse_pickup_blocked": {"rule": "shared_follower_pool",
                                 "follower_pool": PICKUP_POOL},
    "warehouse_pickup_unseen": {"rule": "shared_follower_pool",
                                "follower_pool": PICKUP_POOL},
    "mobile_arm": {"rule": "by_index"},
    "obstacle_arena": {"rule": "by_index"},
}
ROLES = ("leader", "follower")


# ---------------------------------------------------------------------------
# Reading the record (no checkpoints needed)
# ---------------------------------------------------------------------------



def load_selection(results_dir: Path) -> dict:
    path = results_dir / "selection.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m evaluations.paper.cross_scenario.selection` "
            "to select the cells, or pass --allow-missing-selection to analyse "
            "every cell ungated (never for paper tables)."
        )
    return json.loads(path.read_text())


def selected_cells(results_dir: Path, scenario: str) -> list[str]:
    """Selected cell names (``<scenario>_seed<i>``) for ``scenario``."""
    scenarios = load_selection(results_dir)["scenarios"]
    if scenario not in scenarios:
        raise KeyError(f"selection.json has no entry for {scenario!r}")
    return [c["cell"] for c in scenarios[scenario]["selected"]]


def cell_checkpoints(cell: str, results_dir: Path = CS_RESULTS,
                     pool_dir: Path = CS_POOL) -> dict[str, Path]:
    """Pool checkpoints (``leader``, ``follower``) a selected cell uses."""
    scenario = cell.rsplit("_seed", 1)[0]
    for c in load_selection(results_dir)["scenarios"][scenario]["selected"]:
        if c["cell"] == cell:
            return {"leader": pool_dir / c["leader"],
                    "follower": pool_dir / c["follower"]}
    raise KeyError(f"{cell} is not a selected cell")


def apply_selection(
    df: pd.DataFrame, results_dir: Path, *, allow_missing: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """Keep only the selected cells of every scenario in ``df``.

    ``df`` needs ``scenario`` and ``cell`` columns.
    """
    if allow_missing and not (results_dir / "selection.json").is_file():
        print(f"  WARNING: no selection.json under {results_dir}; "
              "all cells kept (ungated)")
        return df
    for scenario in sorted(df["scenario"].unique()):
        keep = set(selected_cells(results_dir, scenario))
        present = set(df.loc[df["scenario"] == scenario, "cell"])
        missing = sorted(keep - present)
        if missing:
            raise FileNotFoundError(
                f"{scenario}: selected cells without results: {missing}")
        df = df[(df["scenario"] != scenario) | df["cell"].isin(keep)]
        if verbose and present != keep:
            print(f"  {scenario}: kept {len(keep)} of {len(present)} cells "
                  "(selection.json)")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_seed(scenario: str, index: int, role: str) -> int:
    """The seed run.py trains (scenario, index, role) with."""
    from critic_sharing.utils.seeding import SeedManager
    return SeedManager(master=MASTER_SEED).cell_seed(
        "cross_scenario", scenario, str(index), role)


def score_policy(job: tuple[str, int, str, str]) -> float:
    """Standalone success rate of one checkpoint: ``critic-sharing competence``."""
    scenario, index, role, zip_path = job
    proc = run_cli(["competence", "--scenario", scenario, "--role", role,
                    "--model", rel(Path(zip_path)),
                    "--train-seed", str(train_seed(scenario, index, role)),
                    "--episodes", str(N_EPISODES), "--json"],
                   capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"competence failed for {zip_path}:\n{proc.stderr[-2000:]}")
    return float(json.loads(proc.stdout.strip().splitlines()[-1])["success"])


def pool_entries(pool_dir: Path) -> list[tuple[str, int, str, Path]]:
    """Every (scenario, index, role, checkpoint) under ``<pool_dir>/candidates``."""
    pool = pool_dir / "candidates"
    if not pool.is_dir():
        raise FileNotFoundError(
            f"{pool} not found: the gate needs every trained candidate "
            "(rejected ones included), see evaluations/paper/cross_scenario/run.py.")
    out = []
    for scenario in sorted({s for s in RULES} | set(PICKUP_POOL)):
        for seed_dir in sorted((pool / scenario).glob("seed*"),
                               key=lambda p: int(p.name[4:])):
            for role in ROLES:
                z = seed_dir / f"{role}.zip"
                if z.is_file():
                    out.append((scenario, int(seed_dir.name[4:]), role, z))
    return out


def score_pool(pool_dir: Path, workers: int) -> dict:
    entries = pool_entries(pool_dir)
    jobs = [(s, i, r, str(z)) for s, i, r, z in entries]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        rates = list(ex.map(score_policy, jobs))
    scores: dict = {}
    for (s, i, r, z), rate in zip(entries, rates):
        scores.setdefault(s, {}).setdefault(i, {})[r] = {
            "path": str(z.relative_to(pool_dir)),
            "sha256": _sha256(z),
            "success": rate,
            "competent": rate >= THRESHOLD,
        }
    return scores


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def _competent(entry: dict | None) -> bool:
    return bool(entry and entry["competent"])


def select(scores: dict) -> dict:
    record: dict = {
        "protocol": {
            "threshold": THRESHOLD,
            "n_select": N_SELECT,
            "n_episodes": N_EPISODES,
            "master_seed": MASTER_SEED,
            "env": "the policy's own training env (env config + training "
                   "env_overrides), policy acting alone, deterministic",
            "reset_seeds": "train_seed + 1_000_000 + k, train_seed = "
                           "SeedManager(master).cell_seed('cross_scenario', "
                           "scenario, index, role)",
            "success": "SUCCESS_INFO_KEY fires at any step of the episode",
            "restrictions": {
                "mobile_arm.follower": "object spawns limited to the arm's "
                                       "reach, r <= L1 + L2a + L2b",
            },
            "selection": "competent = success >= threshold (both roles for "
                         "by_index); then the first n_select in index order",
        },
        "follower_pools": {},
        "scenarios": {},
    }

    pool_name = "warehouse_pickup"
    members = [(s, i) for s in PICKUP_POOL for i in sorted(scores.get(s, {}))
               if "follower" in scores[s][i]]
    pool_competent = [(s, i) for s, i in members
                      if _competent(scores[s][i]["follower"])]
    record["follower_pools"][pool_name] = {
        "order": "blocked seeds, then unseen seeds, each in index order",
        "candidates": [{"id": f"{s}_seed{i}", **scores[s][i]["follower"]}
                       for s, i in members],
        "competent": [f"{s}_seed{i}" for s, i in pool_competent],
        "selected": [f"{s}_seed{i}" for s, i in pool_competent[:N_SELECT]],
    }

    for scenario, rule in RULES.items():
        idx = sorted(scores.get(scenario, {}))
        cand = [{"index": i,
                 "leader": scores[scenario][i].get("leader"),
                 "follower": scores[scenario][i].get("follower")}
                for i in idx]
        if rule["rule"] == "by_index":
            for c in cand:
                c["competent"] = _competent(c["leader"]) and _competent(c["follower"])
            comp = [c for c in cand if c["competent"]][:N_SELECT]
            cells = [{"cell": f"{scenario}_seed{c['index']}",
                      "leader": c["leader"]["path"],
                      "leader_sha256": c["leader"]["sha256"],
                      "follower": c["follower"]["path"],
                      "follower_sha256": c["follower"]["sha256"]}
                     for c in comp]
        else:
            for c in cand:
                c.pop("follower", None)  # scored in the shared pool
                c["competent"] = _competent(c["leader"])
            leaders = [c for c in cand if c["competent"]][:N_SELECT]
            followers = pool_competent[:N_SELECT]
            cells = []
            for c, (fs, fi) in zip(leaders, followers):
                f = scores[fs][fi]["follower"]
                cells.append({"cell": f"{scenario}_seed{c['index']}",
                              "leader": c["leader"]["path"],
                              "leader_sha256": c["leader"]["sha256"],
                              "follower": f["path"],
                              "follower_sha256": f["sha256"]})
        if len(cells) < N_SELECT:
            raise RuntimeError(
                f"{scenario}: only {len(cells)} competent cells, need {N_SELECT}")
        entry = {"rule": rule["rule"], "candidates": cand,
                 "competent": [c["index"] for c in cand if c["competent"]],
                 "selected": cells}
        if rule["rule"] == "shared_follower_pool":
            entry["follower_pool"] = pool_name
        record["scenarios"][scenario] = entry
    return record


# ---------------------------------------------------------------------------
# Check / summary
# ---------------------------------------------------------------------------

def check(results_dir: Path, fresh: dict) -> list[str]:
    """Differences between the recorded selection and a fresh scoring run.

    The record holds every candidate's sha256 and score, so this catches a
    changed checkpoint as well as a changed score or selection."""
    path = results_dir / "selection.json"
    if not path.is_file():
        return [f"{path} missing"]
    recorded = json.loads(path.read_text())
    if recorded == fresh:
        return []
    problems = ["selection.json differs from a fresh scoring run"]
    for scenario in fresh["scenarios"]:
        old = recorded["scenarios"].get(scenario)
        if old != fresh["scenarios"][scenario]:
            problems.append(f"{scenario}: entry differs")
    return problems


def summary(record: dict) -> None:
    def fmt(e):
        return "-" if e is None else f"{e['success']:.2f}"
    print(f"threshold {record['protocol']['threshold']}, "
          f"first {record['protocol']['n_select']} competent\n")
    for name, pool in record["follower_pools"].items():
        print(f"follower pool {name}: {len(pool['candidates'])} trained -> "
              f"{len(pool['competent'])} competent -> {len(pool['selected'])} selected")
        for c in pool["candidates"]:
            tag = "selected" if c["id"] in pool["selected"] else (
                "competent" if c["competent"] else "rejected")
            print(f"    {c['id']:34s} {c['success']:.2f}  {tag}")
    for scenario, e in record["scenarios"].items():
        sel = {c["cell"] for c in e["selected"]}
        print(f"\n{scenario} ({e['rule']}): {len(e['candidates'])} trained -> "
              f"{len(e['competent'])} competent -> {len(sel)} selected")
        for c in e["candidates"]:
            cell = f"{scenario}_seed{c['index']}"
            tag = "selected" if cell in sel else (
                "competent" if c["competent"] else
                "untrained leader (not needed)" if c["leader"] is None else
                "rejected")
            f = fmt(c["follower"]) if "follower" in c else "pool"
            print(f"    seed{c['index']:<3d} leader {fmt(c['leader']):>4s}  "
                  f"follower {f:>4s}  {tag}")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="evaluations.paper.cross_scenario.selection",
                                description=__doc__.splitlines()[0])
    p.add_argument("--pool-dir", type=Path, default=CS_POOL)
    p.add_argument("--results-dir", type=Path, default=CS_RESULTS)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true",
                      help="Re-score the pool and verify selection.json.")
    mode.add_argument("--summary", action="store_true",
                      help="Print selection.json; no scoring.")
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args(argv)
    out = args.results_dir / "selection.json"

    if args.summary:
        summary(json.loads(out.read_text()))
        return 0
    record = select(score_pool(args.pool_dir, args.workers))
    record = json.loads(json.dumps(record))  # int keys -> canonical JSON form
    if args.check:
        problems = check(args.results_dir, record)
        for msg in problems:
            print(f"FAIL {msg}", file=sys.stderr)
        print("selection check: " + ("FAILED" if problems else "OK"))
        return 1 if problems else 0
    out.write_text(json.dumps(record, indent=1) + "\n")
    summary(record)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
