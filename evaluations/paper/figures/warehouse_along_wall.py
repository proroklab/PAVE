"""V_F, OOD and trust-weighted V_F along the warehouse_pickup_blocked wall.

Single PDF, three stacked panels sharing the x-axis (object x along
``wall_y``):

* top:    V_F(x)
* middle: OOD(x)                (kNN density)
* bottom: (1 - OOD(x)) * V_F(x) (what the planner's trust-weighted term sees)

Gap regions are shaded. The curves depend only on the follower and its OOD
detector, not on an episode: the follower and ``follower_ood.pt`` of the
first selected cell (``warehouse_pickup_blocked_seed0``), conditioned on the
follower env's ``reset(seed=0)`` spawn and goal.

Inputs::

    evaluations/pool/cross_scenario/candidates/...      (via selection.json)
    evaluations/results/cross_scenario/ood/<cell>/follower_ood.pt

Output: ``evaluations/reports/figures/warehouse_along_wall.pdf``.

Run: ``python -m evaluations.paper.figures.warehouse_along_wall``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from evaluations._common.paths import CS_OOD, CS_POOL, CS_RESULTS, FIGURES
from evaluations.paper.cross_scenario.selection import cell_checkpoints

CELL = "warehouse_pickup_blocked_seed0"
N_WALL_X = 400  # smoother sweep than the eval default (200)
SPAWN_SEED = 0  # canonical conditioning for V_F's (spawn, goal)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="evaluations.paper.figures.warehouse_along_wall",
                                description=__doc__.splitlines()[0])
    p.add_argument("--cell", default=CELL)
    p.add_argument("--results-dir", type=Path, default=CS_RESULTS)
    p.add_argument("--pool-dir", type=Path, default=CS_POOL)
    p.add_argument("--figures-dir", type=Path, default=FIGURES)
    args = p.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from stable_baselines3 import PPO

    from critic_sharing.config.resolver import load_env_config
    from critic_sharing.scenarios.warehouse import (
        WarehouseFollowerEnv,
        build_warehouse_follower_value_fn,
    )
    from critic_sharing.scenarios.warehouse.common import gap_rects
    from critic_sharing.scenarios.warehouse.ood_query import build_warehouse_ood_metric

    follower_path = cell_checkpoints(args.cell, args.results_dir, args.pool_dir)["follower"]
    ood_path = args.results_dir / CS_OOD.name / args.cell / "follower_ood.pt"
    for path in (follower_path, ood_path):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    print(f"  follower: {follower_path}")
    print(f"  ood:      {ood_path}")

    scenario = args.cell.rsplit("_seed", 1)[0]
    follower_cfg = load_env_config(scenario, "follower")
    follower_env = WarehouseFollowerEnv(follower_cfg)
    follower_model = PPO.load(str(follower_path))

    follower_env.reset(seed=SPAWN_SEED)
    spawn_pos = follower_env._agent_pos.copy()
    spawn_theta = float(follower_env._agent_theta)
    goal = follower_env._goal.copy()
    follower_env.close()

    value_fn = build_warehouse_follower_value_fn(
        follower_model, follower_cfg, spawn_pos, spawn_theta, goal,
    )
    ood_fn = build_warehouse_ood_metric(str(ood_path))

    # Same x-grid conventions as cli/eval/warehouse.py.
    wall_x_grid = np.linspace(
        float(follower_cfg.world_min) + float(follower_cfg.object_radius),
        float(follower_cfg.world_max) - float(follower_cfg.object_radius),
        N_WALL_X,
        dtype=np.float32,
    )
    leader_obs_at_wall = np.zeros((N_WALL_X, 7), dtype=np.float32)
    leader_obs_at_wall[:, 4] = wall_x_grid
    leader_obs_at_wall[:, 5] = float(follower_cfg.wall_y)

    v_curve = np.asarray(value_fn(leader_obs_at_wall), dtype=np.float32).reshape(-1)
    ood_curve = np.asarray(ood_fn(leader_obs_at_wall), dtype=np.float32).reshape(-1)
    trust_v = (v_curve * np.clip(1.0 - ood_curve, 0.0, 1.0)).astype(np.float32)

    fig, axes = plt.subplots(3, 1, figsize=(6.0, 6.5), sharex=True,
                             gridspec_kw={"hspace": 0.18})
    ax_v, ax_ood, ax_trust = axes
    for ax in axes:
        for x1, x2 in gap_rects(follower_cfg):
            ax.axvspan(x1, x2, color="#43A047", alpha=0.18, zorder=1)
        ax.grid(True, alpha=0.3)

    ax_v.plot(wall_x_grid, v_curve, color="#1565C0", linewidth=1.8, zorder=3)
    ax_v.set_ylabel(r"$V_F(x)$")
    ax_ood.plot(wall_x_grid, ood_curve, color="#C62828", linewidth=1.8, zorder=3)
    ax_ood.set_ylim(-0.05, 1.05)
    ax_ood.set_ylabel("OOD$(x)$")
    ax_trust.plot(wall_x_grid, trust_v, color="#6A1B9A", linewidth=1.8, zorder=3)
    ax_trust.set_ylabel(r"$(1-\mathrm{OOD}(x))\cdot V_F(x)$")
    ax_trust.set_xlabel(r"$x$ along wall ($y = {:.2f}$)".format(float(follower_cfg.wall_y)))

    out = args.figures_dir / "warehouse_along_wall.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), bbox_inches="tight", pad_inches=0.05,
                metadata={"CreationDate": None})
    plt.close(fig)
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
