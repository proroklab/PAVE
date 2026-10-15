"""Environment-overview panels for the paper (no checkpoints needed).

Writes four PDFs into ``evaluations/reports/figures/``:

* ``env_mobile_arm_top.pdf``     - leader arena with mobile base, object
                                   and standoff ring
* ``env_mobile_arm_side.pdf``    - follower arm side view in default pose
                                   with an example object on the ground
* ``env_obstacle_arena_top.pdf`` - pusher arena with walls, obstacle, goal
                                   line and agent
* ``env_warehouse_top.pdf``      - warehouse top-down with world bounds,
                                   wall, gaps, shelves, follower at spawn,
                                   object in a gap and goal in the upper aisle

All env states come from ``env.reset(seed=0)``; nothing is hand-picked.
PDF only, no titles or labels (the LaTeX caption annotates).

Run: ``python -m evaluations.paper.figures.env_overviews``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from evaluations._common.paths import FIGURES

OUT_DIR = FIGURES


def _save_pdf(fig, path: Path) -> None:
    fig.savefig(str(path), bbox_inches="tight", pad_inches=0.05,
                metadata={"CreationDate": None})
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  wrote {path}")


# --- Mobile arm: top-down + side -----------------------------------------

def render_mobile_arm() -> None:
    import matplotlib.pyplot as plt

    from critic_sharing.config.resolver import load_env_config
    from critic_sharing.scenarios.mobile_arm import (
        MobileArmFollowerEnv,
        MobileArmLeaderEnv,
    )
    from critic_sharing.scenarios.mobile_arm.render import (
        draw_base_topdown,
        draw_desired_distance_ring,
        draw_follower_side,
        draw_leader_arena,
        draw_object_topdown,
    )

    leader_cfg = load_env_config("mobile_arm", "leader")
    follower_cfg = load_env_config("mobile_arm", "follower")

    # ---- Top-down: leader arena + base + object + standoff ring ----
    leader_env = MobileArmLeaderEnv(leader_cfg)
    leader_env.reset(seed=0)
    base_pos = leader_env._base_pos.copy()
    base_yaw = float(leader_env._base_yaw)
    obj_world = leader_env._object_pos.copy()
    leader_env.close()

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("")
    draw_leader_arena(ax, leader_cfg)
    draw_desired_distance_ring(
        ax, obj_world,
        float(leader_cfg.desired_distance),
        float(leader_cfg.gaussian_sigma),
    )
    draw_object_topdown(ax, obj_world, leader_cfg)
    draw_base_topdown(ax, base_pos, base_yaw, leader_cfg)
    _save_pdf(fig, OUT_DIR / "env_mobile_arm_top.pdf")

    # ---- Side view: arm in default pose + object on ground ----
    follower_env = MobileArmFollowerEnv(follower_cfg)
    follower_obs, _ = follower_env.reset(seed=0)
    theta = follower_env._theta.copy()  # 4-DOF default joint config
    obj_base = np.array(
        [follower_env._object_pos[0], follower_env._object_pos[1], 0.0],
        dtype=np.float32,
    )
    follower_env.close()

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.5))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("")
    draw_follower_side(ax, theta, obj_base, carry=False, config=follower_cfg)
    _save_pdf(fig, OUT_DIR / "env_mobile_arm_side.pdf")


# --- Obstacle arena: top-down --------------------------------------------

def render_obstacle_arena() -> None:
    """Layout:
        * goal = entire bottom edge (shaded band y ∈ [world_min, world_min + r])
        * wall at wall_y with single gap at gap_x
        * obstacle pinned at the ``gap-center`` preset (midpoint between gap
          and goal - the canonical evaluation pose)
        * leader (pusher) below the wall in the lower half (env-natural)
        * follower above the wall, lined up over the gap (canonical task setup)
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    from critic_sharing.config.resolver import load_env_config
    from critic_sharing.scenarios.obstacle_arena import PushObstacleEnv
    from critic_sharing.scenarios.obstacle_arena.common import draw_wall
    from critic_sharing.scenarios.obstacle_arena.initial_conditions import (
        fixed_obstacle_position,
    )

    push_cfg = load_env_config("obstacle_arena", "leader")
    nav_cfg = load_env_config("obstacle_arena", "follower")

    # Leader spawn: env.reset(seed=0) - the env's _sample_agent_pos forces
    # this to be below the wall.
    push_env = PushObstacleEnv(push_cfg)
    push_env.reset(seed=0)
    pusher_pos = push_env._agent_pos.copy()
    push_env.close()

    # Obstacle at gap-center preset (the evaluation-relevant pose).
    obstacle_pos = fixed_obstacle_position("gap-center", nav_cfg)

    # Follower lined up above the gap, with some clearance above the wall.
    gap_x = float(nav_cfg.gap_x)
    wall_y = float(nav_cfg.wall_y)
    follower_clearance = 2.0  # rough mid-upper-half placement
    follower_pos = np.array(
        [gap_x, wall_y + follower_clearance], dtype=np.float32,
    )

    # Goal region: entire bottom edge. Mirrors the band the animation uses.
    goal_top_y = nav_cfg.world_min + nav_cfg.agent_radius

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(nav_cfg.world_min, nav_cfg.world_max)
    ax.set_ylim(nav_cfg.world_min, nav_cfg.world_max)
    ax.set_title("")

    # Outer arena rectangle (the world boundary).
    ax.add_patch(
        mpatches.Rectangle(
            (nav_cfg.world_min, nav_cfg.world_min),
            nav_cfg.world_max - nav_cfg.world_min,
            nav_cfg.world_max - nav_cfg.world_min,
            fill=False, edgecolor="#37474F", linewidth=1.5, zorder=1,
        )
    )

    # Goal region: shaded band along the bottom edge + boundary line.
    ax.axhspan(nav_cfg.world_min, goal_top_y, color="#43A047", alpha=0.18, zorder=1)
    ax.axhline(y=goal_top_y, color="#43A047", linewidth=2, alpha=0.7, zorder=2)

    # Internal wall with the single gap.
    draw_wall(ax, nav_cfg)

    # Obstacle (grey disk) at gap-center.
    ax.add_patch(
        plt.Circle(
            obstacle_pos, float(push_cfg.obstacle_radius),
            facecolor="#757575", edgecolor="#212121", alpha=0.75, zorder=4,
        )
    )

    # Pusher (leader phase): orange disk in lower half.
    ax.add_patch(
        plt.Circle(
            pusher_pos, float(push_cfg.agent_radius),
            facecolor="#E65100", edgecolor="#212121", alpha=0.85, zorder=5,
        )
    )

    # Follower (nav phase): blue disk above the gap.
    ax.add_patch(
        plt.Circle(
            follower_pos, float(nav_cfg.agent_radius),
            facecolor="#1565C0", edgecolor="#212121", alpha=0.85, zorder=5,
        )
    )

    ax.grid(False)
    _save_pdf(fig, OUT_DIR / "env_obstacle_arena_top.pdf")


# --- Warehouse: top-down -------------------------------------------------

def render_warehouse() -> None:
    import matplotlib.pyplot as plt

    from critic_sharing.config.resolver import load_env_config
    from critic_sharing.scenarios.warehouse import (
        WarehouseFollowerEnv,
        WarehouseLeaderEnv,
    )
    from critic_sharing.scenarios.warehouse.render import (
        draw_arena,
        draw_goal,
        make_agent_patch,
        make_object_patch,
        update_agent_patch,
    )

    follower_cfg = load_env_config("warehouse", "follower")
    leader_cfg = load_env_config("warehouse", "leader")

    # Follower spawn comes from follower env reset.
    follower_env = WarehouseFollowerEnv(follower_cfg)
    follower_env.reset(seed=0)
    follower_pos = follower_env._agent_pos.copy()
    follower_theta = float(follower_env._agent_theta)
    follower_goal = follower_env._goal.copy()
    follower_env.close()

    # Leader spawn comes from leader env reset (separate env, same world).
    leader_env = WarehouseLeaderEnv(leader_cfg)
    leader_env.reset(seed=0)
    leader_pos = leader_env._agent_pos.copy()
    leader_theta = float(leader_env._agent_theta)
    leader_obj_pos = leader_env._object_pos.copy()
    leader_env.close()

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("")
    draw_arena(ax, follower_cfg, show_gaps=True, show_shelves=True, show_world_box=True)

    # Goal marker (follower's drop-off zone in upper aisle).
    draw_goal(ax, follower_cfg, follower_goal)

    # Object (in lower half - leader env's reset puts it below the wall;
    # this is a pre-pickup state).
    obj_patch = make_object_patch(ax, follower_cfg)
    obj_patch.center = (float(leader_obj_pos[0]), float(leader_obj_pos[1]))

    # Leader (orange) and follower (blue) agents at their respective
    # spawn poses. Visually conveys "two agents share the same world".
    sq_l, arr_l = make_agent_patch(ax, follower_cfg, color="#E65100", alpha=0.85)
    update_agent_patch(sq_l, arr_l, leader_pos, leader_theta, follower_cfg)
    sq_f, arr_f = make_agent_patch(ax, follower_cfg, color="#1565C0", alpha=0.85)
    update_agent_patch(sq_f, arr_f, follower_pos, follower_theta, follower_cfg)

    _save_pdf(fig, OUT_DIR / "env_warehouse_top.pdf")


# --- Entry point ---------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    global OUT_DIR
    p = argparse.ArgumentParser(prog="evaluations.paper.figures.env_overviews",
                                description=__doc__.splitlines()[0])
    p.add_argument("--figures-dir", type=Path, default=FIGURES)
    OUT_DIR = p.parse_args(argv).figures_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Rendering env overview panels (env.reset(seed=0) for each scenario)...")
    render_mobile_arm()
    render_obstacle_arena()
    render_warehouse()
    print(f"\nDone. Output in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
