"""Initial condition generation for the obstacle-arena scenario.

Two modes of operation:

1. **Training reset** — collision-free random positions for a single env
   (leader below wall, follower above wall, no agent-obstacle overlap).
2. **Eval conditions** — pre-generated consistent initial conditions for
   multi-mode evaluation (same obstacle/leader/follower positions across
   all guidance modes for fair comparison).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from critic_sharing.config.schema import NavEnvConfig, PushEnvConfig
from critic_sharing.scenarios.obstacle_arena.common import (
    agent_bounds,
    circle_collides_wall,
    nav_goal_position,
    push_obstacle_bounds,
)


@dataclass(frozen=True)
class EpisodeConditions:
    """Initial positions for one leader-follower episode."""

    obstacle_pos: np.ndarray   # (2,)
    leader_pos: np.ndarray     # (2,)
    follower_pos: np.ndarray   # (2,)


# ---------------------------------------------------------------------------
# Training reset helpers
# ---------------------------------------------------------------------------

def sample_collision_free_pos(
    rng: np.random.Generator,
    x_low: float,
    x_high: float,
    y_low: float,
    y_high: float,
    avoid_pos: Optional[np.ndarray] = None,
    min_dist: float = 0.0,
    wall_y: float = 0.0,
    wall_radius: float = 0.0,
    config=None,
    max_attempts: int = 1000,
) -> np.ndarray:
    """Sample a 2D position avoiding overlap with ``avoid_pos`` and the wall."""
    for _ in range(max_attempts):
        x = float(rng.uniform(x_low, x_high))
        y = float(rng.uniform(y_low, y_high))
        pos = np.array([x, y], dtype=np.float32)
        # Check obstacle overlap
        if avoid_pos is not None and np.linalg.norm(pos - avoid_pos) < min_dist:
            continue
        # Check wall overlap
        if config is not None and circle_collides_wall(pos, wall_radius, config):
            continue
        return pos
    # Fallback: return last sampled position
    return np.array([float(rng.uniform(x_low, x_high)),
                     float(rng.uniform(y_low, y_high))], dtype=np.float32)


# ---------------------------------------------------------------------------
# Eval condition generation
# ---------------------------------------------------------------------------

def sample_obstacle_near_goal(
    rng: np.random.Generator,
    goal: np.ndarray,
    max_distance: float,
    obstacle_radius: float,
    world_min: float,
    world_max: float,
    wall_y: float = 0.0,
    max_attempts: int = 1000,
) -> np.ndarray:
    """Sample obstacle position within ``max_distance`` of the goal.

    Position is clamped to stay inside the arena and below the wall.
    """
    low = world_min + obstacle_radius + 1e-3
    high = world_max - obstacle_radius - 1e-3
    y_max = wall_y - obstacle_radius - 1e-3
    for _ in range(max_attempts):
        angle = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(0, max_distance)
        pos = goal + np.array([r * np.cos(angle), r * np.sin(angle)], dtype=np.float32)
        pos[0] = float(np.clip(pos[0], low, high))
        pos[1] = float(np.clip(pos[1], low, min(high, y_max)))
        if np.linalg.norm(pos - goal) <= max_distance:
            return pos
    return np.clip(goal.copy(), low, high).astype(np.float32)


def fixed_obstacle_position(
    name: str,
    nav_cfg: NavEnvConfig,
) -> np.ndarray:
    """Return obstacle position for a named fixed-obstacle preset.

    Supported presets:
      - ``"gap-center"``: midpoint between gap centre and goal.
      - ``"x,y"``: explicit coordinates.
    """
    goal = nav_goal_position(nav_cfg)
    if name == "gap-center":
        gap_pos = np.array([nav_cfg.gap_x, nav_cfg.wall_y], dtype=np.float32)
        return ((gap_pos + goal) / 2.0).astype(np.float32)
    # Parse "x,y" coordinates
    parts = name.split(",")
    if len(parts) == 2:
        return np.array([float(parts[0]), float(parts[1])], dtype=np.float32)
    raise ValueError("Unknown fixed_obstacle preset: {name!r}".format(name=name))


def sample_eval_conditions(
    push_cfg: PushEnvConfig,
    nav_cfg: NavEnvConfig,
    n_episodes: int,
    rng: np.random.Generator,
    obstacle_distance: Optional[float] = None,
    fixed_obstacle: Optional[str] = None,
) -> list[EpisodeConditions]:
    """Pre-generate consistent initial conditions for evaluation.

    Returns one ``EpisodeConditions`` per episode. The same conditions
    should be reused across all evaluation modes for fair comparison.

    Parameters
    ----------
    obstacle_distance
        If set, obstacle is sampled within this distance of the goal.
    fixed_obstacle
        If set, all episodes use this obstacle position (preset name or "x,y").
    """
    goal = nav_goal_position(nav_cfg)
    min_dist_leader = push_cfg.obstacle_radius + push_cfg.agent_radius
    min_dist_follower = nav_cfg.obstacle_radius + nav_cfg.agent_radius

    l_low, l_high = agent_bounds(push_cfg)
    f_low, f_high = agent_bounds(nav_cfg)
    ly_max = min(l_high, float(push_cfg.wall_y) - push_cfg.agent_radius - 1e-3)
    fy_min = max(f_low, float(nav_cfg.wall_y) + nav_cfg.agent_radius + 1e-3)

    obs_low, obs_high = push_obstacle_bounds(push_cfg)
    obs_y_max = float(push_cfg.wall_y) - push_cfg.obstacle_radius - 1e-3

    conditions: list[EpisodeConditions] = []

    for _ in range(n_episodes):
        # --- Obstacle position ---
        if fixed_obstacle is not None:
            obs_pos = fixed_obstacle_position(fixed_obstacle, nav_cfg)
        elif obstacle_distance is not None:
            obs_pos = sample_obstacle_near_goal(
                rng=rng,
                goal=goal,
                max_distance=obstacle_distance,
                obstacle_radius=push_cfg.obstacle_radius,
                world_min=push_cfg.world_min,
                world_max=push_cfg.world_max,
                wall_y=push_cfg.wall_y,
            )
        else:
            # Random obstacle below wall
            obs_pos = sample_collision_free_pos(
                rng=rng,
                x_low=obs_low, x_high=obs_high,
                y_low=obs_low, y_high=obs_y_max,
                config=push_cfg,
                wall_radius=push_cfg.obstacle_radius,
            )

        # --- Leader position: below wall, not overlapping obstacle ---
        l_pos = sample_collision_free_pos(
            rng=rng,
            x_low=l_low, x_high=l_high,
            y_low=l_low, y_high=ly_max,
            avoid_pos=obs_pos,
            min_dist=min_dist_leader,
            config=push_cfg,
            wall_radius=push_cfg.agent_radius,
        )

        # --- Follower position: above wall, not overlapping obstacle ---
        f_pos = sample_collision_free_pos(
            rng=rng,
            x_low=f_low, x_high=f_high,
            y_low=fy_min, y_high=f_high,
            avoid_pos=obs_pos,
            min_dist=min_dist_follower,
            config=nav_cfg,
            wall_radius=nav_cfg.agent_radius,
        )

        conditions.append(EpisodeConditions(
            obstacle_pos=obs_pos,
            leader_pos=l_pos,
            follower_pos=f_pos,
        ))

    return conditions


__all__ = [
    "EpisodeConditions",
    "fixed_obstacle_position",
    "sample_collision_free_pos",
    "sample_eval_conditions",
    "sample_obstacle_near_goal",
]
