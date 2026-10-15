"""Navigator (follower) environment for the household scenario.

The follower navigates to the lower boundary (matches the obstacle
arena nav goal). The table is static-per-episode in this env — the
follower must learn to plan around it but never sees the cleaner
displace it. At joint-eval time the cleaner *does* move the table,
which is exactly the leader-induced OOD distribution shift that the
trust-weighted V_F is meant to detect.

Observation (8-D):
  ``[follower_x, follower_y, sin θ_f, cos θ_f,
     table_x, table_y, sin θ_t, cos θ_t]``

Action: ``(vx, vy, ω)``.

Reward (mirrors obstacle_arena nav, slightly more knobs):
  * ``progress_weight * (prev_dist_to_goal - dist_to_goal)``
  * ``- distance_weight * dist_to_goal`` (small standing-penalty pull)
  * ``- distance_weight * gap_penalty`` when above the wall
  * ``- collision_penalty`` per step if SAT-overlapping the table
  * ``+ goal_reward`` and terminate on reaching the goal region
  * Episode terminates on goal reach or table collision.

Spawn region:
  ``options["spawn_region"]`` overrides the default. Default is
  ``"full"`` (uniform full arena) to match obstacle_arena's
  follower-training distribution; joint-eval code passes
  ``"upper"`` to restrict to the deployment distribution.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from critic_sharing.config.schema import HouseholdNavEnvConfig as EnvConfig
from critic_sharing.scenarios.household.common import (
    clamp_to_world,
    nav_goal_position,
    rect_corners,
    resolve_static_collision,
    sat_overlap,
    wall_rects,
)
from critic_sharing.scenarios.household.initial_conditions import (
    sample_pose,
    sample_pose_no_overlap,
)


def _wrap_to_pi(theta: float) -> float:
    return float(((theta + np.pi) % (2.0 * np.pi)) - np.pi)


class HouseholdNavigatorEnv(gym.Env):
    SUCCESS_INFO_KEY = "reached_goal"

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[EnvConfig] = None):
        super().__init__()
        self.config = config or EnvConfig()
        x_min = float(self.config.world_x_min)
        x_max = float(self.config.world_x_max)
        y_min = float(self.config.world_y_min)
        y_max = float(self.config.world_y_max)
        obs_low = np.array(
            [x_min, y_min, -1.0, -1.0, x_min, y_min, -1.0, -1.0],
            dtype=np.float32,
        )
        obs_high = np.array(
            [x_max, y_max, 1.0, 1.0, x_max, y_max, 1.0, 1.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-self.config.max_speed, -self.config.max_speed,
                          -self.config.max_yaw_rate], dtype=np.float32),
            high=np.array([self.config.max_speed, self.config.max_speed,
                           self.config.max_yaw_rate], dtype=np.float32),
            dtype=np.float32,
        )
        self._pos = np.zeros(2, dtype=np.float32)
        self._theta = 0.0
        self._table_pos = np.zeros(2, dtype=np.float32)
        self._table_theta = 0.0
        self._goal = np.zeros(2, dtype=np.float32)
        self._steps = 0

    def set_collision_penalty(self, value: float) -> None:
        """Live setter so a training callback can drive a curriculum.

        The reward formula reads ``self.config.collision_penalty`` each
        step, so mutating the dataclass field takes effect on the next
        ``step()`` call across all parallel envs.
        """

        self.config.collision_penalty = float(value)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        return np.array(
            [
                self._pos[0], self._pos[1],
                float(np.sin(self._theta)), float(np.cos(self._theta)),
                self._table_pos[0], self._table_pos[1],
                float(np.sin(self._table_theta)), float(np.cos(self._table_theta)),
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._steps = 0
        opts = options or {}
        rng = self.np_random

        # Table spawns in the lower half when the wall is enabled
        # (matches the cleaner env's spawn distribution).
        table_region = "lower" if getattr(self.config, "wall_enabled", False) else "full"
        table_pos = opts.get("table_pos")
        table_theta = opts.get("table_theta")
        if table_pos is None or table_theta is None:
            sampled_pos, sampled_theta = sample_pose(
                rng, self.config.table_half_extents, self.config, region=table_region,
            )
            if table_pos is None:
                table_pos = sampled_pos
            if table_theta is None:
                table_theta = sampled_theta
        self._table_pos = np.asarray(table_pos, dtype=np.float32)
        self._table_theta = _wrap_to_pi(float(table_theta))

        # Follower spawn region: defaults to "full" (training); joint-eval
        # passes "upper" to restrict to deployment distribution.
        follower_region = opts.get("spawn_region", "full")
        follower_pos = opts.get("follower_pos")
        if follower_pos is None:
            follower_pos = opts.get("agent_pos")
        follower_theta = opts.get("follower_theta")
        if follower_pos is None or follower_theta is None:
            occupants = [(self._table_pos, self._table_theta,
                          self.config.table_half_extents)]
            sampled_pos, sampled_theta = sample_pose_no_overlap(
                rng, self.config.follower_half_extents, occupants, self.config,
                region=follower_region,
            )
            if follower_pos is None:
                follower_pos = sampled_pos
            if follower_theta is None:
                follower_theta = sampled_theta
        self._pos = np.asarray(follower_pos, dtype=np.float32)
        self._theta = _wrap_to_pi(float(follower_theta))

        self._goal = nav_goal_position(self.config)
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def _goal_hx(self) -> float:
        """Half-width of the centered goal region on the lower boundary.

        ``None`` in config => (world_x_max - world_x_min) / 4, i.e. the goal
        spans the centered *half* of the boundary.
        """
        hx = getattr(self.config, "goal_x_half_extent", None)
        if hx is None:
            return 0.25 * (float(self.config.world_x_max) - float(self.config.world_x_min))
        return float(hx)

    def _goal_low_y(self) -> float:
        from critic_sharing.scenarios.household.common import rect_circumscribed_radius
        r = rect_circumscribed_radius(self.config.follower_half_extents)
        return float(self.config.world_y_min) + r

    def _dist_to_goal(self, pos: np.ndarray) -> float:
        """Distance to the centered goal segment on the lower boundary (≥ 0).

        The goal is the segment ``{(x, low) : |x - mid_x| <= hx}``. The
        distance is to the nearest point of that segment, so the shaping pulls
        the follower both downward and toward the centered band.
        """
        low = self._goal_low_y()
        mid_x = 0.5 * (float(self.config.world_x_min) + float(self.config.world_x_max))
        dx = max(0.0, abs(float(pos[0]) - mid_x) - self._goal_hx())
        dy = max(0.0, float(pos[1]) - low)
        return float(np.hypot(dx, dy))

    def _gap_penalty(self, pos: np.ndarray) -> float:
        if not getattr(self.config, "wall_enabled", False):
            return 0.0
        from critic_sharing.scenarios.household.common import rect_circumscribed_radius
        r = rect_circumscribed_radius(self.config.follower_half_extents)
        wy = float(self.config.wall_y)
        if float(pos[1]) <= wy + r:
            return 0.0
        return abs(float(pos[0]) - float(self.config.gap_x))

    def step(self, action: np.ndarray):
        self._steps += 1
        action = np.asarray(action, dtype=np.float32)
        vx = float(np.clip(action[0], -self.config.max_speed, self.config.max_speed))
        vy = float(np.clip(action[1], -self.config.max_speed, self.config.max_speed))
        omega = float(np.clip(action[2], -self.config.max_yaw_rate, self.config.max_yaw_rate))

        prev_pos = self._pos.copy()
        prev_dist = self._dist_to_goal(prev_pos)

        self._pos = self._pos + np.array([vx, vy], dtype=np.float32) * self.config.dt
        self._theta = _wrap_to_pi(self._theta + omega * self.config.dt)

        self._pos, _ = clamp_to_world(
            self._pos, self.config.follower_half_extents, self.config,
            theta=self._theta,
        )

        # Wall collision: treat hitting a wall like hitting the table —
        # SAT-detect overlap before pushback; if overlapping, register a
        # collision and apply the same collision_penalty / termination.
        follower_corners = rect_corners(
            self._pos, self._theta, self.config.follower_half_extents,
        )
        wall_collided = False
        for (wc, wt, whe) in wall_rects(self.config):
            wcorners = rect_corners(wc, wt, whe)
            ov, _, _ = sat_overlap(follower_corners, wcorners)
            if ov:
                wall_collided = True
                break
        # Resolve any actual penetration so we don't end the episode
        # inside a wall (useful for trajectory recording).
        self._pos, _ = resolve_static_collision(
            self._pos, self._theta,
            self.config.follower_half_extents, wall_rects(self.config),
        )

        # Table collision (SAT). Treat the table as a static obstacle:
        # if the follower overlaps it, push the follower back along the
        # SAT MTV (the table is static-per-episode in this env, so it
        # doesn't absorb the push).
        follower_corners = rect_corners(
            self._pos, self._theta, self.config.follower_half_extents,
        )
        table_corners = rect_corners(
            self._table_pos, self._table_theta, self.config.table_half_extents,
        )
        table_collided, _, _ = sat_overlap(follower_corners, table_corners)
        if table_collided:
            table_as_static = [(
                self._table_pos, self._table_theta,
                tuple(self.config.table_half_extents),
            )]
            self._pos, _ = resolve_static_collision(
                self._pos, self._theta,
                self.config.follower_half_extents, table_as_static,
            )
        collided = bool(wall_collided or table_collided)

        dist = self._dist_to_goal(self._pos)
        reward = self.config.progress_weight * float(prev_dist - dist)
        reward -= self.config.distance_weight * float(dist)
        reward -= self.config.distance_weight * self._gap_penalty(self._pos)
        if collided:
            reward -= self.config.collision_penalty
        # Goal = centered segment of the lower boundary: at the bottom AND
        # within the centered half-width band.
        mid_x = 0.5 * (float(self.config.world_x_min) + float(self.config.world_x_max))
        reached_goal = (
            float(self._pos[1]) <= self._goal_low_y()
            and abs(float(self._pos[0]) - mid_x) <= self._goal_hx()
        )
        if reached_goal:
            reward += self.config.goal_reward

        # Collisions apply ``collision_penalty`` but do not terminate the
        # episode — the follower has been SAT-pushed out of the obstacle
        # by ``resolve_static_collision``, so it remains in a valid state
        # and can keep navigating. This gives PPO recovery experience to
        # learn from instead of short-circuiting to "stand still".
        terminated = bool(reached_goal)
        truncated = self._steps >= self.config.max_steps
        info = {
            "distance": float(dist),
            "reached_goal": bool(reached_goal),
            "collided": bool(collided),
            "table_x": float(self._table_pos[0]),
            "table_y": float(self._table_pos[1]),
            "table_theta": float(self._table_theta),
        }
        return self._get_obs(), float(reward), terminated, truncated, info


__all__ = ["HouseholdNavigatorEnv", "EnvConfig"]
