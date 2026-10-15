"""Cleaner (leader) environment for the household scenario.

The cleaner must pick up a piece of dirt frozen at the table's spawn
centroid. SAT collision means the cleaner physically cannot overlap
the table, so reaching the dirt requires pushing the table aside
first. Each step:

  1. Advance the cleaner pose via ``(vx, vy, ω)`` and ``dt``.
  2. Clamp the cleaner to world bounds (record ``hit_wall``).
  3. If ``wall_enabled``, enforce the horizontal wall.
  4. If the cleaner-rect overlaps the table-rect (SAT), resolve by
     translating the table along the MTV by
     ``push_strength_lin × overlap_depth`` and rotating it by
     ``push_strength_rot × lever_arm × overlap_depth``. Final residual
     overlap is zeroed by an unscaled MTV-translation of the table
     (the cleaner is the active body; the table absorbs the push).
  5. Check pickup: terminate if the dirt point lies inside the
     cleaner-rect.

Reward (mirrors the obstacle_arena pusher pattern, slightly more knobs):
  * ``progress_weight * (prev_dist - dist)`` where ``dist`` = cleaner→dirt
  * ``- step_penalty`` every step
  * ``- wall_collision_penalty`` if step 2/3 clamped the cleaner
  * ``+ pickup_reward`` and terminate on pickup

Observation (10-D):
  ``[cleaner_x, cleaner_y, sin θ_c, cos θ_c,
     table_x, table_y, sin θ_t, cos θ_t,
     dirt_x, dirt_y]``
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from critic_sharing.config.schema import HouseholdCleanerEnvConfig as EnvConfig
from critic_sharing.scenarios.household.common import (
    clamp_to_world,
    clamp_to_world_batched,
    point_in_rect,
    point_in_rect_batched,
    rect_corners,
    rect_corners_batched,
    resolve_static_collision,
    resolve_static_collision_batched,
    sat_contact_point,
    sat_contact_point_batched,
    sat_overlap,
    sat_overlap_batched,
    wall_rects,
)
from critic_sharing.scenarios.household.initial_conditions import (
    sample_pose,
    sample_pose_no_overlap,
)


def _wrap_to_pi(theta: float) -> float:
    return float(((theta + np.pi) % (2.0 * np.pi)) - np.pi)


class HouseholdCleanerEnv(gym.Env):
    SUCCESS_INFO_KEY = "picked_up_dirt"

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[EnvConfig] = None):
        super().__init__()
        self.config = config or EnvConfig()
        x_min = float(self.config.world_x_min)
        x_max = float(self.config.world_x_max)
        y_min = float(self.config.world_y_min)
        y_max = float(self.config.world_y_max)
        obs_low = np.array(
            [x_min, y_min, -1.0, -1.0, x_min, y_min, -1.0, -1.0, x_min, y_min],
            dtype=np.float32,
        )
        obs_high = np.array(
            [x_max, y_max, 1.0, 1.0, x_max, y_max, 1.0, 1.0, x_max, y_max],
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
        self._cleaner_pos = np.zeros(2, dtype=np.float32)
        self._cleaner_theta = 0.0
        self._table_pos = np.zeros(2, dtype=np.float32)
        self._table_theta = 0.0
        self._dirt_pos = np.zeros(2, dtype=np.float32)
        self._steps = 0
        # Step index at which the dirt was first picked up; None until then.
        # Drives the post-pickup aid window (see step()).
        self._pickup_step: Optional[int] = None

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        return np.array(
            [
                self._cleaner_pos[0], self._cleaner_pos[1],
                float(np.sin(self._cleaner_theta)), float(np.cos(self._cleaner_theta)),
                self._table_pos[0], self._table_pos[1],
                float(np.sin(self._table_theta)), float(np.cos(self._table_theta)),
                self._dirt_pos[0], self._dirt_pos[1],
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
        self._pickup_step = None
        region = "lower" if getattr(self.config, "wall_enabled", False) else "full"
        rng = self.np_random

        # Table first (defines dirt position). Same region as cleaner —
        # both spawn in the lower half when the wall is enabled.
        table_pos = options.get("table_pos") if options else None
        table_theta = options.get("table_theta") if options else None
        if table_pos is None or table_theta is None:
            sampled_pos, sampled_theta = sample_pose(
                rng, self.config.table_half_extents, self.config, region=region,
            )
            table_pos = sampled_pos if table_pos is None else np.asarray(table_pos, dtype=np.float32)
            table_theta = sampled_theta if table_theta is None else float(table_theta)
        self._table_pos = np.asarray(table_pos, dtype=np.float32)
        self._table_theta = _wrap_to_pi(float(table_theta))

        # Dirt is frozen at table spawn centroid (5a-i decision).
        self._dirt_pos = self._table_pos.copy()

        # Cleaner avoiding table overlap.
        cleaner_pos = options.get("cleaner_pos") if options else None
        cleaner_theta = options.get("cleaner_theta") if options else None
        if cleaner_pos is None or cleaner_theta is None:
            occupants = [(self._table_pos, self._table_theta,
                          self.config.table_half_extents)]
            sampled_pos, sampled_theta = sample_pose_no_overlap(
                rng, self.config.cleaner_half_extents, occupants, self.config,
                region=region,
            )
            cleaner_pos = sampled_pos if cleaner_pos is None else np.asarray(cleaner_pos, dtype=np.float32)
            cleaner_theta = sampled_theta if cleaner_theta is None else float(cleaner_theta)
        self._cleaner_pos = np.asarray(cleaner_pos, dtype=np.float32)
        self._cleaner_theta = _wrap_to_pi(float(cleaner_theta))

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: np.ndarray):
        self._steps += 1
        action = np.asarray(action, dtype=np.float32)
        vx = float(np.clip(action[0], -self.config.max_speed, self.config.max_speed))
        vy = float(np.clip(action[1], -self.config.max_speed, self.config.max_speed))
        omega = float(np.clip(action[2], -self.config.max_yaw_rate, self.config.max_yaw_rate))

        prev_cleaner_pos = self._cleaner_pos.copy()
        prev_dist = float(np.linalg.norm(self._cleaner_pos - self._dirt_pos))
        walls = wall_rects(self.config)

        # 1. Advance cleaner pose.
        self._cleaner_pos = self._cleaner_pos + np.array([vx, vy], dtype=np.float32) * self.config.dt
        self._cleaner_theta = _wrap_to_pi(self._cleaner_theta + omega * self.config.dt)

        # 2. Clamp to world bounds.
        clamped, hit_boundary = clamp_to_world(
            self._cleaner_pos, self.config.cleaner_half_extents, self.config,
            theta=self._cleaner_theta,
        )
        self._cleaner_pos = clamped

        # 3. SAT-resolve cleaner vs static wall rects.
        self._cleaner_pos, hit_wall = resolve_static_collision(
            self._cleaner_pos, self._cleaner_theta,
            self.config.cleaner_half_extents, walls,
        )
        wall_collided = hit_boundary or hit_wall

        # 4. SAT push of table (also resolves table-vs-wall + cleaner pushback).
        self._resolve_table_push(prev_cleaner_pos, walls)

        # 5. Pickup check. ``in_aid_phase`` is True on every step *after* the
        # one that first picked up the dirt — during it the leader earns no
        # positive reward (penalties still apply) and keeps moving so it can
        # reposition the table to aid the follower (driven by V_F at eval).
        in_aid_phase = self._pickup_step is not None
        picked_up_now = (not in_aid_phase) and point_in_rect(
            self._dirt_pos, self._cleaner_pos, self._cleaner_theta,
            self.config.cleaner_half_extents,
        )
        if picked_up_now:
            self._pickup_step = self._steps

        # Reward uses the real dirt distance on the pickup step; the sentinel
        # move below only takes effect for subsequent (aid-phase) steps.
        dist = float(np.linalg.norm(self._cleaner_pos - self._dirt_pos))
        if in_aid_phase:
            # Post-pickup: penalties only, no progress / pickup reward.
            reward = -self.config.step_penalty
            if wall_collided:
                reward -= self.config.wall_collision_penalty
        else:
            reward = self.config.progress_weight * (prev_dist - dist)
            reward -= self.config.step_penalty
            if wall_collided:
                reward -= self.config.wall_collision_penalty
            if picked_up_now:
                reward += self.config.pickup_reward

        if picked_up_now:
            # Move the (now-collected) dirt out of the arena so neither the
            # env nor the batched CEM dynamics can re-detect a pickup during
            # the aid phase (cleaner is clamped to world bounds, so it can
            # never overlap this sentinel). Keeps obs values modest; applies
            # from the next step's obs onward.
            self._dirt_pos = np.array(
                [float(self.config.world_x_max) + 1.0,
                 float(self.config.world_y_max) + 1.0],
                dtype=np.float32,
            )

        # Terminate ``post_pickup_steps`` after the pickup step (capped at
        # max_steps via truncation). ``post_pickup_steps == 0`` reproduces the
        # legacy terminate-on-pickup behaviour.
        post_pickup_steps = int(getattr(self.config, "post_pickup_steps", 0))
        terminated = (
            self._pickup_step is not None
            and (self._steps - self._pickup_step) >= post_pickup_steps
        )
        truncated = self._steps >= self.config.max_steps
        info = {
            "distance_to_dirt": float(dist),
            # Latches True for the whole aid phase so eval / calibration read
            # success at the final step, not just the pickup step.
            "picked_up_dirt": bool(self._pickup_step is not None),
            "wall_collided": bool(wall_collided),
            "in_aid_phase": bool(in_aid_phase),
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # SAT push helper
    # ------------------------------------------------------------------

    def _resolve_table_push(self, prev_cleaner_pos: np.ndarray, walls) -> None:
        """Apply SAT-MTV penetration resolution with lever-arm torque.

        Each pass:
          1. Push the table along the MTV (with lever-arm torque); clamp
             to world; SAT-resolve table vs wall rects.
          2. If residual cleaner↔table overlap remains (because the
             table is pinned by wall/boundary), push the cleaner back
             along the negative MTV (and re-resolve cleaner vs walls).

        Two passes are usually enough; a third catches edge cases where
        the rotation has changed which axis is shortest.
        """

        for _ in range(2):
            cleaner_corners = rect_corners(
                self._cleaner_pos, self._cleaner_theta,
                self.config.cleaner_half_extents,
            )
            table_corners = rect_corners(
                self._table_pos, self._table_theta,
                self.config.table_half_extents,
            )
            overlapping, mtv_normal, depth = sat_overlap(cleaner_corners, table_corners)
            if not overlapping:
                return

            # --- Try to push the table ---
            lin_step = (self.config.push_strength_lin * depth
                        + max(0.0, depth - 1e-3))
            new_table_pos = self._table_pos + lin_step * mtv_normal
            contact_pt = sat_contact_point(cleaner_corners, table_corners, mtv_normal)
            lever = contact_pt - self._table_pos.astype(np.float32)
            torque = float(lever[0] * mtv_normal[1] - lever[1] * mtv_normal[0])
            d_theta = self.config.push_strength_rot * torque * depth
            new_table_theta = _wrap_to_pi(self._table_theta + d_theta)
            new_table_pos, _ = clamp_to_world(
                new_table_pos, self.config.table_half_extents, self.config,
                theta=new_table_theta,
            )
            new_table_pos, _ = resolve_static_collision(
                new_table_pos, new_table_theta,
                self.config.table_half_extents, walls,
            )
            self._table_pos = new_table_pos.astype(np.float32)
            self._table_theta = float(new_table_theta)

            # --- Push the cleaner back if residual overlap remains ---
            cleaner_corners = rect_corners(
                self._cleaner_pos, self._cleaner_theta,
                self.config.cleaner_half_extents,
            )
            table_corners = rect_corners(
                self._table_pos, self._table_theta,
                self.config.table_half_extents,
            )
            overlapping2, mtv2, residual = sat_overlap(cleaner_corners, table_corners)
            if overlapping2:
                new_cleaner_pos = (
                    self._cleaner_pos - (residual + 1e-3) * mtv2
                )
                new_cleaner_pos, _ = clamp_to_world(
                    new_cleaner_pos, self.config.cleaner_half_extents,
                    self.config, theta=self._cleaner_theta,
                )
                new_cleaner_pos, _ = resolve_static_collision(
                    new_cleaner_pos, self._cleaner_theta,
                    self.config.cleaner_half_extents, walls,
                )
                self._cleaner_pos = new_cleaner_pos.astype(np.float32)


def cleaner_dynamics_batch(
    obs: np.ndarray, action: np.ndarray, config: EnvConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised one-step transition for the cleaner env (matches step())."""

    obs = np.asarray(obs, dtype=np.float32)
    action = np.asarray(action, dtype=np.float32)
    cleaner_he = tuple(config.cleaner_half_extents)
    table_he = tuple(config.table_half_extents)
    walls = wall_rects(config)

    c_pos = obs[:, 0:2].copy()
    c_theta = np.arctan2(obs[:, 2], obs[:, 3]).astype(np.float32)
    t_pos = obs[:, 4:6].copy()
    t_theta = np.arctan2(obs[:, 6], obs[:, 7]).astype(np.float32)
    dirt = obs[:, 8:10].copy()

    vx = np.clip(action[:, 0], -config.max_speed, config.max_speed)
    vy = np.clip(action[:, 1], -config.max_speed, config.max_speed)
    omega = np.clip(action[:, 2], -config.max_yaw_rate, config.max_yaw_rate)

    prev_dist = np.linalg.norm(c_pos - dirt, axis=1)

    c_pos = c_pos + np.stack([vx, vy], axis=1) * config.dt
    c_theta = ((c_theta + omega * config.dt + np.pi) % (2 * np.pi)) - np.pi

    c_pos, hit_boundary = clamp_to_world_batched(c_pos, c_theta, cleaner_he, config)
    c_pos, hit_wall = resolve_static_collision_batched(c_pos, c_theta, cleaner_he, walls)
    wall_collided = hit_boundary | hit_wall

    for _ in range(2):
        cc = rect_corners_batched(c_pos, c_theta, cleaner_he)
        tc = rect_corners_batched(t_pos, t_theta, table_he)
        ov, mtv, depth = sat_overlap_batched(cc, tc)
        if not ov.any():
            break

        lin_step = (config.push_strength_lin * depth
                    + np.maximum(0.0, depth - 1e-3)).astype(np.float32)
        scale = (lin_step * ov).astype(np.float32)
        new_t_pos = t_pos + scale[:, None] * mtv

        contact_pt = sat_contact_point_batched(cc, tc, mtv)
        lever = contact_pt - t_pos
        torque = (lever[:, 0] * mtv[:, 1] - lever[:, 1] * mtv[:, 0]).astype(np.float32)
        d_theta = (config.push_strength_rot * torque * depth * ov).astype(np.float32)
        new_t_theta = ((t_theta + d_theta + np.pi) % (2 * np.pi)) - np.pi

        new_t_pos, _ = clamp_to_world_batched(new_t_pos, new_t_theta, table_he, config)
        new_t_pos, _ = resolve_static_collision_batched(new_t_pos, new_t_theta, table_he, walls)
        t_pos = new_t_pos
        t_theta = new_t_theta

        cc = rect_corners_batched(c_pos, c_theta, cleaner_he)
        tc = rect_corners_batched(t_pos, t_theta, table_he)
        ov2, mtv2, residual = sat_overlap_batched(cc, tc)
        if ov2.any():
            push_back = ((residual + 1e-3) * ov2).astype(np.float32)
            c_pos = c_pos - push_back[:, None] * mtv2
            c_pos, _ = clamp_to_world_batched(c_pos, c_theta, cleaner_he, config)
            c_pos, _ = resolve_static_collision_batched(c_pos, c_theta, cleaner_he, walls)

    picked = point_in_rect_batched(dirt, c_pos, c_theta, cleaner_he)

    dist = np.linalg.norm(c_pos - dirt, axis=1)
    rewards = (config.progress_weight * (prev_dist - dist)).astype(np.float32)
    rewards -= float(config.step_penalty)
    rewards = np.where(wall_collided, rewards - float(config.wall_collision_penalty), rewards)
    rewards = np.where(picked, rewards + float(config.pickup_reward), rewards)

    next_obs = np.column_stack([
        c_pos[:, 0], c_pos[:, 1],
        np.sin(c_theta), np.cos(c_theta),
        t_pos[:, 0], t_pos[:, 1],
        np.sin(t_theta), np.cos(t_theta),
        dirt[:, 0], dirt[:, 1],
    ]).astype(np.float32)

    return next_obs, rewards, picked


__all__ = ["HouseholdCleanerEnv", "EnvConfig", "cleaner_dynamics_batch"]
