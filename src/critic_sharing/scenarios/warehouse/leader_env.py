"""Leader env for the warehouse scenario.

The leader operates in the lower half of the arena (``y < wall_y``). It picks
up an object that spawns at a random location below the wall and drops it
into one of three gaps in the wall. The 3 gaps double as the dropoff areas
that the follower (in the upper half) can pick up from. The leader cannot
cross the wall — gaps are too narrow for the inscribed-circle agent.

**Pickup/drop are automatic**: at the end of each step, the env auto-picks
up if the object is within the pickup cone and the agent isn't already
carrying, and auto-drops in a gap whenever the carrying agent's would-be
drop point falls inside any gap rectangle. The agent therefore only has
to learn navigation.

Reward (gap-agnostic, shaped):
  * ``-step_penalty`` per step.
  * ``progress_weight * (prev_d - d)`` shaping toward the object (when not
    carrying) or toward the *nearest* of the 3 gap centres (when carrying).
  * ``+pickup_reward`` when the object is picked up.
  * ``+goal_reward`` and terminate when the object is auto-dropped inside
    a gap.

Observation (7-D):
  ``[agent_x, agent_y, sin(theta), cos(theta), object_x, object_y, carry_flag]``
  While carrying, ``(object_x, object_y) = (agent_x, agent_y)``.

Action (3-D continuous, ``Box``):
  ``[vx, vy, omega]`` with bounds ``[±max_speed]² × [±max_yaw_rate]``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from critic_sharing.config.schema import WarehouseLeaderEnvConfig as EnvConfig
from critic_sharing.scenarios.warehouse.common import (
    agent_bounds,
    agent_inscribed_radius,
    drop_position,
    enforce_agent_object_collision,
    enforce_agent_object_collision_batch,
    enforce_wall,
    enforce_wall_batch,
    gap_centers,
    gap_index_for_position,
    in_pickup_cone,
    object_bounds,
    snap_to_wall_if_in_gap,
)


class WarehouseLeaderEnv(gym.Env):
    # Per-step ``info`` key naming the bool that flips True on a successful
    # episode. Read by :func:`critic_sharing.utils.calibration.calibrate_range`
    # to filter naive rollouts before computing the leader return range.
    # "leader_success" = object dropped AND drop landed in a gap (not in a
    # wall section). Composed in :meth:`step` from the per-step signals.
    SUCCESS_INFO_KEY = "leader_success"

    """Leader env for the warehouse scenario."""

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[EnvConfig] = None):
        super().__init__()
        self.config = config or EnvConfig()

        big = np.array([
            self.config.world_max,
            self.config.world_max,
            1.0, 1.0,
            self.config.world_max,
            self.config.world_max,
            1.0,
        ], dtype=np.float32)
        small = np.array([
            self.config.world_min,
            self.config.world_min,
            -1.0, -1.0,
            self.config.world_min,
            self.config.world_min,
            0.0,
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low=small, high=big, dtype=np.float32)
        act_low = np.array([
            -self.config.max_speed,
            -self.config.max_speed,
            -self.config.max_yaw_rate,
        ], dtype=np.float32)
        act_high = np.array([
            self.config.max_speed,
            self.config.max_speed,
            self.config.max_yaw_rate,
        ], dtype=np.float32)
        self.action_space = spaces.Box(low=act_low, high=act_high, dtype=np.float32)

        self._agent_pos = np.zeros(2, dtype=np.float32)
        self._agent_theta = 0.0
        self._object_pos = np.zeros(2, dtype=np.float32)
        self._carry = False
        self._steps = 0

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _sample_agent_pos(self) -> np.ndarray:
        low, high = agent_bounds(self.config)
        r = agent_inscribed_radius(self.config)
        y_max = float(self.config.wall_y) - r - 1e-3
        x = float(self.np_random.uniform(low, high))
        y = float(self.np_random.uniform(low, min(high, y_max)))
        return np.array([x, y], dtype=np.float32)

    def _sample_object_pos(self) -> np.ndarray:
        low, high = object_bounds(self.config)
        y_max = float(self.config.wall_y) - float(self.config.object_radius) - 1e-3
        x = float(self.np_random.uniform(low, high))
        y = float(self.np_random.uniform(low, min(high, y_max)))
        return np.array([x, y], dtype=np.float32)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        if self._carry:
            obj = self._agent_pos
        else:
            obj = self._object_pos
        return np.array(
            [
                self._agent_pos[0],
                self._agent_pos[1],
                float(np.sin(self._agent_theta)),
                float(np.cos(self._agent_theta)),
                obj[0],
                obj[1],
                1.0 if self._carry else 0.0,
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Distance helpers (used by reward shaping)
    # ------------------------------------------------------------------

    def _nearest_gap_distance(self, pos: np.ndarray) -> float:
        gaps = gap_centers(self.config)
        diffs = gaps - pos[None, :]
        return float(np.min(np.linalg.norm(diffs, axis=1)))

    def _shaping_distance(self) -> float:
        if self._carry:
            return self._nearest_gap_distance(self._agent_pos)
        return float(np.linalg.norm(self._agent_pos - self._object_pos))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._steps = 0
        self._carry = False
        options = options or {}

        agent_pos = options.get("agent_pos")
        agent_theta = options.get("agent_theta")
        object_pos = options.get("object_pos")

        if object_pos is None:
            object_pos = self._sample_object_pos()
        self._object_pos = np.asarray(object_pos, dtype=np.float32)

        # Resolve agent position; reject overlap with object.
        r = agent_inscribed_radius(self.config)
        min_dist = r + float(self.config.object_radius) + 1e-3
        if agent_pos is None:
            for _ in range(1000):
                candidate = self._sample_agent_pos()
                if np.linalg.norm(candidate - self._object_pos) >= min_dist:
                    agent_pos = candidate
                    break
            else:
                agent_pos = self._sample_agent_pos()
        self._agent_pos = np.asarray(agent_pos, dtype=np.float32)

        if agent_theta is None:
            agent_theta = float(self.np_random.uniform(-np.pi, np.pi))
        self._agent_theta = float(agent_theta)
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self._steps += 1
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        prev_pos = self._agent_pos.copy()
        prev_d = self._shaping_distance()
        prev_carry = self._carry

        # Translation
        vel = action[:2]
        speed = float(np.linalg.norm(vel))
        if speed > self.config.max_speed:
            vel = vel / (speed + 1e-8) * self.config.max_speed
        next_pos = self._agent_pos + vel * self.config.dt
        low, high = agent_bounds(self.config)
        next_pos = np.clip(next_pos, low, high).astype(np.float32)
        next_pos = enforce_wall(
            next_pos, prev_pos, agent_inscribed_radius(self.config), self.config
        )
        # Agent ↔ object collision (only when not carrying — the object
        # follows the agent while carried).
        if not self._carry:
            next_pos = enforce_agent_object_collision(
                next_pos, self._object_pos,
                agent_inscribed_radius(self.config), self.config,
            )
        self._agent_pos = next_pos.astype(np.float32)

        # Rotation
        omega = float(np.clip(action[2], -self.config.max_yaw_rate, self.config.max_yaw_rate))
        self._agent_theta = float(((self._agent_theta + omega * self.config.dt + np.pi) % (2.0 * np.pi)) - np.pi)

        # Auto-pickup: object in cone within range, not already carrying.
        picked_up = False
        if not self._carry and in_pickup_cone(
            self._agent_pos, self._agent_theta, self._object_pos, self.config
        ):
            self._carry = True
            picked_up = True

        # Auto-drop in a gap: carrying AND drop_position falls into a gap rect.
        dropped = False
        gap_idx = -1
        if self._carry:
            drop_pt = drop_position(self._agent_pos, self._agent_theta, self.config)
            gap_idx = gap_index_for_position(drop_pt, self.config)
            if gap_idx >= 0:
                drop_pt = snap_to_wall_if_in_gap(drop_pt, self.config)
                obj_low, obj_high = object_bounds(self.config)
                drop_pt = np.clip(drop_pt, obj_low, obj_high).astype(np.float32)
                self._object_pos = drop_pt
                self._carry = False
                dropped = True

        # Reward — step penalty depends on the carry state at the *start* of
        # the step (the agent moved with/without the object during this step).
        step_pen = (
            self.config.step_penalty_carrying if prev_carry else self.config.step_penalty
        )
        reward = -float(step_pen)
        new_d = self._shaping_distance()
        reward += float(self.config.progress_weight) * float(prev_d - new_d)
        if picked_up:
            reward += float(self.config.pickup_reward)
        terminated = False
        if dropped:
            reward += float(self.config.goal_reward)
            terminated = True

        truncated = self._steps >= int(self.config.max_steps)
        info = {
            "picked_up": bool(picked_up),
            "dropped": bool(dropped),
            "gap_index": int(gap_idx),
            "carry": bool(self._carry),
            "steps": int(self._steps),
            # Composed success signal — see SUCCESS_INFO_KEY class attribute.
            # Drop in a gap (gap_idx >= 0) is the binary "task done" event.
            "leader_success": bool(dropped and gap_idx >= 0),
        }
        return self._get_obs(), float(reward), terminated, truncated, info


# ----------------------------------------------------------------------
# Vectorised CEM dynamics
# ----------------------------------------------------------------------

def warehouse_leader_dynamics_batch(
    obs: np.ndarray,
    action: np.ndarray,
    config: EnvConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure-numpy batched one-step transition for the leader env.

    Compatible with :class:`critic_sharing.planning.cem.CEMController` —
    returns ``(next_obs, rewards, dones)``. ``done[i] = True`` means the
    auto-drop-in-gap fired at this step (object successfully placed in
    one of the 3 gap rectangles); CEM handles the post-terminate freeze
    so the dynamics function is naive (no per-row "already terminated"
    sentinel detection).
    """
    obs = np.asarray(obs, dtype=np.float32)
    action = np.asarray(action, dtype=np.float32)
    if obs.ndim != 2 or action.ndim != 2:
        raise ValueError("obs and action must be (B, *) arrays")
    n = obs.shape[0]

    agent_pos = obs[:, 0:2].copy()
    sin_t = obs[:, 2].copy()
    cos_t = obs[:, 3].copy()
    obj_pos_obs = obs[:, 4:6].copy()
    carry = obs[:, 6] >= 0.5

    # Authoritative object position. (For carrying rows it tracks the agent
    # automatically — value doesn't matter until drop.)
    object_pos = obj_pos_obs.copy()
    half_gap = float(config.gap_width) / 2.0

    # Recover theta from sin/cos for rotation update.
    theta = np.arctan2(sin_t, cos_t)

    # Translation
    vel = action[:, 0:2].copy()
    speed = np.linalg.norm(vel, axis=1, keepdims=True)
    vel = np.where(speed > config.max_speed, vel / (speed + 1e-8) * config.max_speed, vel)
    next_pos = agent_pos + vel * config.dt
    low, high = agent_bounds(config)
    next_pos = np.clip(next_pos, low, high).astype(np.float32)
    next_pos = enforce_wall_batch(
        next_pos, agent_pos, agent_inscribed_radius(config), config
    )
    next_pos = enforce_agent_object_collision_batch(
        next_pos, object_pos, carry, agent_inscribed_radius(config), config,
    )

    # Rotation
    omega = np.clip(action[:, 2], -config.max_yaw_rate, config.max_yaw_rate)
    next_theta = ((theta + omega * config.dt + np.pi) % (2.0 * np.pi)) - np.pi

    # Auto-pickup: not carrying AND object in cone+range.
    cone = _in_cone_batch(next_pos, next_theta, object_pos, config)
    pickup = (~carry) & cone
    new_carry = carry.copy()
    new_carry[pickup] = True

    # Auto-drop in gap: carrying AND drop_position is inside a gap rectangle.
    new_object = object_pos.copy()
    d = float(config.drop_distance)
    drop_x = next_pos[:, 0] + d * np.cos(next_theta)
    drop_y = next_pos[:, 1] + d * np.sin(next_theta)
    drop_pts = np.stack([drop_x, drop_y], axis=1)
    y_tol = float(config.drop_distance)
    wy = float(config.wall_y)
    near_wall = np.abs(drop_pts[:, 1] - wy) <= y_tol
    in_gap_x = np.zeros(n, dtype=bool)
    for gx in config.gap_xs:
        in_gap_x |= np.abs(drop_pts[:, 0] - float(gx)) <= half_gap
    in_gap = new_carry & near_wall & in_gap_x
    if np.any(in_gap):
        drop_pts[in_gap, 1] = wy  # snap to wall line
        obj_low, obj_high = object_bounds(config)
        drop_pts = np.clip(drop_pts, obj_low, obj_high).astype(np.float32)
        new_object[in_gap] = drop_pts[in_gap]
        new_carry[in_gap] = False

    # Per-step reward — step penalty depends on carry state at start of step.
    step_pen_uncarry = -float(config.step_penalty)
    step_pen_carry = -float(config.step_penalty_carrying)
    rewards = np.where(carry, step_pen_carry, step_pen_uncarry).astype(np.float32)
    prev_d = _shaping_distance_batch(agent_pos, object_pos, carry, config)
    new_d = _shaping_distance_batch(next_pos, new_object, new_carry, config)
    rewards += float(config.progress_weight) * (prev_d - new_d).astype(np.float32)
    rewards[pickup] += float(config.pickup_reward)
    rewards[in_gap] += float(config.goal_reward)

    # Build next observation (carrying ⇒ obs object_pos == agent_pos).
    obs_obj_x = np.where(new_carry, next_pos[:, 0], new_object[:, 0])
    obs_obj_y = np.where(new_carry, next_pos[:, 1], new_object[:, 1])
    next_obs = np.stack([
        next_pos[:, 0],
        next_pos[:, 1],
        np.sin(next_theta),
        np.cos(next_theta),
        obs_obj_x,
        obs_obj_y,
        new_carry.astype(np.float32),
    ], axis=1).astype(np.float32)
    # ``in_gap`` is the per-step done mask: object successfully dropped in
    # one of the 3 gap rectangles. CEM handles the post-terminate freeze.
    return next_obs, rewards.astype(np.float32), in_gap.astype(bool)


def _in_cone_batch(
    agent_pos: np.ndarray,
    agent_theta: np.ndarray,
    object_pos: np.ndarray,
    config: EnvConfig,
) -> np.ndarray:
    dx = object_pos[:, 0] - agent_pos[:, 0]
    dy = object_pos[:, 1] - agent_pos[:, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    angle_to_obj = np.arctan2(dy, dx)
    diff = ((angle_to_obj - agent_theta + np.pi) % (2.0 * np.pi)) - np.pi
    return (dist <= config.pickup_max_distance) & (np.abs(diff) <= config.pickup_cone_half_angle)


def _shaping_distance_batch(
    agent_pos: np.ndarray,
    object_pos: np.ndarray,
    carry: np.ndarray,
    config: EnvConfig,
) -> np.ndarray:
    """Distance for progress shaping: to nearest gap if carrying, to object otherwise."""
    gaps = gap_centers(config)  # (G, 2)
    # Distance from agent to each gap centre.
    diffs = agent_pos[:, None, :] - gaps[None, :, :]  # (B, G, 2)
    dists_to_gap = np.linalg.norm(diffs, axis=2)  # (B, G)
    nearest = np.min(dists_to_gap, axis=1)  # (B,)
    to_object = np.linalg.norm(agent_pos - object_pos, axis=1)
    return np.where(carry, nearest, to_object).astype(np.float32)


__all__ = [
    "EnvConfig",
    "WarehouseLeaderEnv",
    "warehouse_leader_dynamics_batch",
]
