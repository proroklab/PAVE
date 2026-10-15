"""Leader env for the mobile_arm scenario.

The leader controls a 2D mobile base in world frame and ends the episode
by firing ``terminate``. Reward peaks (Gaussian) at the configured
``desired_distance`` from the object — the leader is incentivised to
park *at* a stand-off ring around the object.

Action (4-D continuous, ``Box``):
  ``[vx, vy, omega, terminate]`` with bounds
  ``[±max_speed]² × [±max_yaw_rate] × [-1, 1]``.
  ``terminate ≥ interact_threshold`` (≥ 0.5 by default) ends the episode.

Observation (6-D):
  ``[x, y, sin(yaw), cos(yaw), object_x, object_y]``.

Reward:
  * ``-step_penalty`` per step.
  * ``progress_weight * (|prev_d - d*| - |cur_d - d*|)`` shaping.
  * On terminate: ``goal_reward · exp(-(|d - d*|)² / σ²)``.

Dynamics: kinematic; base↔object collision via inscribed-circle clamp.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from critic_sharing.config.schema import MobileArmLeaderEnvConfig as EnvConfig
from critic_sharing.scenarios.mobile_arm.common import (
    base_world_bounds,
    enforce_base_object_collision,
    enforce_base_object_collision_batch,
)


class MobileArmLeaderEnv(gym.Env):
    # Per-step ``info`` key naming the bool that flips True on a successful
    # episode (see :func:`critic_sharing.utils.calibration.calibrate_range`).
    # "leader_success" = leader chose to terminate AND was parked within
    # half the Gaussian envelope of the desired stand-off distance. Sloppy
    # terminations (low bonus) are excluded from the calibration range.
    SUCCESS_INFO_KEY = "leader_success"

    """Leader env for the mobile_arm scenario."""

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
        ], dtype=np.float32)
        small = np.array([
            self.config.world_min,
            self.config.world_min,
            -1.0, -1.0,
            self.config.world_min,
            self.config.world_min,
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low=small, high=big, dtype=np.float32)

        act_low = np.array([
            -self.config.max_speed,
            -self.config.max_speed,
            -self.config.max_yaw_rate,
            -1.0,
        ], dtype=np.float32)
        act_high = np.array([
            self.config.max_speed,
            self.config.max_speed,
            self.config.max_yaw_rate,
            1.0,
        ], dtype=np.float32)
        self.action_space = spaces.Box(low=act_low, high=act_high, dtype=np.float32)

        self._base_pos = np.zeros(2, dtype=np.float32)
        self._base_yaw = 0.0
        self._object_pos = np.zeros(2, dtype=np.float32)
        self._steps = 0

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_object_pos(self) -> np.ndarray:
        low, high = base_world_bounds(self.config)
        return np.array(
            [
                float(self.np_random.uniform(low, high)),
                float(self.np_random.uniform(low, high)),
            ],
            dtype=np.float32,
        )

    def _sample_base_pos(self, object_pos: np.ndarray) -> np.ndarray:
        low, high = base_world_bounds(self.config)
        min_sep = float(self.config.leader_object_min_separation)
        for _ in range(1000):
            cand = np.array(
                [
                    float(self.np_random.uniform(low, high)),
                    float(self.np_random.uniform(low, high)),
                ],
                dtype=np.float32,
            )
            if np.linalg.norm(cand - object_pos) >= min_sep:
                return cand
        return cand

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        return np.array(
            [
                self._base_pos[0],
                self._base_pos[1],
                float(np.sin(self._base_yaw)),
                float(np.cos(self._base_yaw)),
                self._object_pos[0],
                self._object_pos[1],
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _stand_off_diff(self) -> float:
        """``|distance_to_object - desired_distance|`` (lower is better)."""
        d = float(np.linalg.norm(self._base_pos - self._object_pos))
        return abs(d - float(self.config.desired_distance))

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
        options = options or {}

        object_pos = options.get("object_pos")
        if object_pos is None:
            object_pos = self._sample_object_pos()
        self._object_pos = np.asarray(object_pos, dtype=np.float32)

        base_pos = options.get("base_pos")
        if base_pos is None:
            base_pos = self._sample_base_pos(self._object_pos)
        self._base_pos = np.asarray(base_pos, dtype=np.float32)

        base_yaw = options.get("base_yaw")
        if base_yaw is None:
            base_yaw = float(self.np_random.uniform(-np.pi, np.pi))
        self._base_yaw = float(base_yaw)
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self._steps += 1
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        prev_diff = self._stand_off_diff()

        # Translation
        vel = action[:2]
        speed = float(np.linalg.norm(vel))
        if speed > self.config.max_speed:
            vel = vel / (speed + 1e-8) * self.config.max_speed
        next_pos = self._base_pos + vel * self.config.dt
        low, high = base_world_bounds(self.config)
        next_pos = np.clip(next_pos, low, high).astype(np.float32)
        next_pos = enforce_base_object_collision(next_pos, self._object_pos, self.config)
        self._base_pos = next_pos.astype(np.float32)

        # Rotation
        omega = float(np.clip(action[2], -self.config.max_yaw_rate, self.config.max_yaw_rate))
        self._base_yaw = float(((self._base_yaw + omega * self.config.dt + np.pi) % (2.0 * np.pi)) - np.pi)

        # Reward — step penalty + delta shaping toward the stand-off ring.
        reward = -float(self.config.step_penalty)
        new_diff = self._stand_off_diff()
        reward += float(self.config.progress_weight) * float(prev_diff - new_diff)

        # Terminate
        terminate_signal = float(action[3])
        terminated = False
        terminate_bonus = 0.0
        if terminate_signal >= float(self.config.interact_threshold):
            sigma = float(self.config.gaussian_sigma)
            terminate_bonus = float(self.config.goal_reward) * float(
                np.exp(-(new_diff ** 2) / (sigma * sigma))
            )
            reward += terminate_bonus
            terminated = True

        truncated = self._steps >= int(self.config.max_steps)
        info = {
            "stand_off_diff": float(new_diff),
            "terminated_by_action": bool(terminated),
            "terminate_bonus": float(terminate_bonus),
            "steps": int(self._steps),
            # Composed success signal — see SUCCESS_INFO_KEY. The leader
            # is "successful" when it chose to terminate AND was within
            # half a sigma of the desired stand-off ring (i.e. earned
            # >~78% of the goal_reward bonus).
            "leader_success": bool(
                terminated
                and new_diff <= 0.5 * float(self.config.gaussian_sigma)
            ),
        }
        return self._get_obs(), float(reward), terminated, truncated, info


# ----------------------------------------------------------------------
# Vectorised CEM dynamics
# ----------------------------------------------------------------------

def mobile_arm_leader_dynamics_batch(
    obs: np.ndarray,
    action: np.ndarray,
    config: EnvConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure-numpy batched one-step transition for the leader env.

    Compatible with :class:`critic_sharing.planning.cem.CEMController` —
    returns ``(next_obs, rewards, dones)``. ``done[i] = True`` means the
    leader fired ``terminate`` at this step (``action[i, 3] >=
    interact_threshold``); CEM handles the post-terminate freeze
    (state held, subsequent rewards zeroed) so the dynamics function
    is naive (no per-row "already terminated" handling).
    """
    obs = np.asarray(obs, dtype=np.float32)
    action = np.asarray(action, dtype=np.float32)
    if obs.ndim != 2 or action.ndim != 2:
        raise ValueError("obs and action must be (B, *) arrays")
    n = obs.shape[0]

    base_pos = obs[:, 0:2].copy()
    sin_t = obs[:, 2].copy()
    cos_t = obs[:, 3].copy()
    object_pos = obs[:, 4:6].copy()
    yaw = np.arctan2(sin_t, cos_t)

    # Translation
    vel = action[:, 0:2].copy()
    speed = np.linalg.norm(vel, axis=1, keepdims=True)
    vel = np.where(speed > config.max_speed, vel / (speed + 1e-8) * config.max_speed, vel)
    next_pos = base_pos + vel * config.dt
    low, high = base_world_bounds(config)
    next_pos = np.clip(next_pos, low, high).astype(np.float32)
    next_pos = enforce_base_object_collision_batch(next_pos, object_pos, config)

    # Rotation
    omega = np.clip(action[:, 2], -config.max_yaw_rate, config.max_yaw_rate)
    next_yaw = ((yaw + omega * config.dt + np.pi) % (2.0 * np.pi)) - np.pi

    # Step reward
    prev_diff = np.abs(np.linalg.norm(base_pos - object_pos, axis=1) - float(config.desired_distance))
    new_diff = np.abs(np.linalg.norm(next_pos - object_pos, axis=1) - float(config.desired_distance))
    rewards = np.full(n, -float(config.step_penalty), dtype=np.float32)
    rewards += float(config.progress_weight) * (prev_diff - new_diff).astype(np.float32)

    # Terminate bonus
    terminate_signal = action[:, 3]
    fire = terminate_signal >= float(config.interact_threshold)
    if np.any(fire):
        sigma = float(config.gaussian_sigma)
        bonus = float(config.goal_reward) * np.exp(-(new_diff ** 2) / (sigma * sigma))
        rewards[fire] += bonus[fire].astype(np.float32)

    # Build next obs
    next_obs = np.stack([
        next_pos[:, 0],
        next_pos[:, 1],
        np.sin(next_yaw),
        np.cos(next_yaw),
        object_pos[:, 0],
        object_pos[:, 1],
    ], axis=1).astype(np.float32)
    return next_obs, rewards.astype(np.float32), fire.astype(bool)


__all__ = [
    "EnvConfig",
    "MobileArmLeaderEnv",
    "mobile_arm_leader_dynamics_batch",
]
