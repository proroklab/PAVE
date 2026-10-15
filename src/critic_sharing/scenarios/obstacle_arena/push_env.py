"""Pushing environment for the obstacle-arena scenario."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from critic_sharing.config.schema import PushEnvConfig as EnvConfig
from critic_sharing.scenarios.obstacle_arena.common import (
    enforce_wall_batch,
    wall_segments,
)


def _agent_bounds(config: EnvConfig) -> tuple[float, float]:
    low = float(config.world_min + config.agent_radius)
    high = float(config.world_max - config.agent_radius)
    if low > high:
        raise ValueError(
            "Invalid world/agent radius: world_min + agent_radius > world_max - agent_radius."
        )
    return low, high


def _obstacle_bounds(config: EnvConfig) -> tuple[float, float]:
    low = float(config.world_min + config.obstacle_radius + 1e-3)
    high = float(config.world_max - config.obstacle_radius - 1e-3)
    if low > high:
        raise ValueError(
            "Invalid world/agent radius: world_min + agent_radius > world_max - agent_radius."
        )
    return low, high


class PushObstacleEnv(gym.Env):
    # Per-step ``info`` key naming the bool that flips True on a successful
    # episode (pushed obstacle into wall). Read by
    # :func:`critic_sharing.utils.calibration.calibrate_range`.
    SUCCESS_INFO_KEY = "hit_wall"

    """
    Agent pushes a circular obstacle into any wall of the world.
    Observation: [agent_x, agent_y, obstacle_x, obstacle_y]
    Action: desired velocity [vx, vy]
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[EnvConfig] = None):
        super().__init__()
        self.config = config or EnvConfig()

        obs_low = np.array(
            [
                self.config.world_min,
                self.config.world_min,
                self.config.world_min,
                self.config.world_min,
            ],
            dtype=np.float32,
        )
        obs_high = np.array(
            [
                self.config.world_max,
                self.config.world_max,
                self.config.world_max,
                self.config.world_max,
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.config.max_speed,
            high=self.config.max_speed,
            shape=(2,),
            dtype=np.float32,
        )

        self._agent_pos = np.zeros(2, dtype=np.float32)
        self._obstacle_pos = np.zeros(2, dtype=np.float32)
        self._steps = 0

    def _sample_obstacle_pos(self) -> np.ndarray:
        low, high = _obstacle_bounds(self.config)
        # Obstacle spawns below wall
        y_max = float(self.config.wall_y) - self.config.obstacle_radius - 1e-3
        pos = self.np_random.uniform(low, high, size=(2,)).astype(np.float32)
        pos[1] = float(self.np_random.uniform(low, min(high, y_max)))
        return pos

    def _sample_agent_pos(self) -> np.ndarray:
        low, high = _agent_bounds(self.config)
        # Leader spawns below wall
        y_max = float(self.config.wall_y) - self.config.agent_radius - 1e-3
        pos = self.np_random.uniform(low, high, size=(2,)).astype(np.float32)
        pos[1] = float(self.np_random.uniform(low, min(high, y_max)))
        return pos

    def _get_obs(self) -> np.ndarray:
        return np.array(
            [
                self._agent_pos[0],
                self._agent_pos[1],
                self._obstacle_pos[0],
                self._obstacle_pos[1],
            ],
            dtype=np.float32,
        )

    def _distance_to_wall(self, pos: np.ndarray) -> float:
        low = self.config.world_min + self.config.obstacle_radius
        high = self.config.world_max - self.config.obstacle_radius
        d = float(min(pos[0] - low, high - pos[0], pos[1] - low, high - pos[1]))
        # Also consider distance to middle wall segments
        wy = float(self.config.wall_y)
        r = self.config.obstacle_radius
        x = float(pos[0])
        for x1, x2, _ in wall_segments(self.config):
            if x + r > x1 and x - r < x2:
                # Obstacle horizontally overlaps this segment
                d = min(d, abs(float(pos[1]) - wy) - r)
        return d

    def _hit_wall(self, pos: np.ndarray) -> bool:
        low = self.config.world_min + self.config.obstacle_radius
        high = self.config.world_max - self.config.obstacle_radius
        hit_boundary = bool(pos[0] <= low or pos[0] >= high or pos[1] <= low or pos[1] >= high)
        if hit_boundary:
            return True
        # Check middle wall contact
        wy = float(self.config.wall_y)
        r = self.config.obstacle_radius
        x = float(pos[0])
        y = float(pos[1])
        if abs(y - wy) <= r:
            for x1, x2, _ in wall_segments(self.config):
                if x + r > x1 and x - r < x2:
                    return True
        return False

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._steps = 0
        obstacle_pos = None
        agent_pos = None
        if options is not None:
            obstacle_pos = options.get("obstacle_pos")
            agent_pos = options.get("agent_pos")
        if obstacle_pos is None:
            obstacle_pos = self._sample_obstacle_pos()
        self._obstacle_pos = np.asarray(obstacle_pos, dtype=np.float32)
        self._obstacle_pos = np.clip(
            self._obstacle_pos, self.config.world_min, self.config.world_max
        )
        # Reject agent positions that overlap with the obstacle
        min_dist = self.config.obstacle_radius + self.config.agent_radius
        if agent_pos is None:
            for _ in range(1000):
                agent_pos = self._sample_agent_pos()
                if np.linalg.norm(agent_pos - self._obstacle_pos) >= min_dist:
                    break
        else:
            # Validate externally-provided position
            agent_pos = np.asarray(agent_pos, dtype=np.float32)
            if np.linalg.norm(agent_pos - self._obstacle_pos) < min_dist:
                # Nudge agent away from obstacle
                diff = agent_pos - self._obstacle_pos
                dist = np.linalg.norm(diff)
                if dist < 1e-8:
                    diff = np.array([1.0, 0.0], dtype=np.float32)
                    dist = 1.0
                agent_pos = self._obstacle_pos + diff / dist * (min_dist + 1e-3)
        agent_low, agent_high = _agent_bounds(self.config)
        self._agent_pos = np.clip(np.asarray(agent_pos, dtype=np.float32), agent_low, agent_high)
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self._steps += 1
        action = np.asarray(action, dtype=np.float32)

        prev_wall_dist = self._distance_to_wall(self._obstacle_pos)
        next_agent, next_obstacle = pusher_dynamics_batch(
            self._agent_pos[None, :],
            self._obstacle_pos[None, :],
            action[None, :],
            self.config,
        )
        self._agent_pos = next_agent[0]
        self._obstacle_pos = next_obstacle[0]

        wall_dist = self._distance_to_wall(self._obstacle_pos)
        reward = float(prev_wall_dist - wall_dist)
        hit_wall = self._hit_wall(self._obstacle_pos)
        if hit_wall:
            reward += self.config.goal_reward

        terminated = hit_wall
        truncated = self._steps >= self.config.max_steps
        info = {
            "wall_distance": float(wall_dist),
            "hit_wall": bool(hit_wall),
        }
        return self._get_obs(), float(reward), terminated, truncated, info


def pusher_dynamics_batch(
    agent_pos: np.ndarray,
    obstacle_pos: np.ndarray,
    action: np.ndarray,
    config: EnvConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict next agent/obstacle states for a batch of transitions."""

    agent_pos = np.asarray(agent_pos, dtype=np.float32)
    obstacle_pos = np.asarray(obstacle_pos, dtype=np.float32)
    action = np.asarray(action, dtype=np.float32)
    if agent_pos.ndim != 2 or obstacle_pos.ndim != 2 or action.ndim != 2:
        raise ValueError("agent_pos, obstacle_pos, and action must have shape (B, 2).")
    if not (agent_pos.shape == obstacle_pos.shape == action.shape):
        raise ValueError("agent_pos, obstacle_pos, and action must have identical shapes.")
    if agent_pos.shape[1] != 2:
        raise ValueError("Expected state/action vectors with shape (B, 2).")

    speeds = np.linalg.norm(action, axis=1, keepdims=True)
    clipped_action = np.where(
        speeds > config.max_speed,
        action / (speeds + 1e-8) * config.max_speed,
        action,
    )

    agent_delta = clipped_action * config.dt
    next_agent = agent_pos + agent_delta
    agent_low, agent_high = _agent_bounds(config)
    next_agent = np.clip(next_agent, agent_low, agent_high)
    # Enforce wall for agent
    next_agent = enforce_wall_batch(next_agent, agent_pos, config.agent_radius, config)

    vec = obstacle_pos - next_agent
    dist = np.linalg.norm(vec, axis=1)
    contact_dist = config.agent_radius + config.obstacle_radius
    next_obstacle = obstacle_pos.copy()
    mask = dist < contact_dist
    if np.any(mask):
        next_obstacle[mask] = next_obstacle[mask] + agent_delta[mask] * config.push_strength
        mask_pos = mask & (dist > 1e-8)
        if np.any(mask_pos):
            next_obstacle[mask_pos] = (
                next_agent[mask_pos] + vec[mask_pos] / dist[mask_pos][:, None] * contact_dist
            )
        mask_zero = mask & ~mask_pos
        if np.any(mask_zero):
            next_obstacle[mask_zero] = next_obstacle[mask_zero] + np.array(
                [contact_dist, 0.0], dtype=np.float32
            )

    next_obstacle = np.clip(next_obstacle, config.world_min, config.world_max)
    # Enforce wall for obstacle
    next_obstacle = enforce_wall_batch(
        next_obstacle, obstacle_pos, config.obstacle_radius, config,
    )
    return next_agent.astype(np.float32), next_obstacle.astype(np.float32)


def obstacle_wall_distance_batch(obstacle_pos: np.ndarray, config: EnvConfig) -> np.ndarray:
    """Batched :meth:`PushObstacleEnv._distance_to_wall`, shape ``(B,)``."""
    pos = np.asarray(obstacle_pos, dtype=np.float64)
    low = config.world_min + config.obstacle_radius
    high = config.world_max - config.obstacle_radius
    d = np.minimum.reduce([pos[:, 0] - low, high - pos[:, 0], pos[:, 1] - low, high - pos[:, 1]])
    wy = float(config.wall_y)
    r = config.obstacle_radius
    for x1, x2, _ in wall_segments(config):
        overlaps = (pos[:, 0] + r > x1) & (pos[:, 0] - r < x2)
        d = np.where(overlaps, np.minimum(d, np.abs(pos[:, 1] - wy) - r), d)
    return d


def obstacle_hit_wall_batch(obstacle_pos: np.ndarray, config: EnvConfig) -> np.ndarray:
    """Batched :meth:`PushObstacleEnv._hit_wall`, shape ``(B,)`` bool."""
    pos = np.asarray(obstacle_pos, dtype=np.float32)
    low = config.world_min + config.obstacle_radius
    high = config.world_max - config.obstacle_radius
    hit_boundary = (
        (pos[:, 0] <= low) | (pos[:, 0] >= high) | (pos[:, 1] <= low) | (pos[:, 1] >= high)
    )
    wy = float(config.wall_y)
    r = config.obstacle_radius
    near_wall = np.abs(pos[:, 1] - wy) <= r
    overlaps_segment = np.zeros(pos.shape[0], dtype=bool)
    for x1, x2, _ in wall_segments(config):
        overlaps_segment |= (pos[:, 0] + r > x1) & (pos[:, 0] - r < x2)
    return hit_boundary | (near_wall & overlaps_segment)


def push_step_batch(
    agent_pos: np.ndarray,
    obstacle_pos: np.ndarray,
    action: np.ndarray,
    config: EnvConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Batched :meth:`PushObstacleEnv.step` transition, reward and termination.

    Returns ``(next_agent, next_obstacle, rewards, dones)``: the reward is the
    env's wall-distance progress plus ``goal_reward`` on wall contact, and
    ``dones`` is the env's ``terminated`` (obstacle hit a wall). Truncation by
    ``max_steps`` is left to the caller.
    """
    prev_dist = obstacle_wall_distance_batch(obstacle_pos, config)
    next_agent, next_obstacle = pusher_dynamics_batch(agent_pos, obstacle_pos, action, config)
    dones = obstacle_hit_wall_batch(next_obstacle, config)
    rewards = prev_dist - obstacle_wall_distance_batch(next_obstacle, config)
    rewards = rewards + np.where(dones, float(config.goal_reward), 0.0)
    return next_agent, next_obstacle, rewards.astype(np.float32), dones


def rollout_trajectory(env: gym.Env, model, max_steps: int):
    obs, _ = env.reset()
    agent_traj = [obs[:2].copy()]
    obstacle_traj = [obs[2:4].copy()]
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        agent_traj.append(obs[:2].copy())
        obstacle_traj.append(obs[2:4].copy())
        if terminated or truncated:
            break
    return np.array(agent_traj, dtype=np.float32), np.array(obstacle_traj, dtype=np.float32)


__all__ = [
    "EnvConfig",
    "PushObstacleEnv",
    "obstacle_hit_wall_batch",
    "obstacle_wall_distance_batch",
    "push_step_batch",
    "pusher_dynamics_batch",
    "rollout_trajectory",
]
