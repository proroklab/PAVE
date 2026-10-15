"""Navigation environment for the obstacle-arena scenario."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from critic_sharing.config.schema import NavEnvConfig as EnvConfig
from critic_sharing.scenarios.obstacle_arena.common import (
    agent_bounds,
    circle_collides_wall,
    enforce_wall,
    nav_goal_position,
)


class OneObstacleReachEnv(gym.Env):
    # Per-step ``info`` key naming the bool that flips True on a successful
    # episode (reached the goal without colliding). Read by
    # :func:`critic_sharing.utils.calibration.calibrate_range`.
    SUCCESS_INFO_KEY = "reached_goal"

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[EnvConfig] = None):
        super().__init__()
        self.config = config or EnvConfig()
        (
            self._obstacle_x_min,
            self._obstacle_x_max,
            self._obstacle_y_min,
            self._obstacle_y_max,
        ) = self.config.obstacle_bounds()
        obs_low = np.array(
            [
                self.config.world_min,
                self.config.world_min,
                self._obstacle_x_min,
                self._obstacle_y_min,
            ],
            dtype=np.float32,
        )
        obs_high = np.array(
            [
                self.config.world_max,
                self.config.world_max,
                self._obstacle_x_max,
                self._obstacle_y_max,
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

        self._pos = np.zeros(2, dtype=np.float32)
        self._goal = np.zeros(2, dtype=np.float32)
        self._obstacle_center = np.zeros(2, dtype=np.float32)
        self._steps = 0

    def _sample_agent_pos(self) -> np.ndarray:
        low, high = agent_bounds(self.config)
        return np.array(
            [self.np_random.uniform(low, high), self.np_random.uniform(low, high)],
            dtype=np.float32,
        )

    def _sample_obstacle_pos(self) -> np.ndarray:
        for _ in range(1000):
            pos = np.array(
                [
                    self.np_random.uniform(self._obstacle_x_min, self._obstacle_x_max),
                    self.np_random.uniform(self._obstacle_y_min, self._obstacle_y_max),
                ],
                dtype=np.float32,
            )
            if not circle_collides_wall(pos, self.config.obstacle_radius, self.config):
                return pos
        return pos

    def _get_obs(self) -> np.ndarray:
        return np.array(
            [
                self._pos[0],
                self._pos[1],
                self._obstacle_center[0],
                self._obstacle_center[1],
            ],
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._steps = 0
        agent_pos = None if options is None else options.get("agent_pos")
        obstacle_pos = None if options is None else options.get("obstacle_pos")
        if obstacle_pos is None:
            obstacle_pos = self._sample_obstacle_pos()
        obstacle_pos = np.asarray(obstacle_pos, dtype=np.float32)
        self._obstacle_center = np.array(
            [
                np.clip(obstacle_pos[0], self._obstacle_x_min, self._obstacle_x_max),
                np.clip(obstacle_pos[1], self._obstacle_y_min, self._obstacle_y_max),
            ],
            dtype=np.float32,
        )
        # Resolve agent position — reject samples that overlap with the obstacle
        # or collide with the wall. Follower can spawn on either side of the wall
        # (important for training experience below the wall).
        low, high = agent_bounds(self.config)
        min_dist = self.config.obstacle_radius + self.config.agent_radius
        if agent_pos is None:
            for _ in range(1000):
                agent_pos = self._sample_agent_pos()
                if np.linalg.norm(agent_pos - self._obstacle_center) < min_dist:
                    continue
                if circle_collides_wall(agent_pos, self.config.agent_radius, self.config):
                    continue
                break
        self._pos = np.clip(np.asarray(agent_pos, dtype=np.float32), low, high)
        self._goal = nav_goal_position(self.config)
        return self._get_obs(), {}

    def _dist_to_goal(self, pos: np.ndarray) -> float:
        """Distance to the goal region (entire lower boundary)."""
        low = float(self.config.world_min + self.config.agent_radius)
        return float(pos[1]) - low

    def _gap_penalty(self, pos: np.ndarray) -> float:
        """Penalty proportional to horizontal distance from the gap centre.

        Only active when the follower is above the wall. Returns 0.0 if
        the follower is below it.
        """
        wy = float(self.config.wall_y)
        r = float(self.config.agent_radius)
        if float(pos[1]) <= wy + r:
            return 0.0
        gap_x = float(self.config.gap_x)
        return abs(float(pos[0]) - gap_x)

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._steps += 1
        prev_dist = self._dist_to_goal(self._pos)
        action = np.asarray(action, dtype=np.float32)
        speed = np.linalg.norm(action)
        if speed > self.config.max_speed:
            action = action / (speed + 1e-8) * self.config.max_speed
        prev_pos = self._pos.copy()
        self._pos = self._pos + action * self.config.dt
        low, high = agent_bounds(self.config)
        self._pos = np.clip(self._pos, low, high)
        # Enforce wall collision
        self._pos = enforce_wall(self._pos, prev_pos, self.config.agent_radius, self.config)

        dist_to_obs = np.linalg.norm(self._pos - self._obstacle_center)
        collided = dist_to_obs < (self.config.obstacle_radius + self.config.agent_radius)
        dist = self._dist_to_goal(self._pos)
        reward = self.config.progress_weight * float(prev_dist - dist)
        reward += -self.config.distance_weight * float(dist)
        # Penalise horizontal distance from gap while above wall
        reward -= self.config.distance_weight * self._gap_penalty(self._pos)
        if collided:
            reward -= self.config.collision_penalty
        reached_goal = dist <= 0.0
        if reached_goal:
            reward += self.config.goal_reward
        terminated = reached_goal or collided
        truncated = self._steps >= self.config.max_steps
        info = {
            "distance": float(dist),
            "reached_goal": bool(reached_goal),
            "collided": bool(collided),
            "obstacle_x": float(self._obstacle_center[0]),
            "obstacle_y": float(self._obstacle_center[1]),
        }
        return self._get_obs(), float(reward), terminated, truncated, info


def _rollout_policy(
    env: gym.Env,
    model: Any,
    max_steps: int,
    obstacle_pos: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    options = {"obstacle_pos": obstacle_pos} if obstacle_pos is not None else None
    obs, _ = env.reset(options=options)
    traj = [obs[:2].copy()]
    goal = env._goal.copy()
    obstacle = env._obstacle_center.copy()
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        traj.append(obs[:2].copy())
        if terminated or truncated:
            break
    return np.asarray(traj, dtype=np.float32), goal, obstacle


def evaluate_value_grid(
    model: Any,
    config: EnvConfig,
    obstacle_pos: np.ndarray,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(config.world_min, config.world_max, grid_size, dtype=np.float32)
    ys = np.linspace(config.world_min, config.world_max, grid_size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    obs = np.stack(
        [
            grid_x.ravel(),
            grid_y.ravel(),
            np.full(grid_x.size, obstacle_pos[0], dtype=np.float32),
            np.full(grid_x.size, obstacle_pos[1], dtype=np.float32),
        ],
        axis=1,
    )
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, device=model.policy.device)
        values = model.policy.predict_values(obs_tensor).cpu().numpy()
    return grid_x, grid_y, values.reshape(grid_x.shape)


def rollout_trajectories(
    model: Any,
    config: EnvConfig,
    obstacle_pos: np.ndarray,
    num_rollouts: int,
    max_steps: int,
) -> list[np.ndarray]:
    env = OneObstacleReachEnv(config)
    trajectories: list[np.ndarray] = []
    for _ in range(num_rollouts):
        traj, _, _ = _rollout_policy(
            env,
            model,
            max_steps,
            obstacle_pos=np.asarray(obstacle_pos, dtype=np.float32),
        )
        trajectories.append(traj)
    return trajectories


__all__ = [
    "EnvConfig",
    "OneObstacleReachEnv",
    "evaluate_value_grid",
    "rollout_trajectories",
]
