"""Follower env for the mobile_arm scenario.

The follower controls a 4-DOF arm mounted on a static mobile base. The
arm picks an object up off the ground and drops it in a tray attached
to the front of the base. All coordinates are in **base frame** (the
base is the origin, x-axis forward, z-axis up).

**Pickup/drop are automatic** at the end of each step:
* if not carrying and ``||ee - object|| ≤ eps_pickup`` → pickup;
* if carrying and ``||ee - tray|| ≤ eps_tray`` → drop, terminate with
  ``goal_reward``.

Action (4-D continuous, ``Box``): ``[dθ_1, dθ_2, dθ_3, dθ_4]`` joint
velocities in rad/s.

Observation (12-D):
  ``[sin(θ_1), cos(θ_1), sin(θ_2), cos(θ_2), sin(θ_3), cos(θ_3),
     sin(θ_4), cos(θ_4),
     object_x_base, object_y_base, object_z_base, carry_flag]``.
While carrying, ``object_pos = ee_pos`` so the policy sees the object
follow the EE.

Reward (per step):
  * ``-step_penalty``.
  * ``progress_weight * (prev_d - cur_d)`` — to-object when not carrying,
    to-tray when carrying.
  * ``+pickup_reward`` on successful auto-pickup.
  * ``+goal_reward`` and **terminate** on successful auto-drop at tray.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from critic_sharing.config.schema import MobileArmFollowerEnvConfig as EnvConfig
from critic_sharing.scenarios.mobile_arm.common import (
    arm_collides,
    forward_kinematics,
    sample_object_in_base_frame,
    tray_pos_base,
)


class MobileArmFollowerEnv(gym.Env):
    """Follower env for the mobile_arm scenario."""

    metadata = {"render_modes": []}
    # Per-step ``info`` key naming the bool that flips True on a successful
    # episode. The follower terminates on `dropped` (object placed in tray)
    # so the existing flag is exactly the success signal.
    SUCCESS_INFO_KEY = "dropped"

    def __init__(self, config: Optional[EnvConfig] = None):
        super().__init__()
        self.config = config or EnvConfig()

        big = np.array([
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,   # sin/cos for 4 joints
            self.config.world_max,                     # object_x_base
            self.config.world_max,                     # object_y_base
            self.config.world_max,                     # object_z_base
            1.0,                                       # carry
        ], dtype=np.float32)
        small = np.array([
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
            self.config.world_min,
            self.config.world_min,
            self.config.world_min,
            0.0,
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low=small, high=big, dtype=np.float32)

        act_low = np.full(4, -self.config.max_joint_rate, dtype=np.float32)
        act_high = np.full(4, self.config.max_joint_rate, dtype=np.float32)
        self.action_space = spaces.Box(low=act_low, high=act_high, dtype=np.float32)

        self._theta = np.zeros(4, dtype=np.float32)
        self._object_pos = np.zeros(3, dtype=np.float32)
        self._carry = False
        self._steps = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ee_pos(self) -> np.ndarray:
        return forward_kinematics(
            float(self._theta[0]),
            float(self._theta[1]),
            float(self._theta[2]),
            float(self._theta[3]),
            self.config,
        )

    def _shaping_distance(self) -> float:
        ee = self._ee_pos()
        target = tray_pos_base(self.config) if self._carry else self._object_pos
        return float(np.linalg.norm(ee - target))

    def _clip_joints(self, theta: np.ndarray) -> np.ndarray:
        cfg = self.config
        # θ_1 wraps to [-π, π] (arm yaw is unbounded — can rotate freely),
        # mirroring the leader's base-yaw convention. The clip on θ_1 stays
        # as defence-in-depth, but the wrap keeps it in the canonical range.
        t1_wrapped = float(((float(theta[0]) + np.pi) % (2.0 * np.pi)) - np.pi)
        return np.array([
            float(np.clip(t1_wrapped, cfg.theta1_low, cfg.theta1_high)),
            float(np.clip(theta[1], cfg.theta2_low, cfg.theta2_high)),
            float(np.clip(theta[2], cfg.theta3_low, cfg.theta3_high)),
            float(np.clip(theta[3], cfg.theta4_low, cfg.theta4_high)),
        ], dtype=np.float32)

    def _sample_initial_theta(self) -> np.ndarray:
        # All four joints start at their config defaults — the arm always
        # begins pointing at the tray (θ_1 = 0) with the upper arm vertical
        # and elbow / wrist bent forward 45°. This canonical starting pose
        # is what the value-fn closure bakes in.
        return np.array(
            [
                float(self.config.theta1_default),
                float(self.config.theta2_default),
                float(self.config.theta3_default),
                float(self.config.theta4_default),
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        # Object position is the EE position while carrying, so the policy
        # sees the object track the arm (matches warehouse convention).
        if self._carry:
            obj = self._ee_pos()
        else:
            obj = self._object_pos
        return np.array(
            [
                float(np.sin(self._theta[0])), float(np.cos(self._theta[0])),
                float(np.sin(self._theta[1])), float(np.cos(self._theta[1])),
                float(np.sin(self._theta[2])), float(np.cos(self._theta[2])),
                float(np.sin(self._theta[3])), float(np.cos(self._theta[3])),
                float(obj[0]), float(obj[1]), float(obj[2]),
                1.0 if self._carry else 0.0,
            ],
            dtype=np.float32,
        )

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

        theta = options.get("theta")
        if theta is None:
            theta = self._sample_initial_theta()
        self._theta = self._clip_joints(np.asarray(theta, dtype=np.float32))

        object_pos = options.get("object_pos")
        if object_pos is None:
            object_pos = sample_object_in_base_frame(
                self.np_random,
                self.config,
                float(self.config.object_r_min),
                float(self.config.object_r_max),
                phi_low=float(self.config.object_theta_low),
                phi_high=float(self.config.object_theta_high),
            )
        self._object_pos = np.asarray(object_pos, dtype=np.float32)
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self._steps += 1
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        prev_d = self._shaping_distance()
        prev_theta = self._theta.copy()

        # Joint velocity integration
        rate = float(self.config.max_joint_rate)
        dtheta = np.clip(action, -rate, rate) * float(self.config.dt)
        self._theta = self._clip_joints(self._theta + dtheta.astype(np.float32))

        # Arm-vs-environment collision: hard-revert if any joint endpoint
        # would land in the floor or inside the base box.
        floor_collision, base_collision = arm_collides(self._theta, self.config)
        if floor_collision or base_collision:
            self._theta = prev_theta

        # Auto-pickup
        picked_up = False
        if not self._carry:
            ee = self._ee_pos()
            if np.linalg.norm(ee - self._object_pos) <= float(self.config.eps_pickup):
                self._carry = True
                picked_up = True

        # Auto-drop on tray
        dropped = False
        if self._carry:
            ee = self._ee_pos()
            tray = tray_pos_base(self.config)
            if np.linalg.norm(ee - tray) <= float(self.config.eps_tray):
                # Drop succeeds only if we didn't *just* pick up — without
                # this, an object placed exactly at the tray would yield
                # an instant pickup+drop. Realistically the tray is ahead
                # of the arm's resting EE so this is moot, but cheap to
                # guard against.
                if not picked_up:
                    self._object_pos = tray.copy()
                    self._carry = False
                    dropped = True

        # Reward
        reward = -float(self.config.step_penalty)
        new_d = self._shaping_distance()
        reward += float(self.config.progress_weight) * float(prev_d - new_d)
        if picked_up:
            reward += float(self.config.pickup_reward)
        if floor_collision or base_collision:
            reward -= float(self.config.collision_penalty)
        terminated = False
        if dropped:
            reward += float(self.config.goal_reward)
            terminated = True

        truncated = self._steps >= int(self.config.max_steps)
        info = {
            "picked_up": bool(picked_up),
            "dropped": bool(dropped),
            "carry": bool(self._carry),
            "ee_z": float(self._ee_pos()[2]),
            "shaping_distance": float(new_d),
            "arm_floor_collision": bool(floor_collision),
            "arm_base_collision": bool(base_collision),
            "steps": int(self._steps),
        }
        return self._get_obs(), float(reward), terminated, truncated, info


__all__ = [
    "EnvConfig",
    "MobileArmFollowerEnv",
]
