"""Per-episode follower value query for the mobile_arm scenario.

The follower lives in **base frame** but the leader's planner queries V
during a leader rollout in **world frame**. The closure baked here
converts each leader-shaped obs to a follower-shaped obs by:

1. Extracting the leader's pose ``(lx, ly, l_yaw)`` and the object's
   world position ``(ox, oy)`` from the leader obs vector.
2. Transforming the object's position into the base frame:
   ``object_xy_base = R(-l_yaw) · (object_xy_world - base_pos_world)``.
3. Filling a 12-D follower obs with: ``[sin/cos(follower_yaw),
   sin/cos(θ₂_default), sin/cos(θ₃_default), sin/cos(θ₄_default),
   object_x_base, object_y_base, 0_object_z, carry=0]`` — the *follower
   starting state* baked in at episode setup time.
4. Querying the follower PPO critic on the resulting batch.

All four follower joints start at their config defaults at handoff
(``MobileArmFollowerEnv._sample_initial_theta`` returns the constant
``[θ₁_default, θ₂_default, θ₃_default, θ₄_default]``). The closure
therefore bakes ``follower_yaw=θ₁_default=0`` by default to match what
the actual handoff state will look like in
:func:`_run_episode_ma`. The optional ``follower_yaw`` argument exists
solely for the V-heatmap visualisation that scans θ₁ across panels —
it is *not* used at planning time.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch as th
from stable_baselines3 import PPO

from critic_sharing.config.schema import MobileArmFollowerEnvConfig


# Leader obs columns (see MobileArmLeaderEnv._get_obs):
#   [0]=base_x, [1]=base_y, [2]=sin(yaw), [3]=cos(yaw),
#   [4]=object_x, [5]=object_y
LEADER_BASE_X_IDX = 0
LEADER_BASE_Y_IDX = 1
LEADER_SIN_IDX = 2
LEADER_COS_IDX = 3
LEADER_OBJ_X_IDX = 4
LEADER_OBJ_Y_IDX = 5


def build_mobile_arm_follower_value_fn(
    follower_model: PPO,
    follower_config: MobileArmFollowerEnvConfig,
    follower_yaw: float | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a callable mapping leader-obs batches to follower critic values.

    Args:
        follower_model: trained follower PPO model.
        follower_config: follower env config (provides all joint defaults).
        follower_yaw: ``θ_1`` to bake in. Defaults to
            ``follower_config.theta1_default`` — i.e. the canonical
            starting pose the follower env resets to. Pass an explicit
            value to query V at a different arm orientation (e.g. for
            the 2x2 V-heatmap visualisation that varies θ_1 across panels).

    Returns:
        ``value_fn(leader_obs_batch) -> values`` of shape ``(B,)``.
    """
    if follower_yaw is None:
        follower_yaw = float(follower_config.theta1_default)
    s1 = float(follower_yaw)
    s2 = float(follower_config.theta2_default)
    s3 = float(follower_config.theta3_default)
    s4 = float(follower_config.theta4_default)
    sin1, cos1 = float(np.sin(s1)), float(np.cos(s1))
    sin2, cos2 = float(np.sin(s2)), float(np.cos(s2))
    sin3, cos3 = float(np.sin(s3)), float(np.cos(s3))
    sin4, cos4 = float(np.sin(s4)), float(np.cos(s4))

    def value_fn(leader_obs_batch: np.ndarray) -> np.ndarray:
        leader_obs_batch = np.asarray(leader_obs_batch, dtype=np.float32)
        if leader_obs_batch.ndim == 1:
            leader_obs_batch = leader_obs_batch[None, :]
        b = leader_obs_batch.shape[0]

        lx = leader_obs_batch[:, LEADER_BASE_X_IDX]
        ly = leader_obs_batch[:, LEADER_BASE_Y_IDX]
        l_sin = leader_obs_batch[:, LEADER_SIN_IDX]
        l_cos = leader_obs_batch[:, LEADER_COS_IDX]
        ox = leader_obs_batch[:, LEADER_OBJ_X_IDX]
        oy = leader_obs_batch[:, LEADER_OBJ_Y_IDX]

        # World → base-frame rotation: R(-l_yaw) on (object - base).
        dx = ox - lx
        dy = oy - ly
        ox_b = dx * l_cos + dy * l_sin
        oy_b = -dx * l_sin + dy * l_cos

        cf_obs = np.empty((b, 12), dtype=np.float32)
        cf_obs[:, 0] = sin1
        cf_obs[:, 1] = cos1
        cf_obs[:, 2] = sin2
        cf_obs[:, 3] = cos2
        cf_obs[:, 4] = sin3
        cf_obs[:, 5] = cos3
        cf_obs[:, 6] = sin4
        cf_obs[:, 7] = cos4
        cf_obs[:, 8] = ox_b
        cf_obs[:, 9] = oy_b
        cf_obs[:, 10] = 0.0  # object z (always on ground)
        cf_obs[:, 11] = 0.0  # carry_flag = 0 (counterfactual fresh follower)

        obs_tensor, _ = follower_model.policy.obs_to_tensor(cf_obs)
        with th.no_grad():
            values = follower_model.policy.predict_values(obs_tensor)
        return values.detach().cpu().numpy().reshape(-1)

    return value_fn


__all__ = ["build_mobile_arm_follower_value_fn"]
