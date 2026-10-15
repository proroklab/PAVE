"""Per-episode follower value query for the warehouse scenario.

The leader's CEM planner uses ``V_follower(object_pos)`` as a terminal value.
For a given joint episode the follower spawn ``(s_f_pos, s_f_theta)`` and the
goal ``g`` are fixed and known to the planner — we bake them into a closure
and evaluate the follower's PPO critic directly. No marginalisation, no
distillation: a single PPO forward per terminal-state query.

The returned callable is compatible with
:class:`critic_sharing.planning.costs.JointUtilityCost`'s
``follower_value_fn`` parameter — it accepts a ``(B, *)`` batch of *leader*
terminal observations, extracts the object position from each, builds the
counterfactual follower observation, and returns ``(B,)`` values.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch as th
from stable_baselines3 import PPO

from critic_sharing.config.schema import WarehouseFollowerEnvConfig


# Indices in the *leader* observation vector (see leader_env.py):
#   [agent_x, agent_y, sin(theta), cos(theta), object_x, object_y, carry_flag]
LEADER_OBJ_X_IDX = 4
LEADER_OBJ_Y_IDX = 5


def build_warehouse_follower_value_fn(
    follower_model: PPO,
    follower_config: WarehouseFollowerEnvConfig,
    follower_spawn_pos: np.ndarray,
    follower_spawn_theta: float,
    goal: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a callable mapping leader terminal obs batches to follower values.

    Args:
        follower_model: Trained follower PPO model.
        follower_config: Follower env config (currently unused in the obs build,
            kept for symmetry with other scenarios that need world dims).
        follower_spawn_pos: ``(2,)`` follower agent ``(x, y)`` for the episode.
        follower_spawn_theta: Follower agent yaw at spawn (radians).
        goal: ``(2,)`` goal ``(x, y)`` for the episode.

    Returns:
        ``value_fn(leader_obs_batch) -> values`` of shape ``(B,)``.

        ``leader_obs_batch`` is shape ``(B, 7)`` with columns
        ``[agent_x, agent_y, sin(theta), cos(theta), object_x, object_y, carry_flag]``.
        Only the object position columns are read; the rest is ignored.

        The follower obs constructed for the critic is shape ``(B, 9)``:
        ``[s_fx, s_fy, sin(s_fθ), cos(s_fθ), object_x, object_y, 0, goal_x, goal_y]``
        — i.e., the follower at its spawn, not yet carrying, with the candidate
        object position from the leader rollout, and the fixed goal.
    """
    del follower_config  # currently unused; kept for API symmetry

    s_x = float(follower_spawn_pos[0])
    s_y = float(follower_spawn_pos[1])
    s_sin = float(np.sin(follower_spawn_theta))
    s_cos = float(np.cos(follower_spawn_theta))
    g_x = float(goal[0])
    g_y = float(goal[1])

    def value_fn(leader_obs_batch: np.ndarray) -> np.ndarray:
        leader_obs_batch = np.asarray(leader_obs_batch, dtype=np.float32)
        if leader_obs_batch.ndim == 1:
            leader_obs_batch = leader_obs_batch[None, :]
        b = leader_obs_batch.shape[0]
        obj_x = leader_obs_batch[:, LEADER_OBJ_X_IDX]
        obj_y = leader_obs_batch[:, LEADER_OBJ_Y_IDX]

        cf_obs = np.empty((b, 9), dtype=np.float32)
        cf_obs[:, 0] = s_x
        cf_obs[:, 1] = s_y
        cf_obs[:, 2] = s_sin
        cf_obs[:, 3] = s_cos
        cf_obs[:, 4] = obj_x
        cf_obs[:, 5] = obj_y
        cf_obs[:, 6] = 0.0  # carry_flag = 0 (counterfactual fresh follower)
        cf_obs[:, 7] = g_x
        cf_obs[:, 8] = g_y

        obs_tensor, _ = follower_model.policy.obs_to_tensor(cf_obs)
        with th.no_grad():
            values = follower_model.policy.predict_values(obs_tensor)
        return values.detach().cpu().numpy().reshape(-1)

    return value_fn


__all__ = ["build_warehouse_follower_value_fn"]
