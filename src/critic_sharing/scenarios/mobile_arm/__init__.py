"""Mobile base + 3-DOF arm critic-sharing scenario.

Public re-exports:

* :class:`MobileArmLeaderEnv` — leader env: 2D mobile base, terminate
  action, Gaussian terminate-bonus peaked at ``desired_distance``.
* :class:`MobileArmFollowerEnv` — follower env: 3-DOF planar arm in
  base frame, auto-pickup off the ground, auto-drop on tray.
* :func:`mobile_arm_leader_dynamics_batch` — batched analytical leader
  dynamics for use as the dynamics function in
  :class:`critic_sharing.planning.cem.CEMController`.
* :func:`build_mobile_arm_follower_value_fn` — per-episode closure that
  transforms ``(leader_pose, object_world)`` into a follower-shaped obs
  and queries the follower's critic.
"""

from critic_sharing.scenarios.mobile_arm.config import (
    MobileArmFollowerEnvConfig,
    MobileArmLeaderEnvConfig,
)
from critic_sharing.scenarios.mobile_arm.follower_env import MobileArmFollowerEnv
from critic_sharing.scenarios.mobile_arm.leader_env import (
    MobileArmLeaderEnv,
    mobile_arm_leader_dynamics_batch,
)
from critic_sharing.scenarios.mobile_arm.value_query import (
    build_mobile_arm_follower_value_fn,
)

__all__ = [
    "MobileArmFollowerEnv",
    "MobileArmFollowerEnvConfig",
    "MobileArmLeaderEnv",
    "MobileArmLeaderEnvConfig",
    "build_mobile_arm_follower_value_fn",
    "mobile_arm_leader_dynamics_batch",
]
