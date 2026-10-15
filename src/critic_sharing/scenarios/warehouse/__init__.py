"""Continuous warehouse scenario with pickup/drop cooperation.

Mirrors the structure of ``obstacle_arena``:

* :class:`WarehouseLeaderEnv` — leader picks up an object in the lower half
  and drops it into one of three gaps in the wall.
* :class:`WarehouseFollowerEnv` — follower picks up the object at the gap and
  delivers it to a goal point on the boundary of one of three vertical
  shelves in the upper half.
* :func:`warehouse_leader_dynamics_batch` — analytical, batched one-step
  transition for use as the dynamics function in
  :class:`critic_sharing.planning.cem.CEMController`.
* :func:`build_warehouse_follower_value_fn` — per-episode closure that maps
  leader terminal obs batches to follower critic values, baked over the
  episode's fixed follower spawn and goal.
"""

from critic_sharing.scenarios.warehouse.config import (
    WarehouseFollowerEnvConfig,
    WarehouseLeaderEnvConfig,
)
from critic_sharing.scenarios.warehouse.follower_env import (
    GOAL_OFFSET_FROM_FACE,
    WarehouseFollowerEnv,
)
from critic_sharing.scenarios.warehouse.leader_env import (
    WarehouseLeaderEnv,
    warehouse_leader_dynamics_batch,
)
from critic_sharing.scenarios.warehouse.value_query import (
    build_warehouse_follower_value_fn,
)

__all__ = [
    "GOAL_OFFSET_FROM_FACE",
    "WarehouseFollowerEnv",
    "WarehouseFollowerEnvConfig",
    "WarehouseLeaderEnv",
    "WarehouseLeaderEnvConfig",
    "build_warehouse_follower_value_fn",
    "warehouse_leader_dynamics_batch",
]
