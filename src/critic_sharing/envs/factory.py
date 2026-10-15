"""Build a scenario's environment for a role from its config."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from critic_sharing.config.resolver import (
    apply_env_overrides,
    load_env_config,
    load_train_config,
)

_WAREHOUSE = ("warehouse", "warehouse_blocked", "warehouse_pickup_blocked",
              "warehouse_pickup_unseen")

# (scenario, role) -> "module:Class"
_ENV_CLASSES: dict[tuple[str, str], str] = {
    ("household", "leader"): "critic_sharing.scenarios.household.push_env:HouseholdCleanerEnv",
    ("household", "follower"): "critic_sharing.scenarios.household.nav_env:HouseholdNavigatorEnv",
    ("obstacle_arena", "leader"): "critic_sharing.scenarios.obstacle_arena.push_env:PushObstacleEnv",
    ("obstacle_arena", "follower"): "critic_sharing.scenarios.obstacle_arena.nav_env:OneObstacleReachEnv",
    ("mobile_arm", "leader"): "critic_sharing.scenarios.mobile_arm.leader_env:MobileArmLeaderEnv",
    ("mobile_arm", "follower"): "critic_sharing.scenarios.mobile_arm.follower_env:MobileArmFollowerEnv",
    **{(w, "leader"): "critic_sharing.scenarios.warehouse.leader_env:WarehouseLeaderEnv"
       for w in _WAREHOUSE},
    **{(w, "follower"): "critic_sharing.scenarios.warehouse.follower_env:WarehouseFollowerEnv"
       for w in _WAREHOUSE},
}


def env_class(scenario: str, role: str) -> type:
    try:
        target = _ENV_CLASSES[(scenario, role)]
    except KeyError:
        raise ValueError(f"no env for scenario={scenario!r} role={role!r}") from None
    module, name = target.split(":")
    return getattr(import_module(module), name)


def training_env_config(scenario: str, role: str) -> Any:
    """The env config a policy of this (scenario, role) was trained on:
    the scenario's env config plus its training YAML's ``env_overrides``."""
    train = load_train_config(scenario=scenario, role=role,
                              config_override_path=None, cli_overrides={})
    return apply_env_overrides(load_env_config(scenario=scenario, role=role),
                               train.get("env_overrides"))


def make_env(scenario: str, role: str, config: Any = None):
    """The (scenario, role) env; default config: :func:`training_env_config`."""
    if config is None:
        config = training_env_config(scenario, role)
    return env_class(scenario, role)(config)
