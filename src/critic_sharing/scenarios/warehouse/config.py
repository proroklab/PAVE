"""Scenario-local config re-exports for the warehouse package."""

from critic_sharing.config.schema import (
    WarehouseFollowerEnvConfig,
    WarehouseLeaderEnvConfig,
)

__all__ = ["WarehouseFollowerEnvConfig", "WarehouseLeaderEnvConfig"]
