"""Scenario-local config re-exports for the mobile_arm package."""

from critic_sharing.config.schema import (
    MobileArmFollowerEnvConfig,
    MobileArmLeaderEnvConfig,
)

__all__ = ["MobileArmFollowerEnvConfig", "MobileArmLeaderEnvConfig"]
