"""Convenience re-exports for the household scenario configs."""

from critic_sharing.config.schema import (
    HouseholdCleanerEnvConfig as CleanerEnvConfig,
    HouseholdNavEnvConfig as NavEnvConfig,
)

__all__ = ["CleanerEnvConfig", "NavEnvConfig"]
