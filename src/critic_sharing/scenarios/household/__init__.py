"""Household scenario: cleaner picks up dirt frozen under a rectangular table.

All three rigid bodies (cleaner, table, navigator) are rotated
rectangles with full SAT contact resolution and lever-arm torque on
the cleaner→table push. The follower's task is to reach the lower
boundary while avoiding (at training time) a static-per-episode
table; at joint-eval time the cleaner pushes the table around,
generating the leader-induced OOD states the trust-weighted V_F is
designed to handle.
"""

from critic_sharing.scenarios.household.config import (
    CleanerEnvConfig,
    NavEnvConfig,
)
from critic_sharing.scenarios.household.nav_env import HouseholdNavigatorEnv
from critic_sharing.scenarios.household.push_env import HouseholdCleanerEnv
from critic_sharing.scenarios.household.ood_query import (
    build_household_ood_metric,
    household_vae_obs_transform,
)
from critic_sharing.scenarios.household.value_query import (
    NavValueAggregator,
    SHARED_STATE_DIM,
    build_household_follower_value_fn,
    build_nav_value_approximator,
    build_vae_ood_metric,
)

__all__ = [
    "CleanerEnvConfig",
    "HouseholdCleanerEnv",
    "HouseholdNavigatorEnv",
    "NavEnvConfig",
    "NavValueAggregator",
    "SHARED_STATE_DIM",
    "build_household_follower_value_fn",
    "build_household_ood_metric",
    "build_nav_value_approximator",
    "build_vae_ood_metric",
    "household_vae_obs_transform",
]
