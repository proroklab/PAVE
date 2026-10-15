"""Shared helpers for CLI subcommands."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import typer


# ---------------------------------------------------------------------------
# Enums used across multiple CLI modules
# ---------------------------------------------------------------------------


class Scenario(str, Enum):
    obstacle_arena = "obstacle_arena"
    warehouse = "warehouse"
    # warehouse variant where the follower's auto-pickup is disabled at
    # one gap during training AND deployment (config:
    # ``pickup_disabled_gap_indices``). Geometry unchanged; the leader
    # can still drop at the disabled gap, but the follower silently
    # fails to pick up — the planner's V_F overestimates the gap because
    # it never observed pickup-failure outcomes there. OOD trust factor
    # is the protection mechanism.
    warehouse_pickup_blocked = "warehouse_pickup_blocked"
    # Same training as warehouse_pickup_blocked (follower never sees
    # objects at the disabled gap), but deployment leaves pickup
    # ENABLED at that gap. Tests V_F's prediction quality on a
    # deployment state V_F was not trained on, when the env mechanic
    # is normal — the actual pickup outcome depends on the policy's
    # zero-shot generalization. Shares the trained follower + VAE with
    # warehouse_pickup_blocked via the path map below.
    warehouse_pickup_unseen = "warehouse_pickup_unseen"
    mobile_arm = "mobile_arm"
    # Obstacle_arena reshaped: cleaner (leader) picks up dirt frozen
    # under a rectangular table by pushing it aside; follower navigates
    # to the lower boundary. All three rigid bodies are rotated
    # rectangles with full SAT contact + lever-arm torque on the
    # cleaner→table push. Wall + single gap is toggleable
    # (``wall_enabled``); long-table variant runs without the wall by
    # default.
    household = "household"


class Role(str, Enum):
    leader = "leader"
    follower = "follower"


# ---------------------------------------------------------------------------
# Task name mapping  (scenario, role) → artifact task name
# ---------------------------------------------------------------------------

TASK: dict[tuple[str, str], str] = {
    ("obstacle_arena", "leader"): "push",
    ("obstacle_arena", "follower"): "nav",
    ("warehouse", "leader"): "warehouse_leader",
    ("warehouse", "follower"): "warehouse_follower",
    # warehouse_pickup_blocked + warehouse_pickup_unseen share the same
    # leader (vanilla warehouse_leader) AND the same follower artifact
    # namespace (warehouse_pickup_blocked_follower). The two scenarios
    # differ only in deployment-time YAML — V1 keeps
    # ``pickup_disabled_gap_indices=[1]`` set in follower_env.yaml,
    # V2 leaves it empty so pickup is enabled at deployment. The
    # follower training never sees the disabled gap in either case.
    ("warehouse_pickup_blocked", "leader"): "warehouse_leader",
    ("warehouse_pickup_blocked", "follower"): "warehouse_pickup_blocked_follower",
    ("warehouse_pickup_unseen", "leader"): "warehouse_leader",
    ("warehouse_pickup_unseen", "follower"): "warehouse_pickup_blocked_follower",
    ("mobile_arm", "leader"): "mobile_arm_leader",
    ("mobile_arm", "follower"): "mobile_arm_follower",
    ("household", "leader"): "clean",
    ("household", "follower"): "nav",
}

FOLLOWER_TASK: dict[str, str] = {
    "obstacle_arena": "nav",
    "warehouse": "warehouse_follower",
    "warehouse_pickup_blocked": "warehouse_pickup_blocked_follower",
    "warehouse_pickup_unseen": "warehouse_pickup_blocked_follower",
    "mobile_arm": "mobile_arm_follower",
    "household": "nav",
}


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------


def resolve_model(
    path: Optional[Path],
    scenario: str,
    role: str = "follower",
) -> Path:
    """Return *path* if given, otherwise auto-discover the latest model."""
    if path is not None:
        return path
    from critic_sharing.utils.paths import find_latest_model

    task = TASK[(scenario, role)]
    resolved = find_latest_model(task=task)
    typer.echo("Using latest {role} model: {p}".format(role=role, p=resolved))
    return resolved


__all__ = [
    "FOLLOWER_TASK",
    "Role",
    "Scenario",
    "TASK",
    "resolve_model",
]
