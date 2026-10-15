"""Unit tests for the household episode termination predicates.

These exercise the pure-Python predicate factories — no ROS, no
matplotlib. The supervisor itself (which orchestrates rclpy spin +
viz + recording) is integration-tested elsewhere on hardware.
"""

from __future__ import annotations

import numpy as np
import pytest

from critic_sharing.config.schema import (
    HouseholdCleanerEnvConfig,
    HouseholdDeploymentConfig,
    HouseholdNavEnvConfig,
)
from critic_sharing.deployment.household_episode import (
    build_predicate_for_config,
    default_max_steps,
    make_cleaner_predicate,
    make_follower_predicate,
    make_no_op_predicate,
)


CLEANER_CFG = HouseholdCleanerEnvConfig()
FOLLOWER_CFG = HouseholdNavEnvConfig()
ARENA = (
    CLEANER_CFG.world_x_min, CLEANER_CFG.world_x_max,
    CLEANER_CFG.world_y_min, CLEANER_CFG.world_y_max,
)


def _cleaner_state(
    cleaner_xy=(0.0, 0.0),
    cleaner_yaw=0.0,
    dirt_xy=(1.0, 1.0),
    tick=0,
):
    return {
        "cleaner_xy_sim": np.asarray(cleaner_xy, dtype=np.float64),
        "cleaner_yaw_sim": float(cleaner_yaw),
        "table_xy_sim": np.asarray((0.5, 0.5), dtype=np.float64),
        "table_yaw_sim": 0.0,
        "dirt_xy_sim": np.asarray(dirt_xy, dtype=np.float64),
        "tick_count": int(tick),
        "last_action_sim": np.zeros(3),
        "role": "cleaner",
    }


# ---------------------------------------------------------------------------
# Cleaner predicate
# ---------------------------------------------------------------------------

def test_cleaner_pickup_when_dirt_inside_rect():
    pred = make_cleaner_predicate(
        cleaner_half_extents=CLEANER_CFG.cleaner_half_extents,
        arena_bounds=ARENA, arena_margin=0.1, max_steps=200,
    )
    # Cleaner at origin, axis-aligned; dirt at origin — inside the rect.
    # Default post_pickup_steps=0 -> pickup terminates immediately as
    # "aid_complete".
    assert pred(_cleaner_state(cleaner_xy=(0, 0), dirt_xy=(0, 0))) == "aid_complete"


def test_cleaner_no_pickup_when_dirt_outside_rect():
    pred = make_cleaner_predicate(
        cleaner_half_extents=CLEANER_CFG.cleaner_half_extents,
        arena_bounds=ARENA, arena_margin=0.1, max_steps=200,
    )
    # Dirt at (0.5, 0.5) is well outside a cleaner of half-extents
    # (0.165, 0.12) centered at origin.
    assert pred(_cleaner_state(cleaner_xy=(0, 0), dirt_xy=(0.5, 0.5))) is None


def test_cleaner_pickup_respects_rotated_rect():
    pred = make_cleaner_predicate(
        cleaner_half_extents=CLEANER_CFG.cleaner_half_extents,  # (0.165, 0.12)
        arena_bounds=ARENA, arena_margin=0.1, max_steps=200,
    )
    # 90° rotation swaps x/y extents — dirt at (0.13, 0) is now OUTSIDE
    # the rotated rect (rotated hx=0.12 < 0.13), but INSIDE the unrotated.
    # Sanity: at theta=0, |0.13| > 0.12? No, 0.13 > 0.12, so even
    # unrotated it's outside the y-half-extent... let me use clearer values.
    # Cleaner at origin, yaw=π/2. After rotation, the rect's local x-axis
    # points along world +y. A dirt point at (0, 0.16) is at local
    # (lx=0.16, ly=0) which is within hx=0.165 — inside.
    state = _cleaner_state(cleaner_xy=(0, 0), cleaner_yaw=np.pi / 2,
                           dirt_xy=(0.0, 0.16))
    assert pred(state) == "aid_complete"


def test_cleaner_out_of_arena_x_lo():
    pred = make_cleaner_predicate(
        cleaner_half_extents=CLEANER_CFG.cleaner_half_extents,
        arena_bounds=ARENA, arena_margin=0.1, max_steps=200,
    )
    # arena x_min = -1.0, margin 0.1, so x_lo = -0.9. cleaner at x=-0.95 → outside.
    assert pred(_cleaner_state(cleaner_xy=(-0.95, 0))) == "out_of_arena"


def test_cleaner_out_of_arena_x_hi():
    pred = make_cleaner_predicate(
        cleaner_half_extents=CLEANER_CFG.cleaner_half_extents,
        arena_bounds=ARENA, arena_margin=0.1, max_steps=200,
    )
    assert pred(_cleaner_state(cleaner_xy=(0.95, 0))) == "out_of_arena"


def test_cleaner_out_of_arena_y_bounds():
    pred = make_cleaner_predicate(
        cleaner_half_extents=CLEANER_CFG.cleaner_half_extents,
        arena_bounds=ARENA, arena_margin=0.1, max_steps=200,
    )
    # arena y_min=-1.25, margin 0.1, y_lo=-1.15
    assert pred(_cleaner_state(cleaner_xy=(0, -1.2))) == "out_of_arena"
    assert pred(_cleaner_state(cleaner_xy=(0, 1.2))) == "out_of_arena"


def test_cleaner_inside_arena_with_margin_is_fine():
    pred = make_cleaner_predicate(
        cleaner_half_extents=CLEANER_CFG.cleaner_half_extents,
        arena_bounds=ARENA, arena_margin=0.1, max_steps=200,
    )
    # Just inside the shrunk box.
    assert pred(_cleaner_state(cleaner_xy=(-0.85, 1.1))) is None


def test_cleaner_max_steps_fires_at_threshold():
    pred = make_cleaner_predicate(
        cleaner_half_extents=CLEANER_CFG.cleaner_half_extents,
        arena_bounds=ARENA, arena_margin=0.1, max_steps=200,
    )
    assert pred(_cleaner_state(tick=199)) is None
    assert pred(_cleaner_state(tick=200)) == "max_steps"
    assert pred(_cleaner_state(tick=10_000)) == "max_steps"


def test_cleaner_pickup_wins_over_max_steps():
    pred = make_cleaner_predicate(
        cleaner_half_extents=CLEANER_CFG.cleaner_half_extents,
        arena_bounds=ARENA, arena_margin=0.1, max_steps=10,
    )
    # Tick exceeds max_steps AND dirt is inside cleaner → pickup wins
    # (the "good" termination outcome).
    state = _cleaner_state(cleaner_xy=(0, 0), dirt_xy=(0, 0), tick=999)
    assert pred(state) == "aid_complete"


def test_cleaner_pickup_wins_over_out_of_arena():
    pred = make_cleaner_predicate(
        cleaner_half_extents=CLEANER_CFG.cleaner_half_extents,
        arena_bounds=ARENA, arena_margin=0.1, max_steps=200,
    )
    # If we picked up the dirt while leaving the arena, prefer the pickup
    # outcome (aid_complete with window 0) over out_of_arena.
    state = _cleaner_state(cleaner_xy=(-0.95, 0), dirt_xy=(-0.95, 0))
    assert pred(state) == "aid_complete"


# ---------------------------------------------------------------------------
# Follower predicate
# ---------------------------------------------------------------------------

def _follower_state(xy=(0.0, 0.0), tick=0):
    return {
        "cleaner_xy_sim": np.asarray(xy, dtype=np.float64),
        "cleaner_yaw_sim": 0.0,
        "table_xy_sim": np.asarray((0.5, 0.5), dtype=np.float64),
        "table_yaw_sim": 0.0,
        "dirt_xy_sim": np.asarray((0.5, 0.5), dtype=np.float64),
        "tick_count": int(tick),
        "last_action_sim": np.zeros(3),
        "role": "follower",
    }


def test_follower_goal_reached_at_lower_boundary():
    pred = make_follower_predicate(
        follower_half_extents=FOLLOWER_CFG.follower_half_extents,
        arena_bounds=ARENA, arena_margin=0.1, max_steps=250,
    )
    # follower_half_extents = (0.165, 0.12) → r = hypot(0.165, 0.12) ≈ 0.204
    # goal_y = world_y_min + r = -1.25 + 0.204 ≈ -1.046
    # A center at y=-1.05 (just below goal_y but still above y_lo=-1.15) → goal_reached.
    assert pred(_follower_state(xy=(0, -1.05))) == "goal_reached"


def test_follower_goal_requires_centered_x():
    pred = make_follower_predicate(
        follower_half_extents=FOLLOWER_CFG.follower_half_extents,
        arena_bounds=ARENA, arena_margin=0.1, max_steps=250,
    )
    # hx defaults to (x_max - x_min)/4 = 0.5; mid_x = 0.
    # y=-1.1 is past goal_y (~-1.046) and still inside the arena vertically
    # (y_lo=-1.15). Centered x -> goal; off-center x (|x|>0.5) -> NOT goal
    # (continues, returns None).
    assert pred(_follower_state(xy=(0.0, -1.1))) == "goal_reached"
    assert pred(_follower_state(xy=(0.7, -1.1))) is None
    assert pred(_follower_state(xy=(-0.7, -1.1))) is None


def test_follower_no_goal_when_above_threshold():
    pred = make_follower_predicate(
        follower_half_extents=FOLLOWER_CFG.follower_half_extents,
        arena_bounds=ARENA, arena_margin=0.1, max_steps=250,
    )
    # y=-1.0 is above goal_y ≈ -1.046 → no goal.
    assert pred(_follower_state(xy=(0, -1.0))) is None


def test_follower_out_of_arena_x():
    pred = make_follower_predicate(
        follower_half_extents=FOLLOWER_CFG.follower_half_extents,
        arena_bounds=ARENA, arena_margin=0.1, max_steps=250,
    )
    assert pred(_follower_state(xy=(0.95, 0))) == "out_of_arena"
    assert pred(_follower_state(xy=(-0.95, 0))) == "out_of_arena"


def test_follower_goal_wins_over_out_of_arena():
    pred = make_follower_predicate(
        follower_half_extents=FOLLOWER_CFG.follower_half_extents,
        arena_bounds=ARENA, arena_margin=0.1, max_steps=250,
    )
    # y=-1.2 is below both y_lo=-1.15 (out_of_arena) AND below goal_y≈-1.046.
    # Goal should win.
    assert pred(_follower_state(xy=(0, -1.2))) == "goal_reached"


def test_follower_collision_with_table_terminates():
    """Follower SAT-overlapping the table → 'collision'."""
    pred = make_follower_predicate(
        follower_half_extents=FOLLOWER_CFG.follower_half_extents,
        arena_bounds=ARENA, arena_margin=0.1, max_steps=250,
        env_config=FOLLOWER_CFG,
    )
    # Table is at (0.5, 0.5) in _follower_state with yaw 0.
    # Table half_extents from FOLLOWER_CFG default; place follower
    # right on top of the table.
    state = _follower_state(xy=(0.5, 0.5))
    assert pred(state) == "collision"


def test_follower_collision_with_wall_does_not_terminate():
    """Wall contact is intentionally NOT a terminating event — only the
    table is. Walls are static and the resolver keeps the robot inside,
    so wall scrapes are legal during a hardware rollout."""
    pred = make_follower_predicate(
        follower_half_extents=FOLLOWER_CFG.follower_half_extents,
        arena_bounds=ARENA, arena_margin=0.1, max_steps=250,
        env_config=FOLLOWER_CFG,
    )
    # Wall at wall_y=0.25, gap of 0.36 around x=0 → (0.5, 0.25) is
    # inside the right wall segment. Move the table somewhere far so
    # the table check can't trigger.
    state = _follower_state(xy=(0.5, 0.25))
    state["table_xy_sim"] = np.asarray((-5.0, -5.0), dtype=np.float64)
    assert pred(state) is None


def test_follower_collision_skipped_without_env_config():
    """Back-compat: omitting env_config disables the collision check."""
    pred = make_follower_predicate(
        follower_half_extents=FOLLOWER_CFG.follower_half_extents,
        arena_bounds=ARENA, arena_margin=0.1, max_steps=250,
    )
    # Same overlap as above → must NOT terminate.
    state = _follower_state(xy=(0.5, 0.5))
    assert pred(state) is None


def test_follower_goal_wins_over_collision():
    """If the follower is past the goal line AND overlapping the table
    (unlikely but possible), goal_reached should still win — it's
    checked first."""
    pred = make_follower_predicate(
        follower_half_extents=FOLLOWER_CFG.follower_half_extents,
        arena_bounds=ARENA, arena_margin=0.1, max_steps=250,
        env_config=FOLLOWER_CFG,
    )
    # Force the table on top of the goal region.
    state = _follower_state(xy=(0, -1.2))
    state["table_xy_sim"] = np.asarray((0, -1.2), dtype=np.float64)
    assert pred(state) == "goal_reached"


def test_follower_max_steps():
    pred = make_follower_predicate(
        follower_half_extents=FOLLOWER_CFG.follower_half_extents,
        arena_bounds=ARENA, arena_margin=0.1, max_steps=50,
    )
    assert pred(_follower_state(tick=49)) is None
    assert pred(_follower_state(tick=50)) == "max_steps"


# ---------------------------------------------------------------------------
# No-op predicate (debug modes)
# ---------------------------------------------------------------------------

def test_no_op_never_fires():
    pred = make_no_op_predicate()
    assert pred(_cleaner_state(cleaner_xy=(99, 99), tick=10_000)) is None
    assert pred(_follower_state(xy=(0, -10), tick=10_000)) is None


# ---------------------------------------------------------------------------
# build_predicate_for_config dispatch
# ---------------------------------------------------------------------------

def test_default_max_steps_per_role():
    assert default_max_steps("cleaner") == HouseholdCleanerEnvConfig().max_steps
    assert default_max_steps("follower") == HouseholdNavEnvConfig().max_steps


def test_default_max_steps_rejects_unknown_role():
    with pytest.raises(ValueError):
        default_max_steps("bogus")


def test_build_predicate_dispatches_to_cleaner():
    dcfg = HouseholdDeploymentConfig(role="cleaner")
    pred = build_predicate_for_config(dcfg, CLEANER_CFG)
    n_aid = int(CLEANER_CFG.post_pickup_steps)
    assert n_aid > 0  # the new scenario runs an aid window
    # Pickup at tick 0 does NOT terminate — it starts the aid window.
    assert pred(_cleaner_state(cleaner_xy=(0, 0), dirt_xy=(0, 0), tick=0)) is None
    # ... still aiding partway through the window (dirt now elsewhere).
    assert pred(_cleaner_state(cleaner_xy=(0.3, 0.3), dirt_xy=(0.9, 0.9),
                               tick=n_aid - 1)) is None
    # aid_complete fires post_pickup_steps ticks after pickup.
    assert pred(_cleaner_state(cleaner_xy=(0.3, 0.3), dirt_xy=(0.9, 0.9),
                               tick=n_aid)) == "aid_complete"


def test_build_predicate_dispatches_to_follower():
    dcfg = HouseholdDeploymentConfig(role="follower")
    pred = build_predicate_for_config(dcfg, FOLLOWER_CFG)
    # Goal fires for follower, never "pickup".
    assert pred(_follower_state(xy=(0, -1.05))) == "goal_reached"


def test_build_predicate_debug_modes_return_no_op():
    dcfg = HouseholdDeploymentConfig(role="cleaner", goto_origin=True)
    pred = build_predicate_for_config(dcfg, CLEANER_CFG)
    # Even with dirt under the cleaner, no-op never fires.
    assert pred(_cleaner_state(cleaner_xy=(0, 0), dirt_xy=(0, 0), tick=10_000)) is None


def test_build_predicate_max_steps_override_applies():
    dcfg = HouseholdDeploymentConfig(role="cleaner", max_steps_override=5)
    pred = build_predicate_for_config(dcfg, CLEANER_CFG)
    assert pred(_cleaner_state(tick=4)) is None
    assert pred(_cleaner_state(tick=5)) == "max_steps"
