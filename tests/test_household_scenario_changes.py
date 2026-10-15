"""Env-level tests for the two household scenario changes:

1. Cleaner (leader) post-pickup aid window: pickup no longer terminates;
   the episode runs ``post_pickup_steps`` more steps with penalties-only
   reward, then terminates. ``picked_up_dirt`` latches True.
2. Navigator (follower) centered goal: the goal is the centered half of the
   lower boundary; distance is 2D-to-segment and reaching requires
   ``|x - mid_x| <= goal_x_half_extent``.
"""

from __future__ import annotations

import numpy as np

from critic_sharing.config.schema import (
    HouseholdCleanerEnvConfig,
    HouseholdNavEnvConfig,
)
from critic_sharing.scenarios.household.nav_env import HouseholdNavigatorEnv
from critic_sharing.scenarios.household.push_env import HouseholdCleanerEnv


# ---------------------------------------------------------------------------
# Change 1 — cleaner post-pickup aid window
# ---------------------------------------------------------------------------

def _force_pickup_env(post_pickup_steps: int) -> HouseholdCleanerEnv:
    cfg = HouseholdCleanerEnvConfig(post_pickup_steps=post_pickup_steps)
    env = HouseholdCleanerEnv(cfg)
    env.reset(seed=0)
    # Force a pickup on the next step by putting the dirt at the cleaner center.
    env._dirt_pos = env._cleaner_pos.copy()
    return env


def test_pickup_starts_aid_window_not_termination():
    env = _force_pickup_env(post_pickup_steps=3)
    _, reward, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
    # Pickup step: collected, rewarded, but episode continues (window=3).
    assert info["picked_up_dirt"] is True
    assert info["in_aid_phase"] is False
    assert term is False and trunc is False
    assert reward > env.config.pickup_reward - 1.0  # got the pickup reward


def test_aid_phase_reward_is_penalties_only_and_latches_success():
    env = _force_pickup_env(post_pickup_steps=3)
    env.step(np.zeros(3, dtype=np.float32))  # pickup at step 1
    # Steps 2 and 3 are aid phase: penalties only, success latched, no term.
    for _ in range(2):
        _, reward, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
        assert info["in_aid_phase"] is True
        assert info["picked_up_dirt"] is True  # latched
        assert term is False
        # No positive reward; with zero action / no wall hit -> -step_penalty.
        assert reward <= 0.0
        assert abs(reward + env.config.step_penalty) < 1e-5
    # Step 4 == pickup_step + 3 -> terminate.
    _, _, term, _, info = env.step(np.zeros(3, dtype=np.float32))
    assert term is True
    assert info["picked_up_dirt"] is True


def test_post_pickup_zero_reproduces_terminate_on_pickup():
    env = _force_pickup_env(post_pickup_steps=0)
    _, _, term, _, info = env.step(np.zeros(3, dtype=np.float32))
    assert info["picked_up_dirt"] is True
    assert term is True  # legacy: pickup terminates immediately


def test_dirt_moves_out_of_arena_after_pickup():
    env = _force_pickup_env(post_pickup_steps=5)
    env.step(np.zeros(3, dtype=np.float32))  # pickup
    # Dirt sentinel is outside the arena so no re-pickup can occur.
    assert env._dirt_pos[0] > env.config.world_x_max
    assert env._dirt_pos[1] > env.config.world_y_max


# ---------------------------------------------------------------------------
# Change 2 — navigator centered goal
# ---------------------------------------------------------------------------

def test_goal_half_extent_default_is_centered_half():
    env = HouseholdNavigatorEnv(HouseholdNavEnvConfig())
    env.reset(seed=0)
    # (world_x_max - world_x_min) / 4 = (1 - (-1)) / 4 = 0.5
    assert abs(env._goal_hx() - 0.5) < 1e-9


def test_dist_to_goal_is_2d_to_centered_segment():
    env = HouseholdNavigatorEnv(HouseholdNavEnvConfig())
    env.reset(seed=0)
    low = env._goal_low_y()
    hx = env._goal_hx()
    # On the centered segment -> distance 0.
    assert env._dist_to_goal(np.array([0.0, low], dtype=np.float32)) == 0.0
    assert env._dist_to_goal(np.array([hx, low], dtype=np.float32)) == 0.0
    # Off-center at the bottom -> horizontal distance past the band.
    d = env._dist_to_goal(np.array([0.9, low], dtype=np.float32))
    assert abs(d - (0.9 - hx)) < 1e-5
    # Above the goal, centered -> vertical distance.
    d = env._dist_to_goal(np.array([0.0, low + 0.3], dtype=np.float32))
    assert abs(d - 0.3) < 1e-5


def test_reached_goal_requires_centered_x():
    cfg = HouseholdNavEnvConfig()
    env = HouseholdNavigatorEnv(cfg)
    env.reset(seed=0)
    low = env._goal_low_y()
    hx = env._goal_hx()
    # Centered, at/below the goal line -> reached.
    env._pos = np.array([0.0, low - 0.02], dtype=np.float32)
    _, _, term, _, info = env.step(np.zeros(3, dtype=np.float32))
    assert info["reached_goal"] is True and term is True
    # Off-center at the same depth -> not reached.
    env.reset(seed=1)
    env._pos = np.array([hx + 0.3, low - 0.02], dtype=np.float32)
    _, _, term, _, info = env.step(np.zeros(3, dtype=np.float32))
    assert info["reached_goal"] is False
