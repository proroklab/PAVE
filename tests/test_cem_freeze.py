"""Tests for CEM's per-row done-signal freeze in :class:`CEMController`.

Once a row's ``done`` fires at step t, subsequent steps must:
  * hold the row's state (next_obs[row] = current[row]),
  * emit zero reward,
  * never re-fire (the latch stays True even if the dynamics function
    re-reports done=True later).
"""

from __future__ import annotations

import numpy as np
import pytest

import gymnasium as gym

from critic_sharing.planning.cem import CEMController
from critic_sharing.planning.costs import Cost
from critic_sharing.planning.distributions import make_distribution


class _SinkCost(Cost):
    """No-op cost; we only care about the controller's internal trajectory."""

    def evaluate(self, obs, actions, rewards=None, terminated=None):
        return np.zeros(obs.shape[0], dtype=np.float32)


def _make_controller(dynamics_fn, action_dim: int = 1, horizon: int = 5, num_samples: int = 4):
    return CEMController(
        action_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
        dynamics_fn=dynamics_fn,
        cost=_SinkCost(),
        distribution=make_distribution("gaussian", noise_sigma=0.5, min_std=0.05),
        horizon=horizon,
        num_samples=num_samples,
        num_iters=1,
        elite_frac=0.5,
        seed=0,
    )


class TestRolloutFreeze:
    def test_terminated_row_holds_state_and_zero_reward(self):
        """A row that fires done at step 0 has zero reward + frozen state for the rest."""
        # 1-D obs, 1-D action. Each step: next_obs = obs + action; reward = 1.
        # Done fires at step 0 for row 0, never for the others.
        N = 3
        H = 4
        n_calls = {"i": 0}

        def dynamics_fn(obs_batch, act_batch):
            t = n_calls["i"]
            n_calls["i"] += 1
            next_obs = obs_batch + act_batch
            rewards = np.ones(obs_batch.shape[0], dtype=np.float32)
            dones = np.zeros(obs_batch.shape[0], dtype=bool)
            if t == 0:
                dones[0] = True  # row 0 terminates at step 0
            return next_obs, rewards, dones

        ctrl = _make_controller(dynamics_fn, action_dim=1, horizon=H, num_samples=N)
        obs = np.array([0.0], dtype=np.float32)
        # Hand-craft action sequences so we can predict the unfrozen trajectory.
        # All N rollouts use action = +1 every step.
        actions = np.ones((N, H, 1), dtype=np.float32)
        traj, rewards, _ = ctrl._rollout_batch(obs, actions)
        # traj shape: (N, H+1, 1)
        # Row 0 terminated at step 0: trajectory[0, t>=1] should equal trajectory[0, 1].
        assert traj[0, 1, 0] == pytest.approx(1.0)  # 0 + 1 (the terminating step)
        for t in range(2, H + 1):
            assert traj[0, t, 0] == pytest.approx(traj[0, 1, 0])
        # Row 0's rewards beyond step 0 must be zero.
        assert rewards[0, 0] == pytest.approx(1.0)  # the terminating step's reward
        for t in range(1, H):
            assert rewards[0, t] == pytest.approx(0.0)
        # Rows 1, 2 should be unaffected — full action accumulation.
        for r in (1, 2):
            for t in range(1, H + 1):
                assert traj[r, t, 0] == pytest.approx(float(t))
            np.testing.assert_allclose(rewards[r], np.ones(H, dtype=np.float32))

    def test_dones_latch_persists(self):
        """Once latched, a row stays terminated even if dynamics later reports done=False."""
        N = 1
        H = 3

        def dynamics_fn(obs_batch, act_batch):
            next_obs = obs_batch + act_batch
            rewards = np.ones(obs_batch.shape[0], dtype=np.float32)
            # Done fires at step 0 only; subsequent steps return done=False
            # but the row should still be frozen by CEM's latch.
            dones = np.array([n_calls["i"] == 0], dtype=bool)
            n_calls["i"] += 1
            return next_obs, rewards, dones

        n_calls = {"i": 0}
        ctrl = _make_controller(dynamics_fn, action_dim=1, horizon=H, num_samples=N)
        actions = np.ones((N, H, 1), dtype=np.float32)
        traj, rewards, _ = ctrl._rollout_batch(np.array([0.0], dtype=np.float32), actions)
        # Step 0 terminates → traj[0, 1] = 1; remaining frozen.
        assert traj[0, 1, 0] == pytest.approx(1.0)
        assert traj[0, 2, 0] == pytest.approx(1.0)
        assert traj[0, 3, 0] == pytest.approx(1.0)
        # Reward zeroed after step 0.
        assert rewards[0, 0] == pytest.approx(1.0)
        assert rewards[0, 1] == pytest.approx(0.0)
        assert rewards[0, 2] == pytest.approx(0.0)


class TestDynamicsFnSignature:
    def test_three_tuple_returned(self):
        """Sanity: dynamics functions must return a 3-tuple (the breaking change)."""
        def dynamics_fn(obs, act):
            return obs.copy(), np.zeros(obs.shape[0], dtype=np.float32), np.zeros(obs.shape[0], dtype=bool)

        ctrl = _make_controller(dynamics_fn, action_dim=1, horizon=2, num_samples=2)
        # Should run without error.
        traj, rewards, _ = ctrl._rollout_batch(
            np.array([0.0], dtype=np.float32),
            np.zeros((2, 2, 1), dtype=np.float32),
        )
        assert traj.shape == (2, 3, 1)
        assert rewards.shape == (2, 2)
