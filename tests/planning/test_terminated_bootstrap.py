"""Rows that terminate inside a CEM rollout get no terminal-value bootstrap.

* ``JointUtilityCost`` adds ``gamma^H * V_L(terminal)`` only for rows still
  running at the horizon.
* ``CEMController`` hands its termination latch to the cost.
* The prior extension continues that latch instead of restarting it.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from critic_sharing.planning.cem import CEMController
from critic_sharing.planning.costs import Cost, JointUtilityCost
from critic_sharing.planning.distributions import make_distribution
from critic_sharing.planning.prior import PolicyPrior

GAMMA = 0.9
V_LEADER = 10.0


def _cost() -> JointUtilityCost:
    return JointUtilityCost(
        follower_value_fn=lambda obs: np.zeros(len(obs), dtype=np.float32),
        leader_value_fn=lambda obs: np.full(len(obs), V_LEADER, dtype=np.float32),
        leader_gamma=GAMMA,
        alpha=1.0,
    )


def test_bootstrap_skips_terminated_rows():
    n, h = 2, 3
    obs = np.zeros((n, h + 1, 1), dtype=np.float32)
    actions = np.zeros((n, h, 1), dtype=np.float32)
    rewards = np.zeros((n, h), dtype=np.float32)
    costs = _cost().evaluate(obs, actions, rewards=rewards, terminated=np.array([True, False]))
    # Cost is the negated utility; only the running row is bootstrapped.
    assert costs[0] == pytest.approx(0.0)
    assert costs[1] == pytest.approx(-(GAMMA**h) * V_LEADER)


def test_bootstrap_without_latch_is_unchanged():
    n, h = 2, 3
    obs = np.zeros((n, h + 1, 1), dtype=np.float32)
    costs = _cost().evaluate(
        obs, np.zeros((n, h, 1), dtype=np.float32), rewards=np.zeros((n, h), dtype=np.float32),
    )
    np.testing.assert_allclose(costs, -(GAMMA**h) * V_LEADER, rtol=1e-6)


def _dynamics(obs_batch, act_batch):
    """1-D walk; the episode ends once the state reaches 1."""
    next_obs = obs_batch + act_batch
    rewards = np.ones(obs_batch.shape[0], dtype=np.float32)
    return next_obs, rewards, next_obs[:, 0] >= 1.0


class _RecordingCost(Cost):
    def __init__(self):
        self.calls = []

    def evaluate(self, obs, actions, rewards=None, terminated=None):
        self.calls.append((np.asarray(obs).copy(), terminated))
        return np.zeros(obs.shape[0], dtype=np.float32)


def _controller(cost: Cost, prior=None) -> CEMController:
    return CEMController(
        action_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        dynamics_fn=_dynamics,
        cost=cost,
        distribution=make_distribution("gaussian", noise_sigma=0.8, min_std=0.05),
        horizon=3,
        num_samples=64,
        num_iters=2,
        elite_frac=0.25,
        prior=prior,
        seed=0,
    )


def test_controller_passes_latch_to_cost():
    cost = _RecordingCost()
    _controller(cost).command(np.zeros(1, dtype=np.float32))
    assert cost.calls
    seen = np.concatenate([np.asarray(term) for _, term in cost.calls])
    assert seen.any() and not seen.all()
    for traj, term in cost.calls:
        # A row is latched exactly when it reached the terminal region (its
        # state is then held there).
        np.testing.assert_array_equal(term, traj[:, -1, 0] >= 1.0)


def test_prior_extension_continues_the_latch():
    prior = PolicyPrior(
        predict_fn=lambda obs: np.ones(1, dtype=np.float32),
        predict_batch_fn=lambda obs: np.ones((len(obs), 1), dtype=np.float32),
    )
    ctrl = _controller(_RecordingCost(), prior=prior)
    # Row 0 ends at step 0 (state 1); row 1 never moves inside the horizon.
    actions = np.array([[[1.0], [1.0], [1.0]], [[0.0], [0.0], [0.0]]], dtype=np.float32)
    traj, rewards, terminated = ctrl._rollout_batch(np.zeros(1, dtype=np.float32), actions)
    np.testing.assert_array_equal(terminated, [True, False])

    traj, rewards, terminated = ctrl._extend_with_prior(traj, rewards, 2, terminated)
    # Row 0 stays frozen with zero extension reward; row 1 reaches 1 on the
    # first extension step and is latched there.
    assert traj[0, -1, 0] == pytest.approx(1.0)
    np.testing.assert_allclose(rewards[0, 3:], [0.0, 0.0])
    assert traj[1, -1, 0] == pytest.approx(1.0)
    np.testing.assert_allclose(rewards[1, 3:], [1.0, 0.0])
    np.testing.assert_array_equal(terminated, [True, True])
