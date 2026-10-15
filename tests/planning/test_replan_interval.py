"""Tests for executing a cached CEM plan across ``replan_interval`` steps.

Over one replan interval of R steps the controller must execute exactly the
plan it scored, ``plan[0], plan[1], ..., plan[R-1]``, and leave the cache
aligned so the next replan warm-starts from ``plan[R:]``.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from critic_sharing.planning.cem import CEMController
from critic_sharing.planning.costs import Cost
from critic_sharing.planning.distributions import make_distribution
from critic_sharing.planning.prior import PriorDistribution

HORIZON = 8
# Distinct per-step actions, so a repeated or skipped step is visible.
TARGET = np.arange(1, HORIZON + 1, dtype=np.float32).reshape(HORIZON, 1)


class _TargetPrior(PriorDistribution):
    """Warm-starts CEM with TARGET, the unique zero-cost plan."""

    def rollout(self, obs, horizon, dynamics_fn):
        return TARGET[:horizon].copy()


class _DistanceToTargetCost(Cost):
    """Zero only for TARGET; any sampled perturbation costs more."""

    def evaluate(self, obs, actions, rewards=None, terminated=None):
        return np.sum((actions - TARGET[None]) ** 2, axis=(1, 2))


def _dynamics(obs_batch, act_batch):
    n = obs_batch.shape[0]
    return obs_batch.copy(), np.zeros(n, dtype=np.float32), np.zeros(n, dtype=bool)


def _controller(replan_interval: int) -> CEMController:
    return CEMController(
        action_space=gym.spaces.Box(low=-20.0, high=20.0, shape=(1,), dtype=np.float32),
        dynamics_fn=_dynamics,
        cost=_DistanceToTargetCost(),
        distribution=make_distribution("gaussian", noise_sigma=0.5, min_std=0.05),
        horizon=HORIZON,
        num_samples=16,
        num_iters=2,
        elite_frac=0.25,
        replan_interval=replan_interval,
        prior=_TargetPrior(),
        seed=0,
    )


@pytest.mark.parametrize("replan_interval", [1, 3, 5])
def test_executes_the_scored_plan_over_one_interval(replan_interval):
    ctrl = _controller(replan_interval)
    obs = np.zeros(2, dtype=np.float32)
    executed = [float(ctrl.command(obs)[0]) for _ in range(replan_interval)]
    assert executed == TARGET[:replan_interval, 0].tolist()


@pytest.mark.parametrize("replan_interval", [1, 3, 5])
def test_cache_is_aligned_for_the_next_replan(replan_interval):
    ctrl = _controller(replan_interval)
    obs = np.zeros(2, dtype=np.float32)
    for _ in range(replan_interval):
        ctrl.command(obs)
    # The next replan warm-starts from the cache: it must start at the first
    # action not executed yet.
    assert float(ctrl._plan[0, 0]) == float(TARGET[replan_interval, 0])
