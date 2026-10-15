"""Cross-Entropy Method (CEM) controller for online planning.

Plans over continuous (``gym.spaces.Box``) action sequences; sampling
and refitting are delegated to a :class:`GaussianCEM`.
"""

from __future__ import annotations

from typing import Callable, Tuple

import gymnasium as gym
import numpy as np

from critic_sharing.planning.costs import Cost
from critic_sharing.planning.distributions import GaussianCEM
from critic_sharing.planning.prior import PriorDistribution

DynamicsFn = Callable[
    [np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]
"""Signature: ``(obs_batch, act_batch) -> (next_obs_batch, rewards, dones)``.

- obs_batch:  ``(N, *obs_shape)``
- act_batch:  ``(N, *act_shape)`` float32
- returns:
  - ``(N, *obs_shape)``    next observations
  - ``(N,)`` float32       per-step rewards
  - ``(N,)`` bool          per-row done flags. ``done[i] = True`` means
    the action at this step caused row ``i`` to terminate.

Each row's done is OR-ed into a per-rollout latch by
:class:`CEMController`. Once latched:
  * ``next_obs[row]`` is overridden to the row's prior obs (state held).
  * ``reward[row]`` is zeroed before reaching the cost evaluator.
The final latch is passed to the cost as ``terminated``, so terminal-value
bootstraps can skip rows whose episode already ended.

So dynamics functions do *not* need to handle "already-terminated row"
themselves — they only need to report per-step done correctly.
"""


class CEMController:
    """CEM planner over continuous action sequences.

    Args:
        action_space: ``gym.spaces.Box``.
        dynamics_fn: Batched one-step dynamics.
        cost: Composable cost (built via ``Cost + Cost``).
        distribution: CEM sampling distribution.
        horizon: Planning depth in timesteps.
        num_samples: Number of action sequences sampled per CEM iteration.
        num_iters: Number of CEM iterations per planning call.
        elite_frac: Fraction of samples kept as elite.
        replan_interval: Steps between full replans (>1 reuses cached plan).
        prior: Optional warm-start prior (e.g. policy rollout).
        seed: Random seed.
    """

    def __init__(
        self,
        action_space: gym.spaces.Box,
        dynamics_fn: DynamicsFn,
        cost: Cost,
        distribution: GaussianCEM,
        horizon: int = 15,
        num_samples: int = 64,
        num_iters: int = 3,
        elite_frac: float = 0.2,
        replan_interval: int = 1,
        prior: PriorDistribution | None = None,
        seed: int | None = None,
    ) -> None:
        self._action_space = action_space
        self._dynamics_fn = dynamics_fn
        self._cost = cost
        self._distribution = distribution
        self._horizon = int(horizon)
        self._num_samples = int(num_samples)
        self._num_iters = int(num_iters)
        self._elite_frac = float(elite_frac)
        self._replan_interval = int(replan_interval)
        self._prior = prior
        self._rng = np.random.default_rng(seed)

        self._plan: np.ndarray | None = None
        self._plan_step = 0

    def reset(self) -> None:
        """Reset internal state for a new episode."""
        self._plan = None
        self._plan_step = 0

    def _warm_start(self, obs: np.ndarray) -> np.ndarray | None:
        """The cached plan, or on an episode's first replan the prior rollout."""
        if self._plan is not None:
            return self._plan
        if self._prior is not None:
            return self._prior.rollout(obs, self._horizon, self._dynamics_fn)
        return None

    def _step(
        self,
        current: np.ndarray,
        act: np.ndarray,
        terminated: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One batched dynamics step under the termination latch.

        Rows already in ``terminated`` keep their state and get zero reward;
        new ``done`` flags are latched. Returns ``(next_obs, rewards,
        terminated)``.
        """
        next_obs, step_rewards, step_dones = self._dynamics_fn(current, act)
        if np.any(terminated):
            next_obs[terminated] = current[terminated]
            step_rewards[terminated] = 0.0
        return next_obs, step_rewards, terminated | np.asarray(step_dones, dtype=bool)

    def _rollout_batch(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Roll out dynamics for a batch of action sequences.

        Args:
            obs: Current observation, ``(*obs_shape,)``.
            actions: ``(N, H, *act_shape)`` action sequences.

        Returns:
            ``(N, H+1, *obs_shape)`` trajectory, ``(N, H)`` rewards and the
            ``(N,)`` termination latch. A row that terminates is held at its
            post-terminal state with zero reward afterwards, so
            ``trajectory[:, -1]`` is its end-of-episode state.
        """
        n, h = actions.shape[0], actions.shape[1]
        trajectory = np.zeros((n, h + 1, *obs.shape), dtype=np.float32)
        rewards = np.zeros((n, h), dtype=np.float32)
        current = np.repeat(obs[None, :], n, axis=0)
        trajectory[:, 0] = current
        terminated = np.zeros(n, dtype=bool)
        for t in range(h):
            current, rewards[:, t], terminated = self._step(current, actions[:, t], terminated)
            trajectory[:, t + 1] = current
        return trajectory, rewards, terminated

    def _extend_with_prior(
        self,
        trajectory: np.ndarray,
        rewards: np.ndarray,
        extension_steps: int,
        terminated: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Continue each rollout with the prior policy for ``extension_steps``.

        Appends the extension rewards, moves ``trajectory[:, -1]`` to the
        extended end state and continues the termination latch of
        :meth:`_rollout_batch`. Returns ``(trajectory, rewards, terminated)``.
        """
        current = trajectory[:, -1].copy()
        terminated = np.asarray(terminated, dtype=bool).copy()
        ext_rewards = []
        for _ in range(extension_steps):
            acts = self._prior.predict_batch(current)
            current, step_rewards, terminated = self._step(current, acts, terminated)
            ext_rewards.append(step_rewards)
        trajectory[:, -1] = current
        all_rewards = np.concatenate([rewards, np.stack(ext_rewards, axis=1)], axis=1)
        return trajectory, all_rewards, terminated

    def _score(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        steps_remaining: int | None,
    ) -> np.ndarray:
        """Costs ``(N,)`` of ``(N, H, *act_shape)`` action sequences from ``obs``.

        Rolls out the dynamics, extends with the prior to the episode end
        when ``steps_remaining`` exceeds the horizon, and evaluates the cost.
        """
        trajectory, rewards, terminated = self._rollout_batch(obs, actions)
        if (
            self._prior is not None
            and steps_remaining is not None
            and steps_remaining > self._horizon
        ):
            trajectory, rewards, terminated = self._extend_with_prior(
                trajectory, rewards, steps_remaining - self._horizon, terminated,
            )
        costs = self._cost.evaluate(trajectory, actions, rewards=rewards, terminated=terminated)
        return np.asarray(costs, dtype=np.float32).reshape(-1)

    def command(
        self,
        obs: np.ndarray,
        steps_remaining: int | None = None,
    ) -> np.ndarray:
        """Plan and return the next action.

        Args:
            obs: Current observation, ``(*obs_shape,)``.
            steps_remaining: Episode steps remaining (for prior extension).
                If provided and > horizon, the planner extends each
                trajectory with the prior policy to the episode end.

        Returns:
            The action, float32 array.
        """
        obs = np.asarray(obs, dtype=np.float32)

        # Between replans, execute the cached plan. ``self._plan[0]`` is
        # always the next action (the replan below caches the plan shifted
        # past the action it returns).
        if (
            self._replan_interval > 1
            and self._plan is not None
            and (self._plan_step % self._replan_interval) != 0
        ):
            action = self._plan[0].copy()
            self._plan[:-1] = self._plan[1:]
            self._plan_step += 1
            return self._to_action(action)

        # Sample around the warm plan: the prior rollout on an episode's
        # first replan, the cached plan afterwards.
        warm = self._warm_start(obs)
        self._distribution.initialize(self._action_space, self._horizon, warm_start=warm)
        elite_count = max(1, int(np.ceil(self._elite_frac * self._num_samples)))

        best_plan = warm
        best_cost = np.inf
        for _ in range(self._num_iters):
            actions = self._distribution.sample(self._rng, self._num_samples)
            costs = self._score(obs, actions, steps_remaining)
            iter_best = int(np.argmin(costs))
            if costs[iter_best] < best_cost:
                best_cost = costs[iter_best]
                best_plan = actions[iter_best].copy()
            elite_idx = np.argpartition(costs, elite_count - 1)[:elite_count]
            self._distribution.fit(actions[elite_idx])

        # Keep CEM's best sample only if it beats the warm plan, scored the
        # same way, so a replan is never worse than the warm plan.
        if warm is not None:
            warm_cost = float(self._score(obs, warm[None, :], steps_remaining)[0])
            if warm_cost < best_cost:
                best_plan = warm.copy()

        # Execute plan[0] and cache the rest (padded with the last action),
        # so reuse steps execute exactly the plan that was scored.
        best_plan = np.asarray(best_plan)
        action = best_plan[0].copy()
        self._plan = np.concatenate([best_plan[1:], best_plan[-1:]], axis=0)
        self._plan_step += 1
        return self._to_action(action)

    def _to_action(self, action: np.ndarray) -> np.ndarray:
        return np.asarray(action, dtype=np.float32)


__all__ = ["CEMController", "DynamicsFn"]
