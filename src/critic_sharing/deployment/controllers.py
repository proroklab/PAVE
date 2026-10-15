"""Controller adapters for the household deployment node.

The deployment node calls a single ``ControllerAdapter`` per tick to
produce its sim-frame action; the adapter encapsulates whatever
underlying machinery (a trained PPO policy, a CEM planner, etc.).
This is the abstraction boundary that lets us swap planners without
touching the node.

The PPO and CEM adapters keep different per-episode state (e.g. CEM
needs to track discounted incurred return for its cost function), but
the node only cares about the unified ``act(obs)`` interface plus
``reset()`` between episodes.

Debug modes (``--goto-origin``, ``--goto-dirt``) intentionally do NOT
go through this interface — they remain inline branches in
``_tick`` and are mutually exclusive with ``--controller cem`` (hard
error in the CLI).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


class ControllerAdapter:
    """Minimal interface a controller must satisfy to drive the node.

    Subclasses override :meth:`act` and optionally :meth:`reset`.
    """

    def reset(self) -> None:
        """Reset any per-episode state. Called once before the first tick."""

    def act(
        self, obs: np.ndarray, steps_remaining: Optional[int] = None,
    ) -> np.ndarray:
        """Return the next sim-frame action ``(vx, vy, ω)`` for ``obs``.

        ``steps_remaining`` is informational — planners that extend
        trajectories beyond their planning horizon (CEM with a
        policy-prior tail) use it; PPO ignores it.
        """
        raise NotImplementedError


class PPOPolicyAdapter(ControllerAdapter):
    """Wraps a stable-baselines3 PPO model behind the adapter interface.

    Identical behaviour to the original ``policy.predict(obs,
    deterministic=True)[0]`` call site that lived in
    :class:`HouseholdPolicyNode._tick`.
    """

    def __init__(self, model: Any):
        if model is None:
            raise ValueError(
                "PPOPolicyAdapter requires a non-None model. "
                "Use debug modes (--goto-origin/--goto-dirt) on the "
                "node directly if you want to skip the policy."
            )
        self._model = model

    def act(
        self, obs: np.ndarray, steps_remaining: Optional[int] = None,
    ) -> np.ndarray:
        pred, _ = self._model.predict(obs, deterministic=True)
        return np.asarray(pred, dtype=np.float64).reshape(3)


class CEMAdapter(ControllerAdapter):
    """Wraps a :class:`CEMController` + :class:`JointUtilityCost` pair.

    Adds the reward bookkeeping that the sim-side runner does inline in
    ``cli/eval/household.py``: each tick recomputes the cleaner's sim
    reward from the previous and current observations, accumulates it
    discounted, and pushes the running ``(incurred, γ^t)`` pair into
    the cost via :meth:`JointUtilityCost.set_incurred_return` so the
    cost can reconstruct the full from-step-0 return.

    Wall-collision penalty is intentionally NOT included — see grill
    Q3a/(iii). The integrated bias is on the order of 0.1 per episode,
    far below the pickup_reward (25) that dominates the cost.
    """

    def __init__(
        self,
        controller: Any,           # CEMController
        cost: Any,                 # JointUtilityCost
        cleaner_env_config: Any,   # HouseholdCleanerEnvConfig
        leader_gamma: float = 0.99,
    ):
        if controller is None or cost is None:
            raise ValueError("CEMAdapter requires non-None controller and cost.")
        self._controller = controller
        self._cost = cost
        self._gamma = float(leader_gamma)
        # Reward-function parameters (mirrors push_env.step).
        self._progress_weight = float(cleaner_env_config.progress_weight)
        self._step_penalty = float(cleaner_env_config.step_penalty)
        # Per-episode state — cleared in reset().
        self._prev_obs: Optional[np.ndarray] = None
        self._incurred_disc: float = 0.0
        self._gamma_pow_t: float = 1.0

    def reset(self) -> None:
        self._controller.reset()
        self._prev_obs = None
        self._incurred_disc = 0.0
        self._gamma_pow_t = 1.0

    def act(
        self, obs: np.ndarray, steps_remaining: Optional[int] = None,
    ) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)

        # 1. Update incurred return from the (prev_obs → obs) transition.
        #    Cleaner obs layout: [c_x, c_y, sin θ_c, cos θ_c,
        #                         t_x, t_y, sin θ_t, cos θ_t,
        #                         dirt_x, dirt_y]
        #    Reward (mirrors push_env.step):
        #        progress_weight * (prev_dist - dist) - step_penalty
        if self._prev_obs is not None:
            prev_dist = float(np.linalg.norm(
                self._prev_obs[0:2] - self._prev_obs[8:10]
            ))
            dist = float(np.linalg.norm(obs[0:2] - obs[8:10]))
            reward = (
                self._progress_weight * (prev_dist - dist) - self._step_penalty
            )
            self._incurred_disc += self._gamma_pow_t * reward
            self._gamma_pow_t *= self._gamma

        # 2. Sync the cost so its evaluator sees the up-to-date (γ^t,
        #    incurred) pair when scoring this tick's candidates.
        self._cost.set_incurred_return(self._incurred_disc, self._gamma_pow_t)

        # 3. Plan.
        action = self._controller.command(obs, steps_remaining=steps_remaining)

        # 4. Stash for next-tick reward delta.
        self._prev_obs = obs.copy()

        return np.asarray(action, dtype=np.float64).reshape(3)


__all__ = ["ControllerAdapter", "PPOPolicyAdapter", "CEMAdapter"]
