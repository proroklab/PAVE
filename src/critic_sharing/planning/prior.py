"""Prior distributions for warm-starting CEM sampling.

A prior provides an initial action sequence to centre the CEM
distribution around, typically by rolling out a trained policy.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


class PriorDistribution:
    """Base for CEM warm-start priors: a per-step action predictor.

    Subclasses implement :meth:`_predict_single` (and may override
    :meth:`predict_batch` with a vectorised call).
    """

    def rollout(
        self,
        obs: np.ndarray,
        horizon: int,
        dynamics_fn: Callable,
    ) -> np.ndarray:
        """Warm-start action sequence ``(horizon, *act_shape)`` from ``obs``.

        Steps ``dynamics_fn`` (``(obs_batch, act_batch) -> (next_obs_batch,
        rewards, dones)``) with this prior's own actions; rewards and dones
        are ignored, only the actions are kept.
        """
        current_obs = np.asarray(obs, dtype=np.float32).copy()
        actions: list[np.ndarray] = []
        for _ in range(horizon):
            action = self._predict_single(current_obs)
            actions.append(action)
            next_obs, _rewards, _dones = dynamics_fn(
                current_obs[None, :],
                action[None, :] if action.ndim == 1 else action[None],
            )
            current_obs = next_obs[0]
        return np.stack(actions, axis=0)

    def predict_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        """Actions ``(N, *act_shape)`` for ``(N, *obs_shape)`` observations."""
        return np.array([self._predict_single(obs_batch[i]) for i in range(obs_batch.shape[0])])

    def _predict_single(self, obs: np.ndarray) -> np.ndarray:
        """Action for a single observation."""
        raise NotImplementedError


class PolicyPrior(PriorDistribution):
    """Roll out a trained policy to get an initial action sequence.

    Args:
        predict_fn: Maps a single observation to a single action.
            E.g. ``lambda obs: model.predict(obs, deterministic=True)[0]``.
        predict_batch_fn: Optional batched variant mapping
            ``(N, *obs_shape)`` → ``(N, *act_shape)``.  When provided,
            :meth:`predict_batch` uses this instead of looping.
    """

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        predict_batch_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self._predict_fn = predict_fn
        self._predict_batch_fn = predict_batch_fn

    def _predict_single(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(self._predict_fn(obs), dtype=np.float32)

    def predict_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        if self._predict_batch_fn is not None:
            return np.asarray(self._predict_batch_fn(obs_batch), dtype=np.float32)
        return super().predict_batch(obs_batch)


class UniformGaussianPrior(PriorDistribution):
    """State-independent wide Gaussian prior — leader-policy-free baseline.

    Used by the ``EvalMode.uniform`` ablation, which replaces the leader
    policy as the warm start (the leader critic V_L still bootstraps the
    cost). Actions are samples of ``N(0, σ² · I)`` clipped to the action
    box, with ``σ`` a per-dimension width covering the action range.

    Args:
        action_low: ``(*act_shape,)`` lower bound of action space.
        action_high: ``(*act_shape,)`` upper bound.
        sigma_frac: Width of the Gaussian as a fraction of half the
            action range (default 1.0 = σ equal to half-range, so ±2σ
            covers the action box).
        rng: ``np.random.Generator`` for reproducibility (default
            ``np.random.default_rng(0)``).
    """

    def __init__(
        self,
        action_low: np.ndarray,
        action_high: np.ndarray,
        sigma_frac: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self._low = np.asarray(action_low, dtype=np.float32)
        self._high = np.asarray(action_high, dtype=np.float32)
        # σ per dim = sigma_frac × (high − low) / 2.
        self._sigma = sigma_frac * (self._high - self._low) / 2.0
        self._rng = rng if rng is not None else np.random.default_rng(0)

    def _predict_single(self, obs: np.ndarray) -> np.ndarray:
        sample = self._rng.normal(0.0, self._sigma).astype(np.float32)
        return np.clip(sample, self._low, self._high)

    def predict_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        n = obs_batch.shape[0]
        # Broadcast σ across the batch dim.
        samples = self._rng.normal(
            loc=0.0,
            scale=self._sigma[None, :],
            size=(n, *self._sigma.shape),
        ).astype(np.float32)
        return np.clip(samples, self._low[None, :], self._high[None, :])


__all__ = ["PolicyPrior", "PriorDistribution", "UniformGaussianPrior"]
