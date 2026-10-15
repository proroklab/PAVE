"""CEM sampling distribution over action sequences."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class GaussianCEM:
    """Per-timestep Gaussian for continuous (Box) action spaces.

    Maintains a mean and std per timestep. The CEM loop calls
    :meth:`initialize` (around the warm-start plan), then alternates
    :meth:`sample` and :meth:`fit` (on the elite samples). Samples are
    clipped to the action space bounds.
    """

    def __init__(self, noise_sigma: float = 0.5, min_std: float = 0.05) -> None:
        self._noise_sigma = float(noise_sigma)
        self._min_std = float(min_std)
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._low: np.ndarray | None = None
        self._high: np.ndarray | None = None
        self._horizon = 0
        self._act_dim = 0

    def initialize(
        self,
        action_space: gym.Space,
        horizon: int,
        warm_start: np.ndarray | None = None,
    ) -> None:
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError("GaussianCEM requires a Box action space.")
        self._horizon = int(horizon)
        self._act_dim = int(action_space.shape[0])
        self._low = action_space.low.astype(np.float32)
        self._high = action_space.high.astype(np.float32)

        if warm_start is not None:
            self._mean = np.asarray(warm_start, dtype=np.float32).copy()
        else:
            self._mean = np.zeros(
                (self._horizon, self._act_dim), dtype=np.float32
            )
        self._std = np.full_like(
            self._mean, max(self._noise_sigma, self._min_std), dtype=np.float32
        )

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        noise = rng.normal(
            0.0, 1.0, size=(n, self._horizon, self._act_dim)
        ).astype(np.float32)
        actions = self._mean[None, :, :] + self._std[None, :, :] * noise
        # Clip to action space bounds
        actions = np.clip(actions, self._low, self._high)
        return actions

    def fit(self, elite_actions: np.ndarray) -> None:
        self._mean = np.mean(elite_actions, axis=0, dtype=np.float32)
        self._mean = np.clip(self._mean, self._low, self._high)
        self._std = np.maximum(
            np.std(elite_actions, axis=0, dtype=np.float32),
            self._min_std,
        ).astype(np.float32)


def make_distribution(
    name: str,
    noise_sigma: float = 0.5,
    min_std: float = 0.05,
) -> GaussianCEM:
    """Create the CEM distribution named in ``planner.yaml`` (``"gaussian"``)."""
    if name == "gaussian":
        return GaussianCEM(noise_sigma=noise_sigma, min_std=min_std)
    raise ValueError("Unknown distribution '{name}'. Supported: gaussian.".format(name=name))


__all__ = ["GaussianCEM", "make_distribution"]
