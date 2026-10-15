"""Cost modules for trajectory-sampling planners.

Observations have shape ``(N, H+1, *obs_shape)`` — H steps plus the
terminal state.  Actions have shape ``(N, H, *act_shape)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


class Cost(ABC):
    """Base class for all planner costs."""

    @abstractmethod
    def evaluate(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray | None = None,
        terminated: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute per-sample costs.

        Args:
            obs: Trajectory observations, shape ``(N, H+1, *obs_shape)``.
                Index ``[:, -1]`` is the terminal state.
            actions: Actions taken, shape ``(N, H, *act_shape)``.
            rewards: Per-step rewards, shape ``(N, H)`` or ``None``.
            terminated: ``(N,)`` bool, True for rows whose episode ended
                inside the rollout (state held from then on), or ``None``
                when unknown.

        Returns:
            Cost vector of shape ``(N,)``.
        """
        pass


class JointUtilityCost(Cost):
    """Joint utility over the leader's return and the follower's value.

    For ``utility_type="sum"`` the utility of a plan is
    ``r_leader_norm + alpha * trust * r_follower_norm`` (returned negated,
    as a cost):

    * ``r_leader``: the leader's discounted return on full-episode scale,
      ``incurred + γ^t · (Σ_j γ^j r_{t+j} + γ^H · V_L(s_H))``. The runner
      keeps ``incurred`` and ``γ^t`` current through
      :meth:`set_incurred_return`. The ``V_L`` bootstrap is used only with
      a ``leader_value_fn`` (``terminal_value="value"``) and is skipped for
      rows that terminated inside the rollout; with
      ``terminal_value="rollout"`` the controller extends the rollout with
      the prior to the episode end instead.
    * ``r_follower``: ``follower_value_fn`` at the plan's final state.
    * Both are normalized by their calibrated ``(lo, hi)`` range (see
      :func:`critic_sharing.utils.calibration.calibrate_range`) and
      clipped to [0, 1] when ``clip_normalized``.
    * ``trust = 1 - ood_metric(final state)`` with an OOD detector, else 1.

    Args:
        follower_value_fn: Maps ``(B, *obs_shape)`` → ``(B,)`` values.
        leader_gamma: Discount factor for the leader's return.
        utility_type: ``"sum"`` (R_L + R_F) or ``"sabotage"`` (R_L - R_F).
        leader_range: ``(lo, hi)`` of the leader's full-episode discounted
            return, or ``None`` for no leader normalization.
        follower_range: ``(lo, hi)`` of V_F, or ``None``.
        leader_value_fn: Leader critic for the horizon bootstrap, or ``None``.
        ood_metric: Maps final states to OOD scores in [0, 1], or ``None``.
        alpha: Weight of the follower term.
        clip_normalized: Clip both normalized terms to [0, 1].
    """

    UTILITY_TYPES = ("sum", "sabotage")

    def __init__(
        self,
        follower_value_fn: Callable[[np.ndarray], np.ndarray | float],
        leader_gamma: float = 0.99,
        utility_type: str = "sum",
        leader_range: tuple[float, float] | None = None,
        follower_range: tuple[float, float] | None = None,
        leader_value_fn: Callable[[np.ndarray], np.ndarray | float] | None = None,
        ood_metric: Callable[[np.ndarray], np.ndarray | float] | None = None,
        alpha: float = 1.0,
        clip_normalized: bool = True,
    ) -> None:
        if utility_type not in self.UTILITY_TYPES:
            raise ValueError(
                "utility_type must be one of {valid}; got '{got}'.".format(
                    valid=self.UTILITY_TYPES,
                    got=utility_type,
                )
            )
        self._follower_value_fn = follower_value_fn
        self._leader_value_fn = leader_value_fn
        self._leader_gamma = float(leader_gamma)
        self._utility_type = utility_type
        self._leader_range = leader_range
        self._follower_range = follower_range
        self._ood_metric = ood_metric
        self._alpha = float(alpha)
        self._clip_normalized = bool(clip_normalized)
        # Leader return already incurred before the current planning step
        # and the γ^t the upcoming rollout's discounted sum is scaled by.
        self._incurred_return: float = 0.0
        self._partial_discount: float = 1.0

    def set_incurred_return(self, incurred: float, discount: float = 1.0) -> None:
        """Set the leader's return so far and ``γ^t`` for the next planning call.

        Args:
            incurred: ``sum_{i<t} γ^i r_i``, with ``γ`` = ``leader_gamma``.
            discount: ``γ^t``, the factor on the rollout's discounted sum.
        """
        self._incurred_return = float(incurred)
        self._partial_discount = float(discount)

    def _normalize(self, x: np.ndarray, value_range: tuple[float, float] | None) -> np.ndarray:
        if value_range is None:
            return x
        lo, hi = value_range
        x = (x - lo) / max(hi - lo, 1e-8)
        return np.clip(x, 0.0, 1.0) if self._clip_normalized else x

    def evaluate(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray | None = None,
        terminated: np.ndarray | None = None,
    ) -> np.ndarray:
        if rewards is None:
            raise ValueError("JointUtilityCost requires per-step rewards.")
        terminal = obs[:, -1]

        h = rewards.shape[1]
        gammas = np.power(self._leader_gamma, np.arange(h, dtype=np.float32))
        partial = (rewards * gammas[None, :]).sum(axis=1)  # (N,)
        if self._leader_value_fn is not None:
            v_leader = np.asarray(self._leader_value_fn(terminal), dtype=np.float32).reshape(-1)
            # A row that terminated inside the rollout has no tail; V_L at its
            # frozen post-terminal state would count the terminal reward twice.
            if terminated is not None:
                v_leader = np.where(
                    np.asarray(terminated, dtype=bool).reshape(-1), 0.0, v_leader,
                ).astype(np.float32)
            partial = partial + float(self._leader_gamma) ** float(h) * v_leader
        r_leader = self._normalize(
            self._incurred_return + self._partial_discount * partial, self._leader_range,
        )

        r_follower = self._normalize(
            np.asarray(self._follower_value_fn(terminal), dtype=np.float32).reshape(-1),
            self._follower_range,
        )
        if self._ood_metric is not None:
            trust = 1.0 - np.asarray(self._ood_metric(terminal), dtype=np.float32).reshape(-1)
            trust = np.clip(trust, 0.0, 1.0)
            r_follower = self._alpha * trust * r_follower
        else:
            r_follower = self._alpha * r_follower

        # Joint utility, negated: CEM minimizes cost.
        if self._utility_type == "sabotage":
            return -(r_leader - r_follower)
        return -(r_leader + r_follower)


__all__ = [
    "Cost",
    "JointUtilityCost",
]
