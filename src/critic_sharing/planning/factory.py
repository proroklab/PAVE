"""Construction of the CEM planner from a scenario's configs.

A scenario supplies what is specific to it (follower value function, OOD
metric, batched leader dynamics, action space); the cost, prior and
controller are built the same way for every scenario from its
``cost.yaml`` and ``planner.yaml``.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import gymnasium as gym
import numpy as np

from critic_sharing.planning.cem import CEMController, DynamicsFn
from critic_sharing.planning.costs import JointUtilityCost
from critic_sharing.planning.distributions import make_distribution
from critic_sharing.planning.prior import PolicyPrior, PriorDistribution, UniformGaussianPrior

# Planner eval mode -> (utility_type, apply the OOD trust factor, prior kind).
MODE_SETTINGS: dict[str, tuple[str, bool, str]] = {
    "value-only": ("sum", False, "policy"),
    "value-and-ood": ("sum", True, "policy"),
    "uniform": ("sum", True, "uniform"),
    "sabotage": ("sabotage", True, "policy"),
}


def sb3_value_fn(model) -> Callable[[np.ndarray], np.ndarray]:
    """``(B, *obs_shape) -> (B,)`` state values from an SB3 policy's critic."""
    import torch as th

    def value_fn(obs_batch: np.ndarray) -> np.ndarray:
        obs_t, _ = model.policy.obs_to_tensor(np.asarray(obs_batch, dtype=np.float32))
        with th.no_grad():
            values = model.policy.predict_values(obs_t)
        return values.detach().cpu().numpy().reshape(-1)

    return value_fn


def make_prior(
    kind: str,
    leader_model,
    action_space: gym.spaces.Box,
    seed: Optional[int],
) -> PriorDistribution:
    """The CEM warm-start prior.

    ``"policy"``: the leader's deterministic PPO policy. ``"uniform"``: a
    state-independent Gaussian over the action box (the PAVE w/o pi_L
    ablation).
    """
    if kind == "uniform":
        return UniformGaussianPrior(
            action_low=action_space.low,
            action_high=action_space.high,
            sigma_frac=1.0,
            rng=np.random.default_rng(seed),
        )
    return PolicyPrior(
        predict_fn=lambda obs: leader_model.predict(obs, deterministic=True)[0],
        predict_batch_fn=lambda obs_batch: leader_model.predict(
            np.asarray(obs_batch, dtype=np.float32), deterministic=True,
        )[0],
    )


def build_cem_controller(
    scenario: str,
    *,
    leader_model,
    follower_value_fn: Callable[[np.ndarray], np.ndarray],
    dynamics_fn: DynamicsFn,
    action_space: gym.spaces.Box,
    seed: Optional[int],
    utility_type: str = "sum",
    leader_range: Optional[Tuple[float, float]] = None,
    follower_range: Optional[Tuple[float, float]] = None,
    ood_metric: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    alpha: float = 1.0,
    prior_kind: str = "policy",
) -> Tuple[CEMController, JointUtilityCost, str]:
    """Build the joint-utility cost and the CEM controller for ``scenario``.

    Returns ``(controller, cost, terminal_value_mode)``. With
    ``terminal_value_mode == "value"`` the cost bootstraps the leader's tail
    with its critic; with ``"rollout"`` the runner passes ``steps_remaining``
    so the controller extends candidates with the prior instead.
    """
    from critic_sharing.config.resolver import load_cost_config, load_planner_config

    cost_config = load_cost_config(scenario)
    cem_config = load_planner_config(scenario)
    terminal_value_mode = getattr(cem_config, "terminal_value", "value")

    cost = JointUtilityCost(
        follower_value_fn=follower_value_fn,
        leader_gamma=getattr(cost_config, "leader_gamma", 0.99),
        utility_type=utility_type,
        leader_range=leader_range,
        follower_range=follower_range,
        leader_value_fn=sb3_value_fn(leader_model) if terminal_value_mode == "value" else None,
        ood_metric=ood_metric,
        alpha=float(alpha),
        clip_normalized=bool(getattr(cost_config, "clip_normalized", True)),
    )
    controller = CEMController(
        action_space=action_space,
        dynamics_fn=dynamics_fn,
        cost=cost,
        distribution=make_distribution(
            cem_config.distribution,
            noise_sigma=cem_config.noise_sigma,
            min_std=cem_config.min_std,
        ),
        horizon=cem_config.horizon,
        num_samples=cem_config.num_samples,
        num_iters=cem_config.num_iters,
        elite_frac=cem_config.elite_frac,
        replan_interval=cem_config.replan_interval,
        prior=make_prior(prior_kind, leader_model, action_space, seed),
        seed=seed,
    )
    return controller, cost, terminal_value_mode


__all__ = ["MODE_SETTINGS", "build_cem_controller", "make_prior", "sb3_value_fn"]
