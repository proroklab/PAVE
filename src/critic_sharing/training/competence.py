"""Standalone competence: how often a policy solves its own task alone.

One protocol, used by the training-time convergence gate and by the
paper's seed selection (``evaluations/paper/cross_scenario/selection.py``,
via ``critic-sharing competence``):

* the policy acts alone, deterministically, in its training env (env config
  plus the training YAML's ``env_overrides``);
* ``n_episodes`` episodes with reset seeds ``train_seed + 1_000_000 + k``
  (training-disjoint conditions);
* an episode succeeds if the env's ``SUCCESS_INFO_KEY`` event fires at any
  step.

One restriction, fixed by geometry: the mobile_arm follower env spawns
objects beyond the arm's reach on purpose (so V_F learns unreachable
states), so its competence is measured on spawns with
``r <= L1 + L2a + L2b``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

SEED_OFFSET = 1_000_000


def success_rate(model, env, *, n_episodes: int, base_seed: int) -> float:
    """Fraction of ``n_episodes`` deterministic rollouts of ``model`` in
    ``env`` (reset seeds ``base_seed + k``) in which the env's success event
    fires at any step. The caller owns ``env``."""
    key = env.SUCCESS_INFO_KEY
    wins = 0
    for k in range(int(n_episodes)):
        obs, _ = env.reset(seed=int(base_seed) + k)
        done = success = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            success = success or bool(info.get(key, False))
            done = bool(terminated or truncated)
        wins += int(success)
    return wins / max(int(n_episodes), 1)


def competence_env_config(scenario: str, role: str) -> Any:
    """Training env config, with the mobile_arm follower's reach restriction."""
    from critic_sharing.envs.factory import training_env_config

    cfg = training_env_config(scenario, role)
    if scenario == "mobile_arm" and role == "follower":
        cfg = dataclasses.replace(
            cfg, object_r_max=min(cfg.object_r_max, cfg.L1 + cfg.L2a + cfg.L2b))
    return cfg


def competence(scenario: str, role: str, model_path: Path, *, train_seed: int,
               n_episodes: int = 100) -> float:
    """Standalone success rate of the checkpoint at ``model_path``."""
    from stable_baselines3 import PPO

    from critic_sharing.envs.factory import make_env

    env = make_env(scenario, role, competence_env_config(scenario, role))
    try:
        model = PPO.load(str(model_path), device="cpu")
        return success_rate(model, env, n_episodes=n_episodes,
                            base_seed=int(train_seed) + SEED_OFFSET)
    finally:
        env.close()
