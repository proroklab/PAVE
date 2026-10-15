"""Fine-tune a leader policy on the follower's actual (Monte-Carlo) return.

Wraps the leader env so each episode optimises the CEM joint utility, but with
the follower's *rolled-out* return in place of the V_F bootstrap:

    per-step :  r_leader / leader_span                  (-> L_norm; the -lo
                                                          offset is a gradient-
                                                          irrelevant constant)
    on done  :  += (alpha * F_norm) / gamma^T           (F_norm = (R_F - lo_F)/span_F)

The ``/ gamma^T`` cancels PPO's intrinsic gamma^T discounting of the terminal
reward, so the contribution to the leader's expected discounted return is
exactly ``alpha * F_norm`` (undiscounted relative to step 0) — matching how
CEM's ``JointUtilityCost`` composes ``L_norm + alpha * F_norm`` additively at
the handoff. Without this correction the follower term is implicitly
discounted by gamma^T relative to the leader's step-0 reward (≈ 0.5× for a
T=70 episode at gamma=0.99).

``R_F`` is a single (K=1) frozen-follower rollout from the handoff state with
a freshly-sampled follower start, scenario-specific (see the adapters below).
The terminal term is added on EVERY episode end (termination OR truncation).
No OOD trust factor — ``R_F`` is the follower's exact return, nothing to
distrust.

Calibration ranges come from the same ``calibrate_range`` pass the CEM/eval
uses, so the objective matches ``JointUtilityCost`` up to a constant.

Per-scenario handoff/follower-rollout logic lives in ``FinetuneHandoffAdapter``
subclasses below; lifted directly from the corresponding ``_run_episode_*`` in
``cli/eval/*.py`` so the training-time follower MC matches the eval-time one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

from critic_sharing.utils.logging import (
    UnifiedWandbMetricsCallback,
    init_wandb_run,
    upload_run_directory_artifact,
)
from critic_sharing.utils.paths import (
    build_run_metadata,
    prepare_run_bundle,
    write_run_bundle,
)


# ---------------------------------------------------------------------------
# Per-scenario handoff adapters
# ---------------------------------------------------------------------------

def _discounted_rollout(env, model, obs, gamma: float) -> float:
    """Deterministic discounted return until env termination/truncation."""
    total = 0.0
    disc = 1.0
    done = False
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(act)
        total += disc * float(r)
        disc *= gamma
        done = bool(term or trunc)
    return total


class FinetuneHandoffAdapter(ABC):
    """Per-scenario glue for the follower MC rollout used as terminal reward.

    Each adapter knows how to (a) build leader + follower envs from configs,
    and (b) roll the *frozen* follower from the leader env's terminal state.
    Mirrors the corresponding ``_run_episode_*`` in ``cli/eval/*.py``.
    """

    scenario: str = ""
    supports_eval_cond: bool = False

    @abstractmethod
    def make_leader_env(self, cfg) -> gym.Env: ...

    @abstractmethod
    def make_follower_env(self, cfg) -> gym.Env: ...

    @abstractmethod
    def rollout(self, leader_env, follower_env, follower_model,
                rng: np.random.Generator, gamma: float) -> float:
        """Discounted return of one frozen-follower rollout from the leader's
        terminal state. ``leader_env`` and ``follower_env`` are the wrapped
        envs already in the wrapper process — re-used to avoid re-allocating
        on every step."""

    def sample_eval_reset_options(self, rng, leader_cfg, follower_cfg
                                  ) -> Optional[Dict[str, Any]]:
        """Optional: per-scenario eval-condition-upsampling override for the
        leader env reset. Returning ``None`` (default) leaves the leader env's
        own random spawn in charge — the 'broad / training' distribution."""
        return None


class _HouseholdAdapter(FinetuneHandoffAdapter):
    scenario = "household"
    supports_eval_cond = True

    def make_leader_env(self, cfg):
        from critic_sharing.scenarios.household.push_env import HouseholdCleanerEnv
        return HouseholdCleanerEnv(cfg)

    def make_follower_env(self, cfg):
        from critic_sharing.scenarios.household.nav_env import HouseholdNavigatorEnv
        return HouseholdNavigatorEnv(cfg)

    def rollout(self, leader_env, follower_env, follower_model, rng, gamma):
        # Handoff = leader's final table pose (the env attrs are still set
        # to their terminal-step values at this point).
        table_pos = leader_env._table_pos.copy()
        table_theta = float(leader_env._table_theta)
        spawn_region = (
            "upper" if getattr(follower_env.config, "wall_enabled", False) else "full"
        )
        seed = int(rng.integers(0, 2**31 - 1))
        obs, _ = follower_env.reset(seed=seed, options={
            "table_pos": table_pos,
            "table_theta": table_theta,
            "spawn_region": spawn_region,
        })
        return _discounted_rollout(follower_env, follower_model, obs, gamma)

    def sample_eval_reset_options(self, rng, leader_cfg, follower_cfg):
        from critic_sharing.scenarios.household.initial_conditions import (
            sample_eval_conditions,
        )
        # ``fixed_table`` is held on the wrapper; the adapter just samples
        # under whatever preset the caller wired in (see wrapper.__init__).
        ft = getattr(self, "_fixed_table", None)
        cond = sample_eval_conditions(
            leader_cfg, follower_cfg, 1, rng, fixed_table=ft,
        )[0]
        return {
            "table_pos": cond.table_pos,
            "table_theta": cond.table_theta,
            "cleaner_pos": cond.cleaner_pos,
            "cleaner_theta": cond.cleaner_theta,
        }


class _ObstacleArenaAdapter(FinetuneHandoffAdapter):
    scenario = "obstacle_arena"

    def make_leader_env(self, cfg):
        from critic_sharing.scenarios.obstacle_arena.push_env import PushObstacleEnv
        return PushObstacleEnv(cfg)

    def make_follower_env(self, cfg):
        from critic_sharing.scenarios.obstacle_arena.nav_env import OneObstacleReachEnv
        return OneObstacleReachEnv(cfg)

    def rollout(self, leader_env, follower_env, follower_model, rng, gamma):
        # Handoff = leader's final obstacle position.
        obstacle_pos = leader_env._obstacle_pos.copy()
        seed = int(rng.integers(0, 2**31 - 1))
        obs, _ = follower_env.reset(seed=seed, options={"obstacle_pos": obstacle_pos})
        return _discounted_rollout(follower_env, follower_model, obs, gamma)


class _WarehouseAdapter(FinetuneHandoffAdapter):
    scenario = "warehouse"

    def make_leader_env(self, cfg):
        from critic_sharing.scenarios.warehouse.leader_env import WarehouseLeaderEnv
        return WarehouseLeaderEnv(cfg)

    def make_follower_env(self, cfg):
        from critic_sharing.scenarios.warehouse.follower_env import WarehouseFollowerEnv
        return WarehouseFollowerEnv(cfg)

    def rollout(self, leader_env, follower_env, follower_model, rng, gamma):
        # Fresh follower spawn + goal sampled by the follower env itself,
        # then re-reset with the leader's drop position as the object pose.
        seed = int(rng.integers(0, 2**31 - 1))
        follower_env.reset(seed=seed)
        f_spawn_pos = follower_env._agent_pos.copy()
        f_spawn_theta = float(follower_env._agent_theta)
        f_goal = follower_env._goal.copy()
        drop_pos = leader_env._object_pos.copy()
        obs, _ = follower_env.reset(seed=seed, options={
            "agent_pos": f_spawn_pos,
            "agent_theta": f_spawn_theta,
            "object_pos": drop_pos,
            "goal": f_goal,
        })
        return _discounted_rollout(follower_env, follower_model, obs, gamma)


class _MobileArmAdapter(FinetuneHandoffAdapter):
    scenario = "mobile_arm"

    def make_leader_env(self, cfg):
        from critic_sharing.scenarios.mobile_arm.leader_env import MobileArmLeaderEnv
        return MobileArmLeaderEnv(cfg)

    def make_follower_env(self, cfg):
        from critic_sharing.scenarios.mobile_arm.follower_env import MobileArmFollowerEnv
        return MobileArmFollowerEnv(cfg)

    def rollout(self, leader_env, follower_env, follower_model, rng, gamma):
        from critic_sharing.scenarios.mobile_arm.common import world_to_base
        # Transform object world-pos into follower's base frame (parking pose
        # determined by the leader's terminal base pose).
        leader_terminal_pos = leader_env._base_pos.copy()
        leader_terminal_yaw = float(leader_env._base_yaw)
        object_world = leader_env._object_pos.copy()
        obj_base_2d = world_to_base(
            object_world, leader_terminal_pos, leader_terminal_yaw,
        )
        object_pos_base = np.array(
            [float(obj_base_2d[0]), float(obj_base_2d[1]), 0.0], dtype=np.float32,
        )
        fcfg = follower_env.config
        theta_default = np.array(
            [float(fcfg.theta1_default), float(fcfg.theta2_default),
             float(fcfg.theta3_default), float(fcfg.theta4_default)],
            dtype=np.float32,
        )
        seed = int(rng.integers(0, 2**31 - 1))
        obs, _ = follower_env.reset(seed=seed, options={
            "theta": theta_default,
            "object_pos": object_pos_base,
        })
        return _discounted_rollout(follower_env, follower_model, obs, gamma)


_ADAPTERS = {
    "household": _HouseholdAdapter,
    "obstacle_arena": _ObstacleArenaAdapter,
    "warehouse": _WarehouseAdapter,
    "warehouse_pickup_blocked": _WarehouseAdapter,
    "warehouse_pickup_unseen": _WarehouseAdapter,
    "mobile_arm": _MobileArmAdapter,
}


def get_handoff_adapter(scenario: str) -> FinetuneHandoffAdapter:
    if scenario not in _ADAPTERS:
        raise ValueError(
            f"No fine-tune handoff adapter for scenario {scenario!r}. "
            f"Known: {sorted(_ADAPTERS)}"
        )
    return _ADAPTERS[scenario]()


# ---------------------------------------------------------------------------
# Reward wrapper
# ---------------------------------------------------------------------------

class FollowerReturnRewardWrapper(gym.Wrapper):
    """Rewrite the leader env's scalar reward to match CEM's joint utility.

    Per-step:  r_leader / leader_span   (→ L_norm scale; offset dropped)
    On done :  += (α * F_norm) / γ^T    (cancels PPO's intrinsic γ^T discount
                                          so the contribution is α·F_norm
                                          undiscounted, matching CEM)

    Scenario-agnostic: the follower MC rollout is dispatched through
    ``FinetuneHandoffAdapter`` so the same wrapper works for all four
    scenario families.
    """

    def __init__(
        self,
        leader_env: gym.Env,
        *,
        adapter: FinetuneHandoffAdapter,
        follower_model_path: str,
        follower_env_config: Any,
        leader_range: Tuple[float, float],
        follower_range: Tuple[float, float],
        alpha: float,
        gamma: float,
        clip_normalized: bool,
        condition_seed: int = 0,
        eval_condition_prob: float = 0.0,
        fixed_table: Optional[str] = None,
    ) -> None:
        super().__init__(leader_env)
        import os

        from stable_baselines3 import PPO

        self._adapter = adapter
        self._adapter._fixed_table = fixed_table  # household-specific bag
        self._follower = PPO.load(follower_model_path, device="cpu")
        self._fenv = adapter.make_follower_env(follower_env_config)
        self._follower_cfg = follower_env_config

        self._alpha = float(alpha)
        self._gamma = float(gamma)
        self._clip = bool(clip_normalized)
        l_lo, l_hi = leader_range
        self._l_span = max(float(l_hi) - float(l_lo), 1e-8)
        f_lo, f_hi = follower_range
        self._f_lo = float(f_lo)
        self._f_span = max(float(f_hi) - float(f_lo), 1e-8)

        self._eval_prob = float(eval_condition_prob)
        # Per-subprocess RNG so the SubprocVecEnv workers diverge in their
        # follower-spawn / eval-condition streams.
        self._cond_rng = np.random.default_rng([int(condition_seed), os.getpid()])
        self._step_count: int = 0

    # ------------------------------------------------------------------ reset

    def reset(self, *, seed=None, options=None):
        self._step_count = 0
        if (self._eval_prob > 0.0
                and self._adapter.supports_eval_cond
                and float(self._cond_rng.random()) < self._eval_prob):
            opts = self._adapter.sample_eval_reset_options(
                self._cond_rng, self.env.config, self._follower_cfg,
            )
            if opts is not None:
                options = opts
        return self.env.reset(seed=seed, options=options)

    # ------------------------------------------------------------------ step

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        # Leader per-step reward → L_norm scale (offset dropped: constant).
        reward = float(reward) / self._l_span
        if terminated or truncated:
            r_f = self._adapter.rollout(
                self.env, self._fenv, self._follower, self._cond_rng, self._gamma,
            )
            f_norm = (r_f - self._f_lo) / self._f_span
            if self._clip:
                f_norm = float(np.clip(f_norm, 0.0, 1.0))
            # γ^{-T} multiplier: PPO will discount this by γ^T, so pre-scale
            # by γ^{-T} → contribution to leader's return = α·F_norm.
            gamma_pow_T = max(self._gamma ** int(self._step_count), 1e-12)
            reward += (self._alpha * float(f_norm)) / gamma_pow_T
            info = dict(info)
            info["follower_return"] = float(r_f)
            info["follower_return_norm"] = float(f_norm)
            info["finetune_T"] = int(self._step_count)
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# finetune_agent — orchestrates calibration + warm-start (or from-scratch) PPO
# ---------------------------------------------------------------------------

def _config_to_dict(cfg) -> Dict[str, Any]:
    if is_dataclass(cfg):
        return asdict(cfg)
    if hasattr(cfg, "model_dump"):
        return cfg.model_dump()
    if hasattr(cfg, "dict"):
        return cfg.dict()
    return dict(vars(cfg))


def finetune_agent(
    *,
    scenario: str,
    leader_model_path: str,
    follower_model_path: str,
    total_timesteps: int = 500_000,
    n_envs: int = 4,
    seed: int = 42,
    alpha: float = 1.0,
    from_scratch: bool = False,
    reset_exploration: bool = False,
    finetune_ent_coef: float = 0.01,
    eval_condition_prob: float = 0.0,
    fixed_table: Optional[str] = None,
    calibration_episodes: Optional[int] = None,
    save_path: Optional[str] = None,
    artifacts_root: str = "artifacts",
    run_id: Optional[str] = None,
    wandb_project: str = "critic_sharing",
    wandb_entity: Optional[str] = None,
    wandb_name: Optional[str] = None,
    wandb_tags: Optional[Sequence[str]] = None,
) -> str:
    """Warm-start ``leader_model_path`` (or train fresh with ``from_scratch=True``)
    and continue PPO with the follower-return terminal reward. Writes to the
    ``<scenario>_leader_finetune`` artifact namespace and never overwrites the
    warm-start checkpoint. Returns the saved model path."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    from critic_sharing.config.resolver import load_cost_config, load_env_config
    from critic_sharing.utils.calibration import calibrate_range

    adapter = get_handoff_adapter(scenario)
    leader_cfg = load_env_config(scenario, "leader")
    follower_cfg = load_env_config(scenario, "follower")
    cost = load_cost_config(scenario)
    gamma = float(getattr(cost, "leader_gamma", 0.99))
    clip = bool(getattr(cost, "clip_normalized", False))
    n_cal = int(calibration_episodes if calibration_episodes is not None
                else getattr(cost, "calibration_episodes", 200))
    success_only = bool(getattr(cost, "calibrate_on_success_only", False))

    # ---- 1. Calibrate leader + follower ranges (same as CEM/eval). ----
    print(f"[finetune:{scenario}] calibrating ranges ({n_cal} naive episodes, "
          f"success_only={success_only}, gamma={gamma})...", flush=True)
    leader_model = PPO.load(leader_model_path, device="cpu")
    follower_model = PPO.load(follower_model_path, device="cpu")
    lenv = adapter.make_leader_env(leader_cfg)
    try:
        leader_range = calibrate_range(
            lenv, leader_model, n_episodes=n_cal, gamma=gamma,
            success_only=success_only,
        )
    finally:
        lenv.close()
    fenv = adapter.make_follower_env(follower_cfg)
    try:
        follower_range = calibrate_range(
            fenv, follower_model, n_episodes=n_cal, gamma=gamma,
            success_only=success_only,
        )
    finally:
        fenv.close()
    print(f"[finetune:{scenario}]   leader_range={leader_range}  "
          f"follower_range={follower_range}  alpha={alpha}", flush=True)

    # ---- 2. Run bundle (NEW per-scenario namespace; never touches warm-start). ----
    task = f"{scenario}_leader_finetune"
    resolved_config: Dict[str, Any] = {
        "task": task,
        "scenario": scenario,
        "env": _config_to_dict(leader_cfg),
        "finetune": {
            "from_scratch": bool(from_scratch),
            "reset_exploration": bool(reset_exploration),
            "finetune_ent_coef": float(finetune_ent_coef) if reset_exploration else None,
            "leader_model": str(leader_model_path),
            "leader_model_role": "calibration_only" if from_scratch else "warm_start",
            "follower_model": str(follower_model_path),
            "alpha": float(alpha),
            "gamma": gamma,
            "clip_normalized": clip,
            "leader_range": list(leader_range),
            "follower_range": list(follower_range),
            "calibration_episodes": n_cal,
            "calibrate_on_success_only": success_only,
            "eval_condition_prob": float(eval_condition_prob),
            "fixed_table": fixed_table,
            "terminal_reward_undiscounted_via_gamma_inv_T": True,
        },
        "training": {"total_timesteps": int(total_timesteps), "n_envs": int(n_envs)},
        "paths": {"artifacts_root": artifacts_root, "save_path_override": save_path},
        "wandb": {"project": wandb_project, "entity": wandb_entity,
                  "run_name_override": wandb_name},
    }
    run_paths = prepare_run_bundle(
        task=task, resolved_config=resolved_config,
        artifacts_root=artifacts_root, run_id=run_id,
    )
    model_output_path = run_paths.run_dir / "model"
    resolved_config["paths"].update({
        "run_dir": str(run_paths.run_dir),
        "model_path": str(Path(f"{model_output_path}.zip")),
    })
    run = init_wandb_run(
        project=wandb_project, entity=wandb_entity,
        run_name=wandb_name or run_paths.run_id, config=resolved_config,
        run_dir=run_paths.run_dir,
        tags=list(wandb_tags or ()) + [task, scenario,
                                       "from_scratch" if from_scratch else "warm_start"]
             + (["reset_exploration"] if reset_exploration else []),
        job_type="finetune",
    )
    write_run_bundle(
        paths=run_paths, resolved_config=resolved_config, overrides={},
        metadata=build_run_metadata(
            task=task, run_id=run_paths.run_id, resolved_config=resolved_config,
            extra={"wandb": {"run_id": getattr(run, "id", None)}},
        ),
    )

    # ---- 3. Wrapped vec env + warm-started (or fresh) PPO. ----
    def _make_env():
        return Monitor(FollowerReturnRewardWrapper(
            adapter.make_leader_env(leader_cfg),
            adapter=adapter,
            follower_model_path=str(follower_model_path),
            follower_env_config=follower_cfg,
            leader_range=leader_range,
            follower_range=follower_range,
            alpha=alpha, gamma=gamma, clip_normalized=clip,
            condition_seed=seed,
            eval_condition_prob=eval_condition_prob,
            fixed_table=fixed_table,
        ))

    env = None
    try:
        env = make_vec_env(_make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)
        env = VecMonitor(env, info_keywords=(
            "follower_return", "follower_return_norm", "finetune_T",
        ))
        if from_scratch:
            print(f"[finetune:{scenario}] training leader from scratch (no warm-start).",
                  flush=True)
            model = PPO("MlpPolicy", env, verbose=1, seed=seed, device="cpu")
        else:
            model = PPO.load(leader_model_path, env=env, device="cpu")
            if reset_exploration:
                # Warm-start exploration is otherwise capped by the
                # converged-from-base-training state of three things that
                # PPO.load reuses verbatim. We reset them so the policy can
                # actually explore the new (follower-aware) reward landscape:
                #   1. ``policy.log_std`` — Gaussian action stddev (default
                #      init exp(0)=1.0 narrows to ~0.3-0.6 by end of base
                #      training; carry-over leaves the new policy unable to
                #      try anything beyond a small ball around the prior).
                #   2. Adam optimizer state (``exp_avg`` / ``exp_avg_sq``) —
                #      stale second-moment estimates from the OLD gradient
                #      distribution; clearing forces a fresh moment build-up
                #      under the new reward.
                #   3. ``ent_coef`` — bumped from base-training's 0.0 to a
                #      small positive to add an entropy bonus that further
                #      opposes premature determinism during FT.
                import collections
                import torch as th
                with th.no_grad():
                    model.policy.log_std.data.fill_(0.0)
                model.policy.optimizer.state = collections.defaultdict(dict)
                model.ent_coef = float(finetune_ent_coef)
                print(f"[finetune:{scenario}] reset_exploration ON: log_std=0 "
                      f"(σ=1.0); Adam state cleared; ent_coef="
                      f"{model.ent_coef}.", flush=True)
            model.set_random_seed(seed)
        model.learn(
            total_timesteps=total_timesteps, log_interval=1,
            callback=[UnifiedWandbMetricsCallback(
                info_keys=("follower_return", "follower_return_norm"),
            )],
            reset_num_timesteps=True,
        )
        model.save(str(model_output_path))
        print(f"saved_policy={model_output_path}.zip", flush=True)
        if save_path is not None:
            target = Path(save_path).expanduser().with_suffix(".zip")
            if target.resolve() == Path(leader_model_path).expanduser().with_suffix(".zip").resolve():
                raise ValueError(
                    "--save-path must not overwrite the warm-start leader checkpoint."
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(f"{model_output_path}.zip", target)
        upload_run_directory_artifact(
            run=run, run_dir=run_paths.run_dir,
            artifact_name=f"{run_paths.task}-{run_paths.run_id}", aliases=["latest"],
        )
    finally:
        if env is not None:
            env.close()
        run.finish()
    return f"{model_output_path}.zip"


__all__ = [
    "FollowerReturnRewardWrapper",
    "FinetuneHandoffAdapter",
    "finetune_agent",
    "get_handoff_adapter",
]
