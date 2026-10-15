"""Household navigator (follower) training entry point.

Mirrors ``training/nav.py`` for obstacle_arena, with two differences:

  1. ``HouseholdNavigatorEnv`` replaces ``OneObstacleReachEnv``.
  2. The VAE is trained on 4-D table-pose samples
     ``[t_x, t_y, sin θ_t, cos θ_t]`` instead of 2-D obstacle positions.
     The callback collects these from ``info`` each step.
"""

import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor

from critic_sharing.scenarios.household.config import NavEnvConfig as EnvConfig
from critic_sharing.scenarios.household.nav_env import HouseholdNavigatorEnv
from critic_sharing.training.vae import train_vae_on_positions
from critic_sharing.utils.logging import (
    UnifiedWandbMetricsCallback,
    init_wandb_run,
    upload_run_directory_artifact,
)
from critic_sharing.utils.paths import build_run_metadata, prepare_run_bundle, write_run_bundle


class CollisionPenaltyScheduleCallback(BaseCallback):
    """Linear schedule for the follower's collision penalty.

    Holds the live ``collision_penalty`` at ``initial_value`` for
    ``warmup_steps`` timesteps, then linearly interpolates to
    ``final_value`` over ``anneal_steps``. After
    ``warmup_steps + anneal_steps`` the value plateaus at
    ``final_value``.

    Mirrors ``warehouse_follower.CarryPenaltyScheduleCallback``: pushes
    the new value to each parallel env at the start of every rollout
    via ``training_env.env_method("set_collision_penalty", v)`` and
    logs it to wandb.
    """

    def __init__(
        self,
        initial_value: float,
        final_value: float,
        warmup_steps: int,
        anneal_steps: int,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._initial = float(initial_value)
        self._final = float(final_value)
        self._warmup = int(warmup_steps)
        self._anneal = int(anneal_steps)

    def compute_value(self, t: int) -> float:
        if t < self._warmup or self._anneal <= 0:
            return self._initial if t < self._warmup else self._final
        progress = (t - self._warmup) / float(self._anneal)
        if progress >= 1.0:
            return self._final
        return self._initial + progress * (self._final - self._initial)

    def _on_rollout_start(self) -> None:
        v = self.compute_value(self.num_timesteps)
        self.training_env.env_method("set_collision_penalty", v)
        try:
            wandb.log(
                {"train/collision_penalty": v},
                step=int(self.num_timesteps),
            )
        except Exception:
            pass

    def _on_step(self) -> bool:
        return True


class SeenTablePoseCallback(BaseCallback):
    """Collects ``(t_x, t_y, sin θ_t, cos θ_t)`` from per-step infos."""

    def __init__(self):
        super().__init__()
        self._samples = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", ())
        for info in infos:
            t_x = info.get("table_x")
            t_y = info.get("table_y")
            t_theta = info.get("table_theta")
            if t_x is None or t_y is None or t_theta is None:
                continue
            self._samples.append(
                (float(t_x), float(t_y),
                 float(np.sin(t_theta)), float(np.cos(t_theta)))
            )
        return True

    def unique_samples(self) -> np.ndarray:
        if not self._samples:
            return np.empty((0, 4), dtype=np.float32)
        arr = np.asarray(self._samples, dtype=np.float32)
        return np.unique(np.round(arr, decimals=6), axis=0).astype(np.float32)


def _copy_file(source: Path, destination: Optional[str]) -> None:
    if destination is None:
        return
    if not source.exists():
        return
    target = Path(destination).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _copy_model_zip(source_base_path: Path, destination_base_path: Optional[str]) -> None:
    if destination_base_path is None:
        return
    source_zip = Path("{path}.zip".format(path=source_base_path))
    target_zip = Path(destination_base_path).expanduser().with_suffix(".zip")
    target_zip.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_zip, target_zip)


def _build_resolved_config(
    env_config: EnvConfig,
    total_timesteps: int,
    n_envs: int,
    vae_latent_dim: int,
    vae_batch_size: int,
    vae_epochs: int,
    vae_learning_rate: float,
    artifacts_root: str,
    save_path: Optional[str],
    vae_save_path: Optional[str],
    wandb_project: str,
    wandb_entity: Optional[str],
    wandb_name: Optional[str],
) -> Dict[str, Any]:
    return {
        "task": "nav",
        "env": asdict(env_config),
        "training": {
            "total_timesteps": int(total_timesteps),
            "n_envs": int(n_envs),
        },
        "vae": {
            "latent_dim": int(vae_latent_dim),
            "batch_size": int(vae_batch_size),
            "epochs": int(vae_epochs),
            "learning_rate": float(vae_learning_rate),
        },
        "paths": {
            "artifacts_root": artifacts_root,
            "save_path_override": save_path,
            "vae_save_path_override": vae_save_path,
        },
        "wandb": {
            "project": wandb_project,
            "entity": wandb_entity,
            "run_name_override": wandb_name,
        },
    }


def train_agent(
    total_timesteps: int = 200_000,
    env_config: Optional[EnvConfig] = None,
    save_path: Optional[str] = None,
    vae_save_path: Optional[str] = None,
    vae_latent_dim: int = 2,
    vae_batch_size: int = 64,
    vae_epochs: int = 100,
    vae_learning_rate: float = 1e-3,
    n_envs: int = 4,
    seed: int = 42,
    artifacts_root: str = "artifacts",
    run_id: Optional[str] = None,
    wandb_project: str = "critic_sharing",
    wandb_entity: Optional[str] = None,
    wandb_name: Optional[str] = None,
    wandb_tags: Optional[Sequence[str]] = None,
    # Linear collision-penalty schedule (warehouse-style). Set
    # ``collision_penalty_anneal_steps = 0`` to disable the schedule
    # entirely; the env's static ``collision_penalty`` is then used.
    collision_penalty_initial: float = 0.0,
    collision_penalty_final: float = 10.0,
    collision_penalty_warmup_steps: int = 300_000,
    collision_penalty_anneal_steps: int = 900_000,
) -> None:
    config = env_config or EnvConfig()
    resolved_config = _build_resolved_config(
        env_config=config,
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        vae_latent_dim=vae_latent_dim,
        vae_batch_size=vae_batch_size,
        vae_epochs=vae_epochs,
        vae_learning_rate=vae_learning_rate,
        artifacts_root=artifacts_root,
        save_path=save_path,
        vae_save_path=vae_save_path,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_name=wandb_name,
    )
    run_paths = prepare_run_bundle(
        task="nav",
        resolved_config=resolved_config,
        artifacts_root=artifacts_root,
        run_id=run_id,
    )

    model_output_path = run_paths.run_dir / "model"
    # ``follower_ood.pt`` is the new generic name (the inline-trained
    # checkpoint always stores the kNN payload — the default detector).
    # ``_resolve_vae`` also still auto-discovers the legacy
    # ``follower_vae.pt`` / ``nav_env_vae.pt`` names for back-compat.
    vae_output_path = run_paths.run_dir / "follower_ood.pt"
    resolved_config["paths"].update(
        {
            "run_dir": str(run_paths.run_dir),
            "model_path": str(Path("{path}.zip".format(path=model_output_path))),
            "vae_path": str(vae_output_path),
        },
    )
    run = init_wandb_run(
        project=wandb_project,
        entity=wandb_entity,
        run_name=wandb_name or run_paths.run_id,
        config=resolved_config,
        run_dir=run_paths.run_dir,
        tags=list(wandb_tags or ()) + ["nav", "household"],
        job_type="train",
    )
    metadata = build_run_metadata(
        task="nav",
        run_id=run_paths.run_id,
        resolved_config=resolved_config,
        extra={
            "wandb": {
                "run_id": getattr(run, "id", None),
                "project": getattr(run, "project", wandb_project),
                "entity": getattr(run, "entity", wandb_entity),
                "name": getattr(run, "name", wandb_name or run_paths.run_id),
            }
        },
    )
    write_run_bundle(
        paths=run_paths,
        resolved_config=resolved_config,
        overrides={},
        metadata=metadata,
    )

    env = None
    try:
        env = make_vec_env(lambda: Monitor(HouseholdNavigatorEnv(config)), n_envs=n_envs)
        env = VecMonitor(env, info_keywords=("reached_goal", "collided", "distance"))
        seen_callback = SeenTablePoseCallback()
        callbacks = [
            UnifiedWandbMetricsCallback(
                info_keys=("reached_goal", "collided", "distance"),
            ),
            seen_callback,
        ]
        if int(collision_penalty_anneal_steps) > 0 or int(collision_penalty_warmup_steps) > 0:
            print(
                "Collision-penalty schedule: initial={i:.2f} final={f:.2f} "
                "warmup_steps={w} anneal_steps={a}".format(
                    i=float(collision_penalty_initial),
                    f=float(collision_penalty_final),
                    w=int(collision_penalty_warmup_steps),
                    a=int(collision_penalty_anneal_steps),
                )
            )
            callbacks.append(
                CollisionPenaltyScheduleCallback(
                    initial_value=float(collision_penalty_initial),
                    final_value=float(collision_penalty_final),
                    warmup_steps=int(collision_penalty_warmup_steps),
                    anneal_steps=int(collision_penalty_anneal_steps),
                )
            )
        model = PPO("MlpPolicy", env, verbose=1, seed=seed)
        model.learn(
            total_timesteps=total_timesteps,
            log_interval=1,
            callback=callbacks,
        )

        seen_table_states = seen_callback.unique_samples()
        if len(seen_table_states) > 0:
            vae_model, vae_losses = train_vae_on_positions(
                seen_table_states,
                latent_dim=vae_latent_dim,
                batch_size=vae_batch_size,
                num_epochs=vae_epochs,
                learning_rate=vae_learning_rate,
            )
            torch.save(
                {
                    "state_dict": vae_model.state_dict(),
                    "latent_dim": vae_latent_dim,
                    "input_dim": int(seen_table_states.shape[1]),
                    "seen_table_states": seen_table_states,
                    "env_config": asdict(config),
                    "loss_history": vae_losses,
                },
                vae_output_path,
            )
            print(
                "trained_env_vae_samples={count} final_vae_loss={loss:.4f}".format(
                    count=len(seen_table_states),
                    loss=vae_losses[-1],
                )
            )
            print("saved_env_vae={path}".format(path=vae_output_path))
            wandb.log(
                {
                    "vae_dataset_size": int(len(seen_table_states)),
                    "vae_final_loss": float(vae_losses[-1]),
                },
                step=model.num_timesteps,
            )
        else:
            print("Warning: no table-pose samples were collected for VAE training.")
            wandb.log({"vae_dataset_size": 0}, step=model.num_timesteps)

        obs = env.reset()
        for _ in range(5):
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, _, infos = env.step(action)
            avg_rew = float(np.mean(rewards))
            avg_dist = float(np.mean([info["distance"] for info in infos]))
            print(
                "avg_reward={reward:+.3f} avg_distance={distance:.3f}".format(
                    reward=avg_rew,
                    distance=avg_dist,
                )
            )

        model.save(str(model_output_path))
        print("saved_policy={path}.zip".format(path=model_output_path))

        # Distill and save follower value approximator
        value_output_path = run_paths.run_dir / "nav_value_approx.pt"
        try:
            from critic_sharing.config.resolver import load_guidance_config
            from critic_sharing.scenarios.household.value_query import (
                build_nav_value_approximator,
                save_nav_value_approximator,
            )
            guidance = load_guidance_config("household")
            print("Distilling follower value function...")
            value_approx = build_nav_value_approximator(
                nav_model=model,
                nav_config=config,
                obstacle_samples=guidance.distillation.obstacle_samples,
                agent_grid_size=guidance.distillation.agent_grid_size,
                epochs=guidance.distillation.epochs,
                batch_size=guidance.distillation.batch_size,
                lr=guidance.distillation.lr,
            )
            save_nav_value_approximator(
                value_approx,
                str(value_output_path),
                nav_config=config,
                obstacle_samples=guidance.distillation.obstacle_samples,
                agent_grid_size=guidance.distillation.agent_grid_size,
                epochs=guidance.distillation.epochs,
            )
            print("saved_value_approx={path}".format(path=value_output_path))
        except Exception as exc:
            print("Warning: value distillation failed: {e}".format(e=exc))

        _copy_model_zip(model_output_path, save_path)
        _copy_file(vae_output_path, vae_save_path)
        upload_run_directory_artifact(
            run=run,
            run_dir=run_paths.run_dir,
            artifact_name="{task}-{run_id}".format(task=run_paths.task, run_id=run_paths.run_id),
            aliases=["latest"],
        )
    finally:
        if env is not None:
            env.close()
        run.finish()
