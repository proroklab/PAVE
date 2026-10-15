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

from critic_sharing.scenarios.obstacle_arena.nav_env import EnvConfig, OneObstacleReachEnv
from critic_sharing.training.vae import train_vae_on_positions
from critic_sharing.utils.logging import (
    UnifiedWandbMetricsCallback,
    init_wandb_run,
    upload_run_directory_artifact,
)
from critic_sharing.utils.paths import build_run_metadata, prepare_run_bundle, write_run_bundle


class SeenObstacleCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self._positions = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", ())
        for info in infos:
            obstacle_x = info.get("obstacle_x")
            obstacle_y = info.get("obstacle_y")
            if obstacle_x is None or obstacle_y is None:
                continue
            self._positions.append((float(obstacle_x), float(obstacle_y)))
        return True

    def unique_positions(self) -> np.ndarray:
        if not self._positions:
            return np.empty((0, 2), dtype=np.float32)
        positions = np.asarray(self._positions, dtype=np.float32)
        return np.unique(np.round(positions, decimals=6), axis=0).astype(np.float32)


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
    vae_output_path = run_paths.run_dir / "nav_env_vae.pt"
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
        tags=list(wandb_tags or ()) + ["nav"],
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
        env = make_vec_env(lambda: Monitor(OneObstacleReachEnv(config)), n_envs=n_envs)
        env = VecMonitor(env, info_keywords=("reached_goal", "collided", "distance"))
        seen_obstacle_callback = SeenObstacleCallback()
        model = PPO("MlpPolicy", env, verbose=1, seed=seed)
        model.learn(
            total_timesteps=total_timesteps,
            log_interval=1,
            callback=[
                UnifiedWandbMetricsCallback(
                    info_keys=("reached_goal", "collided", "distance"),
                ),
                seen_obstacle_callback,
            ],
        )

        seen_obstacle_positions = seen_obstacle_callback.unique_positions()
        if len(seen_obstacle_positions) > 0:
            vae_model, vae_losses = train_vae_on_positions(
                seen_obstacle_positions,
                latent_dim=vae_latent_dim,
                batch_size=vae_batch_size,
                num_epochs=vae_epochs,
                learning_rate=vae_learning_rate,
            )
            torch.save(
                {
                    "state_dict": vae_model.state_dict(),
                    "latent_dim": vae_latent_dim,
                    "seen_obstacle_positions": seen_obstacle_positions,
                    "env_config": asdict(config),
                    "loss_history": vae_losses,
                },
                vae_output_path,
            )
            print(
                "trained_env_vae_samples={count} final_vae_loss={loss:.4f}".format(
                    count=len(seen_obstacle_positions),
                    loss=vae_losses[-1],
                )
            )
            print("saved_env_vae={path}".format(path=vae_output_path))
            wandb.log(
                {
                    "vae_dataset_size": int(len(seen_obstacle_positions)),
                    "vae_final_loss": float(vae_losses[-1]),
                },
                step=model.num_timesteps,
            )
        else:
            print("Warning: no obstacle positions were collected for VAE training.")
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
            from critic_sharing.scenarios.obstacle_arena.value_query import (
                build_nav_value_approximator,
                save_nav_value_approximator,
            )
            guidance = load_guidance_config("obstacle_arena")
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
