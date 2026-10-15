"""Household cleaner (leader) training entry point.

Mirrors ``training/push.py`` for obstacle_arena — same run-bundle +
wandb plumbing, just swapping in the household cleaner env.
"""

import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor

from critic_sharing.scenarios.household.config import CleanerEnvConfig as EnvConfig
from critic_sharing.scenarios.household.push_env import HouseholdCleanerEnv
from critic_sharing.utils.logging import (
    UnifiedWandbMetricsCallback,
    init_wandb_run,
    upload_run_directory_artifact,
)
from critic_sharing.utils.paths import build_run_metadata, prepare_run_bundle, write_run_bundle


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
    artifacts_root: str,
    save_path: Optional[str],
    wandb_project: str,
    wandb_entity: Optional[str],
    wandb_name: Optional[str],
) -> Dict[str, Any]:
    return {
        "task": "clean",
        "env": asdict(env_config),
        "training": {
            "total_timesteps": int(total_timesteps),
            "n_envs": int(n_envs),
        },
        "paths": {
            "artifacts_root": artifacts_root,
            "save_path_override": save_path,
        },
        "wandb": {
            "project": wandb_project,
            "entity": wandb_entity,
            "run_name_override": wandb_name,
        },
    }


def train_agent(
    total_timesteps: int = 600_000,
    env_config: Optional[EnvConfig] = None,
    save_path: Optional[str] = None,
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
        artifacts_root=artifacts_root,
        save_path=save_path,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_name=wandb_name,
    )
    run_paths = prepare_run_bundle(
        task="clean",
        resolved_config=resolved_config,
        artifacts_root=artifacts_root,
        run_id=run_id,
    )
    model_output_path = run_paths.run_dir / "model"
    resolved_config["paths"].update(
        {
            "run_dir": str(run_paths.run_dir),
            "model_path": str(Path("{path}.zip".format(path=model_output_path))),
        },
    )

    run = init_wandb_run(
        project=wandb_project,
        entity=wandb_entity,
        run_name=wandb_name or run_paths.run_id,
        config=resolved_config,
        run_dir=run_paths.run_dir,
        tags=list(wandb_tags or ()) + ["clean"],
        job_type="train",
    )
    metadata = build_run_metadata(
        task="clean",
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
        env = make_vec_env(lambda: Monitor(HouseholdCleanerEnv(config)), n_envs=n_envs)
        env = VecMonitor(env, info_keywords=("picked_up_dirt", "distance_to_dirt"))
        model = PPO("MlpPolicy", env, verbose=1, seed=seed)
        model.learn(
            total_timesteps=total_timesteps,
            log_interval=1,
            callback=[
                UnifiedWandbMetricsCallback(
                    info_keys=("picked_up_dirt", "distance_to_dirt"),
                ),
            ],
        )
        model.save(str(model_output_path))
        print("saved_policy={path}.zip".format(path=model_output_path))
        _copy_model_zip(model_output_path, save_path)
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
