"""Train the warehouse follower agent (PPO with shaped reward + goal obs)."""

from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor

from critic_sharing.scenarios.warehouse import (
    WarehouseFollowerEnv,
    WarehouseFollowerEnvConfig as EnvConfig,
)


class ConvergenceError(RuntimeError):
    """Raised when the trained warehouse follower converged to a low-success
    local optimum (final ``on_goal`` rate below the configured threshold).

    Carries the measured rate so the orchestrator can report it. A
    ``SUBOPTIMAL_CONVERGENCE.flag`` marker is written into the run dir (and
    next to ``save_path``) before this is raised.
    """

    def __init__(self, message: str, on_goal_rate: float, threshold: float) -> None:
        super().__init__(message)
        self.on_goal_rate = float(on_goal_rate)
        self.threshold = float(threshold)


def _evaluate_on_goal_rate(
    model, config: EnvConfig, *, n_episodes: int, seed: int,
) -> float:
    """Post-training competence gate: the shared standalone protocol
    (:mod:`critic_sharing.training.competence`) on the training config."""
    from critic_sharing.training.competence import SEED_OFFSET, success_rate

    eval_env = WarehouseFollowerEnv(config)
    try:
        return success_rate(model, eval_env, n_episodes=n_episodes,
                            base_seed=int(seed) + SEED_OFFSET)
    finally:
        eval_env.close()


class CarryPenaltyScheduleCallback(BaseCallback):
    """Linear schedule for the follower's per-step carry penalty.

    Holds the live ``step_penalty_carrying`` value at ``initial_value`` (=
    the env's ``step_penalty``) for ``warmup_steps`` timesteps, then linearly
    interpolates to ``final_value`` (= the env's ``step_penalty_carrying``)
    over ``anneal_steps``. After ``warmup_steps + anneal_steps`` the value
    plateaus at ``final_value``.

    Updates fire on ``_on_rollout_start`` and the new value is pushed to all
    parallel envs via ``training_env.env_method('set_step_penalty_carrying', v)``.
    The current value is logged to wandb each rollout.
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
        self.training_env.env_method("set_step_penalty_carrying", v)
        try:
            wandb.log(
                {"train/step_penalty_carrying": v},
                step=int(self.num_timesteps),
            )
        except Exception:
            # wandb may be unavailable in some test contexts — non-fatal.
            pass

    def _on_step(self) -> bool:
        return True
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


_TASK = "warehouse_follower"


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
    step_penalty_carrying_warmup_steps: int,
    step_penalty_carrying_anneal_steps: int,
    task: str = _TASK,
) -> Dict[str, Any]:
    return {
        "task": task,
        "env": asdict(env_config),
        "training": {
            "total_timesteps": int(total_timesteps),
            "n_envs": int(n_envs),
            "step_penalty_carrying_warmup_steps": int(step_penalty_carrying_warmup_steps),
            "step_penalty_carrying_anneal_steps": int(step_penalty_carrying_anneal_steps),
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
    total_timesteps: int = 500_000,
    env_config: Optional[EnvConfig] = None,
    save_path: Optional[str] = None,
    n_envs: int = 8,
    seed: int = 42,
    artifacts_root: str = "artifacts",
    run_id: Optional[str] = None,
    wandb_project: str = "critic_sharing",
    wandb_entity: Optional[str] = None,
    wandb_name: Optional[str] = None,
    wandb_tags: Optional[Sequence[str]] = None,
    step_penalty_carrying_warmup_steps: int = 0,
    step_penalty_carrying_anneal_steps: int = 0,
    convergence_min_on_goal: float = 0.0,
    convergence_eval_episodes: int = 100,
    task: Optional[str] = None,
) -> None:
    # Override the module-level default task name when the caller supplies
    # one (e.g. ``warehouse_pickup_blocked_follower``) so its artifacts land
    # in their own subdirectory rather than mixing with vanilla
    # warehouse_follower checkpoints).
    task_name = task if task is not None else _TASK
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
        step_penalty_carrying_warmup_steps=step_penalty_carrying_warmup_steps,
        step_penalty_carrying_anneal_steps=step_penalty_carrying_anneal_steps,
        task=task_name,
    )
    run_paths = prepare_run_bundle(
        task=task_name,
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
        tags=list(wandb_tags or ()) + [task_name],
        job_type="train",
    )
    metadata = build_run_metadata(
        task=task_name,
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
        env = make_vec_env(lambda: Monitor(WarehouseFollowerEnv(config)), n_envs=n_envs)
        env = VecMonitor(
            env,
            info_keywords=(
                "picked_up",
                "dropped",
                "on_goal",
                "shelf_collision",
                "distance_to_goal",
            ),
        )
        model = PPO("MlpPolicy", env, verbose=1, seed=seed)
        callbacks: list = [
            UnifiedWandbMetricsCallback(
                info_keys=(
                    "picked_up",
                    "dropped",
                    "on_goal",
                    "shelf_collision",
                    "distance_to_goal",
                ),
            ),
        ]
        if (
            int(step_penalty_carrying_warmup_steps) > 0
            or int(step_penalty_carrying_anneal_steps) > 0
        ):
            print(
                "Carry-penalty schedule: initial={i:.4f} final={f:.4f} "
                "warmup_steps={w} anneal_steps={a}".format(
                    i=float(config.step_penalty),
                    f=float(config.step_penalty_carrying),
                    w=int(step_penalty_carrying_warmup_steps),
                    a=int(step_penalty_carrying_anneal_steps),
                )
            )
            callbacks.append(
                CarryPenaltyScheduleCallback(
                    initial_value=float(config.step_penalty),
                    final_value=float(config.step_penalty_carrying),
                    warmup_steps=int(step_penalty_carrying_warmup_steps),
                    anneal_steps=int(step_penalty_carrying_anneal_steps),
                )
            )
        model.learn(
            total_timesteps=total_timesteps,
            log_interval=1,
            callback=callbacks,
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

        # ---- Post-training convergence guard ----
        # Catch seeds that collapsed into a low-success local optimum (the
        # bimodal failure mode) instead of silently shipping a bad policy.
        if float(convergence_min_on_goal) > 0.0:
            on_goal_rate = _evaluate_on_goal_rate(
                model, config,
                n_episodes=int(convergence_eval_episodes), seed=int(seed),
            )
            print(
                "convergence_check on_goal_rate={r:.4f} threshold={t:.4f} "
                "episodes={n}".format(
                    r=on_goal_rate, t=float(convergence_min_on_goal),
                    n=int(convergence_eval_episodes),
                ),
                flush=True,
            )
            try:
                run.log({"convergence/final_on_goal_rate": float(on_goal_rate)})
            except Exception:
                pass
            if on_goal_rate < float(convergence_min_on_goal):
                marker_dirs = [run_paths.run_dir]
                if save_path:
                    marker_dirs.append(Path(save_path).expanduser().parent)
                for md in marker_dirs:
                    try:
                        (Path(md) / "SUBOPTIMAL_CONVERGENCE.flag").write_text(
                            "on_goal_rate={r:.4f}\nthreshold={t:.4f}\n"
                            "seed={s}\ntask={task}\n".format(
                                r=on_goal_rate,
                                t=float(convergence_min_on_goal),
                                s=int(seed), task=task_name,
                            )
                        )
                    except Exception:
                        pass
                msg = (
                    "SUBOPTIMAL CONVERGENCE: {task} seed={s} reached "
                    "on_goal={r:.1%} < {t:.1%} over {n} eval episodes. The "
                    "policy is stuck in a low-success local optimum; checkpoint "
                    "kept for inspection and SUBOPTIMAL_CONVERGENCE.flag "
                    "written.".format(
                        task=task_name, s=int(seed), r=on_goal_rate,
                        t=float(convergence_min_on_goal),
                        n=int(convergence_eval_episodes),
                    )
                )
                print("\n" + "=" * 80 + "\n[!!] " + msg + "\n" + "=" * 80,
                      flush=True)
                raise ConvergenceError(
                    msg, on_goal_rate=on_goal_rate,
                    threshold=float(convergence_min_on_goal),
                )
    finally:
        if env is not None:
            env.close()
        run.finish()
