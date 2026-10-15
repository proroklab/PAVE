"""Shared logging helpers for SB3 training + W&B integration."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import wandb
from stable_baselines3.common.callbacks import BaseCallback

DEFAULT_SB3_METRIC_PREFIXES: Tuple[str, ...] = ("rollout/", "train/", "time/", "eval/")


class UnifiedWandbMetricsCallback(BaseCallback):
    """Common SB3 callback for logging scalar training metrics to W&B.

    Parameters
    ----------
    info_keys
        Custom keys from the env's ``info`` dict to track per-episode.
        For each key a rolling mean (``episode/{key}_mean``) and the
        last-episode value (``episode/{key}``) are logged to W&B.
        Requires the corresponding ``info_keywords`` on ``VecMonitor``.
    """

    def __init__(
        self,
        metric_prefixes: Sequence[str] = DEFAULT_SB3_METRIC_PREFIXES,
        extra_metrics: Optional[Iterable[str]] = None,
        info_keys: Optional[Sequence[str]] = None,
        episode_window: int = 100,
    ) -> None:
        super().__init__()
        self.metric_prefixes = tuple(metric_prefixes)
        self.extra_metrics = tuple(extra_metrics or ())
        self.info_keys = tuple(info_keys or ())
        self.episode_window = max(1, int(episode_window))
        self._episode_rewards: deque[float] = deque(maxlen=self.episode_window)
        self._episode_lengths: deque[float] = deque(maxlen=self.episode_window)
        self._info_buffers: dict[str, deque[float]] = {
            k: deque(maxlen=self.episode_window) for k in self.info_keys
        }
        self._episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", ())
        if not infos:
            return True

        completed_rewards = []
        completed_lengths = []
        completed_times = []
        completed_info: dict[str, list[float]] = {k: [] for k in self.info_keys}
        for info in infos:
            if not isinstance(info, dict):
                continue
            episode = info.get("episode")
            if not isinstance(episode, dict):
                continue

            reward = episode.get("r")
            length = episode.get("l")
            elapsed = episode.get("t")
            try:
                completed_rewards.append(float(reward))
                completed_lengths.append(float(length))
            except (TypeError, ValueError):
                continue
            try:
                completed_times.append(float(elapsed))
            except (TypeError, ValueError):
                pass
            for key in self.info_keys:
                val = info.get(key)
                if val is not None:
                    try:
                        completed_info[key].append(float(val))
                    except (TypeError, ValueError):
                        pass

        if not completed_rewards:
            return True

        self._episode_count += len(completed_rewards)
        self._episode_rewards.extend(completed_rewards)
        self._episode_lengths.extend(completed_lengths)
        for key in self.info_keys:
            self._info_buffers[key].extend(completed_info[key])

        metrics = {
            "episode/reward": completed_rewards[-1],
            "episode/length": completed_lengths[-1],
            "episode/count": int(self._episode_count),
            "rollout/ep_rew_mean": float(sum(self._episode_rewards) / len(self._episode_rewards)),
            "rollout/ep_len_mean": float(sum(self._episode_lengths) / len(self._episode_lengths)),
        }
        if completed_times:
            metrics["episode/time"] = completed_times[-1]
        for key in self.info_keys:
            buf = self._info_buffers[key]
            if buf:
                metrics["episode/{k}_mean".format(k=key)] = float(sum(buf) / len(buf))
            if completed_info[key]:
                metrics["episode/{k}".format(k=key)] = completed_info[key][-1]
        wandb.log(metrics, step=int(self.num_timesteps))
        return True

    def _on_rollout_end(self) -> None:
        metrics = {}
        name_to_value = getattr(self.model.logger, "name_to_value", {})
        for key, value in name_to_value.items():
            if key in ("rollout/ep_rew_mean", "rollout/ep_len_mean"):
                # These are logged directly from info["episode"] in _on_step.
                continue
            if self.metric_prefixes and not key.startswith(self.metric_prefixes):
                if key not in self.extra_metrics:
                    continue
            try:
                metrics[key] = float(value)
            except (TypeError, ValueError):
                continue
        if metrics:
            wandb.log(metrics, step=int(self.num_timesteps))


def init_wandb_run(
    project: str,
    run_name: str,
    config: Mapping[str, Any],
    run_dir: Path,
    entity: Optional[str] = None,
    tags: Optional[Sequence[str]] = None,
    job_type: Optional[str] = None,
) -> Any:
    """Initialize a W&B run configured to write local artifacts into run_dir."""

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=dict(config),
        dir=str(run_dir),
        tags=list(tags) if tags else None,
        job_type=job_type,
    )
    return run


def upload_run_directory_artifact(
    run: Any,
    run_dir: Path,
    artifact_name: Optional[str] = None,
    artifact_type: str = "training-run",
    aliases: Optional[Sequence[str]] = None,
) -> None:
    """Upload the full run directory as a W&B artifact."""

    if run is None:
        return
    resolved_run_dir = Path(run_dir)
    if not resolved_run_dir.exists():
        return

    default_name = "{project}-{run_id}-bundle".format(
        project=getattr(run, "project", "critic-sharing"),
        run_id=getattr(run, "id", "local"),
    )
    artifact = wandb.Artifact(name=artifact_name or default_name, type=artifact_type)
    artifact.add_dir(str(resolved_run_dir))
    run.log_artifact(artifact, aliases=list(aliases) if aliases else None)


__all__ = [
    "DEFAULT_SB3_METRIC_PREFIXES",
    "UnifiedWandbMetricsCallback",
    "init_wandb_run",
    "upload_run_directory_artifact",
]
