"""Experiment result logging — structured output for eval runs.

Mirrors the training artifact pattern (``prepare_run_bundle`` / ``RunPaths``)
but for evaluation experiments.  Every experiment gets its own directory under
``experiments/`` with a snapshot of the resolved config, the artifact paths
that were used, git metadata, and an ``outputs/`` subdirectory for CSVs,
figures, and MP4s.

Directory layout::

    experiments/
      <datetime>_<name>/
        experiment.yaml        # resolved configs (env, planner, cost, guidance)
        experiment_meta.json   # git commit, command, timestamp
        artifacts_used.yaml    # paths to model checkpoints used
        outputs/               # experiment outputs (variable structure)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import yaml

from critic_sharing.utils.paths import (
    _ensure_unique_dir,
    build_run_metadata,
    canonicalize,
    project_root,
    sanitize_segment,
)

_TS_FORMAT = "%Y-%m-%d_%H%M%S"


@dataclass(frozen=True)
class ExperimentPaths:
    """Canonical locations for a single experiment run."""

    name: str
    experiment_id: str
    experiment_dir: Path
    config_path: Path          # experiment.yaml
    metadata_path: Path        # experiment_meta.json
    artifacts_path: Path       # artifacts_used.yaml
    outputs_dir: Path          # outputs/


def _make_experiment_id(
    name: str,
    timestamp: Optional[datetime] = None,
) -> str:
    """Build experiment directory name: ``<datetime>_<name>``."""
    ts = (timestamp or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return "{ts}_{name}".format(
        ts=ts.strftime(_TS_FORMAT),
        name=sanitize_segment(name),
    )


def init_experiment(
    name: str,
    resolved_config: Dict[str, Any],
    artifact_paths: Dict[str, str],
    experiments_root: str = "experiments",
    command: Optional[Sequence[str]] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
    root_dir: Optional[Path] = None,
) -> ExperimentPaths:
    """Create experiment directory and write config + metadata bundle.

    Parameters
    ----------
    name
        Human-readable experiment name (e.g. "guidance-ablation").
    resolved_config
        Full resolved config dict from ``load_run_config()`` or similar.
        Serialised to ``experiment.yaml``.
    artifact_paths
        Mapping of role → model path (e.g. ``{"leader": "artifacts/push/…/model.zip"}``).
        Serialised to ``artifacts_used.yaml``.
    experiments_root
        Top-level experiments directory (relative to project root).
    command
        CLI command that launched this experiment (defaults to ``sys.argv``).
    extra_metadata
        Additional metadata fields merged into ``experiment_meta.json``.
    root_dir
        Project root override (default: auto-detected).

    Returns
    -------
    ExperimentPaths
        Paths to all experiment files and the outputs directory.
    """
    root = root_dir or project_root()
    exp_root = Path(experiments_root)
    if not exp_root.is_absolute():
        exp_root = root / exp_root
    exp_root.mkdir(parents=True, exist_ok=True)

    experiment_id = _make_experiment_id(name)
    experiment_dir = _ensure_unique_dir(exp_root / experiment_id)
    experiment_dir.mkdir(parents=True, exist_ok=False)

    outputs_dir = experiment_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    paths = ExperimentPaths(
        name=name,
        experiment_id=experiment_dir.name,
        experiment_dir=experiment_dir,
        config_path=experiment_dir / "experiment.yaml",
        metadata_path=experiment_dir / "experiment_meta.json",
        artifacts_path=experiment_dir / "artifacts_used.yaml",
        outputs_dir=outputs_dir,
    )

    # Write resolved config
    _write_yaml(paths.config_path, resolved_config)

    # Write artifact references
    _write_yaml(paths.artifacts_path, artifact_paths)

    # Write metadata
    metadata = build_run_metadata(
        task=name,
        run_id=paths.experiment_id,
        resolved_config=resolved_config,
        root_dir=root,
        command=command,
        extra=extra_metadata,
    )
    _write_json(paths.metadata_path, metadata)

    return paths


def _write_yaml(path: Path, payload: Any) -> None:
    serializable = canonicalize(payload)
    text = yaml.safe_dump(serializable, sort_keys=True)
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    text = json.dumps(canonicalize(payload), indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")


__all__ = [
    "ExperimentPaths",
    "init_experiment",
]
