"""Helpers for reproducible training artifact paths and run manifests.

This module standardizes how experiments create artifact folders and record
the exact resolved configuration used for each run.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, TypeVar, Union

import yaml
from omegaconf import OmegaConf

PathLike = Union[str, Path]
T = TypeVar("T")

_RUN_TS_FORMAT = "%Y-%m-%d_%H%M%S"


@dataclass(frozen=True)
class RunPaths:
    """Canonical locations for a single experiment run."""

    task: str
    run_id: str
    root_dir: Path
    artifacts_root: Path
    task_dir: Path
    run_dir: Path
    config_resolved_path: Path
    overrides_path: Path
    metadata_path: Path
    stdout_log_path: Path


def resolve_user_path(base: PathLike, raw_path: PathLike) -> Path:
    """Resolve `raw_path` against `base` if relative, preserving absolute inputs."""

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path(base) / path
    return path


def find_latest_run_dir(
    task: str,
    artifacts_root: PathLike,
    required_files: Sequence[str] = ("model.zip", "config.resolved.yaml"),
) -> Path:
    """Return latest run dir under artifacts/<task> matching required files."""

    task_dir = Path(artifacts_root) / task
    if not task_dir.exists():
        raise FileNotFoundError(
            "No task artifacts found at {path}. Expected runs under '{task}'.".format(
                path=task_dir,
                task=task,
            )
        )

    candidates: list[tuple[float, Path]] = []
    for run_dir in task_dir.iterdir():
        if not run_dir.is_dir():
            continue
        checks = [run_dir / relpath for relpath in required_files]
        if not checks or all(path.exists() for path in checks):
            mtime = (
                run_dir.stat().st_mtime
                if not checks
                else max(path.stat().st_mtime for path in checks)
            )
            candidates.append((mtime, run_dir))

    if not candidates:
        required_csv = ", ".join(required_files)
        raise FileNotFoundError(
            "No valid '{task}' runs found under {path}. Need: {required}.".format(
                task=task,
                path=task_dir,
                required=required_csv,
            )
        )

    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def find_latest_model(
    task: str,
    artifacts_root: PathLike = "artifacts",
) -> Path:
    """Return model.zip path from the latest completed run for *task*.

    Convenience wrapper around :func:`find_latest_run_dir`.
    """
    run_dir = find_latest_run_dir(task=task, artifacts_root=artifacts_root)
    return run_dir / "model.zip"


def load_mapping_file(path: PathLike) -> Dict[str, Any]:
    """Load YAML mapping from disk."""

    path_obj = Path(path)
    text = path_obj.read_text(encoding="utf-8")
    payload = yaml.safe_load(text)

    if not isinstance(payload, dict):
        raise ValueError("Expected mapping in {path}".format(path=path_obj))
    return payload


def mapping_from_omegaconf(cfg: Any, resolve: bool = True) -> Dict[str, Any]:
    """Convert OmegaConf node into a plain mapping."""

    payload = OmegaConf.to_container(cfg, resolve=resolve)
    if not isinstance(payload, dict):
        raise ValueError("Expected mapping-style config section.")
    return payload


def dataclass_from_mapping(mapping: Mapping[str, Any], cls: type[T]) -> T:
    """Instantiate a dataclass from a mapping, dropping unknown keys."""

    field_map = getattr(cls, "__dataclass_fields__", None)
    if field_map is None:
        raise TypeError("Expected dataclass type, got: {cls}".format(cls=cls))
    kwargs = {key: value for key, value in mapping.items() if key in field_map}
    return cls(**kwargs)


def load_dataclass_section(path: PathLike, section: str, cls: type[T]) -> T:
    """Load a named mapping section from file and instantiate dataclass."""

    payload = load_mapping_file(path)
    section_mapping = payload.get(section, {})
    if not isinstance(section_mapping, dict):
        raise ValueError(
            "Missing '{section}' mapping in config: {path}".format(section=section, path=path)
        )
    return dataclass_from_mapping(section_mapping, cls)


def load_resolved_env_config(path: PathLike, cls: type[T]) -> T:
    """Load `env` section from resolved config file into dataclass."""

    return load_dataclass_section(path=path, section="env", cls=cls)


def project_root(start: Optional[PathLike] = None) -> Path:
    """Find repository root by walking upwards for `.git` or `pyproject.toml`."""

    current = Path(start) if start is not None else Path(__file__).resolve()
    if current.is_file():
        current = current.parent

    for candidate in [current] + list(current.parents):
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Could not locate project root from the provided start path.")


def resolve_artifacts_root(
    artifacts_root: PathLike = "artifacts",
    root_dir: Optional[PathLike] = None,
) -> Path:
    """Resolve and create the artifacts root directory."""

    root = Path(root_dir) if root_dir is not None else project_root()
    artifacts = Path(artifacts_root)
    if not artifacts.is_absolute():
        artifacts = root / artifacts
    artifacts.mkdir(parents=True, exist_ok=True)
    return artifacts


def sanitize_segment(raw: str) -> str:
    """Convert arbitrary text to a filesystem-safe segment."""

    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", raw.strip().lower())
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    return normalized or "run"


def canonicalize(value: Any) -> Any:
    """Convert value to a deterministic, JSON-serializable representation."""

    if is_dataclass(value):
        return canonicalize(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): canonicalize(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    if isinstance(value, set):
        items = [canonicalize(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def config_sha256(config: Any) -> str:
    """Hash canonicalized config content for run identity."""

    payload = canonicalize(config)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_commit(root_dir: Optional[PathLike] = None, short: bool = True) -> str:
    """Return current git commit hash, or 'nogit' if unavailable."""

    root = Path(root_dir) if root_dir is not None else project_root()
    cmd = ["git", "-C", str(root), "rev-parse"]
    if short:
        cmd.extend(["--short", "HEAD"])
    else:
        cmd.append("HEAD")
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"
    return result.stdout.strip() or "nogit"


def git_dirty(root_dir: Optional[PathLike] = None) -> Optional[bool]:
    """Return git dirty state; None when git info is unavailable."""

    root = Path(root_dir) if root_dir is not None else project_root()
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return bool(result.stdout.strip())


def make_run_id(
    task: str,
    seed: Optional[int] = None,
    resolved_config: Optional[Any] = None,
    timestamp: Optional[datetime] = None,
    commit: Optional[str] = None,
) -> str:
    """Build deterministic run id: timestamp + commit + optional seed + config hash."""

    ts = (timestamp or datetime.now(timezone.utc)).astimezone(timezone.utc)
    parts = [ts.strftime(_RUN_TS_FORMAT), sanitize_segment(commit or git_commit())[:12]]
    if seed is not None:
        parts.append("seed{seed}".format(seed=seed))
    if resolved_config is not None:
        parts.append(config_sha256(resolved_config)[:8])
    return "_".join(parts)


def _ensure_unique_dir(path: Path) -> Path:
    """Return first non-existing path by appending _vN if needed."""

    if not path.exists():
        return path
    version = 2
    while True:
        candidate = path.parent / "{name}_v{version}".format(
            name=path.name,
            version=version,
        )
        if not candidate.exists():
            return candidate
        version += 1


def init_run_paths(
    task: str,
    artifacts_root: PathLike = "artifacts",
    run_id: Optional[str] = None,
    seed: Optional[int] = None,
    resolved_config: Optional[Any] = None,
    root_dir: Optional[PathLike] = None,
) -> RunPaths:
    """Create run directory tree and return canonical run paths."""

    root = Path(root_dir) if root_dir is not None else project_root()
    artifacts = resolve_artifacts_root(artifacts_root=artifacts_root, root_dir=root)
    task_segment = sanitize_segment(task)
    task_dir = artifacts / task_segment
    task_dir.mkdir(parents=True, exist_ok=True)

    computed_run_id = run_id or make_run_id(
        task=task_segment,
        seed=seed,
        resolved_config=resolved_config,
    )
    run_dir = _ensure_unique_dir(task_dir / computed_run_id)
    run_dir.mkdir(parents=True, exist_ok=False)

    paths = RunPaths(
        task=task_segment,
        run_id=run_dir.name,
        root_dir=root,
        artifacts_root=artifacts,
        task_dir=task_dir,
        run_dir=run_dir,
        config_resolved_path=run_dir / "config.resolved.yaml",
        overrides_path=run_dir / "overrides.yaml",
        metadata_path=run_dir / "run_metadata.json",
        stdout_log_path=run_dir / "stdout.log",
    )
    paths.stdout_log_path.touch(exist_ok=True)
    return paths


def _write_json(path: Path, payload: Any) -> None:
    text = json.dumps(canonicalize(payload), indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")


def _write_yaml(path: Path, payload: Any) -> None:
    serializable = canonicalize(payload)
    text = yaml.safe_dump(serializable, sort_keys=True)
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8")


def build_run_metadata(
    task: str,
    run_id: str,
    resolved_config: Any,
    root_dir: Optional[PathLike] = None,
    command: Optional[Sequence[str]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build standardized run metadata payload."""

    root = Path(root_dir) if root_dir is not None else project_root()
    metadata: Dict[str, Any] = {
        "task": sanitize_segment(task),
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config_sha256": config_sha256(resolved_config),
        "git_commit": git_commit(root_dir=root),
        "git_dirty": git_dirty(root_dir=root),
        "command": list(command) if command is not None else list(sys.argv),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    if extra:
        metadata.update(canonicalize(dict(extra)))
    return metadata


def write_run_bundle(
    paths: RunPaths,
    resolved_config: Any,
    overrides: Optional[Any] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    """Persist resolved config, overrides, and metadata for a run."""

    _write_yaml(paths.config_resolved_path, resolved_config)
    _write_yaml(paths.overrides_path, overrides or {})
    _write_json(paths.metadata_path, metadata or {})


def prepare_run_bundle(
    task: str,
    resolved_config: Any,
    artifacts_root: PathLike = "artifacts",
    run_id: Optional[str] = None,
    seed: Optional[int] = None,
    overrides: Optional[Any] = None,
    command: Optional[Sequence[str]] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
    root_dir: Optional[PathLike] = None,
) -> RunPaths:
    """Create run directory and write config + metadata in one call."""

    paths = init_run_paths(
        task=task,
        artifacts_root=artifacts_root,
        run_id=run_id,
        seed=seed,
        resolved_config=resolved_config,
        root_dir=root_dir,
    )
    metadata = build_run_metadata(
        task=task,
        run_id=paths.run_id,
        resolved_config=resolved_config,
        root_dir=paths.root_dir,
        command=command,
        extra=extra_metadata,
    )
    write_run_bundle(
        paths=paths,
        resolved_config=resolved_config,
        overrides=overrides or {},
        metadata=metadata,
    )
    return paths


__all__ = [
    "RunPaths",
    "build_run_metadata",
    "dataclass_from_mapping",
    "canonicalize",
    "config_sha256",
    "find_latest_model",
    "find_latest_run_dir",
    "git_commit",
    "git_dirty",
    "init_run_paths",
    "load_dataclass_section",
    "load_mapping_file",
    "load_resolved_env_config",
    "make_run_id",
    "mapping_from_omegaconf",
    "prepare_run_bundle",
    "project_root",
    "resolve_artifacts_root",
    "resolve_user_path",
    "sanitize_segment",
    "write_run_bundle",
]
