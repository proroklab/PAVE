"""Tests for the experiment logging system."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from critic_sharing.utils.experiment import ExperimentPaths, init_experiment


def _sample_config() -> dict:
    return {
        "planner": {"horizon": 30, "num_samples": 447},
        "cost": {"alpha": 4.0},
        "nav_env": {"world_min": -5.0, "world_max": 5.0, "wall_y": 0.0},
    }


def _sample_artifacts() -> dict:
    return {
        "leader_model": "artifacts/push/run_abc/model.zip",
        "follower_model": "artifacts/nav/run_xyz/model.zip",
    }


class TestInitExperiment:
    def test_creates_directory_structure(self, tmp_path: Path):
        paths = init_experiment(
            name="test-experiment",
            resolved_config=_sample_config(),
            artifact_paths=_sample_artifacts(),
            experiments_root=str(tmp_path / "experiments"),
        )
        assert paths.experiment_dir.exists()
        assert paths.config_path.exists()
        assert paths.metadata_path.exists()
        assert paths.artifacts_path.exists()
        assert paths.outputs_dir.exists()
        assert paths.outputs_dir.is_dir()

    def test_config_round_trips(self, tmp_path: Path):
        config = _sample_config()
        paths = init_experiment(
            name="config-test",
            resolved_config=config,
            artifact_paths=_sample_artifacts(),
            experiments_root=str(tmp_path / "experiments"),
        )
        loaded = yaml.safe_load(paths.config_path.read_text())
        assert loaded["planner"]["horizon"] == 30
        assert loaded["planner"]["num_samples"] == 447
        assert loaded["cost"]["alpha"] == 4.0

    def test_artifacts_recorded(self, tmp_path: Path):
        artifacts = _sample_artifacts()
        paths = init_experiment(
            name="artifacts-test",
            resolved_config=_sample_config(),
            artifact_paths=artifacts,
            experiments_root=str(tmp_path / "experiments"),
        )
        loaded = yaml.safe_load(paths.artifacts_path.read_text())
        assert loaded["leader_model"] == artifacts["leader_model"]
        assert loaded["follower_model"] == artifacts["follower_model"]

    def test_metadata_has_git_commit(self, tmp_path: Path):
        paths = init_experiment(
            name="meta-test",
            resolved_config=_sample_config(),
            artifact_paths=_sample_artifacts(),
            experiments_root=str(tmp_path / "experiments"),
        )
        meta = json.loads(paths.metadata_path.read_text())
        assert "git_commit" in meta
        assert "created_at_utc" in meta
        assert "config_sha256" in meta

    def test_metadata_includes_extra(self, tmp_path: Path):
        paths = init_experiment(
            name="extra-test",
            resolved_config=_sample_config(),
            artifact_paths=_sample_artifacts(),
            experiments_root=str(tmp_path / "experiments"),
            extra_metadata={"episodes": 50, "seed": 42},
        )
        meta = json.loads(paths.metadata_path.read_text())
        assert meta["episodes"] == 50
        assert meta["seed"] == 42

    def test_experiment_id_contains_name(self, tmp_path: Path):
        paths = init_experiment(
            name="my-experiment",
            resolved_config=_sample_config(),
            artifact_paths=_sample_artifacts(),
            experiments_root=str(tmp_path / "experiments"),
        )
        assert "my-experiment" in paths.experiment_id

    def test_unique_dir_on_collision(self, tmp_path: Path):
        paths1 = init_experiment(
            name="collision",
            resolved_config=_sample_config(),
            artifact_paths=_sample_artifacts(),
            experiments_root=str(tmp_path / "experiments"),
        )
        # Create another with the same name — should get _v2
        paths2 = init_experiment(
            name="collision",
            resolved_config=_sample_config(),
            artifact_paths=_sample_artifacts(),
            experiments_root=str(tmp_path / "experiments"),
        )
        assert paths1.experiment_dir != paths2.experiment_dir
        assert paths2.experiment_dir.exists()

    def test_returns_experiment_paths_type(self, tmp_path: Path):
        paths = init_experiment(
            name="type-test",
            resolved_config=_sample_config(),
            artifact_paths=_sample_artifacts(),
            experiments_root=str(tmp_path / "experiments"),
        )
        assert isinstance(paths, ExperimentPaths)
