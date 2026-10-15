"""Unit tests for the household-comparison harness.

The supervisor wraps ``run_episode`` + recorders; these tests focus on
the configuration plumbing (per-rollout config, init-conditions
serialization, discarded-dir renaming) — the lifecycle itself needs
hardware to exercise end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from critic_sharing.config.schema import HouseholdDeploymentConfig
from critic_sharing.deployment.household_compare import (
    METHODS,
    HouseholdComparisonConfig,
    HouseholdComparisonSupervisor,
    _init_to_json,
)


def _base_cfg() -> HouseholdDeploymentConfig:
    return HouseholdDeploymentConfig(
        model_path="/models/cleaner.zip",
        follower_model_path="/models/follower.zip",
        role="cleaner",
        sim_origin_in_world=(1.25, -1.0, 1.5707963267948966),
        parent_z_down=False,
    )


# ---------------------------------------------------------------------------
# HouseholdComparisonConfig validation
# ---------------------------------------------------------------------------

def test_compare_config_defaults():
    c = HouseholdComparisonConfig()
    assert c.repeats == 1
    assert c.order == ("naive", "ours")
    assert c.webcam_device == "OsmoAction5pro"
    assert c.bag_enabled is True


def test_compare_config_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown method"):
        HouseholdComparisonConfig(order=("naive", "secret_sauce"))


def test_compare_config_rejects_duplicate_method():
    with pytest.raises(ValueError, match="duplicates"):
        HouseholdComparisonConfig(order=("naive", "naive"))


def test_compare_config_rejects_zero_repeats():
    with pytest.raises(ValueError, match="repeats must be >= 1"):
        HouseholdComparisonConfig(repeats=0)


def test_method_spec_shape():
    assert set(METHODS.keys()) == {"naive", "ours"}
    for spec in METHODS.values():
        assert set(spec.keys()) == {"cleaner", "follower"}
    assert METHODS["naive"]["cleaner"] == "policy"
    assert METHODS["ours"]["cleaner"] == "cem"
    # Follower is PPO under both methods (per spec).
    assert METHODS["naive"]["follower"] == "policy"
    assert METHODS["ours"]["follower"] == "policy"


# ---------------------------------------------------------------------------
# _make_rollout_config — model swap by role, no_confirm forced on, etc.
# ---------------------------------------------------------------------------

def test_rollout_config_cleaner_uses_cleaner_model(tmp_path: Path):
    sup = HouseholdComparisonSupervisor(
        base_config=_base_cfg(),
        compare_config=HouseholdComparisonConfig(output_dir=tmp_path),
    )
    cfg = sup._make_rollout_config(
        method="naive", role="cleaner", controller="policy",
        rollout_dir=tmp_path / "p" / "naive_cleaner",
    )
    assert cfg.role == "cleaner"
    assert cfg.controller == "policy"
    assert cfg.model_path == "/models/cleaner.zip"
    assert cfg.episode is True
    assert cfg.no_confirm is True
    assert cfg.record is False
    assert cfg.record_dir == str(tmp_path / "p" / "naive_cleaner")
    # Follower path must remain available for CEM's V_F.
    assert cfg.follower_model_path == "/models/follower.zip"


def test_rollout_config_follower_uses_follower_model(tmp_path: Path):
    sup = HouseholdComparisonSupervisor(
        base_config=_base_cfg(),
        compare_config=HouseholdComparisonConfig(output_dir=tmp_path),
    )
    cfg = sup._make_rollout_config(
        method="naive", role="follower", controller="policy",
        rollout_dir=tmp_path / "p" / "naive_follower",
    )
    assert cfg.role == "follower"
    assert cfg.model_path == "/models/follower.zip"


def test_rollout_config_derives_cmd_topic_from_role_frame(tmp_path: Path):
    base = HouseholdDeploymentConfig(
        model_path="/m/cleaner.zip",
        follower_model_path="/m/follower.zip",
        cleaner_frame="robomaster_3",
        follower_frame="robomaster_1",
    )
    sup = HouseholdComparisonSupervisor(
        base_config=base,
        compare_config=HouseholdComparisonConfig(output_dir=tmp_path),
    )
    cleaner_cfg = sup._make_rollout_config(
        method="naive", role="cleaner", controller="policy",
        rollout_dir=tmp_path / "p" / "naive_cleaner",
    )
    follower_cfg = sup._make_rollout_config(
        method="naive", role="follower", controller="policy",
        rollout_dir=tmp_path / "p" / "naive_follower",
    )
    assert cleaner_cfg.cmd_topic == "/robomaster_3/reference_state"
    assert follower_cfg.cmd_topic == "/robomaster_1/reference_state"


def test_rollout_config_ours_cleaner_is_cem(tmp_path: Path):
    sup = HouseholdComparisonSupervisor(
        base_config=_base_cfg(),
        compare_config=HouseholdComparisonConfig(output_dir=tmp_path),
    )
    cfg = sup._make_rollout_config(
        method="ours", role="cleaner", controller="cem",
        rollout_dir=tmp_path / "p" / "ours_cleaner",
    )
    assert cfg.controller == "cem"
    assert cfg.model_path == "/models/cleaner.zip"
    assert cfg.follower_model_path == "/models/follower.zip"


# ---------------------------------------------------------------------------
# _init_to_json
# ---------------------------------------------------------------------------

def test_init_to_json_round_trips_through_json():
    init = {
        "cleaner_xy_sim": np.array([0.1, -0.2]),
        "cleaner_yaw_sim": np.float64(1.23),
        "follower_xy_sim": np.array([0.5, 0.6]),
        "follower_yaw_sim": np.float64(-0.45),
        "table_xy_sim": np.array([0.0, 0.7]),
        "table_yaw_sim": np.float64(0.0),
    }
    out = _init_to_json(init)
    parsed = json.loads(json.dumps(out))
    assert parsed["cleaner_xy_sim"] == [0.1, -0.2]
    assert parsed["cleaner_yaw_sim"] == pytest.approx(1.23)
    assert parsed["table_yaw_sim"] == 0.0


# ---------------------------------------------------------------------------
# _move_to_discarded — rename behavior
# ---------------------------------------------------------------------------

def test_move_to_discarded_renames(tmp_path: Path):
    sup = HouseholdComparisonSupervisor(
        base_config=_base_cfg(),
        compare_config=HouseholdComparisonConfig(output_dir=tmp_path),
    )
    target = tmp_path / "pair_001" / "naive_cleaner"
    target.mkdir(parents=True)
    (target / "metadata.json").write_text("{}")

    sup._move_to_discarded(target, attempt_idx=1)

    assert not target.exists()
    discarded = target.parent / "naive_cleaner.discarded.1"
    assert discarded.exists()
    assert (discarded / "metadata.json").exists()


def test_move_to_discarded_handles_collisions(tmp_path: Path):
    sup = HouseholdComparisonSupervisor(
        base_config=_base_cfg(),
        compare_config=HouseholdComparisonConfig(output_dir=tmp_path),
    )
    target = tmp_path / "p" / "naive_cleaner"
    target.mkdir(parents=True)
    sup._move_to_discarded(target, attempt_idx=1)

    target.mkdir()
    sup._move_to_discarded(target, attempt_idx=1)

    # Second discard should not have collided with the first.
    siblings = sorted(p.name for p in (tmp_path / "p").iterdir())
    assert "naive_cleaner.discarded.1" in siblings
    # The collision-resolver appends ".1" to the existing suffix.
    assert any(name.startswith("naive_cleaner.discarded.1.") for name in siblings)


def test_move_to_discarded_noop_when_missing(tmp_path: Path):
    sup = HouseholdComparisonSupervisor(
        base_config=_base_cfg(),
        compare_config=HouseholdComparisonConfig(output_dir=tmp_path),
    )
    # Should not raise.
    sup._move_to_discarded(tmp_path / "does_not_exist", attempt_idx=1)


# ---------------------------------------------------------------------------
# Session dir defaulting
# ---------------------------------------------------------------------------

def test_session_dir_defaults_under_experiments(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sup = HouseholdComparisonSupervisor(
        base_config=_base_cfg(),
        compare_config=HouseholdComparisonConfig(),  # no output_dir
    )
    assert sup.session_dir.parts[-2] == "comparisons"
    assert sup.session_dir.parts[-3] == "experiments"


def test_session_dir_honors_explicit(tmp_path: Path):
    custom = tmp_path / "my_session"
    sup = HouseholdComparisonSupervisor(
        base_config=_base_cfg(),
        compare_config=HouseholdComparisonConfig(output_dir=custom),
    )
    assert sup.session_dir == custom
