"""Unit tests for :mod:`evaluations._common.csv_loader`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from evaluations._common.csv_loader import (
    gather_grid,
    load_compare_csv,
    load_run_dir,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_compare_csv_injects_scenario(tmp_path: Path):
    p = tmp_path / "compare_warehouse.csv"
    _write_csv(p, [
        {"mode": "unguided", "episode": 0, "leader_return": 1.0, "follower_return": 2.0},
        {"mode": "unguided", "episode": 1, "leader_return": 1.5, "follower_return": 2.5},
    ])
    df = load_compare_csv(p)
    assert set(df["scenario"].unique()) == {"warehouse"}
    assert len(df) == 2


def test_load_run_dir_concatenates(tmp_path: Path):
    _write_csv(tmp_path / "compare_warehouse.csv",
               [{"mode": "u", "episode": 0, "leader_return": 1.0}])
    _write_csv(tmp_path / "compare_mobile_arm.csv",
               [{"mode": "u", "episode": 0, "leader_return": 2.0}])
    df = load_run_dir(tmp_path)
    assert set(df["scenario"].unique()) == {"warehouse", "mobile_arm"}
    assert len(df) == 2


def test_load_run_dir_empty_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_run_dir(tmp_path)


def test_gather_grid_filters_by_scenario(tmp_path: Path):
    _write_csv(tmp_path / "cell0" / "compare_warehouse.csv",
               [{"mode": "u", "episode": 0, "leader_return": 1.0}])
    _write_csv(tmp_path / "cell0" / "compare_mobile_arm.csv",
               [{"mode": "u", "episode": 0, "leader_return": 2.0}])
    df = gather_grid(tmp_path, scenarios=("warehouse",))
    assert set(df["scenario"].unique()) == {"warehouse"}


def test_gather_grid_filters_by_seed(tmp_path: Path):
    _write_csv(tmp_path / "run_seed0" / "compare_warehouse.csv",
               [{"mode": "u", "episode": 0, "leader_return": 1.0}])
    _write_csv(tmp_path / "run_seed5" / "compare_warehouse.csv",
               [{"mode": "u", "episode": 0, "leader_return": 2.0}])
    _write_csv(tmp_path / "run_seed42" / "compare_warehouse.csv",
               [{"mode": "u", "episode": 0, "leader_return": 3.0}])
    df = gather_grid(tmp_path, seeds=(0, 42))
    cells = set(df["cell"].unique())
    assert cells == {"run_seed0", "run_seed42"}


def test_gather_grid_empty_returns_empty_frame(tmp_path: Path):
    df = gather_grid(tmp_path)
    assert df.empty


def test_gather_grid_preserves_cell_label(tmp_path: Path):
    _write_csv(tmp_path / "alpha_0.5" / "seed_0" / "compare_warehouse.csv",
               [{"mode": "u", "episode": 0, "leader_return": 1.0}])
    df = gather_grid(tmp_path, extra_label="cell")
    assert df["cell"].iloc[0] == "alpha_0.5/seed_0"
