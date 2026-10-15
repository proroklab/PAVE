"""Central config resolver — single source of truth for all config loading.

Resolution chain (later wins):
    common YAML → scenario YAML → --config CLI YAML → CLI flags

All config values ultimately originate from the YAML files under ``configs/``.
Dataclass field defaults are only used as fallbacks when a key is absent from
every layer in the chain.
"""

from __future__ import annotations

import copy
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Type, TypeVar

from critic_sharing.config.schema import (
    PlannerConfig,
    DistillationConfig,
    GuidanceConfig,
    HouseholdCleanerEnvConfig,
    HouseholdNavEnvConfig,
    MobileArmFollowerEnvConfig,
    MobileArmLeaderEnvConfig,
    NavEnvConfig,
    OODConfig,
    OnlineAdaptationCostConfig,
    PushEnvConfig,
    VaeConfig,
    WarehouseFollowerEnvConfig,
    WarehouseLeaderEnvConfig,
)
from critic_sharing.utils.paths import (
    dataclass_from_mapping,
    load_mapping_file,
    project_root,
)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

_CONFIGS_DIR: Path | None = None


def _configs_root() -> Path:
    """Return the ``configs/`` directory at the repository root."""
    global _CONFIGS_DIR
    if _CONFIGS_DIR is None:
        _CONFIGS_DIR = project_root() / "configs"
    return _CONFIGS_DIR


# ---------------------------------------------------------------------------
# Deep-merge utility
# ---------------------------------------------------------------------------


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into *base* (returns a new dict)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


# ---------------------------------------------------------------------------
# Low-level loaders
# ---------------------------------------------------------------------------


def _load_yaml_or_empty(path: Path) -> Dict[str, Any]:
    """Load a YAML file; return ``{}`` if the file does not exist."""
    if not path.exists():
        return {}
    return load_mapping_file(path)


def _load_and_merge(
    *yaml_paths: Path,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load multiple YAML files and merge them left-to-right.

    The final *overrides* dict (if provided) wins over everything.
    """
    merged: Dict[str, Any] = {}
    for path in yaml_paths:
        layer = _load_yaml_or_empty(path)
        merged = _deep_merge(merged, layer)
    if overrides:
        merged = _deep_merge(merged, overrides)
    return merged


def _to_dataclass(mapping: Mapping[str, Any], cls: Type[T]) -> T:
    """Instantiate a dataclass from a mapping, dropping unknown keys."""
    return dataclass_from_mapping(mapping, cls)


# ---------------------------------------------------------------------------
# Public API — env configs
# ---------------------------------------------------------------------------

_ENV_CONFIG_MAP: Dict[str, tuple[str, Type]] = {
    # (scenario, role) -> (yaml filename, dataclass)
    "obstacle_arena:follower": ("nav_env.yaml", NavEnvConfig),
    "obstacle_arena:leader": ("push_env.yaml", PushEnvConfig),
    "warehouse:follower": ("follower_env.yaml", WarehouseFollowerEnvConfig),
    "warehouse:leader": ("leader_env.yaml", WarehouseLeaderEnvConfig),
    # warehouse_pickup_blocked / warehouse_pickup_unseen — same env class
    # as vanilla warehouse, distinct config trees (see configs/<name>/).
    "warehouse_pickup_blocked:follower": ("follower_env.yaml", WarehouseFollowerEnvConfig),
    "warehouse_pickup_blocked:leader": ("leader_env.yaml", WarehouseLeaderEnvConfig),
    "warehouse_pickup_unseen:follower": ("follower_env.yaml", WarehouseFollowerEnvConfig),
    "warehouse_pickup_unseen:leader": ("leader_env.yaml", WarehouseLeaderEnvConfig),
    "mobile_arm:follower": ("follower_env.yaml", MobileArmFollowerEnvConfig),
    "mobile_arm:leader": ("leader_env.yaml", MobileArmLeaderEnvConfig),
    "household:follower": ("nav_env.yaml", HouseholdNavEnvConfig),
    "household:leader": ("clean_env.yaml", HouseholdCleanerEnvConfig),
}


def load_env_config(
    scenario: str,
    role: str = "follower",
    overrides: Optional[Dict[str, Any]] = None,
) -> NavEnvConfig | PushEnvConfig:
    """Load the environment config for a given scenario and role.

    Resolution: scenario YAML → overrides.
    (No common layer — env configs are scenario-specific.)
    """
    key = "{scenario}:{role}".format(scenario=scenario, role=role)
    if key not in _ENV_CONFIG_MAP:
        raise ValueError("Unknown scenario/role: {key}".format(key=key))
    yaml_name, cls = _ENV_CONFIG_MAP[key]
    yaml_path = _configs_root() / scenario / yaml_name
    merged = _load_and_merge(yaml_path, overrides=overrides)
    return _to_dataclass(merged, cls)


# ---------------------------------------------------------------------------
# Public API — training configs
# ---------------------------------------------------------------------------

_TRAIN_YAML_MAP: Dict[str, str] = {
    "obstacle_arena:follower": "train_nav.yaml",
    "obstacle_arena:leader": "train_push.yaml",
    "warehouse:follower": "train_follower.yaml",
    "warehouse:leader": "train_leader.yaml",
    "warehouse_pickup_blocked:follower": "train_follower.yaml",
    "warehouse_pickup_blocked:leader": "train_leader.yaml",
    "warehouse_pickup_unseen:follower": "train_follower.yaml",
    "warehouse_pickup_unseen:leader": "train_leader.yaml",
    "mobile_arm:follower": "train_follower.yaml",
    "mobile_arm:leader": "train_leader.yaml",
    "household:follower": "train_nav.yaml",
    "household:leader": "train_clean.yaml",
}


def load_training_section(
    scenario: str,
    role: str = "follower",
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load training hyperparameters as a flat dict.

    Resolution: common/training.yaml → scenario training YAML → overrides.
    """
    key = "{scenario}:{role}".format(scenario=scenario, role=role)
    if key not in _TRAIN_YAML_MAP:
        raise ValueError("Unknown scenario/role: {key}".format(key=key))
    yaml_name = _TRAIN_YAML_MAP[key]
    common_path = _configs_root() / "common" / "training.yaml"
    scenario_path = _configs_root() / scenario / yaml_name
    return _load_and_merge(common_path, scenario_path, overrides=overrides)


# ---------------------------------------------------------------------------
# Public API — planner configs
# ---------------------------------------------------------------------------


def load_planner_config(
    scenario: str = "obstacle_arena",
    overrides: Optional[Dict[str, Any]] = None,
) -> PlannerConfig:
    """Load CEM planner config.

    Resolution: common/planner.yaml → scenario/planner.yaml → overrides.
    """
    common_path = _configs_root() / "common" / "planner.yaml"
    scenario_path = _configs_root() / scenario / "planner.yaml"
    merged = _load_and_merge(common_path, scenario_path, overrides=overrides)
    return _to_dataclass(merged, PlannerConfig)


def load_cost_config(
    scenario: str = "obstacle_arena",
    overrides: Optional[Dict[str, Any]] = None,
) -> OnlineAdaptationCostConfig:
    """Load online-adaptation cost config.

    Resolution: common/cost.yaml → scenario/cost.yaml → overrides.
    """
    common_path = _configs_root() / "common" / "cost.yaml"
    scenario_path = _configs_root() / scenario / "cost.yaml"
    merged = _load_and_merge(common_path, scenario_path, overrides=overrides)
    return _to_dataclass(merged, OnlineAdaptationCostConfig)


# ---------------------------------------------------------------------------
# Public API — guidance configs
# ---------------------------------------------------------------------------


def load_guidance_config(
    scenario: str = "obstacle_arena",
    overrides: Optional[Dict[str, Any]] = None,
) -> GuidanceConfig:
    """Load guidance config (distillation + OOD).

    Resolution: common/guidance.yaml → scenario/guidance.yaml → overrides.
    """
    common_path = _configs_root() / "common" / "guidance.yaml"
    scenario_path = _configs_root() / scenario / "guidance.yaml"
    merged = _load_and_merge(common_path, scenario_path, overrides=overrides)
    distillation = _to_dataclass(merged.get("distillation", {}), DistillationConfig)
    ood = _to_dataclass(merged.get("ood", {}), OODConfig)
    return GuidanceConfig(distillation=distillation, ood=ood)


# ---------------------------------------------------------------------------
# Public API — VAE config (obstacle_arena specific, nested in training YAML)
# ---------------------------------------------------------------------------


def load_vae_config(
    scenario: str = "obstacle_arena",
    role: str = "follower",
    overrides: Optional[Dict[str, Any]] = None,
) -> VaeConfig:
    """Load VAE config from the training YAML's ``vae`` section."""
    training = load_training_section(scenario, role)
    vae_section = training.get("vae", {})
    if overrides:
        vae_section = _deep_merge(vae_section, overrides)
    return _to_dataclass(vae_section, VaeConfig)


# ---------------------------------------------------------------------------
# Public API — deployment configs (ROS2)
# ---------------------------------------------------------------------------


def load_household_deployment_config(
    config_override_path: Optional[Path] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
):
    """Load the household ROS2 deployment config.

    Resolution chain: ``configs/deployment/household.yaml`` →
    ``--config`` YAML → CLI flags.
    """
    from critic_sharing.config.schema import HouseholdDeploymentConfig

    base_path = _configs_root() / "deployment" / "household.yaml"
    merged = _load_yaml_or_empty(base_path)
    if config_override_path is not None:
        merged = _deep_merge(merged, _load_yaml_or_empty(Path(config_override_path)))
    if cli_overrides:
        merged = _deep_merge(merged, cli_overrides)
    if "sim_origin_in_world" in merged and isinstance(merged["sim_origin_in_world"], list):
        merged["sim_origin_in_world"] = tuple(merged["sim_origin_in_world"])
    return _to_dataclass(merged, HouseholdDeploymentConfig)


# ---------------------------------------------------------------------------
# Public API — full config bundle for a CLI command
# ---------------------------------------------------------------------------


def load_train_config(
    scenario: str,
    role: str,
    config_override_path: Optional[Path] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load the complete training config bundle.

    Returns a dict with keys matching the training YAML sections:
    ``training``, ``env``, ``vae`` (if present), ``paths``, ``wandb``, etc.

    Resolution chain:
        common YAML → scenario YAML → --config file → CLI flags
    """
    # 1. Merge common + scenario training YAML
    key = "{scenario}:{role}".format(scenario=scenario, role=role)
    if key not in _TRAIN_YAML_MAP:
        raise ValueError("Unknown scenario/role: {key}".format(key=key))
    yaml_name = _TRAIN_YAML_MAP[key]
    common_path = _configs_root() / "common" / "training.yaml"
    scenario_path = _configs_root() / scenario / yaml_name

    merged = _load_and_merge(common_path, scenario_path)

    # 2. Apply --config YAML override file
    if config_override_path is not None:
        file_overrides = _load_yaml_or_empty(Path(config_override_path))
        merged = _deep_merge(merged, file_overrides)

    # 3. Apply explicit CLI flags (flat keys merged into top-level)
    if cli_overrides:
        merged = _deep_merge(merged, cli_overrides)

    return merged


def load_run_config(
    scenario: str,
    config_override_path: Optional[Path] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load the complete run/eval config bundle.

    Returns a dict with typed config objects::

        {
            "nav_env": NavEnvConfig,
            "push_env": PushEnvConfig,       # obstacle_arena only
            "planner": PlannerConfig,
            "cost": OnlineAdaptationCostConfig,
            "guidance": GuidanceConfig,
        }
    """
    # File-level overrides
    file_overrides: Dict[str, Any] = {}
    if config_override_path is not None:
        file_overrides = _load_yaml_or_empty(Path(config_override_path))

    # Merge CLI overrides into file overrides
    all_overrides = _deep_merge(file_overrides, cli_overrides or {})

    result: Dict[str, Any] = {
        "planner": load_planner_config(
            scenario, overrides=all_overrides.get("planner")
        ),
        "cost": load_cost_config(
            scenario, overrides=all_overrides.get("cost")
        ),
        "guidance": load_guidance_config(
            scenario, overrides=all_overrides.get("guidance")
        ),
    }

    if scenario == "obstacle_arena":
        result["nav_env"] = load_env_config(
            scenario, "follower", overrides=all_overrides.get("nav_env")
        )
        result["push_env"] = load_env_config(
            scenario, "leader", overrides=all_overrides.get("push_env")
        )
    elif scenario in (
        "warehouse",
        "warehouse_pickup_blocked",
        "warehouse_pickup_unseen",
    ):
        result["leader_env"] = load_env_config(
            scenario, "leader", overrides=all_overrides.get("leader_env")
        )
        result["follower_env"] = load_env_config(
            scenario, "follower", overrides=all_overrides.get("follower_env")
        )
    elif scenario == "mobile_arm":
        result["leader_env"] = load_env_config(
            scenario, "leader", overrides=all_overrides.get("leader_env")
        )
        result["follower_env"] = load_env_config(
            scenario, "follower", overrides=all_overrides.get("follower_env")
        )
    elif scenario == "household":
        result["clean_env"] = load_env_config(
            scenario, "leader", overrides=all_overrides.get("clean_env")
        )
        result["nav_env"] = load_env_config(
            scenario, "follower", overrides=all_overrides.get("nav_env")
        )

    return result


# ---------------------------------------------------------------------------
# Helpers for external use
# ---------------------------------------------------------------------------


def configs_root() -> Path:
    """Return the resolved ``configs/`` directory path."""
    return _configs_root()


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Public deep-merge utility."""
    return _deep_merge(base, override)


def apply_env_overrides(env_config: T, overrides: Optional[Mapping[str, Any]]) -> T:
    """Apply a training YAML's ``env_overrides`` block to an env config.

    ``train_<role>.yaml`` may reshape the training-time env distribution
    (e.g. mobile_arm's spawn cone) without changing the env's deployment
    defaults. Unknown keys raise ``ValueError`` so typos surface. ``None``
    or an empty block is a no-op.
    """
    if not overrides:
        return env_config
    from dataclasses import replace

    unknown = set(overrides) - {f.name for f in fields(env_config)}
    if unknown:
        raise ValueError(
            f"env_overrides has unknown fields for "
            f"{type(env_config).__name__}: {sorted(unknown)}"
        )
    return replace(env_config, **overrides)


__all__ = [
    "apply_env_overrides",
    "configs_root",
    "deep_merge",
    "load_cost_config",
    "load_env_config",
    "load_guidance_config",
    "load_household_deployment_config",
    "load_planner_config",
    "load_run_config",
    "load_train_config",
    "load_training_section",
    "load_vae_config",
]
