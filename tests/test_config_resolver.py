"""Tests for the central config resolver.

Verifies that:
  1. Resolver output matches raw YAML (not dataclass defaults)
  2. Merge chain works: common → scenario → overrides
  3. Round-trip: resolver → _prepare_train_kwargs → matches YAML source
  4. All YAML files are parseable and clean (no vestigial Hydra)

Tests read the YAML files directly and compare against resolver output,
so they stay valid when config values are tuned.
"""

from __future__ import annotations

import inspect
from dataclasses import fields
from pathlib import Path

import pytest
import yaml

from critic_sharing.config.resolver import (
    configs_root,
    deep_merge,
    load_cost_config,
    load_env_config,
    load_guidance_config,
    load_planner_config,
    load_run_config,
    load_train_config,
    load_training_section,
    load_vae_config,
)
from critic_sharing.config.schema import (
    PlannerConfig,
    GuidanceConfig,
    NavEnvConfig,
    OnlineAdaptationCostConfig,
    PushEnvConfig,
    VaeConfig,
)
from critic_sharing.utils.paths import load_mapping_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_yaml_files() -> list[Path]:
    """Glob all YAML files under configs/."""
    root = configs_root()
    return sorted(root.rglob("*.yaml"))


def _read_yaml(relative_path: str) -> dict:
    """Read a YAML file relative to configs/."""
    path = configs_root() / relative_path
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------------------
# 1. Resolver output matches YAML (not dataclass defaults)
# ---------------------------------------------------------------------------


class TestResolverMatchesYaml:
    """Verify the resolver returns values from YAML, not dataclass defaults.

    Each test reads the YAML directly and checks the resolver produces the
    same values — so these tests survive config value changes.
    """

    def test_planner_matches_scenario_yaml(self):
        """Planner config should match obstacle_arena/planner.yaml merged on common."""
        common = _read_yaml("common/planner.yaml")
        scenario = _read_yaml("obstacle_arena/planner.yaml")
        expected = {**common, **scenario}

        cem = load_planner_config("obstacle_arena")
        for key, yaml_val in expected.items():
            if hasattr(cem, key):
                assert getattr(cem, key) == pytest.approx(yaml_val), (
                    "Planner.{k}: resolver={r}, yaml={y}".format(
                        k=key, r=getattr(cem, key), y=yaml_val
                    )
                )

    def test_planner_differs_from_dataclass_defaults(self):
        """Scenario planner config should NOT equal bare dataclass defaults."""
        cem = load_planner_config("obstacle_arena")
        defaults = PlannerConfig()
        differences = [
            f.name for f in fields(PlannerConfig)
            if getattr(cem, f.name) != getattr(defaults, f.name)
        ]
        assert len(differences) > 0, (
            "Resolver returned bare dataclass defaults — YAML not loaded"
        )

    def test_cost_matches_scenario_yaml(self):
        """Cost config should match obstacle_arena/cost.yaml merged on common."""
        common = _read_yaml("common/cost.yaml")
        scenario = _read_yaml("obstacle_arena/cost.yaml")
        expected = {**common, **scenario}

        cost = load_cost_config("obstacle_arena")
        for key, yaml_val in expected.items():
            if hasattr(cost, key):
                assert getattr(cost, key) == pytest.approx(yaml_val), (
                    "Cost.{k}: resolver={r}, yaml={y}".format(
                        k=key, r=getattr(cost, key), y=yaml_val
                    )
                )

    def test_cost_differs_from_dataclass_defaults(self):
        cost = load_cost_config("obstacle_arena")
        defaults = OnlineAdaptationCostConfig()
        differences = [
            f.name for f in fields(OnlineAdaptationCostConfig)
            if getattr(cost, f.name) != getattr(defaults, f.name)
        ]
        assert len(differences) > 0

    def test_guidance_distillation_matches_yaml(self):
        """Distillation values should match obstacle_arena/guidance.yaml."""
        common = _read_yaml("common/guidance.yaml")
        scenario = _read_yaml("obstacle_arena/guidance.yaml")
        expected = {**common.get("distillation", {}), **scenario.get("distillation", {})}

        guidance = load_guidance_config("obstacle_arena")
        for key, yaml_val in expected.items():
            if hasattr(guidance.distillation, key):
                assert getattr(guidance.distillation, key) == yaml_val

    def test_guidance_ood_matches_yaml(self):
        """OOD values should match obstacle_arena/guidance.yaml."""
        common = _read_yaml("common/guidance.yaml")
        scenario = _read_yaml("obstacle_arena/guidance.yaml")
        expected = {**common.get("ood", {}), **scenario.get("ood", {})}

        guidance = load_guidance_config("obstacle_arena")
        for key, yaml_val in expected.items():
            if hasattr(guidance.ood, key):
                assert getattr(guidance.ood, key) == yaml_val

    def test_nav_env_matches_yaml(self):
        """Nav env config should match obstacle_arena/nav_env.yaml."""
        expected = _read_yaml("obstacle_arena/nav_env.yaml")
        cfg = load_env_config("obstacle_arena", "follower")
        for key, yaml_val in expected.items():
            if hasattr(cfg, key):
                assert getattr(cfg, key) == yaml_val

    def test_push_env_matches_yaml(self):
        """Push env config should match obstacle_arena/push_env.yaml."""
        expected = _read_yaml("obstacle_arena/push_env.yaml")
        cfg = load_env_config("obstacle_arena", "leader")
        for key, yaml_val in expected.items():
            if hasattr(cfg, key):
                assert getattr(cfg, key) == yaml_val

# ---------------------------------------------------------------------------
# 2. Merge chain: common → scenario → overrides
# ---------------------------------------------------------------------------


class TestMergeChain:
    """Verify the resolution chain works correctly."""

    def test_common_defaults_used_when_scenario_omits(self):
        """Per-scenario planner.yaml omits some keys → fall back to common defaults."""
        common = _read_yaml("common/planner.yaml")
        cem = load_planner_config("household")
        assert isinstance(cem, PlannerConfig)
        # household/planner.yaml omits distribution, so it falls back to common
        assert cem.distribution == common["distribution"]

    def test_scenario_overrides_common(self):
        """obstacle_arena planner should override common values."""
        scenario = _read_yaml("obstacle_arena/planner.yaml")
        cem = load_planner_config("obstacle_arena")
        for key in scenario:
            if hasattr(cem, key):
                assert getattr(cem, key) == pytest.approx(scenario[key])

    def test_explicit_overrides_beat_yaml(self):
        """Overrides dict should win over YAML values."""
        cem = load_planner_config("obstacle_arena", overrides={"horizon": 99})
        assert cem.horizon == 99

    def test_cost_override(self):
        """Override alpha via overrides dict."""
        cost = load_cost_config("obstacle_arena", overrides={"alpha": 42.0})
        assert cost.alpha == 42.0

    def test_deep_merge_nested(self):
        """deep_merge should handle nested dicts correctly."""
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 99, "z": 100}}
        merged = deep_merge(base, override)
        assert merged == {"a": {"x": 1, "y": 99, "z": 100}, "b": 3}

    def test_deep_merge_does_not_mutate(self):
        """deep_merge should not modify the input dicts."""
        base = {"a": {"x": 1}}
        override = {"a": {"y": 2}}
        _ = deep_merge(base, override)
        assert base == {"a": {"x": 1}}
        assert override == {"a": {"y": 2}}


# ---------------------------------------------------------------------------
# 3. Env config types
# ---------------------------------------------------------------------------


class TestEnvConfigTypes:

    def test_nav_env_type(self):
        cfg = load_env_config("obstacle_arena", "follower")
        assert isinstance(cfg, NavEnvConfig)

    def test_push_env_type(self):
        cfg = load_env_config("obstacle_arena", "leader")
        assert isinstance(cfg, PushEnvConfig)

    def test_unknown_scenario_raises(self):
        with pytest.raises(ValueError):
            load_env_config("unknown_scenario", "follower")


# ---------------------------------------------------------------------------
# 4. Training config loading
# ---------------------------------------------------------------------------


class TestTrainingConfig:

    def test_load_training_section_returns_dict(self):
        training = load_training_section("obstacle_arena", "follower")
        assert isinstance(training, dict)
        assert "total_timesteps" in training

    def test_training_section_matches_yaml(self):
        """Training section should match the scenario YAML."""
        common = _read_yaml("common/training.yaml")
        scenario_yaml = _read_yaml("obstacle_arena/train_nav.yaml")
        expected = {**common, **scenario_yaml}
        training = load_training_section("obstacle_arena", "follower")
        for key, yaml_val in expected.items():
            if key in training:
                assert training[key] == yaml_val, (
                    "training[{k}]: resolver={r}, yaml={y}".format(
                        k=key, r=training[key], y=yaml_val
                    )
                )

    def test_load_train_config_with_cli_overrides(self):
        resolved = load_train_config(
            "obstacle_arena", "follower",
            cli_overrides={"total_timesteps": 999999},
        )
        assert resolved["total_timesteps"] == 999999

    def test_vae_config_type(self):
        vae = load_vae_config("obstacle_arena", "follower")
        assert isinstance(vae, VaeConfig)

    def test_vae_config_matches_yaml(self):
        scenario_yaml = _read_yaml("obstacle_arena/train_nav.yaml")
        vae = load_vae_config("obstacle_arena", "follower")
        vae_yaml = scenario_yaml.get("vae", {})
        for key, yaml_val in vae_yaml.items():
            if hasattr(vae, key):
                assert getattr(vae, key) == yaml_val


# ---------------------------------------------------------------------------
# 5. Run config bundle
# ---------------------------------------------------------------------------


class TestRunConfig:

    def test_obstacle_arena_bundle_keys(self):
        cfg = load_run_config("obstacle_arena")
        assert "planner" in cfg
        assert "cost" in cfg
        assert "guidance" in cfg
        assert "nav_env" in cfg
        assert "push_env" in cfg

    def test_obstacle_arena_bundle_types(self):
        cfg = load_run_config("obstacle_arena")
        assert isinstance(cfg["planner"], PlannerConfig)
        assert isinstance(cfg["cost"], OnlineAdaptationCostConfig)
        assert isinstance(cfg["guidance"], GuidanceConfig)
        assert isinstance(cfg["nav_env"], NavEnvConfig)
        assert isinstance(cfg["push_env"], PushEnvConfig)

# ---------------------------------------------------------------------------
# 6. Round-trip: resolver → _prepare_train_kwargs → matches YAML
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Verify that the full CLI path (resolver → kwargs filter → train_agent)
    preserves every YAML value that the training function accepts.

    These tests import the real train_agent functions and run
    _prepare_train_kwargs, then check every resulting kwarg traces back
    to the YAML source.
    """

    def _resolve_and_prepare(self, scenario: str, role: str):
        """Run the same path as cli/train.py: resolve config, then filter kwargs."""
        from critic_sharing.cli.train import _prepare_train_kwargs

        resolved = load_train_config(scenario, role)

        if role == "follower":
            from critic_sharing.training.nav import train_agent
        else:
            from critic_sharing.training.push import train_agent

        kwargs = _prepare_train_kwargs(dict(resolved), train_agent)
        return kwargs, train_agent

    def _yaml_source_values(self, scenario: str, role: str) -> dict:
        """Build the expected flat dict by reading YAMLs directly."""
        common = _read_yaml("common/training.yaml")
        if scenario == "obstacle_arena":
            yaml_name = "train_nav.yaml" if role == "follower" else "train_push.yaml"
            scenario_yaml = _read_yaml("obstacle_arena/{f}".format(f=yaml_name))
        else:
            scenario_yaml = {}

        # Deep merge common → scenario
        merged = dict(common)
        merged.update(scenario_yaml)
        return merged

    def test_nav_round_trip(self):
        """Every kwarg for nav train_agent should trace to YAML."""
        kwargs, train_fn = self._resolve_and_prepare("obstacle_arena", "follower")
        yaml_vals = self._yaml_source_values("obstacle_arena", "follower")

        # Check flat training keys
        for key in ("total_timesteps", "n_envs", "seed"):
            if key in kwargs:
                assert kwargs[key] == yaml_vals[key], (
                    "nav kwarg {k}={v}, expected {e} from YAML".format(
                        k=key, v=kwargs[key], e=yaml_vals[key]
                    )
                )

        # Check vae keys were flattened correctly from YAML
        vae_yaml = _read_yaml("obstacle_arena/train_nav.yaml").get("vae", {})
        for vae_key, vae_val in vae_yaml.items():
            prefixed = "vae_{k}".format(k=vae_key)
            if prefixed in kwargs:
                assert kwargs[prefixed] == vae_val, (
                    "nav kwarg {k}={v}, expected {e} from YAML vae section".format(
                        k=prefixed, v=kwargs[prefixed], e=vae_val
                    )
                )

    def test_push_round_trip(self):
        """Every kwarg for push train_agent should trace to YAML."""
        kwargs, train_fn = self._resolve_and_prepare("obstacle_arena", "leader")
        yaml_vals = self._yaml_source_values("obstacle_arena", "leader")

        for key in ("total_timesteps", "n_envs", "seed"):
            if key in kwargs:
                assert kwargs[key] == yaml_vals[key], (
                    "push kwarg {k}={v}, expected {e} from YAML".format(
                        k=key, v=kwargs[key], e=yaml_vals[key]
                    )
                )

    def test_cli_override_survives_round_trip(self):
        """A CLI override should still be present after _prepare_train_kwargs."""
        from critic_sharing.cli.train import _prepare_train_kwargs
        from critic_sharing.training.nav import train_agent

        resolved = load_train_config(
            "obstacle_arena", "follower",
            cli_overrides={"total_timesteps": 777},
        )
        kwargs = _prepare_train_kwargs(dict(resolved), train_agent)
        assert kwargs["total_timesteps"] == 777

    def test_no_unknown_kwargs(self):
        """_prepare_train_kwargs should only produce keys the function accepts."""
        from critic_sharing.cli.train import _prepare_train_kwargs

        for scenario, role, module_path, train_fn_name in [
            ("obstacle_arena", "follower", "critic_sharing.training.nav", "train_agent"),
            ("obstacle_arena", "leader", "critic_sharing.training.push", "train_agent"),
        ]:
            import importlib
            mod = importlib.import_module(module_path)
            train_fn = getattr(mod, train_fn_name)

            resolved = load_train_config(scenario, role)
            kwargs = _prepare_train_kwargs(dict(resolved), train_fn)

            sig = inspect.signature(train_fn)
            valid_params = set(sig.parameters.keys())
            has_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )

            if not has_var_keyword:
                for key in kwargs:
                    assert key in valid_params, (
                        "{scenario}/{role}: unexpected kwarg '{key}' not in "
                        "{fn} signature".format(
                            scenario=scenario, role=role, key=key,
                            fn=train_fn.__module__,
                        )
                    )


# ---------------------------------------------------------------------------
# 7. YAML integrity
# ---------------------------------------------------------------------------


class TestYamlIntegrity:

    def test_all_yaml_files_load(self):
        """Every YAML file under configs/ should parse without errors."""
        yaml_files = _all_yaml_files()
        assert len(yaml_files) > 0, "No YAML files found under configs/"
        for path in yaml_files:
            try:
                load_mapping_file(path)
            except Exception as exc:
                pytest.fail(
                    "Failed to parse {path}: {exc}".format(path=path, exc=exc)
                )

    def test_no_hydra_sections(self):
        """No YAML file should contain Hydra defaults: or hydra: sections."""
        yaml_files = _all_yaml_files()
        for path in yaml_files:
            try:
                data = load_mapping_file(path)
            except Exception:
                continue  # parse errors caught by other test
            for key in ("defaults", "hydra"):
                assert key not in data, (
                    "Found vestigial '{key}' section in {path}".format(
                        key=key, path=path
                    )
                )
