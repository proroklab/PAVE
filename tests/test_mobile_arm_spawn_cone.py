"""Spawn-cone restriction for mobile_arm follower env.

The follower env's ``object_theta_low`` / ``object_theta_high`` fields
restrict where objects spawn in base-frame polar coordinates. Defaults
preserve the full circle (back-compat); per-training-run cones flow
through ``configs/<scenario>/train_follower.yaml::env_overrides`` so the
VAE training distribution can be matched to V_F's training distribution.

Tests assert:

* The default config still produces full-circle spawns.
* A configured cone is honored on every reset (no leakage).
* The training-config ``env_overrides`` block actually applies its
  values when loaded via ``load_train_config``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from critic_sharing.config.resolver import load_env_config, load_train_config
from critic_sharing.scenarios.mobile_arm.follower_env import MobileArmFollowerEnv


def _phi_of(obj_pos: np.ndarray) -> float:
    return float(np.arctan2(obj_pos[1], obj_pos[0]))


class TestSpawnCone:
    def test_default_full_circle(self):
        """Schema defaults are ±π → full-circle spawns."""
        cfg = load_env_config("mobile_arm", "follower")
        # Force defaults regardless of any YAML overrides.
        cfg = replace(cfg, object_theta_low=-np.pi, object_theta_high=np.pi)
        env = MobileArmFollowerEnv(cfg)
        phis = []
        for s in range(200):
            env.reset(seed=s)
            phis.append(_phi_of(env._object_pos))
        phis = np.asarray(phis)
        # With 200 samples in [-π, π] we expect to cover both extremes.
        assert phis.min() < -np.pi / 2
        assert phis.max() > np.pi / 2

    def test_cone_restriction_honored(self):
        """A configured cone is honored — no spawn outside the bounds."""
        cfg = load_env_config("mobile_arm", "follower")
        low, high = -1.0, 1.0
        cfg = replace(cfg, object_theta_low=low, object_theta_high=high)
        env = MobileArmFollowerEnv(cfg)
        for s in range(200):
            env.reset(seed=s)
            phi = _phi_of(env._object_pos)
            assert low - 1e-6 <= phi <= high + 1e-6, (
                "phi={p} outside cone [{l}, {h}] at seed={s}".format(
                    p=phi, l=low, h=high, s=s,
                )
            )

    def test_train_follower_yaml_carries_env_overrides(self):
        """The mobile_arm train_follower.yaml ships an env_overrides block
        that, when applied to the env config via the same helper
        ``train follower`` uses, restricts the spawn cone to the
        configured values."""
        from critic_sharing.cli.train import _apply_env_overrides

        resolved = load_train_config(
            scenario="mobile_arm", role="follower",
            config_override_path=None, cli_overrides={},
        )
        overrides = resolved.get("env_overrides")
        assert overrides is not None, (
            "configs/mobile_arm/train_follower.yaml must define env_overrides"
        )
        # Defaults should not equal overrides (otherwise the test would
        # pass trivially and the override block isn't actually doing
        # anything).
        cfg = load_env_config("mobile_arm", "follower")
        cfg = replace(cfg, object_theta_low=-np.pi, object_theta_high=np.pi)
        cfg = _apply_env_overrides(cfg, overrides)
        assert -np.pi < float(cfg.object_theta_low) < float(cfg.object_theta_high) < np.pi
