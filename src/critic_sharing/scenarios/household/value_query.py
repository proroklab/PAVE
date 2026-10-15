"""Follower value queries for the household scenario.

Two V_F builders are exposed:

* :func:`build_household_follower_value_fn` — **per-episode** V_F.
  Bakes the realized follower spawn pose ``(f_x, f_y, θ_f)`` into the
  closure and queries the nav PPO critic directly with the candidate
  table pose extracted from a leader (cleaner) obs. Single PPO
  forward per terminal-state query. This is the canonical eval-side
  V_F (mirrors warehouse / mobile_arm).
* :func:`build_nav_value_approximator` — legacy **marginalized** V_F.
  Averages the nav PPO critic over a grid of follower poses
  ``[f_x, f_y, sin θ_f, cos θ_f]`` (collapsing the follower-specific
  axes of the state space) and distills the result into a small MLP
  (``NavValueAggregator``). Kept for the standalone training /
  visualize paths that pre-compute a scenario-wide V heatmap.
"""

from __future__ import annotations

import os
import sys
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch as th
from stable_baselines3 import PPO

from critic_sharing.config.schema import HouseholdNavEnvConfig as NavEnvConfig
from critic_sharing.guidance.distillation import build_mlp_regressor, fit_pointwise_regressor
from critic_sharing.guidance.ood import (
    OODMetric,
    load_vae_checkpoint,
    load_vae_model,
    reconstruction_losses,
    roc_threshold,
)
from critic_sharing.scenarios.household.common import rect_circumscribed_radius


SHARED_STATE_DIM = 4  # [t_x, t_y, sin θ_t, cos θ_t]


# ---------------------------------------------------------------------------
# Per-episode V_F (canonical eval-side builder)
# ---------------------------------------------------------------------------

# Leader (cleaner) obs layout (see push_env.py::_get_obs):
#   [c_x, c_y, sin θ_c, cos θ_c, T_X, T_Y, SIN_T, COS_T, dirt_x, dirt_y]  (10-D)
# Follower (nav) obs layout (see nav_env.py::_get_obs):
#   [f_x, f_y, sin θ_f, cos θ_f, T_X, T_Y, SIN_T, COS_T]                   (8-D)
# Both share columns 4-7 for table pose.
LEADER_OBS_TABLE_COLS = (4, 5, 6, 7)


def build_household_follower_value_fn(
    follower_model: PPO,
    follower_config: NavEnvConfig,
    follower_spawn_pos: np.ndarray,
    follower_spawn_theta: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a per-episode V_F closure for the household navigator.

    Args:
        follower_model: trained follower PPO model (nav agent).
        follower_config: follower env config (currently unused; kept
            for API symmetry).
        follower_spawn_pos: ``(2,)`` follower ``(x, y)`` for this episode.
        follower_spawn_theta: follower yaw at spawn (radians).

    Returns:
        ``value_fn(leader_obs_batch) -> values`` of shape ``(B,)``. The
        ``leader_obs_batch`` is ``(B, 10)``; only columns 4-7 (table
        pose) are read. The 8-D counterfactual follower obs that is
        built and fed to the PPO critic is
        ``[f_x, f_y, sin θ_f, cos θ_f, T_X, T_Y, SIN_T, COS_T]``.
    """
    del follower_config  # currently unused; kept for API symmetry

    s_x = float(follower_spawn_pos[0])
    s_y = float(follower_spawn_pos[1])
    s_sin = float(np.sin(follower_spawn_theta))
    s_cos = float(np.cos(follower_spawn_theta))

    def value_fn(leader_obs_batch: np.ndarray) -> np.ndarray:
        leader_obs_batch = np.asarray(leader_obs_batch, dtype=np.float32)
        if leader_obs_batch.ndim == 1:
            leader_obs_batch = leader_obs_batch[None, :]
        b = leader_obs_batch.shape[0]
        t_cols = list(LEADER_OBS_TABLE_COLS)
        table_pose = leader_obs_batch[:, t_cols]

        cf_obs = np.empty((b, 8), dtype=np.float32)
        cf_obs[:, 0] = s_x
        cf_obs[:, 1] = s_y
        cf_obs[:, 2] = s_sin
        cf_obs[:, 3] = s_cos
        cf_obs[:, 4:8] = table_pose

        obs_tensor, _ = follower_model.policy.obs_to_tensor(cf_obs)
        with th.no_grad():
            values = follower_model.policy.predict_values(obs_tensor)
        return values.detach().cpu().numpy().reshape(-1)

    return value_fn


# ---------------------------------------------------------------------------
# Distilled value aggregator
# ---------------------------------------------------------------------------

class NavValueAggregator:
    """Fast table-pose-conditioned value approximator for cleaner planning.

    Input is the 4-D shared state ``[t_x, t_y, sin θ_t, cos θ_t]``;
    output is the averaged follower value.
    """

    def __init__(self, model: nn.Module) -> None:
        self._model = model
        self._model.eval()
        self._jit_model: torch.jit.ScriptModule | None = None

    def _ensure_jit(self, example_input: torch.Tensor) -> torch.jit.ScriptModule:
        if self._jit_model is None:
            try:
                self._jit_model = torch.jit.trace(self._model, example_input)
            except Exception:
                self._jit_model = self._model
        return self._jit_model

    def value(self, table_state: np.ndarray) -> float | np.ndarray:
        table_state = np.asarray(table_state, dtype=np.float32)
        with torch.no_grad():
            if table_state.ndim == 1:
                t = torch.as_tensor(table_state).unsqueeze(0)
                m = self._ensure_jit(t)
                return float(m(t).item())
            t = torch.as_tensor(table_state)
            m = self._ensure_jit(t)
            values = m(t).squeeze(-1).cpu().numpy()
        return values.astype(np.float32)

    def __call__(self, table_state: np.ndarray) -> float | np.ndarray:
        return self.value(table_state)


# ---------------------------------------------------------------------------
# Helpers: follower-pose grid + value averaging
# ---------------------------------------------------------------------------

def _make_follower_grid(config: NavEnvConfig, grid_size: int,
                        n_orientations: int) -> np.ndarray:
    """Return a ``(N, 4)`` grid of ``[f_x, f_y, sin θ_f, cos θ_f]`` poses."""

    r = rect_circumscribed_radius(config.follower_half_extents)
    xs = np.linspace(float(config.world_x_min) + r, float(config.world_x_max) - r,
                     grid_size, dtype=np.float32)
    ys = np.linspace(float(config.world_y_min) + r, float(config.world_y_max) - r,
                     grid_size, dtype=np.float32)
    angles = np.linspace(-np.pi, np.pi, n_orientations, endpoint=False, dtype=np.float32)
    grid_x, grid_y, grid_a = np.meshgrid(xs, ys, angles, indexing="ij")
    return np.stack(
        [
            grid_x.ravel(),
            grid_y.ravel(),
            np.sin(grid_a.ravel()),
            np.cos(grid_a.ravel()),
        ],
        axis=1,
    ).astype(np.float32)


def _sample_table_states(config: NavEnvConfig, count: int) -> np.ndarray:
    """Sample ``count`` table poses in the lower-half / full-arena spawn region."""

    r = rect_circumscribed_radius(config.table_half_extents)
    low_x = float(config.world_x_min) + r
    high_x = float(config.world_x_max) - r
    low_y = float(config.world_y_min) + r
    high_y = float(config.world_y_max) - r
    if getattr(config, "wall_enabled", False):
        half_thick = float(getattr(config, "wall_thickness", 0.0)) / 2.0
        high_y = min(high_y, float(config.wall_y) - half_thick - r - 1e-3)
    xs = np.random.uniform(low_x, high_x, size=count).astype(np.float32)
    ys = np.random.uniform(low_y, high_y, size=count).astype(np.float32)
    angles = np.random.uniform(-np.pi, np.pi, size=count).astype(np.float32)
    return np.stack([xs, ys, np.sin(angles), np.cos(angles)], axis=1)


def _average_nav_value(nav_model: PPO, table_state: np.ndarray,
                       follower_grid: np.ndarray) -> float:
    """Query the nav critic for all follower poses at one fixed table state."""

    table_block = np.repeat(table_state[None, :], follower_grid.shape[0], axis=0)
    obs = np.concatenate([follower_grid, table_block], axis=1)
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32,
                                     device=nav_model.policy.device)
        values = nav_model.policy.predict_values(obs_tensor).cpu().numpy()
    return float(np.mean(values))


# ---------------------------------------------------------------------------
# Builder: distill or load cached
# ---------------------------------------------------------------------------

def _load_distilled(path: str) -> dict | object:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")


def build_nav_value_approximator(
    nav_model: PPO,
    nav_config: NavEnvConfig,
    *,
    obstacle_samples: int,
    agent_grid_size: int,
    epochs: int,
    batch_size: int,
    lr: float,
    distilled_path: str | None = None,
    n_orientations: int = 4,
) -> NavValueAggregator:
    """Build (or load) the distilled household nav-value MLP.

    ``obstacle_samples`` is the number of table poses sampled for the
    distillation set; ``agent_grid_size`` × ``agent_grid_size`` × ``n_orientations``
    is the number of follower poses averaged per sample.
    """

    if distilled_path and os.path.exists(distilled_path):
        data = _load_distilled(distilled_path)
        model = build_mlp_regressor(input_dim=SHARED_STATE_DIM)
        state_dict = data["state_dict"] if isinstance(data, dict) and "state_dict" in data else data
        model.load_state_dict(state_dict)
        model.eval()
        return NavValueAggregator(model)

    table_states = _sample_table_states(nav_config, obstacle_samples)
    follower_grid = _make_follower_grid(nav_config, agent_grid_size, n_orientations)
    targets = np.zeros((obstacle_samples, 1), dtype=np.float32)
    for idx in range(obstacle_samples):
        targets[idx, 0] = _average_nav_value(nav_model, table_states[idx], follower_grid)

    model = build_mlp_regressor(input_dim=SHARED_STATE_DIM)
    fit_pointwise_regressor(model, table_states, targets,
                            epochs=epochs, batch_size=batch_size, lr=lr)
    return NavValueAggregator(model)


# ---------------------------------------------------------------------------
# OOD metric builder
# ---------------------------------------------------------------------------

def _config_from_checkpoint(data: dict, fallback: NavEnvConfig) -> NavEnvConfig:
    env_config = data.get("env_config")
    if not isinstance(env_config, dict):
        return fallback
    valid_keys = set(NavEnvConfig.__dataclass_fields__.keys())
    return NavEnvConfig(**{k: v for k, v in env_config.items() if k in valid_keys})


def _table_state_grid(config: NavEnvConfig, grid_size: int,
                      n_orientations: int = 8) -> np.ndarray:
    """Dense grid of ``[t_x, t_y, sin θ_t, cos θ_t]`` over the full arena.

    Used for VAE-OOD calibration: positions outside the training spawn
    region (e.g., upper half when ``wall_enabled``) are labelled OOD.
    """

    r = rect_circumscribed_radius(config.table_half_extents)
    xs = np.linspace(float(config.world_x_min) + r, float(config.world_x_max) - r,
                     grid_size, dtype=np.float32)
    ys = np.linspace(float(config.world_y_min) + r, float(config.world_y_max) - r,
                     grid_size, dtype=np.float32)
    angles = np.linspace(-np.pi, np.pi, n_orientations, endpoint=False, dtype=np.float32)
    grid_x, grid_y, grid_a = np.meshgrid(xs, ys, angles, indexing="ij")
    return np.stack(
        [
            grid_x.ravel(),
            grid_y.ravel(),
            np.sin(grid_a.ravel()),
            np.cos(grid_a.ravel()),
        ],
        axis=1,
    ).astype(np.float32)


def _training_region_mask(states: np.ndarray, config: NavEnvConfig) -> np.ndarray:
    """Bool mask: True iff the table state was within the training spawn region.

    Spawn region: lower half if ``wall_enabled``, else the full arena.
    Orientation is unrestricted (training samples uniformly over [-π, π)).
    """

    r = rect_circumscribed_radius(config.table_half_extents)
    low_x = float(config.world_x_min) + r
    high_x = float(config.world_x_max) - r
    low_y = float(config.world_y_min) + r
    high_y = float(config.world_y_max) - r
    if getattr(config, "wall_enabled", False):
        half_thick = float(getattr(config, "wall_thickness", 0.0)) / 2.0
        high_y = min(high_y, float(config.wall_y) - half_thick - r - 1e-3)
    xs, ys = states[:, 0], states[:, 1]
    return (xs >= low_x) & (xs <= high_x) & (ys >= low_y) & (ys <= high_y)


def build_vae_ood_metric(
    vae_path: str,
    fallback_config: NavEnvConfig,
    grid_size: int,
) -> OODMetric:
    """Build a calibrated VAE-OOD metric from a household VAE checkpoint."""

    checkpoint = load_vae_checkpoint(vae_path)
    if "state_dict" not in checkpoint:
        raise ValueError(f"Unexpected VAE checkpoint format in {vae_path}.")
    checkpoint_config = _config_from_checkpoint(checkpoint, fallback_config)
    model = load_vae_model(vae_path)
    states = _table_state_grid(checkpoint_config, grid_size)
    losses = reconstruction_losses(model, states)
    in_distribution = _training_region_mask(states, checkpoint_config)
    ood_labels = (~in_distribution).astype(np.int32)
    auroc, threshold, max_loss = roc_threshold(ood_labels, losses)
    print(
        f"Loaded household VAE OOD metric: auroc={auroc:.4f} "
        f"threshold={threshold:.6f} max_loss={max_loss:.6f}",
        file=sys.stderr,
    )
    return OODMetric(model, threshold=threshold, max_loss=max_loss)


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_nav_value_approximator(
    aggregator: NavValueAggregator,
    path: str,
    *,
    nav_config: NavEnvConfig | None = None,
    obstacle_samples: int | None = None,
    agent_grid_size: int | None = None,
    epochs: int | None = None,
) -> None:
    payload: dict = {"state_dict": aggregator._model.state_dict()}
    if nav_config is not None:
        from dataclasses import asdict
        payload["nav_config"] = asdict(nav_config)
    if obstacle_samples is not None:
        payload["obstacle_samples"] = obstacle_samples
    if agent_grid_size is not None:
        payload["agent_grid_size"] = agent_grid_size
    if epochs is not None:
        payload["epochs"] = epochs
    torch.save(payload, path)


def load_nav_value_approximator(path: str) -> NavValueAggregator:
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    model = build_mlp_regressor(input_dim=SHARED_STATE_DIM)
    state_dict = data["state_dict"] if isinstance(data, dict) and "state_dict" in data else data
    model.load_state_dict(state_dict)
    model.eval()
    return NavValueAggregator(model)


__all__ = [
    "NavValueAggregator",
    "SHARED_STATE_DIM",
    "build_household_follower_value_fn",
    "build_nav_value_approximator",
    "build_vae_ood_metric",
    "load_nav_value_approximator",
    "save_nav_value_approximator",
]
