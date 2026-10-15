"""Follower (nav agent) value query for the obstacle-arena scenario.

Two V_F builders are exposed:

* :func:`build_obstacle_arena_follower_value_fn` — **per-episode** V_F.
  Bakes the realized follower spawn position into the closure and
  queries the nav PPO critic directly with the candidate obstacle
  position. Single PPO forward per terminal-state query. This is the
  canonical eval-side V_F (mirrors warehouse / mobile_arm).
* :func:`build_nav_value_approximator` — legacy **marginalized** V_F.
  Averages the PPO critic over a grid of agent positions and distills
  the result into a small MLP (``NavValueAggregator``) keyed on
  obstacle position only. Used by the standalone training / visualize
  paths that pre-compute a scenario-wide V heatmap.
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch as th
from stable_baselines3 import PPO

from critic_sharing.config.schema import NavEnvConfig
from critic_sharing.guidance.distillation import build_mlp_regressor, fit_pointwise_regressor
from critic_sharing.guidance.ood import load_vae_checkpoint, load_vae_model, reconstruction_losses, roc_threshold, OODMetric


# ---------------------------------------------------------------------------
# Per-episode V_F (canonical eval-side builder)
# ---------------------------------------------------------------------------

# Leader (push) obs layout (see push_env.py::_get_obs):
#   [agent_x, agent_y, obstacle_x, obstacle_y]    (4-D)
# Follower (nav) obs layout (see nav_env.py::_get_obs):
#   [agent_x, agent_y, obstacle_x, obstacle_y]    (4-D, same shape, different agent)
LEADER_OBS_OBSTACLE_X = 2
LEADER_OBS_OBSTACLE_Y = 3


def build_obstacle_arena_follower_value_fn(
    follower_model: PPO,
    follower_config: NavEnvConfig,
    follower_spawn_pos: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a per-episode V_F closure mapping leader obs → follower critic value.

    Args:
        follower_model: trained follower PPO model (nav agent).
        follower_config: follower env config (currently unused, kept for
            API symmetry with the other scenarios' builders).
        follower_spawn_pos: ``(2,)`` follower agent ``(x, y)`` for this
            episode. Baked into the closure so every V_F query asks
            "what is V_F at this candidate obstacle pos given this
            specific follower spawn?".

    Returns:
        ``value_fn(leader_obs_batch) -> values`` of shape ``(B,)``.
        ``leader_obs_batch`` is ``(B, 4)``; only columns 2-3 (obstacle
        pos) are read. The 4-D counterfactual follower obs that is
        built and fed to the PPO critic is
        ``[s_fx, s_fy, obstacle_x, obstacle_y]``.
    """
    del follower_config  # currently unused; kept for API symmetry

    s_x = float(follower_spawn_pos[0])
    s_y = float(follower_spawn_pos[1])

    def value_fn(leader_obs_batch: np.ndarray) -> np.ndarray:
        leader_obs_batch = np.asarray(leader_obs_batch, dtype=np.float32)
        if leader_obs_batch.ndim == 1:
            leader_obs_batch = leader_obs_batch[None, :]
        b = leader_obs_batch.shape[0]
        ox = leader_obs_batch[:, LEADER_OBS_OBSTACLE_X]
        oy = leader_obs_batch[:, LEADER_OBS_OBSTACLE_Y]

        cf_obs = np.empty((b, 4), dtype=np.float32)
        cf_obs[:, 0] = s_x
        cf_obs[:, 1] = s_y
        cf_obs[:, 2] = ox
        cf_obs[:, 3] = oy

        obs_tensor, _ = follower_model.policy.obs_to_tensor(cf_obs)
        with th.no_grad():
            values = follower_model.policy.predict_values(obs_tensor)
        return values.detach().cpu().numpy().reshape(-1)

    return value_fn


# ---------------------------------------------------------------------------
# Distilled value aggregator
# ---------------------------------------------------------------------------

class NavValueAggregator:
    """Fast obstacle-conditioned value approximation for pusher (leader) planning.

    Wraps a small MLP that maps obstacle_pos -> expected nav value,
    averaged over the agent-position dimension.

    The model is JIT-traced on first call for faster inference during
    CEM planning loops.
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
                # Fallback: some models can't be traced
                self._jit_model = self._model
        return self._jit_model

    def value(self, obstacle_pos: np.ndarray) -> float | np.ndarray:
        obstacle_pos = np.asarray(obstacle_pos, dtype=np.float32)
        with torch.no_grad():
            if obstacle_pos.ndim == 1:
                t = torch.as_tensor(obstacle_pos).unsqueeze(0)
                m = self._ensure_jit(t)
                return float(m(t).item())
            t = torch.as_tensor(obstacle_pos)
            m = self._ensure_jit(t)
            values = m(t).squeeze(-1).cpu().numpy()
        return values.astype(np.float32)

    # Make it callable so it satisfies the ValueFn protocol directly.
    def __call__(self, obstacle_pos: np.ndarray) -> float | np.ndarray:
        return self.value(obstacle_pos)


# ---------------------------------------------------------------------------
# Value query: average nav critic over agent-position grid
# ---------------------------------------------------------------------------

def _make_agent_grid(config: NavEnvConfig, grid_size: int) -> np.ndarray:
    xs = np.linspace(config.world_min, config.world_max, grid_size, dtype=np.float32)
    ys = np.linspace(config.world_min, config.world_max, grid_size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)


def _sample_obstacle_positions(config: NavEnvConfig, count: int) -> np.ndarray:
    x_min, x_max, y_min, y_max = config.obstacle_bounds()
    positions = np.empty((count, 2), dtype=np.float32)
    positions[:, 0] = np.random.uniform(x_min, x_max, size=count)
    positions[:, 1] = np.random.uniform(y_min, y_max, size=count)
    return positions


def _average_nav_value(
    nav_model: PPO,
    obstacle_pos: np.ndarray,
    agent_grid: np.ndarray,
) -> float:
    """Query the nav critic for all agent positions at a fixed obstacle pos."""
    obs = np.concatenate(
        [agent_grid, np.repeat(obstacle_pos[None, :], agent_grid.shape[0], axis=0)],
        axis=1,
    )
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=nav_model.policy.device)
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
) -> NavValueAggregator:
    """Build or load a distilled nav-value MLP for the pusher planner.

    If ``distilled_path`` points to an existing file it is loaded directly,
    skipping the expensive distillation step.
    """
    if distilled_path and os.path.exists(distilled_path):
        data = _load_distilled(distilled_path)
        model = build_mlp_regressor(input_dim=2)
        state_dict = data["state_dict"] if isinstance(data, dict) and "state_dict" in data else data
        model.load_state_dict(state_dict)
        model.eval()
        return NavValueAggregator(model)

    obstacle_positions = _sample_obstacle_positions(nav_config, obstacle_samples)
    agent_grid = _make_agent_grid(nav_config, agent_grid_size)
    targets = np.zeros((obstacle_samples, 1), dtype=np.float32)
    for idx, obstacle_pos in enumerate(obstacle_positions):
        targets[idx, 0] = _average_nav_value(nav_model, obstacle_pos, agent_grid)

    model = build_mlp_regressor(input_dim=2)
    fit_pointwise_regressor(model, obstacle_positions, targets, epochs=epochs, batch_size=batch_size, lr=lr)
    return NavValueAggregator(model)


# ---------------------------------------------------------------------------
# OOD metric builder (convenience, wires to guidance.ood)
# ---------------------------------------------------------------------------

def _config_from_checkpoint(data: dict, fallback: NavEnvConfig) -> NavEnvConfig:
    env_config = data.get("env_config")
    if not isinstance(env_config, dict):
        return fallback
    valid_keys = set(NavEnvConfig.__dataclass_fields__.keys())
    return NavEnvConfig(**{k: v for k, v in env_config.items() if k in valid_keys})


def _obstacle_grid(config: NavEnvConfig, grid_size: int) -> np.ndarray:
    xs = np.linspace(config.world_min, config.world_max, grid_size, dtype=np.float32)
    ys = np.linspace(config.world_min, config.world_max, grid_size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)


def _training_region_mask(obstacle_positions: np.ndarray, config: NavEnvConfig) -> np.ndarray:
    x_min, x_max, y_min, y_max = config.obstacle_bounds()
    xs, ys = obstacle_positions[:, 0], obstacle_positions[:, 1]
    return (xs >= x_min) & (xs <= x_max) & (ys >= y_min) & (ys <= y_max)


def build_vae_ood_metric(
    vae_path: str,
    fallback_config: NavEnvConfig,
    grid_size: int,
) -> OODMetric:
    """Build a calibrated VAE-OOD metric from a saved VAE checkpoint."""
    import sys
    checkpoint = load_vae_checkpoint(vae_path)
    if "state_dict" not in checkpoint:
        raise ValueError("Unexpected VAE checkpoint format in {path}.".format(path=vae_path))
    checkpoint_config = _config_from_checkpoint(checkpoint, fallback_config)
    model = load_vae_model(vae_path)
    obstacle_positions = _obstacle_grid(checkpoint_config, grid_size)
    losses = reconstruction_losses(model, obstacle_positions)
    in_distribution = _training_region_mask(obstacle_positions, checkpoint_config)
    ood_labels = (~in_distribution).astype(np.int32)
    auroc, threshold, max_loss = roc_threshold(ood_labels, losses)
    print(
        "Loaded VAE OOD metric: auroc={:.4f} threshold={:.6f} max_loss={:.6f}".format(
            auroc, threshold, max_loss,
        ),
        file=sys.stderr,
    )
    return OODMetric(model, threshold=threshold, max_loss=max_loss)


def save_nav_value_approximator(
    aggregator: NavValueAggregator,
    path: str,
    *,
    nav_config: NavEnvConfig | None = None,
    obstacle_samples: int | None = None,
    agent_grid_size: int | None = None,
    epochs: int | None = None,
) -> None:
    """Save a distilled NavValueAggregator to disk.

    The checkpoint contains the MLP state dict plus optional metadata
    so it can be loaded later without re-distilling.
    """
    payload: dict = {
        "state_dict": aggregator._model.state_dict(),
    }
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
    """Load a previously saved NavValueAggregator from disk."""
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older PyTorch without weights_only kwarg
        data = torch.load(path, map_location="cpu")
    model = build_mlp_regressor(input_dim=2)
    state_dict = data["state_dict"] if isinstance(data, dict) and "state_dict" in data else data
    model.load_state_dict(state_dict)
    model.eval()
    return NavValueAggregator(model)


__all__ = [
    "NavValueAggregator",
    "build_nav_value_approximator",
    "build_obstacle_arena_follower_value_fn",
    "build_vae_ood_metric",
    "load_nav_value_approximator",
    "save_nav_value_approximator",
]
