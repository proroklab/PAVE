"""Per-scenario VAE/OOD plumbing for obstacle_arena.

Mirrors the existing obstacle_arena ``value_query.py::build_vae_ood_metric``
but exposes the same interface as warehouse + mobile_arm so all three
scenarios plug into ``JointUtilityCost`` the same way.

The follower's V_F query at terminal uses obs columns ``[2:4]`` —
the obstacle position (the agent + obstacle 4-D obs has agent_xy in
[0:2] and obstacle_xy in [2:4]). The VAE is trained on those obstacle
positions: that's exactly what V_F's training distribution depended on
(agent positions are dense and uniform; obstacle positions are the
sparse, scenario-shaped variable).

This module also offers a back-compat shim for legacy ``nav_env_vae.pt``
checkpoints (saved before the percentile-calibration refactor): when the
checkpoint lacks ``threshold`` and ``max_loss`` keys, we synthesize them
from the existing ``seen_obstacle_positions`` field via percentile
calibration, mirroring what ``train vae`` would do today.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np


# Leader obs layout: [agent_x, agent_y, obstacle_x, obstacle_y].
LEADER_OBSTACLE_X_IDX = 2
LEADER_OBSTACLE_Y_IDX = 3


def obstacle_arena_vae_obs_transform(follower_obs: np.ndarray) -> np.ndarray:
    """Project follower obs to the (obstacle_x, obstacle_y) slice the VAE models.

    Input is ``(N, 4)`` follower obs (or full leader-shape obs, same
    layout); output is ``(N, 2)`` obstacle positions.
    """
    obs = np.asarray(follower_obs, dtype=np.float32)
    if obs.ndim == 1:
        obs = obs[None, :]
    return obs[:, [LEADER_OBSTACLE_X_IDX, LEADER_OBSTACLE_Y_IDX]].astype(np.float32)


VAE_OBS_TRANSFORM = obstacle_arena_vae_obs_transform


def _obstacle_features(obs_batch: np.ndarray) -> np.ndarray:
    """Map either pre-transformed (2-col) or raw leader-obs (4-col) to (obstacle_x, obstacle_y).

    ``JointUtilityCost.evaluate`` applies ``obs_transform`` to ``terminal``
    before calling the OOD closure, so obstacle_arena passes the
    obs-to-obstacle slice. We tolerate both shapes for callers that
    bypass the transform.
    """
    obs = np.asarray(obs_batch, dtype=np.float32)
    if obs.ndim == 1:
        obs = obs[None, :]
    if obs.shape[-1] == 2:
        return obs.astype(np.float32)
    return obs[:, [LEADER_OBSTACLE_X_IDX, LEADER_OBSTACLE_Y_IDX]].astype(np.float32)


def build_obstacle_arena_ood_metric(
    vae_path: str | Path,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a closure that scores leader-shaped CEM terminal obs.

    Uses **kNN density** in the (obstacle_x, obstacle_y) slice the VAE
    was trained on when the checkpoint includes the kNN payload
    (``train_samples``, ``knn_threshold``, ``knn_max_dist``). Falls back
    to VAE reconstruction loss for new-without-kNN checkpoints, and to
    on-the-fly percentile calibration for legacy ``nav_env_vae.pt``
    checkpoints (which only have ``state_dict`` /
    ``seen_obstacle_positions``).

    The kNN path matches the warehouse + mobile_arm reasoning: the VAE
    decoder smoothly interpolates across the bounded training-spawn
    region, producing false in-distribution flags at the boundary where
    V_F is actually untrustworthy. kNN distance is local and flags any
    query that's far from every training neighbour.
    """
    from critic_sharing.guidance.ood import (
        OODMetric,
        load_vae_checkpoint,
        load_vae_model,
        reconstruction_losses,
    )
    from critic_sharing.training.vae_from_rollouts import percentile_calibration

    checkpoint = load_vae_checkpoint(str(vae_path))
    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Unexpected VAE checkpoint format in {p}".format(p=vae_path),
        )

    train_samples = checkpoint.get("train_samples")
    knn_threshold = checkpoint.get("knn_threshold")
    knn_max_dist = checkpoint.get("knn_max_dist")
    knn_k = int(checkpoint.get("knn_k", 5))

    if (
        train_samples is not None
        and knn_threshold is not None
        and knn_max_dist is not None
    ):
        # kNN-density OOD path (preferred).
        from scipy.spatial import cKDTree

        train_samples = np.asarray(train_samples, dtype=np.float32)
        tree = cKDTree(train_samples)
        threshold = float(knn_threshold)
        max_dist = float(knn_max_dist)
        span = max(max_dist - threshold, 1e-8)

        def ood_fn(obs_batch: np.ndarray) -> np.ndarray:
            features = _obstacle_features(obs_batch)
            dists, _ = tree.query(features, k=knn_k)
            mean_dists = dists.mean(axis=1)
            scores = (mean_dists - threshold) / span
            return np.clip(scores, 0.0, 1.0).astype(np.float32)

        return ood_fn

    # ---- VAE-reconstruction fallback (legacy + new-without-kNN). ----
    vae = load_vae_model(str(vae_path))

    if "threshold" in checkpoint and "max_loss" in checkpoint:
        threshold = float(checkpoint["threshold"])
        max_loss = float(checkpoint["max_loss"])
    else:
        # Legacy nav_env_vae.pt — synthesize calibration from the saved
        # training set. Falls back to percentile defaults if no positions.
        seen = checkpoint.get("seen_obstacle_positions")
        if seen is None:
            raise ValueError(
                "Legacy VAE checkpoint at {p} has neither (threshold, "
                "max_loss) nor seen_obstacle_positions — can't calibrate.".format(
                    p=vae_path,
                )
            )
        seen_arr = np.asarray(seen, dtype=np.float32)
        train_losses = reconstruction_losses(vae, seen_arr)
        cal = percentile_calibration(train_losses)
        threshold, max_loss = cal.threshold, cal.max_loss

    metric = OODMetric(vae, threshold=threshold, max_loss=max_loss)

    def ood_fn(obs_batch: np.ndarray) -> np.ndarray:
        features = _obstacle_features(obs_batch)
        return np.asarray(metric.metric(features), dtype=np.float32).reshape(-1)

    return ood_fn


__all__ = [
    "VAE_OBS_TRANSFORM",
    "build_obstacle_arena_ood_metric",
    "obstacle_arena_vae_obs_transform",
]
