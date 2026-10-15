"""Per-scenario VAE/OOD plumbing for mobile_arm.

The follower's V_F is queried at ``(object_x_base, object_y_base)`` —
the object position in the **base frame** of the leader's parking pose.
That's also the right slice for OOD: at planning time we want to flag
parking poses that put the object at base-frame positions V_F has not
been trained on. The arm joint angles add another two dimensions; we
include them so the VAE captures the full V_F input shape (12-D) but
the practical OOD signal is dominated by the object position when the
follower's joint defaults are fixed.

The follower env's natural reset already samples object positions
uniformly in an annulus — that's exactly the V_F training distribution
we want to model.

The world→base transform happens here at planning time (when the
planner queries OOD on a leader's CEM-terminal world-frame obs); the
VAE itself is trained on follower-obs samples which are already in
base frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np


# Leader obs columns (matches build_mobile_arm_follower_value_fn).
LEADER_BASE_X_IDX = 0
LEADER_BASE_Y_IDX = 1
LEADER_SIN_IDX = 2
LEADER_COS_IDX = 3
LEADER_OBJ_X_IDX = 4
LEADER_OBJ_Y_IDX = 5

# Follower obs columns (12-D total). The VAE models the (object_x_base,
# object_y_base) 2-D slice — see module docstring.
FOLLOWER_OBJ_X_BASE_IDX = 8
FOLLOWER_OBJ_Y_BASE_IDX = 9


def mobile_arm_vae_obs_transform(follower_obs: np.ndarray) -> np.ndarray:
    """Project follower obs to (object_x_base, object_y_base).

    Used by ``train vae`` to build the VAE's training dataset from a
    follower-policy rollout. Input is ``(N, 12)`` follower obs in base
    frame; output is ``(N, 2)`` object positions.
    """
    obs = np.asarray(follower_obs, dtype=np.float32)
    if obs.ndim == 1:
        obs = obs[None, :]
    return obs[:, [FOLLOWER_OBJ_X_BASE_IDX, FOLLOWER_OBJ_Y_BASE_IDX]].astype(np.float32)


VAE_OBS_TRANSFORM = mobile_arm_vae_obs_transform


def _world_to_base_features(leader_obs_batch: np.ndarray) -> np.ndarray:
    """Map leader-shaped obs to (object_x_base, object_y_base) — the VAE feature.

    Same world→base rotation as
    :func:`build_mobile_arm_follower_value_fn`. Lifted out of the OOD
    closure so the kNN and VAE-recon code paths share the geometry.
    """
    obs = np.asarray(leader_obs_batch, dtype=np.float32)
    if obs.ndim == 1:
        obs = obs[None, :]
    lx = obs[:, LEADER_BASE_X_IDX]
    ly = obs[:, LEADER_BASE_Y_IDX]
    l_sin = obs[:, LEADER_SIN_IDX]
    l_cos = obs[:, LEADER_COS_IDX]
    ox = obs[:, LEADER_OBJ_X_IDX]
    oy = obs[:, LEADER_OBJ_Y_IDX]
    dx = ox - lx
    dy = oy - ly
    ox_b = dx * l_cos + dy * l_sin
    oy_b = -dx * l_sin + dy * l_cos
    return np.stack([ox_b, oy_b], axis=1).astype(np.float32)


def build_mobile_arm_ood_metric(
    vae_path: str | Path,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a closure that scores leader-shaped CEM terminal obs.

    Uses **kNN density** in the (object_x_base, object_y_base) slice the
    follower saw during training when the VAE checkpoint includes the
    kNN payload (``train_samples``, ``knn_threshold``, ``knn_max_dist``).
    Falls back to VAE reconstruction loss for legacy checkpoints.

    The kNN path matches the warehouse-side rationale: the follower's
    training distribution is a bounded cone (φ ∈ [-1, 1] in base frame).
    The VAE smoothly reconstructs across the cone boundary (it has
    decoder-side interpolation freedom), producing false in-distribution
    flags for handoffs that fall outside the cone where V_F is actually
    untrustworthy. kNN-distance flags any query point that's far from
    every training neighbour, which is exactly the planner-relevant
    failure mode here.

    Args:
        vae_path: path to the checkpoint produced by ``train vae``.

    Returns:
        ``ood_fn(leader_obs_batch) -> scores`` of shape ``(B,)``.
    """
    from critic_sharing.guidance.ood import (
        OODMetric,
        load_vae_checkpoint,
        load_vae_model,
    )

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

        def ood_fn(leader_obs_batch: np.ndarray) -> np.ndarray:
            features = _world_to_base_features(leader_obs_batch)
            dists, _ = tree.query(features, k=knn_k)
            mean_dists = dists.mean(axis=1)
            scores = (mean_dists - threshold) / span
            return np.clip(scores, 0.0, 1.0).astype(np.float32)

        return ood_fn

    # Legacy fallback: VAE reconstruction loss as the OOD signal.
    threshold = float(checkpoint["threshold"])
    max_loss = float(checkpoint["max_loss"])
    vae = load_vae_model(str(vae_path))
    metric = OODMetric(vae, threshold=threshold, max_loss=max_loss)

    def ood_fn(leader_obs_batch: np.ndarray) -> np.ndarray:
        features = _world_to_base_features(leader_obs_batch)
        return np.asarray(metric.metric(features), dtype=np.float32).reshape(-1)

    return ood_fn


__all__ = [
    "VAE_OBS_TRANSFORM",
    "build_mobile_arm_ood_metric",
    "mobile_arm_vae_obs_transform",
]
