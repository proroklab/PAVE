"""Per-scenario VAE/OOD plumbing for household.

The shared state the VAE models is the table pose
``[t_x, t_y, sin θ_t, cos θ_t]`` (4-D). The follower's V_F query at
each CEM terminal projects the leader-shaped obs (which has the same
table-pose columns as the follower obs) onto this slice; the VAE/kNN
trust score flags terminals where the table pose drifts off the
follower's training-spawn distribution.

Mirrors ``scenarios/obstacle_arena/ood_query.py`` — only the obs
projection (4-D instead of 2-D) and the column indices differ.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np


# Both leader and follower obs share the same table-pose columns:
#
#   cleaner obs (10-D): [c_x, c_y, sin θ_c, cos θ_c,
#                        T_X, T_Y, SIN_T, COS_T,
#                        dirt_x, dirt_y]
#   follower obs (8-D): [f_x, f_y, sin θ_f, cos θ_f,
#                        T_X, T_Y, SIN_T, COS_T]
#
# In both, the table-pose slice is columns [4:8].
TABLE_POSE_COLS = (4, 5, 6, 7)


def household_vae_obs_transform(obs: np.ndarray) -> np.ndarray:
    """Project leader/follower obs to the 4-D table-pose slice.

    Input is ``(N, 10)`` cleaner obs or ``(N, 8)`` follower obs (or
    1-D); output is ``(N, 4)`` table poses.
    """

    arr = np.asarray(obs, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr[:, list(TABLE_POSE_COLS)].astype(np.float32)


VAE_OBS_TRANSFORM = household_vae_obs_transform


def _table_features(obs_batch: np.ndarray) -> np.ndarray:
    """Map either pre-transformed (4-col) or raw obs to table pose (4-col)."""

    obs = np.asarray(obs_batch, dtype=np.float32)
    if obs.ndim == 1:
        obs = obs[None, :]
    if obs.shape[-1] == 4:
        return obs.astype(np.float32)
    return obs[:, list(TABLE_POSE_COLS)].astype(np.float32)


def build_household_ood_metric(
    vae_path: str | Path,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build the OOD scoring closure that ``JointUtilityCost`` will call.

    Prefers kNN density in the 4-D table-pose space when the
    checkpoint includes the kNN payload (``train_samples``,
    ``knn_threshold``, ``knn_max_dist``). Falls back to VAE
    reconstruction loss otherwise.
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
        raise ValueError(f"Unexpected VAE checkpoint format in {vae_path}")

    train_samples = checkpoint.get("train_samples")
    knn_threshold = checkpoint.get("knn_threshold")
    knn_max_dist = checkpoint.get("knn_max_dist")
    knn_k = int(checkpoint.get("knn_k", 5))

    if (
        train_samples is not None
        and knn_threshold is not None
        and knn_max_dist is not None
    ):
        from scipy.spatial import cKDTree

        train_samples = np.asarray(train_samples, dtype=np.float32)
        tree = cKDTree(train_samples)
        threshold = float(knn_threshold)
        max_dist = float(knn_max_dist)
        span = max(max_dist - threshold, 1e-8)

        def ood_fn(obs_batch: np.ndarray) -> np.ndarray:
            features = _table_features(obs_batch)
            dists, _ = tree.query(features, k=knn_k)
            mean_dists = dists.mean(axis=1)
            scores = (mean_dists - threshold) / span
            return np.clip(scores, 0.0, 1.0).astype(np.float32)

        return ood_fn

    # ---- VAE-reconstruction fallback. ----
    vae = load_vae_model(str(vae_path))

    if "threshold" in checkpoint and "max_loss" in checkpoint:
        threshold = float(checkpoint["threshold"])
        max_loss = float(checkpoint["max_loss"])
    else:
        seen = checkpoint.get("seen_table_states")
        if seen is None:
            seen = checkpoint.get("train_samples")
        if seen is None:
            raise ValueError(
                f"VAE checkpoint at {vae_path} has neither (threshold, max_loss) "
                "nor seen_table_states/train_samples — can't calibrate."
            )
        seen_arr = np.asarray(seen, dtype=np.float32)
        train_losses = reconstruction_losses(vae, seen_arr)
        cal = percentile_calibration(train_losses)
        threshold, max_loss = cal.threshold, cal.max_loss

    metric = OODMetric(vae, threshold=threshold, max_loss=max_loss)

    def ood_fn(obs_batch: np.ndarray) -> np.ndarray:
        features = _table_features(obs_batch)
        return np.asarray(metric.metric(features), dtype=np.float32).reshape(-1)

    return ood_fn


__all__ = [
    "TABLE_POSE_COLS",
    "VAE_OBS_TRANSFORM",
    "build_household_ood_metric",
    "household_vae_obs_transform",
]
