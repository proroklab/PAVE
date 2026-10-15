"""Per-scenario VAE/OOD plumbing for warehouse.

Two pieces:

* :data:`VAE_OBS_TRANSFORM` — the projector applied to **follower obs**
  before training the VAE during ``critic-sharing train vae --scenario
  warehouse``. We model only the (object_x, object_y) slice because V_F's
  scale-relevant input — once the leader's drop is decided — is exactly
  the object position; agent and goal positions are constant within an
  episode and don't tell us anything about V_F's reliability.

* :func:`build_warehouse_ood_metric` — the planner-side closure that
  takes a batch of **leader-shaped CEM terminal obs** and returns a
  per-row OOD score in ``[0, 1]``. Mirrors the existing
  :func:`build_warehouse_follower_value_fn` pattern but evaluates the
  VAE OOD detector instead of V_F.

The leader obs layout (see ``WarehouseLeaderEnv._get_obs``):
``[agent_x, agent_y, sin(theta), cos(theta), object_x, object_y, carry]``.
The follower env shares this exact layout, so for warehouse the
"transform from leader's CEM-terminal obs to V_F's input frame" is a
no-op — we just slice columns 4:6 (the object position) before scoring
with the VAE.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np


# Leader obs columns (matches build_warehouse_follower_value_fn).
LEADER_OBJ_X_IDX = 4
LEADER_OBJ_Y_IDX = 5


def warehouse_vae_obs_transform(follower_obs: np.ndarray) -> np.ndarray:
    """Project follower obs to the (object_x, object_y) slice the VAE models.

    Used by ``train vae`` to build the VAE's training dataset from a
    follower-policy rollout. Input is ``(N, 7)`` follower obs;
    output is ``(N, 2)`` object positions.
    """
    obs = np.asarray(follower_obs, dtype=np.float32)
    if obs.ndim == 1:
        obs = obs[None, :]
    return obs[:, [LEADER_OBJ_X_IDX, LEADER_OBJ_Y_IDX]].astype(np.float32)


VAE_OBS_TRANSFORM = warehouse_vae_obs_transform


def build_warehouse_ood_metric(
    vae_path: str | Path,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a closure that scores leader-shaped CEM terminal obs.

    Uses **kNN density** in the (object_x, object_y) slice the follower
    saw during training, NOT the VAE's reconstruction loss. The VAE's
    smooth-manifold reconstruction smears across multimodal training
    distributions (warehouse_blocked / warehouse_pickup_blocked have
    objects at exactly 2 of 3 gaps; the VAE happily reconstructs the
    third gap because the latent decoder smoothly interpolates between
    the two trained modes). kNN distance is local: a query "between
    modes" is far from every training neighbour, so the OOD signal
    fires correctly.

    Score formula:

        d_query = mean k-nearest-neighbour distance from query to the
                  training samples (in post-``obs_transform`` 2-D space)
        score   = clip((d_query - threshold) / (max_dist - threshold),
                       0, 1)

    where ``threshold`` is the 95th percentile of intra-training-set
    kNN distances (computed at VAE-train time and stashed in the
    checkpoint).

    Falls back to VAE-reconstruction OOD if ``train_samples`` is
    missing from the checkpoint (legacy checkpoints from before the
    kNN refactor).

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
            obs_batch = np.asarray(leader_obs_batch, dtype=np.float32)
            if obs_batch.ndim == 1:
                obs_batch = obs_batch[None, :]
            features = obs_batch[:, [LEADER_OBJ_X_IDX, LEADER_OBJ_Y_IDX]]
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
        leader_obs_batch = np.asarray(leader_obs_batch, dtype=np.float32)
        if leader_obs_batch.ndim == 1:
            leader_obs_batch = leader_obs_batch[None, :]
        features = leader_obs_batch[:, [LEADER_OBJ_X_IDX, LEADER_OBJ_Y_IDX]]
        return np.asarray(metric.metric(features), dtype=np.float32).reshape(-1)

    return ood_fn


__all__ = [
    "VAE_OBS_TRANSFORM",
    "build_warehouse_ood_metric",
    "warehouse_vae_obs_transform",
]
