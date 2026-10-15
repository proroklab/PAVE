"""Train a VAE on the follower env's natural starting-state distribution.

This is the generic "build a model of V_F's *query* distribution" pipeline
behind ``critic-sharing train ood --detector vae``. It exists alongside the
older ``training/vae.py::train_vae_on_positions``; this helper:

* takes the follower env (a config is enough — no policy needed),
* samples ``n_samples`` initial observations by calling
  ``env.reset(seed=...)`` repeatedly with reproducible seeds,
* optionally projects each obs through a per-scenario
  ``vae_obs_transform``,
* fits an MLP :class:`~critic_sharing.models.vae.VAE` to that dataset,
* calibrates the OOD threshold + max_loss either via percentiles of
  training reconstruction losses (default — fully scenario-agnostic) or
  via :func:`~critic_sharing.guidance.ood.roc_threshold` against an
  optional supplied OOD sample set,
* saves a checkpoint dict carrying everything the eval-time loader needs:
  ``state_dict, latent_dim, obs_dim, threshold, max_loss, scenario,
  transform_name, dataset_size``.

**Why initial states only.** The CEM planner queries V_F at the leader's
CEM-terminal obs (transformed to the follower's frame), which
corresponds to the **follower's initial state at deployment**. The
follower env's natural reset distribution is exactly the set of states
V_F gets asked about. Mid-trajectory states the follower visits during
its rollouts aren't queried by the planner, so modelling them just
widens the VAE's "in-distribution" region beyond what V_F actually
needs to be trusted on. Restricting the training dataset to ``env.reset``
outputs gives the sharpest OOD signal at the planner's query manifold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import torch as th

from critic_sharing.models.vae import VAE


# ---------------------------------------------------------------------------
# Initial-state collection
# ---------------------------------------------------------------------------


def collect_initial_obs(
    env,
    n_samples: int,
    seed_offset: int = 0,
) -> np.ndarray:
    """Sample initial observations from ``env``'s natural reset distribution.

    No policy rollout — we only call ``env.reset(seed=seed_offset + i)``
    ``n_samples`` times and stack the resulting initial obs. This matches
    the planner's V_F query distribution: at deployment the leader hands
    off to the follower with some leader-induced terminal obs, which the
    follower env interprets as its initial state. Modelling exactly that
    distribution gives the sharpest OOD signal.

    Args:
        env: gym env (will be ``reset()``-only — never stepped).
        n_samples: number of reset seeds to sample. Each sample contributes
            one obs; with low-dim 2-D feature slices a few hundred is
            usually enough.
        seed_offset: integer offset added to each sample index for
            ``env.reset(seed=...)``. Reproducible per
            ``(seed_offset, n_samples)`` pair.

    Returns:
        ``(n_samples, obs_dim)`` array of stacked initial observations.
    """
    obs_buffer: list[np.ndarray] = []
    for i in range(int(n_samples)):
        obs, _ = env.reset(seed=int(seed_offset + i))
        obs_buffer.append(np.asarray(obs, dtype=np.float32))

    if not obs_buffer:
        raise RuntimeError(
            "collect_initial_obs: no observations collected (n_samples={n}).".format(
                n=int(n_samples),
            )
        )
    return np.stack(obs_buffer, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


@dataclass
class CalibrationResult:
    """Result of calibrating the OOD detector's (threshold, max_loss).

    ``method`` describes how the numbers were produced ("percentile" or
    "roc"); ``auroc`` is filled when method="roc".
    """

    threshold: float
    max_loss: float
    method: str
    auroc: Optional[float] = None


def percentile_calibration(
    train_losses: np.ndarray,
    threshold_pct: float = 95.0,
    max_loss_pct: float = 99.9,
) -> CalibrationResult:
    """Pure-percentile calibration from training reconstruction losses.

    The OOD score is ``clip((loss - threshold) / (max_loss - threshold), 0, 1)``,
    so ``threshold`` ≈ "this many % of training data sits at-or-below
    here" (i.e. comfortable in-distribution). ``max_loss`` is set above
    the highest training losses so that anything larger maps to a fully
    OOD score of 1.0.

    Args:
        train_losses: per-sample reconstruction losses from the trained VAE
            evaluated on its own training set, shape ``(N,)``.
        threshold_pct: percentile for ``threshold``. Default 95 — the worst
            5% of training data is "borderline".
        max_loss_pct: percentile for ``max_loss``. Default 99.9.

    Returns:
        :class:`CalibrationResult` with method="percentile".
    """
    losses = np.asarray(train_losses, dtype=np.float32).reshape(-1)
    threshold = float(np.percentile(losses, threshold_pct))
    max_loss = float(np.percentile(losses, max_loss_pct))
    if max_loss <= threshold:
        # Degenerate: all training losses identical, or pcts too close.
        # Force a non-zero span so the OOD denominator stays sensible.
        max_loss = threshold + max(1e-3, abs(threshold) * 1e-3)
    return CalibrationResult(
        threshold=threshold, max_loss=max_loss, method="percentile",
    )


def roc_calibration(
    in_dist_losses: np.ndarray,
    ood_losses: np.ndarray,
) -> CalibrationResult:
    """ROC-based calibration when an OOD sample set is supplied.

    Wraps :func:`~critic_sharing.guidance.ood.roc_threshold`. Returns
    AUROC alongside (threshold, max_loss) so the CLI can print a sanity
    score.
    """
    from critic_sharing.guidance.ood import roc_threshold

    in_dist_losses = np.asarray(in_dist_losses, dtype=np.float32).reshape(-1)
    ood_losses = np.asarray(ood_losses, dtype=np.float32).reshape(-1)
    all_losses = np.concatenate([in_dist_losses, ood_losses], axis=0)
    labels = np.concatenate([
        np.zeros(in_dist_losses.shape[0], dtype=np.int32),
        np.ones(ood_losses.shape[0], dtype=np.int32),
    ], axis=0)
    auroc, threshold, max_loss = roc_threshold(labels, all_losses)
    return CalibrationResult(
        threshold=float(threshold),
        max_loss=float(max_loss),
        method="roc",
        auroc=float(auroc),
    )


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------


def train_vae_from_rollouts(
    env,
    n_samples: int = 1000,
    latent_dim: int = 8,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    obs_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    transform_name: str = "identity",
    ood_samples: Optional[np.ndarray] = None,
    threshold_pct: float = 95.0,
    max_loss_pct: float = 99.9,
    seed: int = 0,
    fit_vae: bool = True,
) -> Tuple[Optional[VAE], Optional[CalibrationResult], dict]:
    """Sample initial obs, train an MLP VAE, calibrate the OOD threshold.

    The function name is kept for backward import compatibility, but the
    pipeline no longer rolls a policy — it only samples ``env.reset``
    outputs (see :func:`collect_initial_obs`). The follower policy is not
    needed at this stage.

    The transform contract: ``obs_transform`` (when provided) maps a
    ``(N, *raw_obs_shape)`` array to a ``(N, feature_dim)`` array. The VAE
    is built on the post-transform shape. Identity (None) means the raw
    follower obs vector is the VAE input.

    Args:
        env: follower env (will be ``reset()``-only; never stepped).
        n_samples: number of reset seeds to sample. Default 1000 — gives
            plenty of coverage for low-dim feature slices (warehouse:
            2-D, mobile_arm: 2-D, obstacle_arena: 2-D). Bump for richer
            feature spaces.
        latent_dim: VAE latent dimension.
        epochs, batch_size, learning_rate: VAE training schedule.
        obs_transform: optional projector (see contract above).
        transform_name: human-readable identifier saved into the checkpoint
            for traceability (e.g. "warehouse_object_pos_2d").
        ood_samples: optional ``(M, feature_dim)`` array of pre-computed
            OOD samples (already in post-transform shape) for ROC
            calibration. When ``None`` we fall back to percentile
            calibration on the training set itself.
        threshold_pct, max_loss_pct: percentile-calibration params; only
            consulted when ``ood_samples is None``.
        seed: RNG seed for VAE training (numpy + torch).

    Returns:
        ``(model, calibration, metadata)`` where ``metadata`` is a dict
        of training stats / config (``loss_history``, ``dataset_size``,
        etc.) the caller can fold into the checkpoint.
    """
    # Reproducibility — only seed locally, don't touch global state.
    th.manual_seed(int(seed))
    np.random.seed(int(seed))

    print("Sampling initial obs from env.reset: {n} seeds...".format(n=int(n_samples)))
    raw_obs = collect_initial_obs(env, n_samples=int(n_samples))
    if obs_transform is not None:
        features = np.asarray(obs_transform(raw_obs), dtype=np.float32)
    else:
        features = raw_obs
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    feature_dim = int(features.shape[1])
    print("  collected {n} initial obs samples, feature dim = {d}".format(
        n=int(features.shape[0]), d=feature_dim,
    ))

    # --- VAE training (optional) ----------------------------------------
    # The kNN-density detector below is the default and doesn't need a
    # neural network. We only fit a VAE when the caller explicitly
    # requests it (``--detector vae`` in the CLI). The MLP VAE uses a
    # fixed (64, 32) hidden topology — see ``models/vae.py``.
    vae: Optional[VAE] = None
    calibration: Optional[CalibrationResult] = None
    loss_history: list = []
    if fit_vae:
        print("Training VAE: latent_dim={ld} epochs={e}".format(
            ld=int(latent_dim), e=int(epochs),
        ))
        vae = VAE(input_dim=feature_dim, latent_dim=int(latent_dim))

        from critic_sharing.training.vae import train_vae

        dataset = th.utils.data.TensorDataset(th.from_numpy(features))
        loader = th.utils.data.DataLoader(
            dataset, batch_size=int(batch_size), shuffle=True,
        )
        loss_history = train_vae(
            model=vae,
            dataloader=loader,
            optimizer=th.optim.Adam(vae.parameters(), lr=float(learning_rate)),
            num_epochs=int(epochs),
        )

        # --- VAE calibration (only when we trained the VAE) -------------
        from critic_sharing.guidance.ood import reconstruction_losses

        train_losses = reconstruction_losses(vae, features)
        if ood_samples is not None:
            ood_features = np.asarray(ood_samples, dtype=np.float32)
            if ood_features.ndim == 1:
                ood_features = ood_features.reshape(-1, 1)
            if ood_features.shape[1] != feature_dim:
                raise ValueError(
                    "ood_samples feature_dim ({a}) != VAE feature_dim ({b}). "
                    "Apply the same obs_transform to OOD samples before "
                    "passing them in.".format(
                        a=int(ood_features.shape[1]), b=feature_dim,
                    )
                )
            ood_losses = reconstruction_losses(vae, ood_features)
            calibration = roc_calibration(train_losses, ood_losses)
            print(
                "Calibration (ROC): auroc={a:.4f} threshold={t:.6f} "
                "max_loss={m:.6f}".format(
                    a=calibration.auroc, t=calibration.threshold,
                    m=calibration.max_loss,
                )
            )
        else:
            calibration = percentile_calibration(
                train_losses,
                threshold_pct=float(threshold_pct),
                max_loss_pct=float(max_loss_pct),
            )
            print(
                "Calibration (percentile {tp:.1f}%/{mp:.1f}%): "
                "threshold={t:.6f} max_loss={m:.6f}".format(
                    tp=float(threshold_pct), mp=float(max_loss_pct),
                    t=calibration.threshold, m=calibration.max_loss,
                )
            )
    else:
        print("Skipping VAE training (kNN-only detector). Fitting kNN tree...")

    # --- kNN-OOD threshold ---------------------------------------------
    # Compute the 95th percentile of intra-training-set kNN distances
    # (k=5). At inference, query distance > this threshold → OOD. kNN
    # is robust to the multimodal-training-data failure mode the VAE
    # hits (smoothing across modes makes "between-modes" reconstruct
    # well, false-negatives at OOD points). See ``warehouse/ood_query.py
    # ::build_warehouse_knn_ood_metric``.
    knn_k = 5
    knn_threshold = float("nan")
    knn_max_dist = float("nan")
    if features.shape[0] > knn_k:
        # For each training sample, exclude itself from neighbours via
        # k+1 lookup then drop the distance-0 self-match.
        from scipy.spatial import cKDTree

        tree = cKDTree(features)
        dists, _ = tree.query(features, k=knn_k + 1)
        intra_knn_means = dists[:, 1:].mean(axis=1)  # drop self at index 0
        knn_threshold = float(np.quantile(intra_knn_means, 0.95))
        knn_max_dist = float(np.quantile(intra_knn_means, 0.999))
        print(
            "Calibration (kNN k={k}): threshold={t:.6f} max_dist={m:.6f} "
            "(95th / 99.9th percentile of intra-training-set "
            "{k}-nn mean distance)".format(
                k=int(knn_k), t=knn_threshold, m=knn_max_dist,
            )
        )

    metadata = {
        "loss_history": loss_history,
        "dataset_size": int(features.shape[0]),
        "feature_dim": feature_dim,
        "transform_name": transform_name,
        "n_samples": int(n_samples),
        "train_samples": features.copy(),
        "knn_k": int(knn_k),
        "knn_threshold": knn_threshold,
        "knn_max_dist": knn_max_dist,
    }
    return vae, calibration, metadata


__all__ = [
    "CalibrationResult",
    "collect_initial_obs",
    "percentile_calibration",
    "roc_calibration",
    "train_vae_from_rollouts",
]
