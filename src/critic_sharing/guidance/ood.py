"""VAE-based OOD detection for follower value generalisation.

The OOD metric tells the leader how far the current shared state is from the
follower's training distribution, so the planner can down-weight value
estimates that are likely to be unreliable.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from critic_sharing.models.vae import VAE


# ---------------------------------------------------------------------------
# Core metric class
# ---------------------------------------------------------------------------

class OODMetric:
    """Maps shared-state inputs to normalised OOD scores in [0, 1]."""

    def __init__(
        self,
        model: nn.Module,
        threshold: float,
        max_loss: float,
        loss_fn: object | None = None,
    ) -> None:
        self._model = model
        self._model.eval()
        self._threshold = float(threshold)
        self._max_loss = float(max_loss)
        # Default to the flat-vector reconstruction_losses if not specified.
        self._loss_fn = loss_fn if loss_fn is not None else reconstruction_losses

    def metric(self, inputs: np.ndarray) -> float | np.ndarray:
        inputs = np.asarray(inputs, dtype=np.float32)
        single = inputs.ndim == 1
        batch = inputs[None, :] if single else inputs
        losses = self._loss_fn(self._model, batch)
        if self._max_loss <= self._threshold:
            metrics = np.zeros_like(losses, dtype=np.float32)
        else:
            metrics = (losses - self._threshold) / (self._max_loss - self._threshold)
            metrics = np.clip(metrics, 0.0, 1.0).astype(np.float32)
        return float(metrics[0]) if single else metrics


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_vae_checkpoint(path: str) -> dict[str, object]:
    """Load a VAE checkpoint dict from ``path``."""
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    except Exception:
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError("Unexpected checkpoint format in {path}.".format(path=path))
    return data


def load_vae_model(
    path: str,
    default_input_dim: int = 2,
    default_latent_dim: int = 2,
) -> VAE:
    """Instantiate and load a VAE from a checkpoint file."""
    checkpoint = load_vae_checkpoint(path)
    if "state_dict" not in checkpoint:
        raise ValueError("Missing state_dict in VAE checkpoint: {path}.".format(path=path))
    latent_dim = int(checkpoint.get("latent_dim", default_latent_dim))
    input_dim = int(checkpoint.get("input_dim", default_input_dim))
    model = VAE(input_dim=input_dim, latent_dim=latent_dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def reconstruction_losses(model: nn.Module, inputs: np.ndarray) -> np.ndarray:
    """Return per-sample VAE reconstruction losses for ``inputs``."""
    batch = np.asarray(inputs, dtype=np.float32)
    with torch.no_grad():
        input_tensor = torch.as_tensor(batch)
        encoded = model.encoder(input_tensor)
        mu, _ = encoded.chunk(2, dim=-1)
        recon = model.decoder(mu)
        losses = torch.mean((recon - input_tensor) ** 2, dim=1).cpu().numpy()
    return losses.astype(np.float32)


def roc_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[float, float, float]:
    """Compute AUROC and the Youden-optimal threshold.

    Returns:
        (auroc, threshold, max_loss) — used to calibrate ``OODMetric``.
    """
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if labels.size == 0 or labels.size != scores.size:
        raise ValueError("Labels and scores must be non-empty and have the same length.")
    positives = int(np.sum(labels))
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("Need both positive and negative examples to compute AUROC.")

    order = np.argsort(scores)[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    distinct = np.where(np.diff(sorted_scores))[0]
    threshold_idxs = np.r_[distinct, sorted_labels.size - 1]
    tps = np.cumsum(sorted_labels)[threshold_idxs].astype(np.float64)
    fps = (1 + threshold_idxs - tps).astype(np.float64)
    thresholds = sorted_scores[threshold_idxs]
    tpr = np.r_[0.0, tps / positives, 1.0]
    fpr = np.r_[0.0, fps / negatives, 1.0]
    thresholds = np.r_[np.inf, thresholds, -np.inf]
    _trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    auroc = float(_trapezoid(tpr, fpr))
    best_idx = int(np.argmax(tpr - fpr))
    return auroc, float(thresholds[best_idx]), float(np.max(scores))


__all__ = [
    "OODMetric",
    "load_vae_checkpoint",
    "load_vae_model",
    "reconstruction_losses",
    "roc_threshold",
]
