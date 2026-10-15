"""Guidance utilities shared across all leader/follower experiments."""

from critic_sharing.guidance.distillation import build_mlp_regressor, fit_pointwise_regressor
from critic_sharing.guidance.ood import (
    OODMetric,
    load_vae_checkpoint,
    load_vae_model,
    reconstruction_losses,
    roc_threshold,
)
__all__ = [
    # Distillation
    "build_mlp_regressor",
    "fit_pointwise_regressor",
    # OOD
    "OODMetric",
    "load_vae_checkpoint",
    "load_vae_model",
    "reconstruction_losses",
    "roc_threshold",
]
