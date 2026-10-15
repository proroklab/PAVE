"""Generic helpers for fitting pointwise value approximators.

These are scenario-agnostic: any scenario's value_query.py can generate
(input, target) pairs and pass them to ``fit_pointwise_regressor``.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def build_mlp_regressor(
    input_dim: int,
    hidden_dims: Iterable[int] = (64, 64),
    output_dim: int = 1,
) -> nn.Module:
    """Build a small MLP for value regression."""
    layers: list[nn.Module] = []
    prev_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, int(hidden_dim)))
        layers.append(nn.ReLU())
        prev_dim = int(hidden_dim)
    layers.append(nn.Linear(prev_dim, int(output_dim)))
    return nn.Sequential(*layers)


def fit_pointwise_regressor(
    model: nn.Module,
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
) -> nn.Module:
    """Train ``model`` to regress ``targets`` from ``inputs`` with MSE loss."""
    features = np.asarray(inputs, dtype=np.float32)
    labels = np.asarray(targets, dtype=np.float32)
    dataset = TensorDataset(
        torch.as_tensor(features),
        torch.as_tensor(labels),
    )
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(int(epochs)):
        for feature_batch, label_batch in loader:
            prediction = model(feature_batch)
            loss = loss_fn(prediction, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return model


__all__ = ["build_mlp_regressor", "fit_pointwise_regressor"]
