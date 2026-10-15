import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from critic_sharing.models.vae import VAE


def _batch_tensor(batch):
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def vae_loss(recon_x, x, mu, log_var):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_divergence


def train_vae(model, dataloader, optimizer, num_epochs=100):
    model.train()
    epoch_losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            batch = _batch_tensor(batch)
            optimizer.zero_grad()
            recon_batch, mu, log_var = model(batch)
            loss = vae_loss(recon_batch, batch, mu, log_var)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        mean_loss = total_loss / len(dataloader.dataset)
        epoch_losses.append(mean_loss)
        print(f"Epoch {epoch + 1}, Loss: {mean_loss:.4f}")
    return epoch_losses


def evaluate_vae_recon(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = _batch_tensor(batch)
            recon_batch, mu, log_var = model(batch)
            loss = nn.MSELoss()(recon_batch, batch).item()
            total_loss += loss
    print(f"Evaluation Loss: {total_loss / len(dataloader.dataset):.4f}")
    return total_loss / len(dataloader.dataset)


def sample_from_vae(model, num_samples=10):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.encoder[-1].out_features // 2)
        samples = model.decoder(z)
    return samples.numpy()


def make_position_dataloader(positions, batch_size=64, shuffle=True):
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 2:
        raise ValueError("Expected positions with shape (N, D).")
    dataset = TensorDataset(torch.as_tensor(positions))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_vae_on_positions(
    positions,
    latent_dim=2,
    batch_size=64,
    num_epochs=100,
    learning_rate=1e-3,
):
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 2:
        raise ValueError("Expected positions with shape (N, D).")
    if len(positions) == 0:
        raise ValueError("Cannot train a VAE on an empty position set.")

    dataloader = make_position_dataloader(positions, batch_size=batch_size, shuffle=True)
    model = VAE(input_dim=positions.shape[1], latent_dim=latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = train_vae(model, dataloader, optimizer, num_epochs=num_epochs)
    return model, losses
