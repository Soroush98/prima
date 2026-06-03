"""Donut-style VAE for univariate time-series anomaly detection.

The same idea behind a production multi-task edge anomaly detector: learn to
*reconstruct* normal windows of the signal; points the model reconstructs
poorly (high error) are anomalous. No labels required — it's unsupervised.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DonutVAE(nn.Module):
    """Donut-style variational autoencoder for seasonal KPIs (Xu et al., WWW'18).

    A dense VAE over a flattened window: encoder → (μ, logσ²) → reparameterized
    latent z → decoder reconstruction. Trained with the ELBO (reconstruction +
    KL). Well matched to seasonal business metrics; the probabilistic latent
    regularizes the "normal" manifold it learns.
    """

    def __init__(self, window: int, hidden: int = 64, latent: int = 8):
        super().__init__()
        self.window = window
        self.enc = nn.Sequential(
            nn.Linear(window, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.dec = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, window),
        )

    def encode(self, x: torch.Tensor):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x: torch.Tensor):  # x: (B, W)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.dec(z)
        return recon, mu, logvar
