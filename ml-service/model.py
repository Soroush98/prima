"""Donut VAE for univariate time-series anomaly detection.

Faithful PyTorch port of Donut (Xu et al., WWW'18 — github.com/NetManAIOps/donut):
a variational auto-encoder over sliding windows of a seasonal KPI whose decoder
is a *Gaussian* p(x|z) (mean AND std), trained with the modified ELBO (M-ELBO)
and missing-data injection, and scored by reconstruction *probability* under
MCMC missing-data imputation. No labels required — it's unsupervised.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

STD_EPSILON = 1e-4  # softplus floor on predicted stds (matches Donut's std_epsilon)


class DonutVAE(nn.Module):
    """Donut variational autoencoder for seasonal KPIs (Xu et al., WWW'18).

    Encoder q(z|x) → (μ_z, σ_z); reparameterized latent z; decoder p(x|z) →
    (μ_x, σ_x), i.e. a diagonal-Gaussian likelihood over the window. Both the
    encoder and decoder predict standard deviations via softplus (Donut emits a
    full Gaussian rather than a point reconstruction — this is what makes the
    reconstruction-probability score well-defined).
    """

    def __init__(self, window: int, hidden: int = 100, latent: int = 8):
        super().__init__()
        self.window = window
        self.latent = latent
        self.enc = nn.Sequential(
            nn.Linear(window, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.z_mean = nn.Linear(hidden, latent)
        self.z_std_pre = nn.Linear(hidden, latent)
        self.dec = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.x_mean = nn.Linear(hidden, window)
        self.x_std_pre = nn.Linear(hidden, window)

    def encode(self, x: torch.Tensor):
        """q(z|x) → (μ_z, σ_z)."""
        h = self.enc(x)
        return self.z_mean(h), F.softplus(self.z_std_pre(h)) + STD_EPSILON

    def decode(self, z: torch.Tensor):
        """p(x|z) → (μ_x, σ_x). Works for z of shape (..., latent)."""
        h = self.dec(z)
        return self.x_mean(h), F.softplus(self.x_std_pre(h)) + STD_EPSILON

    @staticmethod
    def reparameterize(mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return mu + std * torch.randn_like(std)

    def forward(self, x: torch.Tensor):  # x: (B, W)
        z_mean, z_std = self.encode(x)
        z = self.reparameterize(z_mean, z_std)
        x_mean, x_std = self.decode(z)
        return x_mean, x_std, z, z_mean, z_std
