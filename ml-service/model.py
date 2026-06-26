"""OmniAnomaly: a stochastic-RNN VAE for *multivariate* time-series anomaly detection.

PyTorch port of OmniAnomaly (Su et al., KDD'19 — "Robust Anomaly Detection for
Multivariate Time Series through Stochastic Recurrent Neural Network",
github.com/NetManAIOps/OmniAnomaly). Unlike a per-channel detector, one model
sees all D channels jointly, so it scores *correlations* breaking, not just
single channels moving — which is exactly what SMD's multivariate anomalies are.

Signature mechanisms (each load-bearing; a port that drops one is not OmniAnomaly):

  1. **GRU in both nets.** A GRU reads the input window in the inference net
     (qnet) and a GRU drives the generative net (pnet) — the latent carries
     *temporal* context, it is not inferred per-timestep i.i.d. This is the
     "recurrent" in stochastic-RNN.
  2. **Temporally-connected stochastic latent.** The prior is a learned
     linear-Gaussian transition p(z_t | z_{t-1}) (a Linear Gaussian State Space
     Model), so latents are stochastically linked across time rather than an
     independent N(0,I) per step.
  3. **Planar Normalizing Flows** (Rezende & Mohamed '15) enrich the posterior
     q(z_t|·) beyond a diagonal Gaussian — OmniAnomaly's stated way of getting a
     non-Gaussian, higher-capacity posterior.
  4. **Gaussian reconstruction** p(x_t | z_t) = N(μ_x, σ_x): the decoder emits a
     mean *and* a spread, so the anomaly score is a genuine reconstruction
     *probability*, not an L2 error.

Trained by maximizing the per-timestep ELBO (SGVB); scored by the reconstruction
probability at each window's last step (high −log p ⇒ anomalous), thresholded by
EVT/POT downstream. Unsupervised — no labels in training.

Deviations from the reference (flagged per mle-practices):
  * the latent transition is a single learned linear-Gaussian map rather than the
    original's full LGSSM/Kalman parameterization;
  * planar flows default to K=4; the paper uses ~20 NF layers on long streams.
  Both are cost trade-offs for CPU/per-request training and are documented next to
  the benchmark numbers in the README.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

STD_EPSILON = 1e-4  # softplus floor on predicted stds (matches OmniAnomaly's std_epsilon)


def gaussian_log_prob(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Elementwise log N(x; mean, std)."""
    log_2pi = 1.8378770664093453  # math.log(2*pi)
    return -0.5 * (((x - mean) / std) ** 2 + log_2pi) - torch.log(std)


class PlanarFlow(nn.Module):
    """One planar normalizing-flow step f(z) = z + u·h(wᵀz + b), h = tanh
    (Rezende & Mohamed 2015, eqs. 10–12). `u` is reparameterized so wᵀû ≥ −1,
    which keeps the map invertible and the log-det well defined."""

    def __init__(self, dim: int):
        super().__init__()
        self.u = nn.Parameter(torch.randn(dim) * 0.01)
        self.w = nn.Parameter(torch.randn(dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # z: (..., dim).  Returns (f(z), log|det ∂f/∂z|) with log-det shape (...).
        wu = (self.w * self.u).sum()
        u_hat = self.u + (F.softplus(wu) - 1 - wu) * self.w / (self.w.pow(2).sum() + 1e-8)
        lin = (z * self.w).sum(-1, keepdim=True) + self.b      # (..., 1)
        f = z + u_hat * torch.tanh(lin)
        psi = (1 - torch.tanh(lin) ** 2) * self.w              # (..., dim)
        det = (1 + (psi * u_hat).sum(-1)).abs() + 1e-8         # (...)
        return f, torch.log(det)


class OmniAnomaly(nn.Module):
    """Stochastic-RNN VAE over windows of a D-channel series (Su et al., KDD'19).

    Input windows are (B, W, D). The inference GRU produces a per-step posterior
    q(z_t|·) that is pushed through K planar flows; the generative GRU maps the
    flowed latent back to a Gaussian reconstruction of the window. The latent
    prior is a learned linear-Gaussian transition across time.
    """

    def __init__(self, dim: int, window: int, hidden: int = 32, latent: int = 8, n_flows: int = 4):
        super().__init__()
        self.dim = dim
        self.window = window
        self.latent = latent
        # qnet — GRU inference net
        self.enc_rnn = nn.GRU(dim, hidden, batch_first=True)
        self.enc_mu = nn.Linear(hidden, latent)
        self.enc_std_pre = nn.Linear(hidden, latent)
        self.flows = nn.ModuleList(PlanarFlow(latent) for _ in range(n_flows))
        # linear-Gaussian latent transition  p(z_t | z_{t-1})
        self.trans = nn.Linear(latent, latent)
        self.trans_std_pre = nn.Parameter(torch.zeros(latent))
        # pnet — GRU generative net
        self.dec_rnn = nn.GRU(latent, hidden, batch_first=True)
        self.dec_mu = nn.Linear(hidden, dim)
        self.dec_std_pre = nn.Linear(hidden, dim)

    @staticmethod
    def _std(pre: torch.Tensor) -> torch.Tensor:
        return F.softplus(pre) + STD_EPSILON

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, _ = self.enc_rnn(x)                         # (B, W, H)
        return self.enc_mu(h), self._std(self.enc_std_pre(h))

    def apply_flows(self, z0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = z0
        log_det = torch.zeros(z.shape[:-1], device=z.device)
        for flow in self.flows:
            z, ld = flow(z)
            log_det = log_det + ld
        return z, log_det                              # zK, Σ log|det| over flows  (B, W)

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        d, _ = self.dec_rnn(z)                         # (B, W, H)
        return self.dec_mu(d), self._std(self.dec_std_pre(d))

    def transition_log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """log p(z_t | z_{t-1}); z_0 ~ N(0, I). z: (B, W, L) → (B, W)."""
        prior_mean = torch.zeros_like(z)
        prior_mean[:, 1:, :] = self.trans(z[:, :-1, :])
        std = self._std(self.trans_std_pre).expand_as(z).clone()
        std[:, 0, :] = 1.0                             # z_0 prior is the standard normal
        return gaussian_log_prob(z, prior_mean, std).sum(-1)

    def elbo(self, x: torch.Tensor) -> torch.Tensor:
        """Mean per-window ELBO: Σ_t E_q[log p(x_t|z_t)] + log p(z_t|z_{t-1}) − log q(z_t)."""
        mu_q, std_q = self.encode(x)
        z0 = mu_q + std_q * torch.randn_like(std_q)             # reparameterize
        log_q0 = gaussian_log_prob(z0, mu_q, std_q).sum(-1)     # (B, W)
        zK, log_det = self.apply_flows(z0)
        log_q = log_q0 - log_det                                # density of the flowed latent
        log_pz = self.transition_log_prob(zK)                  # (B, W)
        x_mu, x_std = self.decode(zK)
        log_px = gaussian_log_prob(x, x_mu, x_std).sum(-1)     # (B, W)
        elbo = (log_px + log_pz - log_q).sum(-1)               # sum over the window → (B,)
        return elbo.mean()

    @torch.no_grad()
    def reconstruction_log_prob(self, x: torch.Tensor, n_z: int) -> torch.Tensor:
        """E_q[log p(x_t|z_t)], MC-averaged over n_z posterior samples. → (B, W)."""
        mu_q, std_q = self.encode(x)
        acc = torch.zeros(x.shape[:-1])                        # (B, W)
        for _ in range(n_z):
            z0 = mu_q + std_q * torch.randn_like(std_q)
            zK, _ = self.apply_flows(z0)
            x_mu, x_std = self.decode(zK)
            acc += gaussian_log_prob(x, x_mu, x_std).sum(-1)
        return acc / n_z


# --------------------------------------------------------------------------- #
# Shared train / score utilities — used by both the live detector and the bench #
# --------------------------------------------------------------------------- #

def windows(values: torch.Tensor, window: int) -> torch.Tensor:
    """Stride-1 sliding windows. values: (T, D) → (T-W+1, W, D)."""
    return values.unfold(0, window, 1).transpose(1, 2)


def fit(
    model: OmniAnomaly,
    x: torch.Tensor,
    epochs: int,
    lr: float = 1e-3,
    batch: int = 128,
    seed: int = 42,
    anneal: bool = True,
) -> float:
    """SGVB training loop (Adam, minibatched, paper-style LR anneal, grad-clip).
    x: (N, W, D). Returns the final minibatch −ELBO."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    n = x.shape[0]
    final = 0.0
    for ep in range(epochs):
        if anneal:
            for g in opt.param_groups:                 # OmniAnomaly anneals LR ×0.75 / 10 epochs
                g["lr"] = lr * (0.75 ** (ep // 10))
        perm = torch.randperm(n, generator=gen)
        for s in range(0, n, batch):
            b = x[perm[s : s + batch]]
            loss = -model.elbo(b)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            final = float(loss.item())
    return final


@torch.no_grad()
def window_scores(model: OmniAnomaly, x: torch.Tensor, n_z: int = 16, batch: int = 1024):
    """Per-window reconstruction log-prob. Returns (last_lp[N], first_window_lp[W]):
    the last-step log-prob of every window, plus the full first window for warm-up."""
    model.eval()
    n = x.shape[0]
    last = torch.empty(n)
    first = None
    for s in range(0, n, batch):
        b = x[s : s + batch]
        lp = model.reconstruction_log_prob(b, n_z)     # (chunk, W)
        last[s : s + b.shape[0]] = lp[:, -1]
        if s == 0:
            first = lp[0].clone()
    return last.numpy(), first.numpy()


def localize(last_lp, first_lp, n_points: int, window: int):
    """Map per-window log-probs to a per-timestep NLL (high = anomalous): each point
    gets the score of the window *ending* on it; the first window seeds the warm-up."""
    import numpy as np

    nll = np.zeros(n_points)
    nll[: window - 1] = -first_lp[: window - 1]
    for i in range(len(last_lp)):
        nll[i + window - 1] = -last_lp[i]
    return nll
