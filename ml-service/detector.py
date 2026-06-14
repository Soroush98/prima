"""Training + scoring pipeline for the Donut anomaly detector.

Faithful port of Donut (Xu et al., WWW'18 — github.com/NetManAIOps/donut):

  * Training  — modified ELBO (M-ELBO): per-point reconstruction log-prob is
    masked by alpha = 1 - y (y = injected-missing indicator) and the prior term
    is scaled by beta = mean(alpha); points are randomly injected as "missing"
    each step (missing-data injection).
  * Scoring   — reconstruction *probability*: mean over n_z Monte-Carlo z-samples
    of log p(x|z) at the last step of each window, optionally after MCMC
    missing-data imputation. We report the negative log-prob as the anomaly
    score (high = anomalous) so the downstream EVT/POT and MAD thresholding,
    which expect "high = anomalous", are unchanged.

Trained per request (cached by series hash). Thresholds with a robust
median/MAD cutoff or a self-calibrating EVT/POT threshold (SPOT, Siffer 2017).
"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from model import DonutVAE

torch.manual_seed(42)

_LOG_2PI = math.log(2 * math.pi)
_CACHE: dict[str, "TrainedModel"] = {}


@dataclass
class TrainedModel:
    model: nn.Module
    kind: str
    mean: float
    std: float
    window: int
    train_ms: float
    final_loss: float


def _windows(values: np.ndarray, window: int) -> np.ndarray:
    n = len(values)
    return np.stack([values[i : i + window] for i in range(n - window + 1)])


def _series_key(values: list[float], window: int, epochs: int, kind: str) -> str:
    h = hashlib.sha1()
    h.update(np.asarray(values, dtype=np.float32).tobytes())
    h.update(f"{window}-{epochs}-{kind}".encode())
    return h.hexdigest()


def _gaussian_log_prob(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Elementwise log N(x; mean, std)."""
    return -0.5 * (((x - mean) / std) ** 2 + _LOG_2PI) - torch.log(std)


def _train(values, window, epochs, hidden, latent, kind, inject_rate=0.01) -> TrainedModel:
    key = _series_key(values, window, epochs, kind)
    if key in _CACHE:
        return _CACHE[key]

    torch.manual_seed(42)
    start = time.perf_counter()
    arr = np.asarray(values, dtype=np.float32)
    mean, std = float(arr.mean()), float(arr.std() or 1.0)
    norm = (arr - mean) / std
    wins = _windows(norm, window)                        # (N, W)

    # Tight bottleneck (latent << window) so the VAE learns the *normal* manifold
    # instead of trivially copying anomalies through z.
    vae_latent = min(latent, max(2, window // 4))
    model = DonutVAE(window=window, hidden=max(hidden, 48), latent=vae_latent)
    x = torch.tensor(wins, dtype=torch.float32)          # (N, W)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)  # Donut: 1e-3
    model.train()
    final_loss = 0.0
    for _ in range(epochs):
        opt.zero_grad()

        # ---- missing-data injection: mark ~inject_rate points "missing" (y=1),
        #      zero them in the *input*; reconstruction target stays the clean x.
        y = (torch.rand_like(x) < inject_rate).float()   # 1 = injected-missing
        alpha = 1.0 - y                                  # observed weight
        beta = alpha.mean(dim=-1)                         # (N,) = fraction observed
        x_in = x * alpha                                  # zero the missing inputs

        x_mean, x_std, z, z_mean, z_std = model(x_in)

        # ---- M-ELBO (Donut eq.):  E_q[ sum(alpha * log p(x|z)) + beta*log p(z) - log q(z|x) ]
        x_log_prob = _gaussian_log_prob(x, x_mean, x_std)            # (N, W) — clean target
        recon = (alpha * x_log_prob).sum(dim=-1)                     # masked reconstruction
        log_pz = (-0.5 * (z ** 2 + _LOG_2PI)).sum(dim=-1)            # N(z;0,I)
        log_qz = _gaussian_log_prob(z, z_mean, z_std).sum(dim=-1)    # q(z|x)
        elbo = recon + beta * log_pz - log_qz
        loss = -elbo.mean()

        loss.backward()
        opt.step()
        final_loss = float(loss.item())

    tm = TrainedModel(model, kind, mean, std, window, round((time.perf_counter() - start) * 1000, 1), round(final_loss, 5))
    _CACHE[key] = tm
    return tm


@torch.no_grad()
def _masked_reconstruct(model: nn.Module, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """One MCMC step: replace masked (missing) positions with the decoder mean,
    keep observed positions fixed. mask: 1 = missing."""
    z_mean, _ = model.encode(x)
    x_mean, _ = model.decode(z_mean)          # deterministic decode from the posterior mean
    return torch.where(mask.bool(), x_mean, x)


@torch.no_grad()
def _iterative_masked_reconstruct(model, x, mask, iters: int) -> torch.Tensor:
    """Donut's iterative_masked_reconstruct: refine the imputed values `iters` times."""
    for _ in range(iters):
        x = _masked_reconstruct(model, x, mask)
    return x


@torch.no_grad()
def _reconstruction_prob(tm: TrainedModel, wins: np.ndarray, n_z: int, missing=None, mcmc_iter: int = 10) -> np.ndarray:
    """Reconstruction probability log p(x|z), MC-averaged over n_z samples.

    Returns (N_win, W) of per-position log-probs. When a `missing` mask is given,
    MCMC missing-data imputation refines the input first (Donut). With no missing
    data — our default detection path — imputation is a no-op (Donut's
    `missing=None` branch), matching the reference behaviour exactly.
    """
    tm.model.eval()
    x = torch.tensor(wins, dtype=torch.float32)           # (N, W)
    if missing is not None:
        x = _iterative_masked_reconstruct(tm.model, x, torch.tensor(missing, dtype=torch.float32), mcmc_iter)

    z_mean, z_std = tm.model.encode(x)                    # (N, L)
    # n_z latent samples: z ~ q(z|x) → (n_z, N, L)
    eps = torch.randn(n_z, *z_mean.shape)
    zs = z_mean.unsqueeze(0) + z_std.unsqueeze(0) * eps
    x_mean, x_std = tm.model.decode(zs)                   # (n_z, N, W)
    target = x.unsqueeze(0)                               # (1, N, W)
    log_prob = _gaussian_log_prob(target, x_mean, x_std)  # (n_z, N, W)
    return log_prob.mean(dim=0).numpy()                   # (N, W) — averaged over z-samples


def _pot_threshold(scores: np.ndarray, init_q: float = 0.90, q: float = 0.02) -> float:
    """EVT/POT threshold via a method-of-moments GPD fit on the tail."""
    n = len(scores)
    t = float(np.quantile(scores, init_q))
    peaks = scores[scores > t] - t
    if len(peaks) < 10:
        return float(np.quantile(scores, 1 - q))
    m, v = float(peaks.mean()), float(peaks.var(ddof=1) or 1e-9)
    xi = 0.5 * (1 - m * m / v)
    sigma = m * (1 - xi)
    if sigma <= 0 or not np.isfinite(xi):
        return float(np.quantile(scores, 1 - q))
    ratio = (q * n) / len(peaks)
    if abs(xi) < 1e-6:
        zq = t - sigma * np.log(ratio)
    else:
        zq = t + (sigma / xi) * (ratio ** (-xi) - 1)
    return float(zq) if np.isfinite(zq) and zq >= t else float(np.quantile(scores, 1 - q))


def detect(values, window, epochs, k, hidden, latent, kind="vae", threshold="evt", q=0.02, n_z=256, mcmc_iter=10):
    n = len(values)
    if n < window * 2:
        return {"error": f"need at least {window * 2} points, got {n}", "anomalies": [], "scores": []}

    tm = _train(values, window, epochs, hidden, latent, kind)
    arr = np.asarray(values, dtype=np.float32)
    norm = (arr - tm.mean) / tm.std
    wins = _windows(norm, window)

    # reconstruction probability per window position; anomaly score = negative
    # log-prob (high = anomalous), so EVT/POT + MAD thresholding stay unchanged.
    log_prob = _reconstruction_prob(tm, wins, n_z=n_z, mcmc_iter=mcmc_iter)   # (N_win, W)
    nll = -log_prob

    # localize: last-step score of the window ending on each point (no bleed)
    scores = np.zeros(n)
    scores[: window - 1] = nll[0, : window - 1]
    for i in range(nll.shape[0]):
        scores[i + window - 1] = nll[i, window - 1]

    med = float(np.median(scores))
    mad = float(np.median(np.abs(scores - med))) or 1e-9
    robust_sigma = 1.4826 * mad
    if threshold == "evt":
        thr = _pot_threshold(scores, 0.90, q)
    else:
        thr = med + k * robust_sigma

    anomalies = []
    for idx in range(n):
        if scores[idx] <= thr:
            continue
        z = (scores[idx] - med) / robust_sigma
        anomalies.append(
            {
                "index": idx,
                "value": float(values[idx]),
                "score": round(float(scores[idx]), 5),
                "zscore": round(float(z), 2),
                "severity": "high" if z > 6 else "medium" if z > 4 else "low",
            }
        )

    return {
        "anomalies": anomalies,
        "scores": [round(float(s), 5) for s in scores],
        "threshold": round(float(thr), 5),
        "model": kind,
        "thresholding": threshold,
        "params": {"window": window, "epochs": epochs, "hidden": hidden, "latent": latent, "k": k, "q": q, "n_z": n_z, "mcmc_iter": mcmc_iter},
        "train_ms": tm.train_ms,
        "final_loss": tm.final_loss,
    }
