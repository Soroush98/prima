"""Training + scoring pipeline for the deep anomaly detector.

Uses a Donut-style VAE, trained per request (cached by series hash), localizes
each point's reconstruction error to the last step of its window, and thresholds
either with a robust median/MAD cutoff or a self-calibrating EVT/POT threshold
(SPOT, Siffer et al. 2017).
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from model import DonutVAE

torch.manual_seed(42)

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


def _train(values, window, epochs, hidden, latent, kind) -> TrainedModel:
    key = _series_key(values, window, epochs, kind)
    if key in _CACHE:
        return _CACHE[key]

    torch.manual_seed(42)
    start = time.perf_counter()
    arr = np.asarray(values, dtype=np.float32)
    mean, std = float(arr.mean()), float(arr.std() or 1.0)
    norm = (arr - mean) / std
    wins = _windows(norm, window)                        # (N, W)

    # Tight bottleneck (latent << window) so the VAE can't trivially copy
    # anomalies; Donut-style denoising (missing-data injection + noise) forces
    # it to learn the *normal* manifold.
    vae_latent = min(latent, max(2, window // 4))
    model: nn.Module = DonutVAE(window=window, hidden=max(hidden, 48), latent=vae_latent)
    x = torch.tensor(wins, dtype=torch.float32)          # (N, W)

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    mse = nn.MSELoss()
    model.train()
    final_loss = 0.0
    for ep in range(epochs):
        opt.zero_grad()
        # missing-data injection (mask ~10%) + input noise → reconstruct clean
        mask = (torch.rand_like(x) > 0.1).float()
        x_in = x * mask + 0.1 * torch.randn_like(x)
        recon, mu, logvar = model(x_in)
        recon_loss = mse(recon, x)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        beta = min(1.0, ep / max(1, epochs // 2)) * 0.05  # light KL, warmed up
        loss = recon_loss + beta * kl
        loss.backward()
        opt.step()
        final_loss = float(loss.item())

    tm = TrainedModel(model, kind, mean, std, window, round((time.perf_counter() - start) * 1000, 1), round(final_loss, 5))
    _CACHE[key] = tm
    return tm


def _reconstruction(tm: TrainedModel, wins: np.ndarray) -> np.ndarray:
    """Return reconstructed windows (N, W)."""
    tm.model.eval()
    with torch.no_grad():
        x = torch.tensor(wins, dtype=torch.float32)
        mu, logvar = tm.model.encode(x)
        recon = tm.model.dec(mu)                          # deterministic decode from the mean
        return recon.numpy()


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


def detect(values, window, epochs, k, hidden, latent, kind="vae", threshold="evt", q=0.02):
    n = len(values)
    if n < window * 2:
        return {"error": f"need at least {window * 2} points, got {n}", "anomalies": [], "scores": []}

    tm = _train(values, window, epochs, hidden, latent, kind)
    arr = np.asarray(values, dtype=np.float32)
    norm = (arr - tm.mean) / tm.std
    wins = _windows(norm, window)
    recon = _reconstruction(tm, wins)
    err = (recon - wins) ** 2                             # (N_win, W)

    # localize: last-step error of the window ending on each point (no bleed)
    scores = np.zeros(n)
    scores[: window - 1] = err[0, : window - 1]
    for i in range(err.shape[0]):
        scores[i + window - 1] = err[i, window - 1]

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
        "params": {"window": window, "epochs": epochs, "hidden": hidden, "latent": latent, "k": k, "q": q},
        "train_ms": tm.train_ms,
        "final_loss": tm.final_loss,
    }
