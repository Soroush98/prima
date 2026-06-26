"""Training + scoring pipeline for the OmniAnomaly detector (live /detect path).

OmniAnomaly is natively *multivariate*; the live HTTP API takes a single KPI, so
this path runs it as a one-channel series (D=1). That keeps the service contract
unchanged and the deep detector available as a benchmarked reference — but the
regime where OmniAnomaly actually earns its keep is many correlated channels at
once (SMD), which `bench_smd.py` exercises directly.

  * Training — maximize the per-timestep ELBO (SGVB) over sliding windows.
  * Scoring  — reconstruction *probability*: mean over n_z Monte-Carlo posterior
    samples of log p(x|z) at the last step of each window. We report the negative
    log-prob as the anomaly score (high = anomalous) so the downstream EVT/POT and
    MAD thresholding, which expect "high = anomalous", are unchanged.

Trained per request (cached by series hash). Thresholds with a robust median/MAD
cutoff or a self-calibrating EVT/POT threshold (SPOT, Siffer 2017).
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from model import OmniAnomaly, fit, localize, window_scores, windows

# Reproducibility: training re-seeds locally in _train and scoring uses a seeded
# generator so a given (series, params) is deterministic.
_SEED = 42

_EPS = 1e-9
_MAD_TO_SIGMA = 1.4826           # MAD → σ for a Gaussian
_MIN_PEAKS_FOR_GPD = 10          # min tail exceedances to fit the GPD, else fall back to a quantile
_SEVERITY_HIGH_Z = 6.0
_SEVERITY_MED_Z = 4.0


@dataclass
class TrainedModel:
    model: nn.Module
    kind: str
    mean: float
    std: float
    window: int
    train_ms: float
    final_loss: float


_CACHE: dict[str, TrainedModel] = {}


def _series_key(
    values: list[float], window: int, epochs: int, kind: str, hidden: int, latent: int
) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(np.asarray(values, dtype=np.float32).tobytes())
    # every hyperparameter that changes the trained weights must be in the key,
    # else a cached model trained with different hidden/latent is returned (bug).
    h.update(f"{window}-{epochs}-{kind}-{hidden}-{latent}".encode())
    return h.hexdigest()


def _train(
    values: list[float],
    window: int,
    epochs: int,
    hidden: int,
    latent: int,
    kind: str,
) -> TrainedModel:
    key = _series_key(values, window, epochs, kind, hidden, latent)
    if key in _CACHE:
        return _CACHE[key]

    torch.manual_seed(_SEED)
    start = time.perf_counter()
    arr = np.asarray(values, dtype=np.float32)
    mean, std = float(arr.mean()), float(arr.std() or 1.0)
    norm = (arr - mean) / std

    # Tight bottleneck (latent << window) so the VAE learns the *normal* manifold
    # instead of trivially copying anomalies through z.
    z_latent = min(latent, max(2, window // 4))
    model = OmniAnomaly(dim=1, window=window, hidden=max(hidden, 16), latent=z_latent)
    x_win = windows(torch.tensor(norm, dtype=torch.float32).reshape(-1, 1), window)  # (N, W, 1)

    final_loss = fit(model, x_win, epochs=epochs, batch=128, seed=_SEED)

    tm = TrainedModel(model, kind, mean, std, window, round((time.perf_counter() - start) * 1000, 1), round(final_loss, 5))
    _CACHE[key] = tm
    return tm


def _pot_threshold(scores: np.ndarray, init_q: float = 0.90, q: float = 0.02) -> float:
    """EVT/POT threshold via a method-of-moments GPD fit on the tail."""
    n = len(scores)
    t = float(np.quantile(scores, init_q))
    peaks = scores[scores > t] - t
    if len(peaks) < _MIN_PEAKS_FOR_GPD:
        return float(np.quantile(scores, 1 - q))
    m, v = float(peaks.mean()), float(peaks.var(ddof=1) or _EPS)
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


def detect(
    values: list[float],
    window: int,
    epochs: int,
    k: float,
    hidden: int,
    latent: int,
    kind: str = "omni",
    threshold: str = "evt",
    q: float = 0.02,
    n_z: int = 256,
) -> dict[str, Any]:
    n = len(values)
    if n < window * 2:
        return {"error": f"need at least {window * 2} points, got {n}", "anomalies": [], "scores": []}

    tm = _train(values, window, epochs, hidden, latent, kind)
    arr = np.asarray(values, dtype=np.float32)
    norm = (arr - tm.mean) / tm.std

    x_win = windows(torch.tensor(norm, dtype=torch.float32).reshape(-1, 1), window)  # (N, W, 1)

    # reconstruction probability per window; anomaly score = negative log-prob
    # (high = anomalous), so EVT/POT + MAD thresholding stay unchanged.
    torch.manual_seed(_SEED)
    last_lp, first_lp = window_scores(tm.model, x_win, n_z=n_z)
    scores = localize(last_lp, first_lp, n, window)

    med = float(np.median(scores))
    mad = float(np.median(np.abs(scores - med))) or _EPS
    robust_sigma = _MAD_TO_SIGMA * mad
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
                "severity": "high" if z > _SEVERITY_HIGH_Z else "medium" if z > _SEVERITY_MED_Z else "low",
            }
        )

    return {
        "anomalies": anomalies,
        "scores": [round(float(s), 5) for s in scores],
        "threshold": round(float(thr), 5),
        "model": kind,
        "thresholding": threshold,
        "params": {"window": window, "epochs": epochs, "hidden": hidden, "latent": latent, "k": k, "q": q, "n_z": n_z},
        "train_ms": tm.train_ms,
        "final_loss": tm.final_loss,
    }
