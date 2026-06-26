"""z-score vs OmniAnomaly benchmark on the real SMD (Server Machine Dataset).

SMD ships 28 server entities, each with 38 metric channels, a clean `train` split
(assumed anomaly-free), a `test` split, and per-point `test_label`s. Its anomalies
are *multivariate* — channels that are individually plausible but jointly
impossible (a correlation breaking). SMD is the dataset OmniAnomaly was introduced
on (Su et al., KDD'19), so this is its design regime.

The two detectors graded here:
  * baseline   — a causal rolling-median robust z-score scored *per channel*, then
                 max-aggregated. Dependency-free; the honest simple comparison.
                 It can only see one channel move at a time.
  * OmniAnomaly— ONE multivariate model over all 38 channels at once, trained on
                 the clean `train` split and scored on `test` (real split, no
                 peeking). One model, one score per timestep — it can see
                 *correlations* break, which the per-channel baseline cannot.

We report a strict metric (point-wise AUC-PR), a lenient one (point-adjusted
best-F1, the paper's headline), AND the deployed EVT/POT threshold's F1 — never
just the flattering number. See `metrics.py` for what each means.

Run:  ml-service/.venv/bin/python ml-service/bench_smd.py
Env:  MACHINES (csv of entity ids, default machine-1-1)
      EPOCHS (10)  WIN (60)  N_Z (16)  N (cap test pts, 0=all)
      TRAIN_N (cap train pts, 0=all)  CH_LIMIT (cap channels, 0=all)
      HIDDEN (32)  LATENT (8)  FLOWS (4)  BATCH (64)
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from model import OmniAnomaly, fit, localize, window_scores, windows
from metrics import auc_pr, best_f1_adjusted, adjusted_f1_at
from detector import _pot_threshold

torch.manual_seed(42)

SMD_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "smd")
EPOCHS = int(os.environ.get("EPOCHS", 10))
WIN = int(os.environ.get("WIN", 60))
N = int(os.environ.get("N", 0)) or None              # cap test points
TRAIN_N = int(os.environ.get("TRAIN_N", 0)) or None  # cap train points
CH_LIMIT = int(os.environ.get("CH_LIMIT", 0)) or None
N_Z = int(os.environ.get("N_Z", 16))                 # MC samples for recon prob
HIDDEN = int(os.environ.get("HIDDEN", 32))
LATENT = int(os.environ.get("LATENT", 8))
FLOWS = int(os.environ.get("FLOWS", 4))
BATCH = int(os.environ.get("BATCH", 64))
ROLL = 100                                           # baseline rolling-median window


def load_entity(name: str):
    """Return (train[T1,38], test[T2,38], label[T2]) for one SMD entity."""
    train = np.loadtxt(os.path.join(SMD_DIR, "train", f"{name}.txt"), delimiter=",")
    test = np.loadtxt(os.path.join(SMD_DIR, "test", f"{name}.txt"), delimiter=",")
    label = np.loadtxt(os.path.join(SMD_DIR, "test_label", f"{name}.txt"), delimiter=",")
    if train.ndim == 1:  # single-channel guard
        train, test = train[:, None], test[:, None]
    return train, test, label.astype(np.int64)


# ---- non-seasonal robust z-score baseline (per channel) -----------------
def zscore_channel(test_vals: np.ndarray) -> np.ndarray:
    """Residual from a causal rolling median, MAD-standardized. No seasonality
    assumed (SMD has none) — the honest simple baseline for it."""
    n = len(test_vals)
    expected = np.full(n, np.nan)
    for i in range(ROLL, n):
        expected[i] = np.median(test_vals[i - ROLL : i])
    resid = test_vals - expected
    valid = ~np.isnan(resid)
    med = np.median(resid[valid]) if valid.any() else 0.0
    mad = np.median(np.abs(resid[valid] - med)) if valid.any() else 0.0
    # Floor sigma: SMD channels are normalized [0,1] and often piecewise-constant,
    # so MAD~0; without a floor a tiny move divides by ~0 and z explodes. 0.01 caps
    # a full-range jump at z≈100. (Matches src/lib/smdWarehouse.ts channelZ.)
    sigma = max(1.4826 * mad, 0.01)
    out = np.zeros(n)
    out[valid] = np.abs((resid[valid] - med) / sigma)
    return out


def _standardize(s: np.ndarray) -> np.ndarray:
    """Robust per-channel standardization so one noisy channel can't dominate the
    baseline's max-aggregation."""
    med = np.median(s)
    mad = np.median(np.abs(s - med)) or 1e-9
    return (s - med) / (1.4826 * mad)


# ---- OmniAnomaly: ONE multivariate model, train on `train`, score `test` ----
def omni_entity(train: np.ndarray, test: np.ndarray) -> np.ndarray:
    """Per-timestep anomaly NLL on `test` from a single OmniAnomaly trained on the
    clean `train` split — all channels jointly. Standardize per channel by *train*
    statistics so the model sees a zero-mean/unit-scale manifold."""
    mean = train.mean(0)
    std = train.std(0)
    std[std < 1e-9] = 1.0                              # constant channels → flat after norm
    tr = (train - mean) / std
    te = (test - mean) / std
    if len(tr) < 2 * WIN or len(te) < WIN:
        return np.zeros(len(test))

    d = train.shape[1]
    latent = min(LATENT, max(3, WIN // 4))
    model = OmniAnomaly(dim=d, window=WIN, hidden=HIDDEN, latent=latent, n_flows=FLOWS)
    x = windows(torch.tensor(tr, dtype=torch.float32), WIN)        # (N, W, D)
    fit(model, x, epochs=EPOCHS, batch=BATCH, seed=42)

    xte = windows(torch.tensor(te, dtype=torch.float32), WIN)      # (M, W, D)
    last_lp, first_lp = window_scores(model, xte, n_z=N_Z, batch=1024)
    return localize(last_lp, first_lp, len(test), WIN)


def main():
    machines = os.environ.get("MACHINES", "machine-1-1").split(",")
    print(f"SMD benchmark · OmniAnomaly (multivariate, one model/entity) vs rolling-median z · W={WIN} · {EPOCHS} epochs")
    print(f"  best-F1 = point-adjusted (lenient/oracle) · AUC-PR = strict point-wise · o-EVT = point-adj F1 at deployed POT cutoff")
    print(f"{'entity':>14}  {'chans':>5}  {'pts':>6}  {'anom':>6}  "
          f"{'z-F1':>6}  {'o-F1':>6}  {'z-AUC':>6}  {'o-AUC':>6}  {'o-EVT':>6}  {'t(s)':>5}")
    zf_all, of_all, za_all, oa_all, oe_all = [], [], [], [], []
    for name in machines:
        name = name.strip()
        train, test, label = load_entity(name)
        if TRAIN_N:
            train = train[:TRAIN_N]
        if N:
            test, label = test[:N], label[:N]
        if CH_LIMIT is not None:
            train, test = train[:, :CH_LIMIT], test[:, :CH_LIMIT]
        if label.sum() == 0:
            print(f"{name:>14}  (no anomalies in test — skipped)")
            continue

        t0 = time.perf_counter()

        # baseline: per-channel rolling-median z, max-aggregated over channels
        z_agg = np.full(len(label), -np.inf)
        used = 0
        for c in range(test.shape[1]):
            if train[:, c].std() < 1e-9:               # constant channel carries no signal
                continue
            used += 1
            z_agg = np.maximum(z_agg, _standardize(zscore_channel(test[:, c])))

        # OmniAnomaly: one multivariate model over all channels at once
        o = omni_entity(train, test)

        warm = WIN - 1                                 # ignore warm-up for both detectors
        ev, zs, os_ = label[warm:], z_agg[warm:], o[warm:]
        zf, of = best_f1_adjusted(zs, ev), best_f1_adjusted(os_, ev)
        za, oa = auc_pr(zs, ev), auc_pr(os_, ev)
        oe = adjusted_f1_at(os_, ev, _pot_threshold(os_, 0.90, 0.02))  # deployed EVT/POT cutoff
        zf_all.append(zf); of_all.append(of); za_all.append(za); oa_all.append(oa); oe_all.append(oe)
        print(f"{name:>14}  {used:>5}  {len(label):>6}  {int(label.sum()):>6}  "
              f"{zf:>6.3f}  {of:>6.3f}  {za:>6.3f}  {oa:>6.3f}  {oe:>6.3f}  {time.perf_counter()-t0:>5.1f}")
    if za_all:
        print("-" * 92)
        print(f"{'MACRO MEAN':>14}  {'':>5}  {'':>6}  {'':>6}  "
              f"{np.mean(zf_all):>6.3f}  {np.mean(of_all):>6.3f}  {np.mean(za_all):>6.3f}  "
              f"{np.mean(oa_all):>6.3f}  {np.mean(oe_all):>6.3f}   (n={len(za_all)})")


if __name__ == "__main__":
    main()
