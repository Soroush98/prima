"""z-score vs Donut benchmark on the real SMD (Server Machine Dataset).

SMD ships 28 server entities, each with 38 metric channels, a clean `train`
split (assumed anomaly-free), a `test` split, and per-point `test_label`s.
Anomalies are multivariate, so we run a univariate Donut *per channel*, then
aggregate channel scores into one entity-level anomaly score per timestep and
grade it against the real labels.

Differences from bench_aiops.py (which this reuses primitives from):
  * multivariate  -> Donut per channel, max-aggregate standardized scores
  * real split    -> train Donut on `train/`, score `test/` (no peeking)
  * no seasonality-> baseline is a causal rolling-median robust z-score
                     (SMD has no timestamps / daily period to exploit)

Run:  ml-service/.venv/bin/python ml-service/bench_smd.py
Env:  MACHINES (csv of entity ids, default machine-1-1)
      EPOCHS (30)  WIN (60)  N (cap test pts, default 0=all)
      TRAIN_N (cap train pts, default 0=all)  CH_LIMIT (cap channels, 0=all)
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from model import DonutVAE
# reuse the exact metric + Donut primitives the AIOps bench is graded on
from bench_aiops import auc_pr, best_f1_adjusted, _windows, _glp

torch.manual_seed(42)

SMD_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "smd")
EPOCHS = int(os.environ.get("EPOCHS", 30))
WIN = int(os.environ.get("WIN", 60))
N = int(os.environ.get("N", 0)) or None              # cap test points
TRAIN_N = int(os.environ.get("TRAIN_N", 0)) or None  # cap train points
CH_LIMIT = int(os.environ.get("CH_LIMIT", 0)) or None
N_Z = 64                                             # MC samples for recon prob
ROLL = 100                                           # baseline rolling-median window


def load_entity(name: str):
    """Return (train[T1,38], test[T2,38], label[T2]) for one SMD entity."""
    train = np.loadtxt(os.path.join(SMD_DIR, "train", f"{name}.txt"), delimiter=",")
    test = np.loadtxt(os.path.join(SMD_DIR, "test", f"{name}.txt"), delimiter=",")
    label = np.loadtxt(os.path.join(SMD_DIR, "test_label", f"{name}.txt"), delimiter=",")
    if train.ndim == 1:  # single-channel guard
        train, test = train[:, None], test[:, None]
    return train, test, label.astype(np.int64)


# ---- non-seasonal robust z-score baseline -------------------------------
def zscore_channel(test_vals: np.ndarray) -> np.ndarray:
    """Residual from a causal rolling median, MAD-standardized. No seasonality
    assumed (SMD has none) — this is the honest simple baseline for it."""
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


# ---- Donut: train on `train`, score `test` ------------------------------
def donut_channel(train_vals: np.ndarray, test_vals: np.ndarray) -> np.ndarray:
    """Per-point reconstruction NLL on `test`, with a Donut trained only on the
    (clean) `train` split. Mirrors bench_aiops.donut_scores but respects the
    real SMD split instead of training and scoring on one series."""
    mean, std = float(train_vals.mean()), float(train_vals.std() or 1.0)
    tr = (train_vals - mean) / std
    te = (test_vals - mean) / std
    if len(tr) < 2 * WIN or len(te) < WIN:
        return np.zeros(len(test_vals))

    latent = min(8, max(3, WIN // 8))
    model = DonutVAE(window=WIN, hidden=100, latent=latent)
    x = torch.tensor(_windows(tr, WIN), dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(42)

    model.train()
    nb, bs = x.shape[0], 256
    for ep in range(EPOCHS):
        for g in opt.param_groups:                       # paper LR anneal
            g["lr"] = 1e-3 * (0.75 ** (ep // 10))
        idx = rng.permutation(nb)
        for s in range(0, nb, bs):
            b = x[idx[s : s + bs]]
            y = (torch.rand_like(b) < 0.01).float()      # missing-data injection
            alpha = 1.0 - y
            beta = alpha.mean(dim=-1)
            x_mean, x_std, z, z_mean, z_std = model(b * alpha)
            recon = (alpha * _glp(b, x_mean, x_std)).sum(-1)
            log_pz = (-0.5 * (z ** 2 + float(np.log(2 * np.pi)))).sum(-1)
            log_qz = _glp(z, z_mean, z_std).sum(-1)
            loss = -(recon + beta * log_pz - log_qz).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    # ---- score test windows (batched) ----
    model.eval()
    xte = torch.tensor(_windows(te, WIN), dtype=torch.float32)
    last_lp = np.zeros(xte.shape[0])
    first_win_lp = None
    with torch.no_grad():
        for s in range(0, xte.shape[0], 1024):
            b = xte[s : s + 1024]
            z_mean, z_std = model.encode(b)
            eps = torch.randn(N_Z, *z_mean.shape)
            zs = z_mean.unsqueeze(0) + z_std.unsqueeze(0) * eps
            xm, xs_ = model.decode(zs)
            lp = _glp(b.unsqueeze(0), xm, xs_).mean(0)   # (chunk, W)
            last_lp[s : s + b.shape[0]] = lp[:, -1].numpy()
            if s == 0:
                first_win_lp = lp[0].numpy()
    n = len(test_vals)
    nll = np.zeros(n)
    nll[: WIN - 1] = -first_win_lp[: WIN - 1]
    for i in range(xte.shape[0]):
        nll[i + WIN - 1] = -last_lp[i]
    return nll


def _standardize(s: np.ndarray) -> np.ndarray:
    """Robust per-channel standardization so one noisy channel can't dominate
    the max-aggregation."""
    med = np.median(s)
    mad = np.median(np.abs(s - med)) or 1e-9
    return (s - med) / (1.4826 * mad)


def main():
    machines = os.environ.get("MACHINES", "machine-1-1").split(",")
    print(f"SMD benchmark · per-channel Donut, max-aggregated · W={WIN} · {EPOCHS} epochs")
    print(f"  best-F1 = point-adjusted (Donut §4.2) · AUC-PR = strict point-wise · baseline = rolling-median z")
    print(f"{'entity':>14}  {'chans':>5}  {'pts':>6}  {'anom':>6}  {'z-F1':>6}  {'d-F1':>6}  {'z-AUC':>6}  {'d-AUC':>6}  {'t(s)':>5}")
    zf_all, df_all, z_all, d_all = [], [], [], []
    for name in machines:
        name = name.strip()
        train, test, label = load_entity(name)
        if TRAIN_N:
            train = train[:TRAIN_N]
        if N:
            test, label = test[:N], label[:N]
        chans = range(test.shape[1] if CH_LIMIT is None else min(CH_LIMIT, test.shape[1]))
        if label.sum() == 0:
            print(f"{name:>14}  (no anomalies in test — skipped)")
            continue
        t0 = time.perf_counter()
        z_agg = np.full(len(label), -np.inf)
        d_agg = np.full(len(label), -np.inf)
        used = 0
        for c in chans:
            tr_c, te_c = train[:, c], test[:, c]
            if tr_c.std() < 1e-9:          # constant channel carries no signal
                continue
            used += 1
            z_agg = np.maximum(z_agg, _standardize(zscore_channel(te_c)))
            d_agg = np.maximum(d_agg, _standardize(donut_channel(tr_c, te_c)))
        # ignore warm-up region for both detectors
        warm = WIN - 1
        ev, zs, ds = label[warm:], z_agg[warm:], d_agg[warm:]
        zf, df = best_f1_adjusted(zs, ev), best_f1_adjusted(ds, ev)
        za, da = auc_pr(zs, ev), auc_pr(ds, ev)
        zf_all.append(zf); df_all.append(df); z_all.append(za); d_all.append(da)
        print(f"{name:>14}  {used:>5}  {len(label):>6}  {int(label.sum()):>6}  "
              f"{zf:>6.3f}  {df:>6.3f}  {za:>6.3f}  {da:>6.3f}  {time.perf_counter()-t0:>5.1f}")
    if z_all:
        print("-" * 86)
        print(f"{'MACRO MEAN':>14}  {'':>5}  {'':>6}  {'':>6}  "
              f"{np.mean(zf_all):>6.3f}  {np.mean(df_all):>6.3f}  {np.mean(z_all):>6.3f}  "
              f"{np.mean(d_all):>6.3f}   (n={len(z_all)})")


if __name__ == "__main__":
    main()
