"""z-score vs Donut benchmark on the real AIOps 2018 KPI dataset.

Standalone (no HTTP): trains a faithful Donut per KPI and a seasonal-naive
robust z-score, scores both against the dataset's labels, reports AUC-PR per
KPI and the macro mean. Minute-level data, so the z-score's seasonal period is
daily (1440). Memory-safe: minibatch training + window-batched scoring.

Run:  ml-service/.venv/bin/python ml-service/bench_aiops.py
Env:  N (cap pts/KPI, default 25000), EPOCHS (30), KPI_LIMIT (all), WIN (120)
"""
from __future__ import annotations

import csv
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from model import DonutVAE

torch.manual_seed(42)
_LOG_2PI = float(np.log(2 * np.pi))

CSV = os.path.join(os.path.dirname(__file__), "..", "data", "aiops-kpi", "preliminary_train.csv")
N = int(os.environ.get("N", 25000))          # cap points per KPI (tractability)
EPOCHS = int(os.environ.get("EPOCHS", 50))
WIN = int(os.environ.get("WIN", 120))        # Donut's window
KPI_LIMIT = int(os.environ.get("KPI_LIMIT", 0)) or None
LOOKBACK = 3                                  # prior days for the z-score baseline
N_Z = 64                                      # MC samples for reconstruction prob


def load():
    series = defaultdict(list)  # kpi -> list[(ts, value, label)]
    with open(CSV) as f:
        for d in csv.DictReader(f):
            series[d["KPI ID"]].append((int(d["timestamp"]), float(d["value"]), int(d["label"])))
    for k in series:
        series[k].sort(key=lambda r: r[0])
    return series


# ---- metric -------------------------------------------------------------
def auc_pr(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores)
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    total_pos = labels.sum()
    if total_pos == 0:
        return float("nan")
    precision = tp / (tp + fp)
    recall = tp / total_pos
    # trapezoid as recall advances, starting from (0,1)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.sum((recall[1:] - recall[:-1]) * (precision[1:] + precision[:-1]) / 2))


def best_f1_adjusted(scores: np.ndarray, labels: np.ndarray) -> float:
    """Donut paper's metric (§4.2): point-adjusted best F-score. A ground-truth
    anomaly *segment* counts as fully detected if ANY point in it is alerted;
    points outside segments are scored as usual. Sweep all thresholds, take best
    F1. Best F1 can only occur at a segment-max threshold (lowering it between
    segment maxes only adds false positives), so we enumerate those."""
    n = len(labels)
    segs, i = [], 0
    while i < n:
        if labels[i] == 1:
            j = i
            while j + 1 < n and labels[j + 1] == 1:
                j += 1
            segs.append((i, j)); i = j + 1
        else:
            i += 1
    P = sum(e - s + 1 for s, e in segs)
    if P == 0:
        return float("nan")
    seg_max = np.array([scores[s : e + 1].max() for s, e in segs])
    seg_len = np.array([e - s + 1 for s, e in segs], dtype=np.float64)
    normal = np.sort(scores[labels == 0])            # ascending
    order = np.argsort(-seg_max)
    sm, tp_cum = seg_max[order], np.cumsum(seg_len[order])
    best = 0.0
    for k in range(len(sm)):
        tp = tp_cum[k]
        fp = len(normal) - np.searchsorted(normal, sm[k], side="left")
        f1 = 2 * tp / (2 * tp + fp + (P - tp))       # FN = P - tp
        best = max(best, float(f1))
    return best


# ---- seasonal-naive robust z-score -------------------------------------
def zscore_scores(vals: np.ndarray, period: int) -> np.ndarray:
    n = len(vals)
    expected = np.full(n, np.nan)
    for i in range(period, n):
        refs = [vals[i - k * period] for k in range(1, LOOKBACK + 1) if i - k * period >= 0]
        if len(refs) >= 2:
            expected[i] = np.median(refs)
    resid = vals - expected
    valid = ~np.isnan(resid)
    med = np.median(resid[valid]) if valid.any() else 0.0
    mad = np.median(np.abs(resid[valid] - med)) if valid.any() else 0.0
    sigma = 1.4826 * (mad or 1e-9)
    scores = np.zeros(n)
    scores[valid] = np.abs((resid[valid] - med) / sigma)
    return scores


# ---- Donut --------------------------------------------------------------
def _windows(x: np.ndarray, w: int) -> np.ndarray:
    return np.stack([x[i : i + w] for i in range(len(x) - w + 1)])


def _glp(x, mean, std):
    return -0.5 * (((x - mean) / std) ** 2 + _LOG_2PI) - torch.log(std)


def donut_scores(vals: np.ndarray) -> np.ndarray:
    n = len(vals)
    mean, std = float(vals.mean()), float(vals.std() or 1.0)
    norm = (vals - mean) / std
    wins = _windows(norm, WIN)                       # (Nw, W)
    latent = min(8, max(3, WIN // 8))
    model = DonutVAE(window=WIN, hidden=100, latent=latent)
    x = torch.tensor(wins, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(42)

    model.train()
    nb = x.shape[0]
    bs = 256
    for ep in range(EPOCHS):
        for g in opt.param_groups:                   # LR anneal x0.75 / 10 epochs (paper)
            g["lr"] = 1e-3 * (0.75 ** (ep // 10))
        idx = rng.permutation(nb)
        for s in range(0, nb, bs):
            b = x[idx[s : s + bs]]
            y = (torch.rand_like(b) < 0.01).float()  # missing-data injection
            alpha = 1.0 - y
            beta = alpha.mean(dim=-1)
            x_mean, x_std, z, z_mean, z_std = model(b * alpha)
            recon = (alpha * _glp(b, x_mean, x_std)).sum(-1)
            log_pz = (-0.5 * (z ** 2 + _LOG_2PI)).sum(-1)
            log_qz = _glp(z, z_mean, z_std).sum(-1)
            loss = -(recon + beta * log_pz - log_qz).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    # ---- reconstruction probability (batched over windows) ----
    model.eval()
    last_lp = np.zeros(x.shape[0])
    first_win_lp = None
    with torch.no_grad():
        for s in range(0, x.shape[0], 1024):
            b = x[s : s + 1024]
            z_mean, z_std = model.encode(b)
            eps = torch.randn(N_Z, *z_mean.shape)
            zs = z_mean.unsqueeze(0) + z_std.unsqueeze(0) * eps
            xm, xs_ = model.decode(zs)
            lp = _glp(b.unsqueeze(0), xm, xs_).mean(0)   # (chunk, W)
            last_lp[s : s + b.shape[0]] = lp[:, -1].numpy()
            if s == 0:
                first_win_lp = lp[0].numpy()
    # localize to last point of each window; warm-up from first window
    nll = np.zeros(n)
    nll[: WIN - 1] = -first_win_lp[: WIN - 1]
    for i in range(x.shape[0]):
        nll[i + WIN - 1] = -last_lp[i]
    return nll


def main():
    series = load()
    kpis = sorted(series.keys())
    if KPI_LIMIT:
        kpis = kpis[:KPI_LIMIT]
    print(f"AIOps KPI benchmark · N≤{N}/KPI · W={WIN} · {EPOCHS} epochs · per-KPI daily period")
    print(f"  best-F1 = point-adjusted (paper §4.2) · AUC-PR = strict point-wise")
    print(f"{'KPI':>16}  {'pts':>6}  {'dt':>4}  {'anom':>5}  {'z-F1':>6}  {'d-F1':>6}  {'z-AUC':>6}  {'d-AUC':>6}  {'t(s)':>4}")
    zf_all, df_all, z_all, d_all = [], [], [], []
    for k in kpis:
        rows = series[k][:N]
        vals = np.array([r[1] for r in rows], dtype=np.float32)
        labels = np.array([r[2] for r in rows], dtype=np.int64)
        ts = np.array([r[0] for r in rows], dtype=np.int64)
        if len(vals) < 2 * WIN:
            continue
        dt = int(np.median(np.diff(ts))) or 60          # sampling interval (s)
        period = max(2, round(86400 / dt))               # points per day
        warmup = max(WIN - 1, LOOKBACK * period)
        ev = labels[warmup:]
        if ev.sum() == 0:
            print(f"{k:>16}  {len(vals):>6}  {dt:>4}  {int(ev.sum()):>5}  (no anomalies in eval range — skipped)")
            continue
        t0 = time.perf_counter()
        zs = zscore_scores(vals, period)[warmup:]
        ds = donut_scores(vals)[warmup:]
        zf, df = best_f1_adjusted(zs, ev), best_f1_adjusted(ds, ev)
        za, da = auc_pr(zs, ev), auc_pr(ds, ev)
        zf_all.append(zf); df_all.append(df); z_all.append(za); d_all.append(da)
        print(f"{k:>16}  {len(vals):>6}  {dt:>4}  {int(ev.sum()):>5}  {zf:>6.3f}  {df:>6.3f}  {za:>6.3f}  {da:>6.3f}  {time.perf_counter()-t0:>4.1f}")
    if z_all:
        print("-" * 74)
        print(f"{'MACRO MEAN':>16}  {'':>6}  {'':>4}  {'':>5}  {np.mean(zf_all):>6.3f}  {np.mean(df_all):>6.3f}  {np.mean(z_all):>6.3f}  {np.mean(d_all):>6.3f}   (n={len(z_all)})")


if __name__ == "__main__":
    main()
