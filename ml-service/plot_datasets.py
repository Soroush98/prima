"""Render dataset figures for the README:
  docs/synthetic.png — the synthetic Prima benchmark series (large + subtle regimes)
  docs/aiops.png     — representative real AIOps KPIs (clean / Donut-wins / coarse)

Run: ml-service/.venv/bin/python ml-service/plot_datasets.py
"""
from __future__ import annotations

import csv
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
DOCS = os.path.join(HERE, "docs")
os.makedirs(DOCS, exist_ok=True)
AX = "#1f6feb"
RED = "#e5484d"


# ---- faithful port of the TS mulberry32 + makeLabeledSeries -------------
def mulberry32(seed: int):
    state = seed & 0xFFFFFFFF

    def rng():
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        s = state
        t = ((s ^ (s >> 15)) * (1 | s)) & 0xFFFFFFFF
        t = (((t + (((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF)) & 0xFFFFFFFF) ^ t) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    return rng


DAYS = 180
LARGE = [[22, 1, 0.45], [40, 1, 1.7], [58, 3, 0.5], [85, 1, 1.8], [105, 5, 0.6], [130, 1, 0.4], [152, 1, 1.65]]
SUBTLE = [[18, 1, 0.85], [30, 1, 1.15], [44, 1, 0.88], [60, 1, 1.12], [74, 3, 0.88],
          [92, 1, 1.18], [108, 5, 0.87], [128, 1, 0.82], [140, 1, 1.2], [158, 1, 0.9]]


def make_series(seed: int, template):
    rand = mulberry32(seed)

    def laplace(b):
        u = rand() - 0.5
        return -b * (1 if u > 0 else -1 if u < 0 else 0) * math.log(1 - 2 * abs(u))

    PHI, TARGET_STD = 0.45, 30.0
    b = math.sqrt((TARGET_STD * TARGET_STD * (1 - PHI * PHI)) / 2)
    vals, noise = [], 0.0
    for d in range(DAYS):
        noise = PHI * noise + laplace(b)
        dow = (d + 3) % 7
        weekend = dow in (5, 6)
        v = 1000 + d * 2
        v *= 0.8 if weekend else 1.0
        v *= 1 + 0.04 * math.sin((d / 7) * 2 * math.pi)
        v += noise
        vals.append(max(50, v))
    truth = set()
    for start0, length, factor in template:
        jitter = int(rand() * 9) - 4
        start = min(DAYS - length - 1, max(1, start0 + jitter))
        for j in range(length):
            vals[start + j] *= factor
            truth.add(start + j)
    return np.array(vals), sorted(truth)


def plot_synthetic():
    # sanity-check the RNG matches the TS benchmark (seed 1000)
    chk = mulberry32(1000)
    assert abs(chk() - 0.795195) < 1e-5, "mulberry32 port drifted from TS"

    fig, axes = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True)
    for ax, (name, tmpl) in zip(axes, [("large (40–80% deviations)", LARGE), ("subtle (10–20% deviations)", SUBTLE)]):
        vals, truth = make_series(1000, tmpl)
        ax.plot(range(DAYS), vals, color=AX, lw=1.1)
        ax.scatter(truth, vals[truth], color=RED, s=34, zorder=5, label="injected anomaly")
        ax.set_title(f"Synthetic Prima DAU — {name}", fontsize=10, loc="left")
        ax.set_ylabel("DAU")
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        ax.grid(alpha=0.15)
    axes[1].set_xlabel("day")
    fig.suptitle("Benchmark A data — one clean daily series with a single weekly season", fontsize=11, y=0.99)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "synthetic.png"), dpi=120)
    print("wrote docs/synthetic.png")


def load_aiops():
    path = os.path.join(HERE, "..", "data", "aiops-kpi", "preliminary_train.csv")
    series = defaultdict(list)
    with open(path) as f:
        for d in csv.DictReader(f):
            series[d["KPI ID"]].append((int(d["timestamp"]), float(d["value"]), int(d["label"])))
    for k in series:
        series[k].sort()
    return series


def plot_aiops():
    series = load_aiops()
    # (kpi, caption) — clean→z-score wins, Donut-wins, coarse 5-min→Donut wins
    picks = [
        ("18fbb1d5a5dc099d", "1-min, sharp spike — z-score ranks far better (AUC-PR 0.69 vs 0.11)"),
        ("9ee5879409dccef9", "1-min, irregular shape — Donut wins (best-F1 0.97 vs 0.74)"),
        ("07927a9a18fa19ae", "5-min sampling — Donut wins big (best-F1 0.86 vs 0.14)"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(11, 7))
    for ax, (kpi, cap) in zip(axes, picks):
        rows = series[kpi]
        vals = np.array([r[1] for r in rows])
        labs = np.array([r[2] for r in rows])
        anom = np.where(labs == 1)[0]
        # window around the first anomaly so seasonality + an incident are both visible
        first = anom[0] if len(anom) else 0
        lo, hi = max(0, first - 1500), max(0, first - 1500) + 4000
        x = np.arange(lo, hi)
        ax.plot(x, vals[lo:hi], color=AX, lw=0.7)
        seg = anom[(anom >= lo) & (anom < hi)]
        ax.scatter(seg, vals[seg], color=RED, s=12, zorder=5, label="labeled anomaly")
        ax.set_title(f"{kpi[:10]}… — {cap}", fontsize=9.5, loc="left")
        ax.set_ylabel("value")
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        ax.grid(alpha=0.15)
    axes[-1].set_xlabel("sample index")
    fig.suptitle("Benchmark B data — real AIOps KPIs (heterogeneous: sampling rate & shape vary)", fontsize=11, y=0.995)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "aiops.png"), dpi=120)
    print("wrote docs/aiops.png")


if __name__ == "__main__":
    plot_synthetic()
    plot_aiops()
