"""Render dataset figures for the README:
  docs/synthetic.png — synthetic series behind `npm run bench:detectors` (ensemble study)
  docs/smd.png       — real SMD server telemetry: raw channels + ground-truth anomaly span
  docs/aiops.png     — representative real AIOps KPIs (clean / Donut-wins / coarse)

Each plot is skipped (with a note) if its dataset isn't present locally, so the
script runs even when only some datasets are fetched.

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


def _shade_labeled(ax, label: np.ndarray, lo: int, hi: int) -> None:
    """Shade contiguous label==1 runs within [lo, hi) as the ground-truth anomaly."""
    i, first = lo, True
    while i < hi:
        if label[i] == 1:
            j = i
            while j + 1 < hi and label[j + 1] == 1:
                j += 1
            ax.axvspan(i, j, color=RED, alpha=0.18, lw=0, label="labeled anomaly" if first else None)
            first = False
            i = j + 1
        else:
            i += 1


def plot_smd(entity: str = "machine-1-1") -> None:
    """Real SMD telemetry: a few of the entity's 38 channels around its largest
    labeled anomaly, with the ground-truth culprit channels (interpretation_label)
    plotted and the labeled span shaded — the multivariate signal Donut scores
    per-channel, and the root-cause ground truth the agent is graded against."""
    base = os.path.join(HERE, "..", "data", "smd")
    if not os.path.isdir(os.path.join(base, "test")):
        print("skip docs/smd.png — data/smd absent (see README to fetch the dataset)")
        return
    test = np.loadtxt(os.path.join(base, "test", f"{entity}.txt"), delimiter=",")
    label = np.loadtxt(os.path.join(base, "test_label", f"{entity}.txt"), delimiter=",").astype(int)

    segs: list[tuple[int, int, list[int]]] = []
    with open(os.path.join(base, "interpretation_label", f"{entity}.txt")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rng, dims = line.split(":")
            s, e = (int(x) for x in rng.split("-"))
            chans = [int(x) for x in dims.split(",") if x]
            segs.append((s, e, chans))

    s, e, chans = max(segs, key=lambda seg: seg[1] - seg[0])  # largest segment
    lo, hi = max(0, s - 1500), min(len(test), e + 1500)
    picks = [c - 1 for c in chans[:3]]  # interpretation_label is 1-indexed

    fig, axes = plt.subplots(len(picks), 1, figsize=(11, 6), sharex=True)
    x = np.arange(lo, hi)
    for ax, c in zip(axes, picks):
        ax.plot(x, test[lo:hi, c], color=AX, lw=0.7)
        _shade_labeled(ax, label, lo, hi)
        ax.set_title(f"metric_{c + 1:02d} — ground-truth culprit channel", fontsize=9.5, loc="left")
        ax.set_ylabel("value")
        ax.legend(loc="upper right", fontsize=8, frameon=False)
        ax.grid(alpha=0.15)
    axes[-1].set_xlabel("timestep")
    fig.suptitle(
        f"SMD data — {entity}: 3 of 38 channels around its worst anomaly "
        "(shaded = labeled span; interpretation_label names the culprit channels)",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "smd.png"), dpi=120)
    print("wrote docs/smd.png")


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
    if not os.path.exists(os.path.join(HERE, "..", "data", "aiops-kpi", "preliminary_train.csv")):
        print("skip docs/aiops.png — data/aiops-kpi absent (see README to fetch the dataset)")
        return
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
    plot_smd()
    plot_aiops()
