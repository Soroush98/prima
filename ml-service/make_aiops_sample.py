"""Extract a small, committable AIOps KPI sample for the UI viewer.

For each chosen KPI, find the densest-anomaly window (W points) and store the
rounded values + the anomaly indices within that window. Output is a few hundred
KB → ships in the Docker image (unlike the 360MB full dataset, which is gitignored).

Run: ml-service/.venv/bin/python ml-service/make_aiops_sample.py
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, "..", "data", "aiops-kpi", "preliminary_train.csv")
OUT = os.path.join(HERE, "..", "data", "aiops-sample.json")
W = 1200  # window length to display

# (kpi id, short caption) — a representative spread of shapes/sampling rates
PICKS = [
    ("18fbb1d5a5dc099d", "Sharp spike on a noisy baseline (1-min)"),
    ("9ee5879409dccef9", "Smooth, irregular waves — anomalies at the peaks (1-min)"),
    ("88cf3a776ba00e7c", "Volatile KPI, frequent anomalies (1-min)"),
    ("da403e4e3f87c9e0", "Strong daily cycle with dips (1-min)"),
    ("07927a9a18fa19ae", "Coarse 5-min sampling, clear oscillation"),
    ("76f4550c43334374", "Coarse 5-min sampling, low-volume dips"),
]


def load():
    series = defaultdict(list)
    with open(SRC) as f:
        for d in csv.DictReader(f):
            series[d["KPI ID"]].append((int(d["timestamp"]), float(d["value"]), int(d["label"])))
    for k in series:
        series[k].sort()
    return series


def densest_window(labels: list[int], w: int) -> int:
    """Start index of the length-w window with the most anomalies, but capped at
    35% anomalous so the viewer keeps enough normal context to be readable."""
    n = len(labels)
    if n <= w:
        return 0
    cap = int(0.35 * w)
    cur = sum(labels[:w])
    best, best_i = -1, 0
    for i in range(0, n - w + 1):
        if i > 0:
            cur += labels[i + w - 1] - labels[i - 1]
        if 0 < cur <= cap and cur > best:
            best, best_i = cur, i
    return best_i


def main():
    series = load()
    out = {"kpis": []}
    for kpi, caption in PICKS:
        rows = series[kpi]
        ts = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        labs = [r[2] for r in rows]
        interval = int(round((ts[-1] - ts[0]) / max(1, len(ts) - 1)))
        s = densest_window(labs, W)
        win_vals = vals[s : s + W]
        win_labs = labs[s : s + W]
        anomalies = [i for i, a in enumerate(win_labs) if a == 1]
        out["kpis"].append({
            "id": kpi,
            "caption": caption,
            "intervalSec": interval,
            "startTs": ts[s],
            "values": [round(v, 4) for v in win_vals],
            "anomalies": anomalies,
        })
        print(f"{kpi}  interval={interval}s  window@{s}  anomalies={len(anomalies)}/{len(win_vals)}")

    with open(OUT, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"\nwrote {OUT}  ({os.path.getsize(OUT) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
