"""Zero-shot forecasting with a time-series foundation model.

Uses **Chronos-Bolt** (Amazon; the efficient Chronos-2 lineage) — a pretrained
transformer that forecasts *without any training on our data*. It returns
probabilistic quantiles, which map directly onto the forecast prediction band.

The model (~48MB) downloads from HuggingFace on first use and is cached.
"""
from __future__ import annotations

import time

import torch

MODEL_ID = "amazon/chronos-bolt-small"
_pipe = None


def get_pipe():
    global _pipe
    if _pipe is None:
        from chronos import BaseChronosPipeline

        _pipe = BaseChronosPipeline.from_pretrained(MODEL_ID, device_map="cpu", dtype=torch.float32)
    return _pipe


def forecast(values: list[float], horizon: int = 14, quantiles=(0.1, 0.5, 0.9)) -> dict:
    start = time.perf_counter()
    pipe = get_pipe()
    ctx = torch.tensor(values, dtype=torch.float32)
    q, _mean = pipe.predict_quantiles(ctx, prediction_length=horizon, quantile_levels=list(quantiles))
    arr = q[0].numpy()  # (H, len(quantiles))
    return {
        "lower": [round(float(x), 2) for x in arr[:, 0]],
        "median": [round(float(x), 2) for x in arr[:, 1]],
        "upper": [round(float(x), 2) for x in arr[:, 2]],
        "model": MODEL_ID,
        "zero_shot": True,
        "infer_ms": round((time.perf_counter() - start) * 1000, 1),
    }
