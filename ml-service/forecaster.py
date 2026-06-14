"""Zero-shot forecasting with a time-series foundation model.

Uses **Chronos-Bolt** (Amazon) — a pretrained transformer that forecasts
*without any training on our data*. It returns probabilistic quantiles, which
map directly onto the forecast prediction band.

The model (~48MB) downloads from HuggingFace on first use and is cached.
"""
from __future__ import annotations

import threading
import time
from typing import Any

import torch

MODEL_ID = "amazon/chronos-bolt-small"
_pipe: Any = None
_pipe_lock = threading.Lock()


def get_pipe() -> Any:
    """Lazily load the pipeline once. Double-checked locking so concurrent first
    requests (uvicorn's sync threadpool) don't race to download the model."""
    global _pipe
    if _pipe is None:
        with _pipe_lock:
            if _pipe is None:
                try:
                    from chronos import BaseChronosPipeline

                    _pipe = BaseChronosPipeline.from_pretrained(
                        MODEL_ID, device_map="cpu", dtype=torch.float32
                    )
                except Exception as exc:  # download/load failure is transient & external
                    raise RuntimeError(f"failed to load {MODEL_ID}: {exc}") from exc
    return _pipe


def forecast(
    values: list[float], horizon: int = 14, quantiles: tuple[float, float, float] = (0.1, 0.5, 0.9)
) -> dict[str, Any]:
    if len(quantiles) != 3:
        raise ValueError(f"expected (lower, median, upper) quantiles, got {quantiles}")
    start = time.perf_counter()
    pipe = get_pipe()
    ctx = torch.tensor(values, dtype=torch.float32)
    q, _mean = pipe.predict_quantiles(ctx, prediction_length=horizon, quantile_levels=list(quantiles))
    arr = q[0].numpy()  # (H, 3) — columns align with the requested (lower, median, upper)
    lower_i, median_i, upper_i = 0, 1, 2
    return {
        "lower": [round(float(x), 2) for x in arr[:, lower_i]],
        "median": [round(float(x), 2) for x in arr[:, median_i]],
        "upper": [round(float(x), 2) for x in arr[:, upper_i]],
        "model": MODEL_ID,
        "zero_shot": True,
        "infer_ms": round((time.perf_counter() - start) * 1000, 1),
    }
