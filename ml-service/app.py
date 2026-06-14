"""FastAPI service exposing the Donut-VAE anomaly detector.

This is a deliberately separate Python microservice (mirroring an edge model
served behind its own inference API): the Node/LangGraph agent fleet calls it
over HTTP, so the deep-learning detector composes with the rest of the system
without coupling languages.
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from detector import detect
from forecaster import forecast as tsfm_forecast

app = FastAPI(title="Prima ML Service", version="1.1.0")


class Point(BaseModel):
    date: str
    value: float


class DetectRequest(BaseModel):
    series: list[Point]
    model: str = Field(default="vae")              # "vae"
    threshold: str = Field(default="evt")          # "evt" (SPOT/POT) | "mad"
    window: int = Field(default=28, ge=3, le=60)  # ~4 weekly cycles; Donut uses 120 on long high-freq streams
    epochs: int = Field(default=150, ge=10, le=1000)
    k: float = Field(default=5.0, ge=1.0, le=10.0)
    q: float = Field(default=0.02, ge=1e-5, le=0.2)
    hidden: int = Field(default=100, ge=4, le=256)   # Donut uses ~100-unit hidden layers
    latent: int = Field(default=8, ge=2, le=128)     # clamped to <= window//4 in detector
    n_z: int = Field(default=256, ge=1, le=4096)     # MC z-samples for reconstruction prob (Donut: 1024)
    mcmc_iter: int = Field(default=10, ge=0, le=50)  # MCMC missing-data imputation iterations


class ForecastRequest(BaseModel):
    series: list[Point]
    horizon: int = Field(default=14, ge=1, le=90)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": ["vae"],
        "thresholding": ["evt", "mad"],
        "forecaster": "chronos-bolt",
    }


@app.post("/forecast")
def forecast_endpoint(req: ForecastRequest):
    values = [p.value for p in req.series]
    if len(values) < 8:
        return {"error": f"need at least 8 points, got {len(values)}"}
    return tsfm_forecast(values, req.horizon)


@app.post("/detect")
def detect_endpoint(req: DetectRequest):
    values = [p.value for p in req.series]
    dates = [p.date for p in req.series]
    result = detect(
        values, req.window, req.epochs, req.k, req.hidden, req.latent,
        kind=req.model, threshold=req.threshold, q=req.q,
        n_z=req.n_z, mcmc_iter=req.mcmc_iter,
    )
    for a in result.get("anomalies", []):
        a["date"] = dates[a["index"]]
    return result
