# Prima ML Service — Donut-VAE Anomaly Detector

A standalone **Python / FastAPI + PyTorch** microservice that the Node/LangGraph
agent fleet calls over HTTP. It detects anomalies in a univariate time series with
an unsupervised deep model — a **Donut-style VAE** — that learns to reconstruct
normal windows; points reconstructed poorly are flagged. The cutoff is set by
**EVT/POT (SPOT)**, a self-calibrating tail threshold, not a hand-picked constant.
No labels needed.

## Run

```bash
npm run ml          # from repo root — creates the uv venv on first run
# or directly:
cd ml-service && ./run.sh
```

Service listens on `http://127.0.0.1:8000` (override with `ML_PORT`).

## API

`POST /detect`
```jsonc
{
  "series": [{"date": "2011-01-01", "value": 63}, ...],
  "model": "vae",       // "vae"
  "threshold": "evt",   // "evt" (SPOT/POT) | "mad"
  "window": 14,         // sequence length
  "epochs": 150,        // training epochs (model is cached per series)
  "q": 0.02             // EVT target false-alarm rate (or "k" for mad)
}
```
Returns per-point reconstruction-error scores, the anomaly list (with severity and
an error z-score), and training telemetry (`train_ms`, `final_loss`).

`POST /forecast`
```jsonc
{ "series": [{"date":"2011-01-01","value":63}, ...], "horizon": 14 }
```
Zero-shot forecast from **Chronos-Bolt** (`amazon/chronos-bolt-small`, a pretrained
time-series foundation model). Returns `median` + `lower`/`upper` (p10/p90) arrays —
no training on the input series. The ~48MB model downloads from HuggingFace on first
use and is cached.

`GET /health` → models, thresholding options, and forecaster.

## Design notes

- **Localized scoring** — each point is scored by the reconstruction error at the
  last step of the window ending on it, so an anomaly's error isn't smeared onto its
  neighbours.
- **Robust threshold** — median/MAD on the error signal (same robust statistic used
  on the Node side), so the anomalies don't inflate their own cutoff.
- **Cached training** — the trained model is memoised by a hash of the series, so
  repeat requests are instant.
- **Reproducible** — `torch.manual_seed(42)`.

Mirrors a production pattern: an unsupervised anomaly model served behind its own
inference API, consumed by the rest of the system over the network.
