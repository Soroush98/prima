# Prima: Agentic DAU/WAU Anomaly & Forecasting Platform

A **LangGraph** agent fleet that does the analysis a static BI dashboard can't. Prima
watches product engagement (Daily and Weekly Active Users), writes and runs its own SQL,
finds anomalies, forecasts where the metric is heading, works out *why* it moved, and
writes you a short brief in plain English. It comes with an observability layer and an
LLM evaluation scorecard.

It runs on **Node 24 + Next.js 16 (App Router) + TypeScript**. The agent graph is built
with **LangGraph.js**, **Claude** is the reasoning engine, and a **Python / FastAPI +
PyTorch** service hosts a **Donut-VAE** anomaly detector that the agents call over HTTP.

> **Live Claude only.** The SQL-Analyst, Root-Cause, and Narrator agents call a real
> Claude model, so you need an `ANTHROPIC_API_KEY` (`cp .env.example .env`). The
> statistical engine (anomaly detection and forecasting) is plain deterministic code.

---

## What this demonstrates

A full, end-to-end take on an enterprise analytical agent ecosystem, and where each
capability lives in the code:

| Capability | Where it lives in this repo |
| --- | --- |
| Autonomous analytical agents using **LangGraph** | [src/agents/graph.ts](src/agents/graph.ts), [src/agents/nodes.ts](src/agents/nodes.ts) |
| Agents that **autonomously leverage SQL** to extract/manipulate large datasets | `sql_analyst` node generates, **guards**, and executes SQL ([src/agents/nodes.ts](src/agents/nodes.ts), [src/lib/db.ts](src/lib/db.ts)) |
| **Standardize DAU / WAU** tracking | Real `COUNT(DISTINCT)` + trailing-7-day window SQL ([src/agents/util.ts](src/agents/util.ts)) |
| Move beyond chat to **proactive, prescriptive** actions | `root_cause` + `narrator` agents emit diagnosis + recommendations ([src/agents/nodes.ts](src/agents/nodes.ts)) |
| **Predictive / prescriptive** analytics (vs legacy PowerBI) | Zero-shot **Chronos-Bolt** foundation-model forecasting (Holt-Winters fallback) + prescriptive narration ([ml-service/forecaster.py](ml-service/forecaster.py)) |
| **Prompt analysis metrics & LLM performance evaluation** | Golden-set eval harness: routing, SQL validity, recall, hallucination, latency, cost ([src/eval/](src/eval/)) |
| **Observability dashboards** (feature usage, perf, agent health) | Next.js dashboard with live agent trace ([src/app/page.tsx](src/app/page.tsx)) |
| **Statistical modeling for predictive analytics** (preferred) | Seasonal-trend decomposition, robust MAD z-scores, Holt-Winters ([src/lib/stats.ts](src/lib/stats.ts)) |
| **Deep learning** (PyTorch) | Donut-VAE anomaly detector in a FastAPI microservice, ensembled with the statistical detector ([ml-service/](ml-service/)) |
| **GCP / BigQuery** style SQL (preferred) | SQLite warehouse with the same SQL surface (CTEs, window fns, `COUNT DISTINCT`) |
| **Technical writing / documentation** | This README + heavily documented modules |

---

## Architecture

```
                       ┌─────────────────────────────────────────────┐
  "Why did DAU drop    │              LangGraph state graph            │
   for EU mobile?"  ─► │                                               │
                       │  orchestrator → sql_analyst → anomaly_detector│
                       │        → forecaster → root_cause → narrator   │
                       └───────┬───────────────┬──────────────┬────────┘
                               │               │              │
                        ┌──────▼─────┐  ┌───────▼──────┐ ┌─────▼──────┐
                        │  SQLite     │  │  stats.ts    │ │   Claude    │
                        │ (warehouse) │  │ decompose /  │ │  (Sonnet)   │
                        │ 600k+ rows  │  │ HW forecast  │ │             │
                        └─────────────┘  └──────────────┘ └─────────────┘
                               │
                       ┌───────▼──────────────────────────────────────┐
                       │   Next.js dashboard + /api + eval scorecard   │
                       └───────────────────────────────────────────────┘
```

### The six agents

1. **Orchestrator** parses the natural-language question into a metric (DAU/WAU) and a dimensional scope (region / platform / feature).
2. **SQL Analyst** writes a read-only SQL query for the metric, **guards** it (single SELECT/WITH, no DDL/DML), executes it, and returns the time series. If generation fails it falls back to a validated template.
3. **Anomaly Detector** is an **ensemble**: a statistical **seasonal-naive week-over-week** z-score (robust median/MAD) combined with a **PyTorch Donut-VAE** served from the Python microservice. When both agree, the anomaly is marked high-confidence. See [Deep-learning detector & ensemble](#deep-learning-detector--ensemble-pytorch).
4. **Forecaster** uses zero-shot **Chronos-Bolt** (a pretrained time-series foundation model from Amazon) to forecast 14 days with probabilistic quantile bands, with *no training on our data*. It falls back to a hand-written Holt-Winters model when the ML service is offline. See [ml-service/forecaster.py](ml-service/forecaster.py).
5. **Root-Cause** lines up the worst drop against the `deploys` table and the worst-hit `region / platform` segments (week over week), then forms hypotheses grounded in that evidence.
6. **Narrator** writes a short executive briefing with a recommendation.

Every node also writes an **observability trace** entry (status, latency, mode, detail)
and adds up token/cost telemetry through LangGraph state reducers.

The dashboard has two tabs: the live analysis view, and an **Architecture** tab that
diagrams the LangGraph runtime and how a request flows through the model. It's there to
walk someone through the internals.

---

## Quickstart: zero setup

```bash
npm install
npm run dev      # open http://localhost:3000  (no seeding step needed)
```

**No `npm run seed`.** On the first request the app builds its own warehouse: if
`data/prima.db` is missing it downloads the real **UCI "Online Retail"** dataset from
GitHub and builds it (about 1 to 8 seconds, then cached), see
[src/lib/bootstrap.ts](src/lib/bootstrap.ts). After that every query runs on real data.

> Port 3000 busy? `PORT=3100 npm run dev`. Requires `ANTHROPIC_API_KEY`:
> `cp .env.example .env` and add your key (the agents call Claude on every request).

> **Optional: deep-learning detector.** Start the PyTorch service in a second
> terminal (`npm run ml`) and the anomaly agent automatically ensembles it with the
> statistical detector. If it's offline the graph runs statistical-only, so the ML
> detector adds to the result but never blocks it.

### The data: real DAU, computed live

| | Auto-bootstrapped (default) |
| --- | --- |
| Source | UCI **Online Retail** CSV via [this GitHub repo](https://github.com/amir-hojjati/Data-Analysis-Online-Retail-Transactions) (541,909 rows) |
| DAU = | `COUNT(DISTINCT CustomerID)` per day, written and run by the SQL agent |
| Span | 305 active days (Dec 2010 to Dec 2011), DAU 11 / 63 / 154 |
| Regions | 38 real countries (United Kingdom, Germany, France, EIRE, …) |
| Features | product category derived from item Description |

The orchestrator reads each dataset's *actual* dimension vocabulary from the DB, so
questions like *"How is DAU trending in the United Kingdom?"* route correctly with no
hardcoding. On real data the fleet finds genuine signals on its own. For example, the
**-82.5% DAU drop on 2010-12-22** (the pre-Christmas wind-down), with per-country
segment attribution.

### Optional commands

```bash
npm run eval         # LLM evaluation scorecard → data/eval-report.json
npm run seed:github  # force-rebuild the real warehouse now (instead of lazily)
npm run seed         # build the synthetic, storyline-rich dataset instead
```

`npm run eval` is intentionally **isolated**: it builds a *synthetic* dataset (with
injected, known anomalies) in its own `data/eval-synthetic.db` and scores the agents
against that ground truth. It never touches the real runtime warehouse. Real data has
no labels to assert against, so evaluation always uses the synthetic fixture.

Optional, go live with Claude:

```bash
cp .env.example .env      # then set ANTHROPIC_API_KEY=...
```

> If port 3000 is busy: `PORT=3100 npm run dev`.

---

## The synthetic fixture (for evaluation)

The synthetic generator ([src/lib/synthetic.ts](src/lib/synthetic.ts), exposed via
`npm run seed`) builds a raw `user_activity` fact table (one row per user / day /
feature) so DAU/WAU are computed with **real SQL**, not pre-aggregated. It's what the
eval harness scores against, because three storylines are deliberately baked in as
known ground truth:

- **AI-assistant launch** (~day 90): a usage spike after `v3.9.0`.
- **Bad EU/mobile deploy** (`v4.2.0`, ~day 150): a sharp DAU drop isolated to EU mobile, the platform's headline incident.
- **Export decay**: the legacy `export` feature slowly bleeding users.

A `deploys` table provides the release events the Root-Cause agent correlates against.

---

## Deep-learning detector & ensemble (PyTorch)

Anomaly detection runs as a **two-detector ensemble**, each scored continuously and
thresholded with **EVT/POT (SPOT)**, a self-calibrating threshold instead of a
hand-picked `k`:

1. **Statistical** ([src/lib/stats.ts](src/lib/stats.ts)): robust seasonal-naive week-over-week z-score. Explainable, no dependencies, always on.
2. **Donut-style VAE** ([ml-service/](ml-service/)): a variational autoencoder for seasonal KPIs (tight bottleneck plus missing-data-injection denoising), the better-matched deep model for this kind of signal.

**EVT/POT thresholding** ([src/lib/evt.ts](src/lib/evt.ts)) fits a Generalized Pareto
Distribution to the *tail* of the score distribution and derives the cutoff for a target
false-alarm rate `q`. A noisy metric gets a higher bar and a clean one a lower bar, all
from one interpretable risk knob (Siffer et al., KDD 2017).

The LangGraph `anomaly_detector` node calls the Python service for the deep model,
combines both detectors, and tags each anomaly with the detectors that agreed plus a
**confidence = fraction of detectors in agreement**. The dashboard shows a `conf N% · vae`
badge.

### Benchmark: `npm run bench:detectors`

A controlled experiment, **averaged over 12 independent seeds** (mean ± std), on
synthetic series with **known injected anomalies** (labels by construction). To keep the
numbers honest, the generator is made harder than the detectors' own assumptions:
**AR(1)-autocorrelated, Laplace (heavy-tailed) noise**, about 7% contamination, and a mix
of **point, short-collective, and 5-day level-shift (regime)** anomalies. Scored with the
**TSB-AD threshold-free metrics**, **VUS-PR** and **AUC-PR** (robust to temporal offset;
the point-adjusted F1 that flattered older leaderboards was [shown unreliable at NeurIPS
2024](https://openreview.net/forum?id=R6kJtWsTGy)), plus point-F1 for contrast:

| Detector | VUS-PR | AUC-PR | point-F1 |
|---|---|---|---|
| statistical (EVT) | 0.67 ± 0.02 | **1.00 ± 0.00** | 0.36 ± 0.06 |
| donut_vae | 0.50 ± 0.05 | 0.53 ± 0.11 | 0.42 ± 0.10 |
| **ensemble (mean score)** | **0.68 ± 0.02** | **1.00 ± 0.00** | n/a |
| └ union (≥1 vote) | n/a | n/a | **0.52 ± 0.09** |
| └ intersect (≥2 votes) | n/a | n/a | 0.22 ± 0.13 |

The combined ensemble score ranks anomalies almost perfectly (AUC-PR about 1.0) and is
the steadiest ranker (tightest std). The single statistical detector is precision-first
but misses on recall; the **union** op recovers it (best F1) while the **intersection**
stays precision-tight. That's the quantified case for combining statistics with deep
learning, and for picking the ensemble op by the cost of a miss versus a false alarm.
Note the VAE's AUC-PR drops to about 0.53 under the harder noise (versus about 0.72 on a
single easy seed), which is exactly the optimism the multi-seed, harder-noise setup is
meant to expose.

> The deep detector mirrors a production multi-task anomaly model deployed on IoT edge
> devices: an unsupervised autoencoder behind its own inference API, used by the rest of
> the system over the network. The metric and threshold choices (VUS-PR, EVT, Donut-VAE)
> track the 2024-25 time-series anomaly-detection state of the art.

---

## The evaluation harness

The piece most GenAI demos skip. `npm run eval` runs a golden set of questions through
the live agent graph and scores each on:

- **Routing accuracy**: did the orchestrator resolve the right metric and scope?
- **SQL validity & execution**: does the generated SQL pass the read-only guard and return rows?
- **Anomaly recall**: were the known injected anomalies detected?
- **Root-cause attribution**: was the correct deploy implicated?
- **Hallucination rate**: did the narrative invent entities outside the known vocab?
- **Latency** (avg + p95) and **token cost**.

Results are written to `data/eval-report.json` and shown on the dashboard.

Latest local run (live Claude): **100% pass, 100% routing, 100% SQL validity, 100%
anomaly recall, 0% hallucination.**

---

## Deployment

Prima runs on **Fly.io** as two small apps in the same org:

- **`prima-web`**: the Next.js app and agent fleet. It keeps the SQLite warehouse on a 1 GB volume, so the data survives restarts.
- **`prima-ml`**: the FastAPI + PyTorch service. The web app reaches it over Fly's private network at `prima-ml.internal:8000`. If it's down, the web app drops back to the statistical detector and keeps working.

Both deploy to the Toronto region (`yyz`). The only required secret is
`ANTHROPIC_API_KEY`, set with `fly secrets set`.

```bash
make deploy        # build and ship both apps
make logs          # tail web logs
make status        # show both apps
```

Full one-time setup, plus a GitHub Actions workflow that ships on every push to `main`,
is in [DEPLOY.md](DEPLOY.md).

> The first request after a fresh deploy takes a bit longer while the app downloads the
> dataset and builds `prima.db` on the volume once. Every request after that is fast.

Production builds use the webpack builder (set in the `build` script). The `standalone`
output it produces is what the Docker image runs.

---

## Project layout

```
ml-service/          Python FastAPI + PyTorch microservice
  model.py           Donut-style VAE
  detector.py        train + recon-error scoring + EVT/POT threshold (cached)
  forecaster.py      zero-shot Chronos-Bolt foundation-model forecasting
  app.py             /detect, /forecast, /health
  run.sh             `npm run ml`: creates the uv venv and starts uvicorn
data/
  seed.ts            synthetic telemetry generator (deterministic PRNG)
src/
  lib/
    db.ts            SQLite connection + read-only SQL guard
    stats.ts         seasonal-naive scoring, EVT-thresholded detector, Holt-Winters
    evt.ts           EVT/POT (SPOT) self-calibrating threshold
    mlClient.ts      HTTP client for the PyTorch detectors (graceful if offline)
    llm.ts           Claude provider (Anthropic SDK) + cost model
  agents/
    state.ts         LangGraph shared-state annotation
    nodes.ts         the six agent nodes (anomaly_detector = 2-detector ensemble)
    graph.ts         StateGraph wiring + runPrima() entrypoint
    util.ts          intent parsing + SQL builder + schema doc
  eval/
    golden.ts        golden dataset
    evaluate.ts      scoring engine
    runEval.ts       CLI scorecard
    metrics.ts       VUS-PR / AUC-PR / point-F1
    benchmarkDetectors.ts   statistical vs VAE vs ensemble
  app/
    api/             /analyze, /feature-usage, /eval route handlers
    page.tsx         observability dashboard (Dashboard + Architecture tabs)
    components/      Recharts metric + feature-usage charts, Architecture diagrams
```

---

## Notes & honest limitations

- The SQLite warehouse stands in for BigQuery to keep the demo free and offline. The SQL surface (CTEs, window functions, `COUNT DISTINCT`) maps directly.
- Forecasting uses the zero-shot **Chronos-Bolt** foundation model when the ML service is up, with Holt-Winters as a transparent fallback that has no dependencies. The choice is deliberate: foundation models are strong for zero-shot and cold-start, while the `forecaster` node stays a clean seam (you can swap in Prophet, NeuralProphet, or Vertex just as easily).
- Clean separation of concerns: the SQL authoring and the narrative come from a live Claude model, while the numeric results (anomalies, forecast, segment attribution) are computed in plain code, so the figures never depend on the LLM.
