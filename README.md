# Prima: Agentic DAU/WAU Anomaly & Forecasting Platform

A **LangGraph** agent fleet that does the analysis a static BI dashboard can't. Prima
watches product engagement (Daily and Weekly Active Users), writes and runs its own SQL,
finds anomalies, forecasts where the metric is heading, works out *why* it moved, and
writes you a short brief in plain English. It comes with an observability layer and an
LLM evaluation scorecard.

It runs on **Node 24 + Next.js 16 (App Router) + TypeScript**. The agent graph is built
with **LangGraph.js**, **Claude** is the reasoning engine, and a **Python / FastAPI +
PyTorch** service provides zero-shot **Chronos-Bolt** forecasting and a benchmarked
**Donut VAE** anomaly detector.

> **Live Claude only.** The SQL-Analyst, Root-Cause, and Narrator agents call a real
> Claude model, so you need an `ANTHROPIC_API_KEY` (`cp .env.example .env`). The
> statistical engine (anomaly detection and forecasting) is plain deterministic code.

---

## What this shows

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
| **Observability dashboards** (feature usage, perf, agent health) | Next.js dashboard with live agent trace + **dataset picker** ([src/app/page.tsx](src/app/page.tsx)) |
| **Statistical modeling for predictive analytics** (preferred) | Seasonal-trend decomposition, robust MAD z-scores, Holt-Winters ([src/lib/stats.ts](src/lib/stats.ts)) |
| **Deep learning** (PyTorch) | Faithful **Donut** VAE (WWW'18) in a FastAPI microservice — benchmarked against the statistical detector on synthetic + real AIOps KPIs ([ml-service/README.md](ml-service/README.md)) |
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
3. **Anomaly Detector** runs a statistical **seasonal-naive week-over-week** z-score (robust median/MAD) with a self-calibrating **EVT/POT (SPOT)** threshold. Benchmarks showed the Donut VAE ensemble adds nothing for Prima's single clean daily KPI — and the "agreement" vote actively hurt — so the agent is **statistical-only**; the deep model is kept as a benchmarked reference. See [Deep-learning detector](#deep-learning-detector-pytorch-benchmarked).
4. **Forecaster** uses zero-shot **Chronos-Bolt** (a pretrained time-series foundation model from Amazon) to forecast 14 days with probabilistic quantile bands, with *no training on our data*. It falls back to a hand-written Holt-Winters model when the ML service is offline. See [ml-service/forecaster.py](ml-service/forecaster.py).
5. **Root-Cause** lines up the worst drop against the `deploys` table and the worst-hit `region / platform` segments (week over week), then forms hypotheses grounded in that evidence.
6. **Narrator** writes a short executive briefing with a recommendation.

Every node also writes an **observability trace** entry (status, latency, mode, detail)
and adds up token/cost telemetry through LangGraph state reducers.

The dashboard has two tabs: the live analysis view, and an **Architecture** tab that
diagrams the LangGraph runtime and how a request flows through the model. It's there to
walk someone through the internals.

A **dataset picker** in the header switches the warehouse the agents run against, with
the selection threaded per-request all the way to the SQL layer ([src/lib/datasetContext.ts](src/lib/datasetContext.ts)):

| Dataset | Kind | What it is |
| --- | --- | --- |
| **Online Retail (real)** | agent | The auto-bootstrapped UCI dataset — genuine e-commerce DAU/WAU (default). |
| **Synthetic storyline** | agent | Seeded telemetry with a known EU/mobile deploy incident — best for showing root-cause attribution. |
| **AIOps KPI (real)** | viewer | 2018 AIOps Challenge web-service KPIs with ground-truth anomaly labels. Different shape (per-minute value/label, no users/deploys), so it's a **read-only viewer**, not analyzed by the agent. |

The two *agent* datasets share the `user_activity` / `deploys` schema, so the full
LangGraph fleet runs against either; each warehouse is built on demand the first time
it's selected. The registry lives in [src/lib/datasets.ts](src/lib/datasets.ts).

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

> **Optional: ML service.** Start the PyTorch service in a second terminal
> (`npm run ml`) to enable zero-shot **Chronos-Bolt** forecasting — the `forecaster`
> falls back to Holt-Winters when it's offline. The service also hosts the benchmarked
> **Donut** detector (`npm run bench:detectors`); it is not in the live anomaly path.

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

## Deep-learning detector (PyTorch, benchmarked)

The Python service hosts a **faithful port of Donut** (Xu et al., WWW'18) — a
variational autoencoder for seasonal KPIs with the paper's three techniques: a
**Gaussian decoder**, **M-ELBO + missing-data injection**, and **MCMC imputation**.
Both detectors are thresholded with **EVT/POT (SPOT)** ([src/lib/evt.ts](src/lib/evt.ts)),
which fits a Generalized Pareto Distribution to the score *tail* for a target
false-alarm rate `q` — one interpretable risk knob instead of a hand-picked `k`
(Siffer et al., KDD 2017).

**It is not in the live agent path** — two benchmarks make the case for why:

- **Synthetic Prima** (one clean daily KPI, `npm run bench:detectors`): the statistical
  seasonal z-score matches or beats Donut (AUC-PR 1.00 vs 1.00 on large anomalies; 0.95
  vs 0.84 on subtle). The ensemble adds nothing, and the "agreement = confirmed" vote
  was the *worst* performer. So the agent runs **statistical-only**.
- **Real AIOps KPIs** (26 heterogeneous web-service KPIs, `bench_aiops.py`): here Donut
  *wins* — point-adjusted best-F1 **0.91 vs 0.54**, landing in the paper's reported
  0.75–0.90 range. You can't hand-tune one seasonal period across 26 KPIs of different
  sampling rates and shapes; Donut learns each.

The deep model earns its keep on **many heterogeneous, high-frequency KPIs**, not on
Prima's single clean daily series. The full tables, dataset plots, window sweep, and the
metric caveats (why point-adjusted best-F1 ≫ strict AUC-PR) live in
**[ml-service/README.md](ml-service/README.md)**.

> The deep detector mirrors a production unsupervised anomaly model behind its own
> inference API. The metric and threshold choices (AUC-PR/VUS-PR, EVT, Donut) track the
> 2024-25 time-series anomaly-detection state of the art.

---

## The evaluation harness

`npm run eval` runs a golden set of questions through the live agent graph and scores
each on:

- **Routing accuracy**: did the orchestrator resolve the right metric and scope?
- **SQL validity & execution**: does the generated SQL pass the read-only guard and return rows?
- **Anomaly recall**: were the known injected anomalies detected?
- **Root-cause attribution**: was the correct deploy implicated?
- **Hallucination rate**: did the narrative invent entities outside the known vocab?
- **Latency** (avg + p95) and **token cost**.

Results are written to `data/eval-report.json` and shown on the dashboard.

Latest local run (live Claude, 5-case golden set): **100% pass, 100% routing, 100% SQL
validity, 100% anomaly recall, 0% hallucination.** The set is small and meant to grow —
it covers the core routing/SQL/anomaly/root-cause paths rather than exhaustive coverage.

---

## Deployment

Prima runs on **Fly.io** as two small apps in the same org:

- **`prima-web`**: the Next.js app and agent fleet. It keeps the SQLite warehouse on a 1 GB volume, so the data survives restarts.
- **`prima-ml`**: the FastAPI + PyTorch service (Chronos-Bolt forecaster + Donut benchmark). The web app reaches it over Fly's private network at `prima-ml.internal:8000`. If it's down, the `forecaster` drops back to Holt-Winters and the app keeps working (anomaly detection is statistical-only and doesn't depend on it).

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
  model.py           Donut VAE (faithful WWW'18 port: Gaussian decoder)
  detector.py        M-ELBO train + reconstruction-probability scoring + EVT/POT (cached)
  forecaster.py      zero-shot Chronos-Bolt foundation-model forecasting
  app.py             /detect, /forecast, /health
  bench_aiops.py     z-score vs Donut on the real AIOps KPI dataset
  plot_datasets.py   renders the dataset figures in ml-service/README
  run.sh             `npm run ml`: creates the uv venv and starts uvicorn
data/
  seed.ts            synthetic telemetry generator (deterministic PRNG)
src/
  lib/
    db.ts            SQLite connection + read-only SQL guard
    datasets.ts      dataset registry (real / synthetic / aiops) + per-dataset warehouse
    datasetContext.ts  per-request DB selection (AsyncLocalStorage) for the picker
    stats.ts         seasonal-naive scoring, EVT-thresholded detector, Holt-Winters
    evt.ts           EVT/POT (SPOT) self-calibrating threshold
    mlClient.ts      HTTP client for the PyTorch detectors (graceful if offline)
    llm.ts           Claude provider (Anthropic SDK) + cost model + concurrency gate
    rateLimit.ts     per-caller quota (token bucket) + global LLM concurrency semaphore
  agents/
    state.ts         LangGraph shared-state annotation
    nodes.ts         the six agent nodes (anomaly_detector = seasonal z-score / EVT)
    graph.ts         StateGraph wiring + runPrima() entrypoint
    util.ts          intent parsing + SQL builder + schema doc
  eval/
    golden.ts        golden dataset
    evaluate.ts      scoring engine
    runEval.ts       CLI scorecard
    metrics.ts       VUS-PR / AUC-PR / point-F1
    benchmarkDetectors.ts   statistical vs VAE vs ensemble
  app/
    api/             /analyze, /feature-usage, /eval, /datasets, /kpi route handlers
    page.tsx         observability dashboard (Dashboard + Architecture tabs, dataset picker)
    components/      Recharts metric + feature-usage charts, KpiViewer, Architecture diagrams
```

---

## Notes & honest limitations

- The SQLite warehouse stands in for BigQuery to keep the project free and offline. The SQL surface (CTEs, window functions, `COUNT DISTINCT`) maps directly.
- Forecasting uses the zero-shot **Chronos-Bolt** foundation model when the ML service is up, with Holt-Winters as a transparent fallback that has no dependencies. The choice is deliberate: foundation models are strong for zero-shot and cold-start, while the `forecaster` node stays a clean seam (you can swap in Prophet, NeuralProphet, or Vertex just as easily).
- Clean separation of concerns: the SQL authoring and the narrative come from a live Claude model, while the numeric results (anomalies, forecast, segment attribution) are computed in plain code, so the figures never depend on the LLM.
