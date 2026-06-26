# Prima: Agentic Server-Health Anomaly & Root-Cause Platform

A **LangGraph** agent fleet that does the analysis a static monitoring dashboard can't.
Prima watches real server telemetry, writes and runs its own SQL, finds anomalies in a
machine's health, forecasts where it's heading, works out *which metric channels caused*
each anomaly — and **grades that root-cause attribution against ground truth** — then
writes a short reliability brief in plain English. It comes with an observability layer
and an LLM evaluation scorecard.

It runs on **Node 24 + Next.js 16 (App Router) + TypeScript**. The agent graph is built
with **LangGraph.js**, **Claude** is the reasoning engine, and a **Python / FastAPI +
PyTorch** service provides zero-shot **Chronos-Bolt** forecasting and a benchmarked
**OmniAnomaly** (KDD'19) multivariate anomaly detector.

> **Live Claude only.** The SQL-Analyst, Root-Cause, and Narrator agents call a real
> Claude model, so you need an `ANTHROPIC_API_KEY` (`cp .env.example .env`). The
> statistical engine (anomaly detection and forecasting) is plain deterministic code.

---

## The data: real, labeled server telemetry (SMD)

Prima runs on the **Server Machine Dataset (SMD)** from the OmniAnomaly project (Su et
al., KDD'19) — real telemetry from a large internet company:

| | SMD |
| --- | --- |
| Entities | 28 server machines (Prima loads 8 by default; `SMD_ENTITIES=all` for every one) |
| Channels | 38 metric channels per machine, normalized `[0,1]` |
| Labels | per-timestep `test_label` (real anomalies — no synthetic injection) |
| **Root-cause ground truth** | `interpretation_label`: the exact channels responsible for each anomaly segment |

That last row is the point. A clean KPI series tells you *when* something broke; SMD's
`interpretation_label` tells you *which metric* broke — so the root-cause agent becomes a
**gradeable** task, not a plausible-sounding story.

**Get the data** (place under `data/smd/`):

```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/NetManAIOps/OmniAnomaly.git /tmp/omni \
  && (cd /tmp/omni && git sparse-checkout set ServerMachineDataset) \
  && mv /tmp/omni/ServerMachineDataset data/smd && rm -rf /tmp/omni
```

The warehouse (`data/prima.db`) is then built from `data/smd` on the first request — no
manual seed step.

---

## What this shows

A full, end-to-end enterprise analytical-agent ecosystem, and where each capability lives:

| Capability | Where it lives in this repo |
| --- | --- |
| Autonomous analytical agents using **LangGraph** | [src/agents/graph.ts](src/agents/graph.ts), [src/agents/nodes.ts](src/agents/nodes.ts) |
| Agents that **autonomously leverage SQL** | `sql_analyst` generates, **guards**, and executes the health query ([src/agents/nodes.ts](src/agents/nodes.ts), [src/lib/db.ts](src/lib/db.ts)) |
| Move beyond chat to **proactive, prescriptive** actions | `root_cause` + `narrator` emit diagnosis + recommendations ([src/agents/nodes.ts](src/agents/nodes.ts)) |
| **Root-cause attribution graded against ground truth** | `root_cause` ranks channels by deviation and scores precision/recall vs `interpretation_label` ([src/agents/nodes.ts](src/agents/nodes.ts)) |
| **Predictive** analytics | Zero-shot **Chronos-Bolt** forecasting (Holt-Winters fallback) ([ml-service/forecaster.py](ml-service/forecaster.py)) |
| **LLM performance evaluation** | Golden-set eval on the real SMD warehouse: routing, SQL validity, anomaly recall, root-cause grading, hallucination, latency, cost ([src/eval/](src/eval/)) |
| **Detector benchmarking** (the benchmark is the justification) | The app's actual detection path scored on real labels ([src/eval/benchmarkSmdDetector.ts](src/eval/benchmarkSmdDetector.ts)); multivariate OmniAnomaly vs z-score ([ml-service/bench_smd.py](ml-service/bench_smd.py)) |
| **Statistical modeling** | Robust median/MAD z-scores, EVT/POT thresholding, Holt-Winters ([src/lib/stats.ts](src/lib/stats.ts), [src/lib/evt.ts](src/lib/evt.ts)) |
| **Deep learning** (PyTorch) | **OmniAnomaly** stochastic-RNN VAE (KDD'19: GRU + planar flows + linear-Gaussian latent transition, multivariate) in a FastAPI microservice ([ml-service/README.md](ml-service/README.md)) |
| **BigQuery-style SQL** | SQLite warehouse with the same SQL surface (CTEs, window fns, `COUNT DISTINCT`) |

---

## Architecture

```
                       ┌─────────────────────────────────────────────┐
 "Worst anomaly on     │              LangGraph state graph            │
  machine-1-1 and   ─► │                                               │
  its root cause?"     │  orchestrator → sql_analyst → anomaly_detector│
                       │        → forecaster → root_cause → narrator   │
                       └───────┬───────────────┬──────────────┬────────┘
                               │               │              │
                        ┌──────▼─────┐  ┌───────▼──────┐ ┌─────▼──────┐
                        │  SQLite     │  │  stats.ts    │ │   Claude    │
                        │ health /    │  │ EVT detector │ │             │
                        │ metrics /   │  │ HW forecast  │ │             │
                        │ interp      │  │              │ │             │
                        └─────────────┘  └──────────────┘ └─────────────┘
                               │
                       ┌───────▼──────────────────────────────────────┐
                       │   Next.js dashboard + /api + eval scorecard   │
                       └───────────────────────────────────────────────┘
```

The warehouse holds three tables built from SMD ([src/lib/smdWarehouse.ts](src/lib/smdWarehouse.ts)):

- **`health(entity, t, value, label)`** — the 1-D series the agents analyze. `value` is
  the entity **health score** = per-timestep **max robust z-score across all 38
  channels** (causal rolling-median / MAD). `label` is the ground-truth anomaly flag.
- **`metrics(entity, t, m01…m38)`** — the raw channels, used for root-cause attribution.
- **`interp(entity, seg_start, seg_end, channel)`** — the ground-truth culprit channels.

### The six agents

1. **Orchestrator** parses the question into the server **entity** to analyze (e.g. `machine-1-1`).
2. **SQL Analyst** writes a read-only query for that entity's health series, **guards** it (single SELECT/WITH, no DDL/DML), executes it, and falls back to a validated template if generation fails.
3. **Anomaly Detector** thresholds the health-score tail with a self-calibrating **EVT/POT (SPOT)** rule ([src/lib/stats.ts](src/lib/stats.ts) `detectScoreAnomalies`). No seasonal baseline — server metrics aren't weekly.
4. **Forecaster** uses zero-shot **Chronos-Bolt** (Amazon's pretrained time-series foundation model) for a 14-step projection with quantile bands, falling back to Holt-Winters when the ML service is offline.
5. **Root-Cause** takes the worst anomaly, ranks the 38 channels by their robust deviation at that timestep, looks up the ground-truth culprit channels from `interp`, and **scores its attribution (precision / recall)** — then asks Claude for evidence-grounded hypotheses about the likely fault.
6. **Narrator** writes a short reliability briefing with a recommendation.

Every node writes an **observability trace** entry (status, latency, mode, detail) and
accumulates token/cost telemetry through LangGraph state reducers. The dashboard has a
live analysis view and an **Architecture** tab that diagrams the runtime.

A **dataset picker** threads the selection per-request to the data layer
([src/lib/datasetContext.ts](src/lib/datasetContext.ts)):

| Dataset | Kind | What it is |
| --- | --- | --- |
| **Server Machine Dataset (real)** | agent | The SMD warehouse — the full LangGraph fleet runs against it (default). |
| **AIOps KPI (real)** | viewer | 2018 AIOps Challenge web-service KPIs with ground-truth labels. A univariate KPI dataset — **read-only viewer**, not analyzed by the agent. |

The registry lives in [src/lib/datasets.ts](src/lib/datasets.ts).

---

## Quickstart

```bash
npm install
# fetch SMD into data/smd (see "The data" above), then:
npm run dev      # open http://localhost:3000
```

On the first request the app builds `data/prima.db` from `data/smd` (a few seconds, then
cached). Requires `ANTHROPIC_API_KEY` — `cp .env.example .env` and add your key (the
agents call Claude on every request).

> Port 3000 busy? `PORT=3100 npm run dev`.

> **Optional: ML service.** Start the PyTorch service in a second terminal
> (`npm run ml`) to enable zero-shot **Chronos-Bolt** forecasting — the `forecaster`
> falls back to Holt-Winters when it's offline.

### Commands

```bash
npm run bench:smd        # benchmark Prima's ACTUAL detection path on real SMD labels
npm run eval             # LLM evaluation scorecard on the SMD warehouse → data/eval-report.json
npm run bench:detectors  # statistical vs OmniAnomaly vs ensemble on synthetic series
```

---

## Is the detection any good? (the benchmark is the justification)

There are two distinct benchmarks, because the app and the deep model do different things.

### 1. The app's actual path — `npm run bench:smd`

The agent doesn't run OmniAnomaly. It analyzes one 1-D series (the health score)
and thresholds it with EVT/POT. [src/eval/benchmarkSmdDetector.ts](src/eval/benchmarkSmdDetector.ts)
benchmarks **that exact path** against `test_label`, reporting strict, lenient, *and*
deployed metrics (per mle-practices — never report only the flattering number):

| metric | macro (8 entities) | what it means |
| --- | --- | --- |
| **AUC-PR** | 0.17 | strict, point-wise, threshold-free — several× the 2–15% prevalence floor, but the raw score ranks loosely |
| **best-F1\*** | 0.84 | point-adjusted best-F1 under an *oracle* threshold — the score separates anomalies well |
| **EVT-F1\*** | 0.62 | point-adjusted F1 at the threshold the app **actually** picks |

**Honest finding:** the gap between the oracle (0.84) and the deployed EVT threshold
(0.62) is real — the self-calibrating SPOT threshold over-flags on 2 of 8 entities
(`machine-3-11`, `machine-1-4`: precision 0.02–0.11, recall 1.0). The health score is a
good *signal*; the per-entity threshold is the weak link, and that's where future work
should go.

### 2. The deep detector — `ml-service/bench_smd.py`

The deep model is **OmniAnomaly** (Su et al., KDD'19) — a *stochastic-RNN VAE* with its
signature mechanisms intact (GRU inference/generation, a linear-Gaussian latent
transition, planar normalizing flows, a Gaussian reconstruction-probability score).
**One OmniAnomaly model sees all 38 SMD channels at once**, so it scores *correlations
breaking* — which is what SMD's multivariate anomalies actually are, and what a
per-channel detector is blind to. [ml-service/bench_smd.py](ml-service/bench_smd.py)
trains it on SMD's clean `train` split, scores `test`, and grades against `test_label`
vs a causal rolling-median z-score baseline, thresholding both with **EVT/POT (SPOT)**
([src/lib/evt.ts](src/lib/evt.ts); Siffer et al., KDD 2017).

It's a benchmark, not the live path — but it's a decisive one in its design regime. On a
capped 3-entity validation run, OmniAnomaly nearly triples the baseline's strict
point-wise **AUC-PR (0.33 vs 0.12)** and edges its lenient point-adjusted best-F1 (0.97
vs 0.89), landing in OmniAnomaly's published SMD range. The flip side, from the synthetic
benchmark (`npm run bench:detectors`): on Prima's *one clean univariate daily metric*
OmniAnomaly is **out of its design regime** and the z-score wins outright on subtle
anomalies (AUC-PR 0.95 vs 0.24) — right tool, right regime. Full tables, plots, and metric
caveats (why point-adjusted best-F1 ≫ strict AUC-PR) live in
**[ml-service/README.md](ml-service/README.md)**.

---

## The evaluation harness

`npm run eval` runs a golden set of questions through the live agent graph against an
isolated copy of the **real SMD warehouse** (`data/eval-smd.db`), scoring each on:

- **Routing accuracy**: did the orchestrator resolve the right entity?
- **SQL validity & execution**: does the generated SQL pass the read-only guard and return rows?
- **Anomaly recall**: were anomalies detected on entities that have them?
- **Root-cause attribution**: did the worst anomaly land in a labeled segment, so its channel attribution is graded against `interpretation_label`?
- **Hallucination rate**: did the narrative invent entity ids outside the warehouse vocab?
- **Latency** (avg + p95) and **token cost**.

Because SMD ships real labels, the eval asserts against genuine ground truth (no synthetic
fixture). Results are written to `data/eval-report.json` and shown on the dashboard. Run
`npm run eval` to generate the scorecard ([src/eval/golden.ts](src/eval/golden.ts),
[src/eval/evaluate.ts](src/eval/evaluate.ts)).

---

## Deployment

Prima runs on **Fly.io** as two small apps in the same org:

- **`prima-web`**: the Next.js app and agent fleet. It keeps the SQLite warehouse on a 1 GB volume, so the data survives restarts.
- **`prima-ml`**: the FastAPI + PyTorch service (Chronos-Bolt forecaster + OmniAnomaly benchmark), reached over Fly's private network at `prima-ml.internal:8000`. If it's down, the `forecaster` drops back to Holt-Winters (anomaly detection is statistical-only and doesn't depend on it).

The only required secret is `ANTHROPIC_API_KEY`.

```bash
make deploy        # build and ship both apps
make logs          # tail web logs
make status        # show both apps
```

Full setup + a GitHub Actions workflow that ships on every push to `main` is in
[DEPLOY.md](DEPLOY.md). Production builds use the webpack builder (`build` script); the
`standalone` output it produces is what the Docker image runs.

---

## Project layout

```
ml-service/          Python FastAPI + PyTorch microservice
  model.py           OmniAnomaly (KDD'19 port: GRU stochastic-RNN VAE + planar flows, multivariate)
  detector.py        ELBO train + reconstruction-probability scoring + EVT/POT (cached, univariate /detect)
  metrics.py         AUC-PR + point-adjusted best-F1 + deployed-threshold F1 (bench metrics)
  forecaster.py      zero-shot Chronos-Bolt foundation-model forecasting
  app.py             /detect, /forecast, /health
  bench_smd.py       multivariate OmniAnomaly vs rolling-median z on real SMD (train/test split)
  run.sh             `npm run ml`: creates the uv venv and starts uvicorn
data/
  smd/               Server Machine Dataset (train / test / test_label / interpretation_label)
  aiops-kpi/         2018 AIOps Challenge KPIs (univariate read-only viewer dataset)
src/
  lib/
    smdWarehouse.ts  builds health / metrics / interp from data/smd; computeHealthSeries
    db.ts            SQLite connection + read-only SQL guard + entity vocab
    datasets.ts      dataset registry (smd / aiops)
    datasetContext.ts  per-request DB selection (AsyncLocalStorage) for the picker
    stats.ts         robust scoring, EVT-thresholded score detector, Holt-Winters
    evt.ts           EVT/POT (SPOT) self-calibrating threshold
    mlClient.ts      HTTP client for the PyTorch detectors (graceful if offline)
    llm.ts           Claude provider (Anthropic SDK) + cost model + concurrency gate
    rateLimit.ts     per-caller quota (token bucket) + global LLM concurrency semaphore
  agents/
    state.ts         LangGraph shared-state annotation
    nodes.ts         the six agent nodes (root_cause = channel attribution + grading)
    graph.ts         StateGraph wiring + runPrima() entrypoint
    util.ts          entity intent parsing + health-query builder + schema doc
  eval/
    golden.ts        golden dataset (SMD entities)
    evaluate.ts      scoring engine (routing / SQL / recall / root-cause grade / hallucination)
    runEval.ts       CLI scorecard (builds the SMD eval warehouse)
    benchmarkSmdDetector.ts  Prima's actual detection path scored on real SMD labels
    benchmarkDetectors.ts    statistical vs OmniAnomaly vs ensemble (synthetic)
  app/
    api/             /analyze, /feature-usage, /eval, /datasets, /kpi route handlers
    page.tsx         observability dashboard (Dashboard + Architecture tabs, dataset picker)
    components/      Recharts metric chart, channel-frequency chart, KpiViewer, Architecture
```

---

## Notes & honest limitations

- The SQLite warehouse stands in for BigQuery to keep the project free and offline. The SQL surface (CTEs, window functions, `COUNT DISTINCT`) maps directly.
- **The EVT threshold is the weak link**, not the signal. The health score earns a 0.84 oracle point-adjusted F1, but the self-calibrating SPOT threshold over-flags on a couple of entities (see the benchmark above). Per-entity threshold calibration is the obvious next step.
- SMD ships **no human-readable metric names** — channels are `metric_01…38`. Root-cause attribution is therefore at the channel-index level (and gradeable against `interpretation_label`); turning that into subsystem-level narratives ("disk", "NIC") would need a metric taxonomy SMD doesn't provide.
- Clean separation of concerns: SQL authoring and the narrative come from a live Claude model, while the numeric results (health score, anomalies, forecast, channel attribution) are computed in plain, benchmarked code — the figures never depend on the LLM.
```
