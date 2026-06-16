# Prima ML Service — Donut VAE Anomaly Detector

A small standalone service (**Python / FastAPI + PyTorch**) that finds anomalies in a
time series. It uses a deep model called a **Donut VAE** that learns what "normal"
looks like, then flags any point it can't reconstruct well. The line between "normal"
and "anomaly" is set automatically by **EVT/POT (SPOT)** — a method that calibrates the
cutoff from the data instead of using a number you pick by hand. No labels needed.

> **Quick status:** Prima's live agent uses the **simple statistical detector only**.
> The benchmarks below show the Donut model adds nothing for Prima's one clean daily
> metric, so it (and the old "agreement vote") were removed from the agent. This service
> stays around as a **benchmarked reference** — and it still runs the Chronos forecaster.
> Donut earns its place on *many, fast-changing, varied* metrics — see Benchmark B.

## How it works (the model)

This is a faithful PyTorch rebuild of **Donut** (Xu et al., *Unsupervised Anomaly
Detection via Variational Auto-Encoder for Seasonal KPIs*, WWW'18), following the
authors' code: <https://github.com/NetManAIOps/donut>. Three things make Donut more
than a plain autoencoder, and all three are here:

1. **The decoder predicts a mean *and* a spread.** Because it outputs `p(x|z)` with both
   a mean and a standard deviation, the anomaly score is a real *probability*, not just
   a raw error. ([`model.py`](model.py))
2. **M-ELBO + missing-data injection.** During training we hide a few random points each
   step and tell the loss to ignore them (`α = 1 − y`, prior scaled by `β = mean(α)`).
   This stops anomalies in the training data from poisoning the model.
   ([`detector.py`](detector.py), `_train`)
3. **MCMC fill-in before scoring** (`iterative_masked_reconstruct`). Like the original,
   this only runs when you actually mark points as missing; the normal detection path
   (no missing points) skips it.

The final anomaly score is the **negative** average log-probability over `n_z` samples,
taken at the last point of each window (higher = more anomalous). That keeps both
threshold methods (EVT/POT and MAD) working the same way.

## Run

```bash
npm run ml          # from repo root — sets up the venv on first run
# or directly:
cd ml-service && ./run.sh
```

Listens on `http://127.0.0.1:8000` (change with `ML_PORT`).

## API

`POST /detect`
```jsonc
{
  "series": [{"date": "2011-01-01", "value": 63}, ...],
  "model": "vae",       // "vae"
  "threshold": "evt",   // "evt" (SPOT/POT, self-calibrating) | "mad" (median ± k·MAD)
  "window": 28,         // how many points the model looks at together (~4 weekly cycles)
  "epochs": 150,        // training epochs (the model is cached per series)
  "q": 0.02,            // EVT target false-alarm rate (use "k" instead for mad)
  "n_z": 256,           // samples used for the reconstruction probability (Donut uses 1024)
  "mcmc_iter": 10       // fill-in steps (only runs when data is marked missing)
}
```
Returns: per-point scores (negative reconstruction log-prob), the list of anomalies
(each with a severity and a robust z-score), and training info (`train_ms`,
`final_loss` = the converged −ELBO).

`POST /forecast`
```jsonc
{ "series": [{"date":"2011-01-01","value":63}, ...], "horizon": 14 }
```
A zero-shot forecast from **Chronos-Bolt** (`amazon/chronos-bolt-small`, a pretrained
time-series model). Returns `median` plus `lower`/`upper` (p10/p90) — it does **not**
train on your series. The ~48MB model downloads from HuggingFace the first time and is
then cached.

`GET /health` → lists the models, threshold options, and the forecaster.

## Design notes

- **Localized scoring** — each point is scored using the window that *ends* on it, so one
  anomaly's signal doesn't smear onto its neighbors.
- **Robust threshold** — the cutoff uses median/MAD (the same robust statistic the Node
  side uses), so a few big anomalies can't drag the cutoff up around themselves.
- **Cached training** — the trained model is saved by a hash of the input series, so the
  same request comes back instantly.
- **Reproducible** — `torch.manual_seed(42)`.

This mirrors a common production setup: an unsupervised anomaly model served behind its
own API, used by the rest of the system over the network.

## Benchmark A — synthetic Prima series (do we even need the ensemble?)

![Synthetic Prima benchmark data: one clean daily series; large anomalies pop out, subtle ones blend into the weekly season](docs/synthetic.png)

`npm run bench:detectors` runs two settings (12 seeds each, AR(1)/Laplace noise, with
point + collective + level-shift anomalies). AUC-PR/VUS-PR are averaged per seed
(mean±std); precision/recall/F1 are **pooled across seeds** so they rest on 150–190 real
anomaly points, not the ~13 a single seed would give.

Both tables use the default **window = 28** (~4 weekly cycles — see the sweep below).
Note: for AUC-PR, the "no skill" floor is the share of points that are actually anomalies
(~0.08 here), not 0.5.

**Large anomalies** — 40–80% deviations (156 pooled positives):

| detector | AUC-PR | P / R / F1 |
|---|---|---|
| statistical (EVT) | **1.000 ± 0.000** | 1.00 / 0.22 / 0.36 |
| donut_vae | **1.000 ± 0.000** | 1.00 / 0.26 / 0.41 |
| ensemble — mean score | 1.000 ± 0.000 | — |
| ensemble — union (≥1) | — | 1.00 / 0.29 / **0.46** |
| ensemble — intersect (≥2) | — | 1.00 / 0.18 / 0.30 |

**Subtle anomalies** — 10–20% deviations, about the size of the normal weekly swing
(192 pooled positives):

| detector | AUC-PR | P / R / F1 |
|---|---|---|
| statistical (EVT) | **0.949 ± 0.033** | 1.00 / 0.18 / 0.30 |
| donut_vae | 0.836 ± 0.057 | 1.00 / 0.27 / 0.43 |
| ensemble — mean score | 0.952 ± 0.034 | — |
| ensemble — union (≥1) | — | 1.00 / 0.31 / **0.48** |
| ensemble — intersect (≥2) | — | 1.00 / 0.14 / 0.24 |

**Window size matters a lot for Donut.** Donut was built for *long, fast* metric streams
with a 120-wide window. A daily series (~180 points) can't use W=120, so we pick the
largest window that still leaves most points scorable. Sweeping it on the subtle case
shows the choice really matters:

| window | VAE subtle AUC-PR | VAE subtle F1 |
|---|---|---|
| 14 | 0.465 | 0.35 |
| 21 | 0.653 | 0.39 |
| **28** | **0.836** | **0.43** |
| 40 | 0.847 | 0.36 |

At W=14 the model sees only two weekly cycles and can't tell a 15% dip from normal
seasonal movement (AUC-PR 0.465 — better than random's ~0.08, but weak). At W=28 (4
cycles) most of the signal comes back (0.836). W=40 is slightly *worse* than 28: on a
180-point series, a longer window leaves more early points unscorable, and some anomalies
fall in that warm-up zone. Best spot ≈ 28. (`VAE_WINDOW=21 npm run bench:detectors`
reproduces a row.)

**So, do we need the ensemble?**

- **Large anomalies — no.** Both detectors score a perfect 1.000 AUC-PR; combining them
  barely moves F1 (0.46 vs 0.41).
- **Subtle anomalies — yes, but only the UNION (OR) rule.** The statistical detector
  ranks better overall (AUC-PR 0.949 vs 0.836), but the two detectors catch *different*
  subtle anomalies, so taking the **union** lifts pooled F1 to **0.48** — above either one
  alone (stat 0.30, Donut 0.43), at recall 0.31 with precision still 1.00. That's the
  ensemble actually helping.

**Why the old "agreement" rule was wrong.** Prima used to treat *agreement* (≥2 detectors)
as "confirmed / high confidence." That **intersect** rule is the **worst** on subtle
anomalies (F1 0.24, vs 0.48 for union) — demanding agreement throws away the very
different-but-correct detections that make an ensemble worth having. That's why the
ensemble was dropped from the agent (see the Verdict); the agent now runs the statistical
detector only.

## Benchmark B — real AIOps metrics (z-score vs Donut)

![Real AIOps KPIs: three heterogeneous series with different sampling rates and shapes, labeled anomalies in red](docs/aiops.png)

Benchmark A is the z-score's best case — one clean daily series with a single known weekly
cycle — and Donut's worst case (a short series forces W=28, far below its design). To test
the setting Donut was actually *built* for, `bench_aiops.py` runs it on the **2018 AIOps
Challenge KPI dataset** (26 real web-service metrics from Tsinghua/Alibaba; sampled every
1 or 5 minutes; ~300k points each; download `Preliminary_dataset/train.csv` from
<https://github.com/NetManAIOps/KPI-Anomaly-Detection> into
`data/aiops-kpi/preliminary_train.csv`).

```bash
ml-service/.venv/bin/python ml-service/bench_aiops.py   # N≤25000/KPI, W=120, 50 epochs
```

Each metric is scored on its own; the z-score's seasonal period is worked out per metric
from its sampling interval (this is the "you must tune it per series" step). Two scores,
averaged over the usable metrics (of the 26 loaded, any metric whose test range has no
labeled anomaly is skipped — `bench_aiops.py` prints the exact `n`):

| metric | z-score | Donut |
|---|---|---|
| **point-adjusted best-F1** (paper §4.2) | 0.541 | **0.910** |
| strict point-wise AUC-PR | 0.184 | **0.194** |

> **Important caveat (read before trusting these numbers).** Benchmark B is **in-sample**:
> `donut_scores()` trains the model on the *whole* metric and then scores those same points
> (only a leading warm-up window is dropped — there's no separate held-out test split), and
> the z-score is likewise fit over the full series. So these are best-case, single-seed
> numbers — **not** a held-out generalization estimate — and best-F1 uses an oracle (perfect)
> threshold. They show the rebuild *runs in Donut's home turf and ranks anomalies sensibly*;
> they do **not** by themselves prove it matches the paper's held-out best-F1. A fully
> faithful test would split each metric in time (train on the earlier part, score a later
> part) and average over several seeds. That's tracked as a follow-up (see ../SKILLS_AUDIT.md).

**Two takeaways:**

1. **The rebuild behaves like Donut in its home turf.** At W=120 on minute-level data,
   faithful Donut reaches in-sample **best-F1 0.910**, inside the paper's reported
   **0.75–0.90** range. The model behaves as the paper describes — but remember this is an
   in-sample number (see caveat), so treat it as a sanity check on the code, not proof of
   held-out fidelity.
2. **The setting flips the winner.** In Benchmark A (one clean series) the z-score wins; on
   26 varied real metrics Donut wins — clearly on the paper's metric, narrowly on AUC-PR.
   You can't hand-pick one seasonal period that fits metrics with different sampling rates
   and shapes; Donut learns each one. It wins **every** coarse (5-min) metric, where the
   fixed z-score template fails worst (e.g. `07927`: best-F1 0.137 → 0.858).

**Read best-F1 carefully.** It's an *oracle-threshold*, *point-adjusted* score: a whole
anomaly stretch counts as caught if *any single point* in it crosses the best-chosen
threshold. That inflates the headline number and favors a learned detector that reliably
spikes *somewhere* in each stretch — which is exactly why Donut's best-F1 (0.910) towers
over its strict AUC-PR (0.194) on the same runs (e.g. `18fbb`: best-F1 0.996 but AUC-PR
0.114). best-F1 answers "with a perfect cutoff, can you get one alert per incident?";
AUC-PR answers "is the ranking actually well-ordered?" The honest move is to report both —
the paper uses best-F1, so we show it for comparison, but AUC-PR/VUS-PR (Benchmark A) is
the stricter view.

## Benchmark C — real SMD (the dataset the agent actually runs on)

![SMD telemetry: three ground-truth culprit channels of machine-1-1 around its worst anomaly, with the labeled span shaded](docs/smd.png)

Benchmarks A/B justify *which detector to use*. This one tests the **exact path the web app
ships**: the agent squeezes an SMD machine's 38 channels down to a single **health score**
(per timestep, the max robust-z across channels, `src/lib/smdWarehouse.ts`) and flags
anomalies with the same EVT/POT rule. SMD is real server telemetry with two kinds of ground
truth — per-point `test_label`s *and* `interpretation_label`s naming which channels caused
each anomaly (shaded above) — so we can grade both detection and root-cause.

**The shipped path — `npm run bench:smd`** scores that health-score → EVT pipeline against
`test_label`, averaged over 8 machines (we report strict, lenient, *and* the actual deployed
setting — never just the flattering one):

| metric | macro | what it means |
|---|---|---|
| strict point-wise AUC-PR | 0.26 | several× above the 2–15% anomaly floor; the raw ranking is loose |
| point-adjusted best-F1 (oracle threshold) | **0.97** | with a good cutoff, the score *separates* anomalies very well |
| point-adjusted F1 at the **deployed** EVT threshold | 0.69 | what actually ships |

**Honest finding:** the gap between the perfect cutoff (0.97) and the self-calibrating EVT
cutoff (0.69) is real — EVT over-flags on 2 of the 8 machines (precision 0.1–0.2, recall
1.0). The health *signal* is strong; **the weak link is picking a per-machine threshold**,
and that's the obvious next step. (A robustness fix lives here too: SMD channels are scaled
to `[0,1]` and are often flat, so median/MAD divided by a near-zero MAD blew up into the
millions until we floored sigma at 0.01 — see `channelZ`.)

**The deep detector on SMD — `ml-service/bench_smd.py`** trains a Donut *per channel* on
SMD's clean `train` split, scores `test`, takes the max across channels, and grades against
`test_label` (reusing the AIOps metric code). A quick partial run (machine-1-1, 8 channels,
8 epochs) gives z-score best-F1 0.939 vs Donut 0.948 — neck and neck, just like on the clean
AIOps series. A full per-channel run across all machines is a follow-up.

## Verdict — do we need the ensemble / agreement rule?

Putting the benchmarks together, the honest answer is **mostly no**, with one narrow
exception:

- **The agreement *gate* ("confirmed ≥2") — no, remove it.** It's the worst performer
  everywhere we measured (subtle F1 0.24 vs 0.48 for union). Requiring two detectors to
  agree throws away the different-but-correct catches that are the only reason to run two
  detectors. At most keep agreement as a soft *confidence note*, never as a gate on whether
  to flag.
- **The ensemble itself — only narrowly worth it.** It helps in exactly one case: when both
  detectors are individually weak but catch *different* anomalies (subtle 10–20% dips), where
  the union lifts F1 to 0.48 above either alone. On large/easy anomalies it's redundant (both
  ≈ 1.0), and on one clean daily series (Prima's real metric) the statistical detector alone
  is enough — Donut and the ensemble are overkill there. The deep detector only pulls ahead in
  Donut's home turf: many varied metrics at higher frequency (Benchmark B).
- **What's actually worth keeping isn't about accuracy:** (1) graceful fallback — the
  statistical detector has no dependencies and is always on, the VAE adds to it when the
  service is up; (2) a confidence signal for the narrator. Neither needs the gate.

**Decision (already applied).** The ensemble and the agreement vote were **removed from the
agent**: [`src/agents/nodes.ts`](../src/agents/nodes.ts) now runs the seasonal z-score
(EVT/POT) only — no VAE call, no `confidence`/`detectors` fields, no "confirmed ≥2" gate. The
Donut VAE stays here purely as a benchmarked reference; it would only be worth wiring back in
if Prima started monitoring many varied, higher-frequency metrics (the Benchmark B case).
