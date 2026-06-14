---
name: mle-practices
description: Senior machine-learning-engineering workflow for the Prima repo — faithfully reproduce techniques against the source paper/repo, adversarially benchmark them vs a simple baseline across multiple regimes, report strict AND lenient metrics with their caveats, and ship only what the evidence justifies. Use when implementing or evaluating models, detectors, forecasters, metrics, or benchmarks here.
---

# Senior Machine Learning Engineering — Prima

The throughline: **don't trust a model or a metric — reproduce it faithfully, benchmark it adversarially, and let the evidence (not the hype) decide what ships.** These are the moves we used implementing and benchmarking Donut vs the statistical detector.

## 1. Faithful reproduction
- When implementing a published method, **pull the reference repo/paper** and verify each *signature* technique is actually present in the code. Don't accept an "X-style" label the code doesn't earn. (Donut needs: Gaussian decoder `p(x|z)`, M-ELBO with missing-data injection, MCMC imputation, and reconstruction-*probability* scoring — not MSE.)
- **Flag every deviation** from the paper's hyperparameters and *test its impact* — hyperparameters are load-bearing. (window=14 vs the paper's 120 → a sweep moved AUC-PR 0.465 → 0.836.)
- **Validate the port** by matching the paper's reported metric range (Donut best-F1 0.75–0.90 → 0.91 on real in-regime data confirms fidelity).
- Reproducibility: fix seeds; if you port an RNG or generator, validate its output against the source (we matched `mulberry32` against the TS/node values before plotting).

## 2. Adversarial, multi-regime benchmarking
- Always benchmark the model against a **simple, dependency-free baseline** (a seasonal z-score). The baseline often wins in the easy regime — that's a finding, not an embarrassment.
- **Test more than one regime — one dataset lies.** (large/obvious vs subtle 10–20%; a clean synthetic single series vs real heterogeneous KPIs.) Verdicts are **regime-dependent**: a model isn't "good" or "bad," it's good *in its design regime*. State when you'd use it and when you wouldn't.
- Make the data **hostile to the model's assumptions** (AR(1) + heavy-tailed Laplace noise) so the numbers don't flatter a generator that matches the model.
- **Pool across seeds** for a believable N; report mean±std; state contamination/prevalence.

## 3. Metric rigor (don't get fooled)
- **AUC-PR's no-skill baseline is the positive prevalence, not 0.5.** Low absolute AUC-PR on rare events (~2% anomalies → ~0.02 floor) is expected, not "broken." Don't call 5× chance "coin-flip."
- **Point-adjusted best-F1 is an oracle-threshold, lenient metric**: one in-segment hit credits the whole segment, inflating numbers and favoring any model that spikes once per incident. Cite it **only** for paper-comparability, and **always pair it with strict AUC-PR/VUS-PR**. The truth is between them (e.g. 0.91 best-F1 vs 0.19 AUC-PR on the same scores).
- Use **temporal tolerance** (VUS-PR) so a detection one step early/late isn't double-counted as FP+FN.
- Report **both** the flattering and the strict number, and explain the gap.

## 4. Ship only what the evidence justifies
- If an ensemble/model doesn't beat the **better single component** on the *actual* workload, don't ship it. (We removed the ensemble; the "agreement = confirmed" gate was the single worst performer because it discarded complementary detections.)
- Right tool for the regime: a statistical baseline for one clean seasonal series; a learned model for many heterogeneous, high-frequency KPIs you can't hand-tune per series.
- Be honest about compromises — capped data, fewer epochs, single config, omitted training tricks (LR anneal, L2, grad-clip) — state them as caveats, never bury them.

## 5. Keep the benchmark as the decision's evidence
- The benchmark **is** the justification. When you pull a model from the live path, keep its benchmark + plots + the metric caveats in the README so the decision (and the conditions that would reverse it) stay documented.
