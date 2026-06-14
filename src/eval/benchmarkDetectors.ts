/**
 * Controlled detector benchmark — statistical vs. Donut VAE vs. ensemble.
 *
 * Generates synthetic DAU series with KNOWN injected anomalies (labels by
 * construction) and scores each detector with the TSB-AD-recommended,
 * threshold-free metrics **VUS-PR** and **AUC-PR** (robust to temporal offset),
 * alongside point precision/recall/F1.
 *
 * Two anomaly regimes are tested:
 *   - **large**  — the screaming anomalies (40–80% deviations) most detectors ace.
 *   - **subtle** — 10–20% deviations, comparable to the weekly seasonal swing, so
 *     a detector must separate anomaly from *seasonality*, not just from noise.
 *     This is where the statistical baseline and the VAE are expected to diverge —
 *     i.e. where an ensemble could actually earn its keep.
 *
 * Believability: threshold-free AUC-PR/VUS-PR are averaged per-seed (mean ± std),
 * but the operational point metric is **pooled across all seeds** into a single
 * confusion matrix, so precision/recall/F1 rest on ~100+ true anomaly points
 * rather than the ~13 a single seed provides.
 *
 * Hardened vs. a naive benchmark: sustained **level-shift** anomalies (not just
 * points) and **AR(1)-autocorrelated, Laplace (heavy-tailed)** noise — structure
 * neither detector assumes, so the scores don't flatter a matched generator.
 *
 * Requires the Python ML service running:  npm run ml
 * Run with:  npm run bench:detectors
 */
import { detectAnomalies, scoreAnomalies, type Point } from "@/lib/stats";
import { detectAnomaliesML, mlServiceHealthy } from "@/lib/mlClient";
import { aucPr, vusPr, pointF1 } from "./metrics";

const DAYS = 180;
const SEEDS = 12; // independent realizations to average over
const TOLERANCE = 1;
const VAE_WINDOW = Number(process.env.VAE_WINDOW) || 28; // ~4 weekly cycles; Donut uses 120 on long high-freq streams

/** [startDay, length, factor] — an injected anomaly. */
type AnomalySpec = [number, number, number];

/** Screaming anomalies (40–80% deviations): the easy regime. */
const LARGE_TEMPLATE: AnomalySpec[] = [
  [22, 1, 0.45], [40, 1, 1.7], [58, 3, 0.5], [85, 1, 1.8],
  [105, 5, 0.6], [130, 1, 0.4], [152, 1, 1.65],
];

/** Subtle anomalies (10–20% deviations) — on the order of the weekend swing, so
 *  they're confusable with seasonality. Mostly points + one short collective dip
 *  and one 5-day level shift. ~14 positives/seed → ~165 pooled over 12 seeds. */
const SUBTLE_TEMPLATE: AnomalySpec[] = [
  [18, 1, 0.85], [30, 1, 1.15], [44, 1, 0.88], [60, 1, 1.12],
  [74, 3, 0.88], [92, 1, 1.18], [108, 5, 0.87],
  [128, 1, 0.82], [140, 1, 1.2], [158, 1, 0.9],
];

function mulberry32(seed: number) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function isoDate(i: number): string {
  const d = new Date("2025-01-01T00:00:00Z");
  d.setUTCDate(d.getUTCDate() + i);
  return d.toISOString().slice(0, 10);
}

interface Labeled {
  series: Point[];
  labels: number[];
  truthIdx: Set<number>;
}

/**
 * One labelled realization. The noise is AR(1) (phi=0.45) with Laplace
 * innovations — autocorrelated and heavier-tailed than the Gaussian both
 * detectors implicitly assume — scaled to a stationary std of ~30. The anomaly
 * magnitudes come from the supplied `template` (large vs. subtle regime).
 */
function makeLabeledSeries(seed: number, template: AnomalySpec[]): Labeled {
  const rand = mulberry32(seed);
  // Laplace(0, b): heavier tails than Gaussian (excess kurtosis 3).
  const laplace = (b: number) => {
    const u = rand() - 0.5;
    return -b * Math.sign(u) * Math.log(1 - 2 * Math.abs(u));
  };

  const PHI = 0.45;
  const TARGET_STD = 30;
  // stationary var of AR(1) = var(innov)/(1-phi^2); Laplace var = 2b^2.
  const b = Math.sqrt((TARGET_STD * TARGET_STD * (1 - PHI * PHI)) / 2);

  const values: number[] = [];
  let noise = 0;
  for (let d = 0; d < DAYS; d++) {
    noise = PHI * noise + laplace(b);
    const dow = (d + 3) % 7;
    const weekend = dow === 5 || dow === 6;
    let v = 1000 + d * 2; // linear trend
    v *= weekend ? 0.8 : 1.0; // weekly seasonality
    v *= 1 + 0.04 * Math.sin((d / 7) * 2 * Math.PI);
    v += noise;
    values.push(Math.max(50, v));
  }

  // positions jittered per seed so we don't test fixed indices.
  const truthIdx = new Set<number>();
  for (const [start0, len, factor] of template) {
    const jitter = Math.floor(rand() * 9) - 4; // ±4 days
    const start = Math.min(DAYS - len - 1, Math.max(1, start0 + jitter));
    for (let j = 0; j < len; j++) {
      values[start + j] *= factor;
      truthIdx.add(start + j);
    }
  }

  const series = values.map((v, i) => ({ date: isoDate(i), value: Math.round(v) }));
  const labels = values.map((_, i) => (truthIdx.has(i) ? 1 : 0));
  return { series, labels, truthIdx };
}

/* ---- metric collection ----------------------------------------------- */

// Threshold-free scores are averaged per-seed; point flags are POOLED across
// seeds (indices namespaced as seed*OFFSET + i) into one confusion matrix, so
// precision/recall/F1 rest on ~100+ positives instead of ~13.
const OFFSET = 10000;

interface Acc {
  vus: number[];
  auc: number[];
}
const newAcc = (): Acc => ({ vus: [], auc: [] });
function recordScore(acc: Acc, scores: number[], labels: number[]) {
  acc.vus.push(vusPr(scores, labels, 5));
  acc.auc.push(aucPr(scores, labels));
}

function meanStd(xs: number[]): { m: number; s: number } {
  if (xs.length === 0) return { m: NaN, s: 0 };
  const m = xs.reduce((a, b) => a + b, 0) / xs.length;
  const s = Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / xs.length);
  return { m, s };
}
function fmt({ m, s }: { m: number; s: number }, places = 3): string {
  if (Number.isNaN(m)) return "—".padEnd(11);
  return `${m.toFixed(places)}±${s.toFixed(places)}`;
}

/** A row with both threshold-free (per-seed) and pooled point metrics. */
function printRow(name: string, acc: Acc | null, pooledFlags: number[] | null, truth: Set<number>) {
  const score = acc
    ? `VUS-PR=${fmt(meanStd(acc.vus))}  AUC-PR=${fmt(meanStd(acc.auc))}`
    : `${" ".repeat(19)}  ${" ".repeat(19)}`;
  let point = "";
  if (pooledFlags) {
    const p = pointF1(pooledFlags, truth, TOLERANCE);
    point = ` | P=${p.precision.toFixed(2)} R=${p.recall.toFixed(2)} F1=${p.f1.toFixed(2)} (flagged ${pooledFlags.length})`;
  }
  console.log(`  ${name.padEnd(22)} ${score}${point}`);
}

const norm = (xs: number[]) => {
  const mx = Math.max(...xs) || 1;
  return xs.map((v) => v / mx);
};

async function runScenario(label: string, template: AnomalySpec[]) {
  const accStat = newAcc();
  const accVae = newAcc();
  const accEns = newAcc(); // ensemble mean-score (threshold-free)

  const truth = new Set<number>();
  const flStat: number[] = [];
  const flVae: number[] = [];
  const flUnion: number[] = []; // ≥1 vote (recall-oriented)
  const flInter: number[] = []; // ≥2 votes (precision-oriented)

  let contamSum = 0;
  let vaeRuns = 0;
  let lastLoss = 0;
  let lastTrainMs = 0;

  for (let s = 0; s < SEEDS; s++) {
    const { series, labels, truthIdx } = makeLabeledSeries(1000 + s, template);
    contamSum += truthIdx.size / DAYS;
    const off = s * OFFSET;
    truthIdx.forEach((i) => truth.add(off + i));
    const dateToIdx = new Map(series.map((p, i) => [p.date, i]));

    // statistical — continuous |z| scores + EVT-thresholded flags
    const statScores = scoreAnomalies(series).map((x) => x.score);
    const statFlags = detectAnomalies(series, { method: "evt" }).map((a) => dateToIdx.get(a.date)!);
    recordScore(accStat, statScores, labels);
    statFlags.forEach((i) => flStat.push(off + i));

    // Donut VAE (skip this seed's VAE/ensemble rows if the service hiccups)
    const vae = await detectAnomaliesML(series, { model: "vae", threshold: "evt", window: VAE_WINDOW });
    if (!vae) continue;
    vaeRuns++;
    lastLoss = vae.finalLoss;
    lastTrainMs = vae.trainMs;
    const vaeFlags = vae.anomalies.map((a) => a.index);
    recordScore(accVae, vae.scores, labels);
    vaeFlags.forEach((i) => flVae.push(off + i));

    // ensemble — mean of min-max-normalized scores + vote-based flag sets
    const sset = new Set(statFlags);
    const vset = new Set(vaeFlags);
    const votes = (i: number) => (sset.has(i) ? 1 : 0) + (vset.has(i) ? 1 : 0);
    const ns = norm(statScores);
    const nv = norm(vae.scores);
    recordScore(accEns, series.map((_, i) => (ns[i] + nv[i]) / 2), labels);
    series.forEach((_, i) => {
      if (votes(i) >= 1) flUnion.push(off + i);
      if (votes(i) >= 2) flInter.push(off + i);
    });
  }

  const avgContam = ((contamSum / SEEDS) * 100).toFixed(1);
  console.log(
    `\n  ── ${label.toUpperCase()} regime — ${DAYS}d × ${SEEDS} seeds · ~${avgContam}% contamination · ` +
      `${truth.size} pooled anomaly points`,
  );
  console.log("  " + "─".repeat(82));
  printRow("statistical (EVT)", accStat, flStat, truth);
  if (vaeRuns) {
    printRow("donut_vae", accVae, flVae, truth);
    printRow("ensemble (mean score)", accEns, null, truth);
    printRow("  └ union (≥1 vote)", null, flUnion, truth);
    printRow("  └ intersect (≥2)", null, flInter, truth);
  }
  console.log("  " + "─".repeat(82));
  console.log(`  VAE trained on ${vaeRuns}/${SEEDS} seeds · last −ELBO ${lastLoss} (${lastTrainMs}ms)`);
}

async function main() {
  if (!(await mlServiceHealthy())) {
    console.error("\n  ✗ ML service not reachable. Start it first:  npm run ml\n");
    process.exit(1);
  }
  console.log(`\n  Detector benchmark — AR(1) Laplace noise · point + collective + level-shift anomalies`);
  console.log(`  AUC-PR/VUS-PR averaged per-seed (mean±std); P/R/F1 pooled across seeds (±${TOLERANCE}d)`);

  await runScenario("large", LARGE_TEMPLATE);
  await runScenario("subtle", SUBTLE_TEMPLATE);
  console.log("");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
