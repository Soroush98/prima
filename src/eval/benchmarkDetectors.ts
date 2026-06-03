/**
 * Controlled detector benchmark — statistical vs. Donut-VAE vs. ensemble.
 *
 * Generates synthetic DAU series with KNOWN injected anomalies (labels by
 * construction) and scores each detector with the TSB-AD-recommended,
 * threshold-free metrics **VUS-PR** and **AUC-PR** (robust to temporal offset),
 * alongside the classic point-F1 for contrast.
 *
 * Hardened vs. a naive single-run benchmark:
 *   - averages over MANY seeds and reports mean ± std (single-seed numbers are
 *     anecdotal — N=~13 positives makes each detection worth ~8% recall);
 *   - injects a sustained **level-shift / regime** anomaly, not just points;
 *   - drives the series with **AR(1)-autocorrelated, Laplace (heavy-tailed)**
 *     noise — structure neither the seasonal-naive z-score nor the VAE assumes,
 *     so the scores don't flatter a generator that matches the model.
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
 * detectors implicitly assume — scaled to a stationary std of ~30. Anomalies
 * include point spikes/drops, a short collective drop, and a 5-day level shift.
 */
function makeLabeledSeries(seed: number): Labeled {
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

  // [startDay, length, factor] — jittered per seed so we don't test fixed indices.
  // The 5-day 0.6× block is a sustained level-shift / regime anomaly.
  const template: [number, number, number][] = [
    [22, 1, 0.45],
    [40, 1, 1.7],
    [58, 3, 0.5],
    [85, 1, 1.8],
    [105, 5, 0.6], // level shift
    [130, 1, 0.4],
    [152, 1, 1.65],
  ];
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

/* ---- per-seed metric collection -------------------------------------- */

interface Acc {
  vus: number[];
  auc: number[];
  f1: number[];
  flagged: number[];
}
const newAcc = (): Acc => ({ vus: [], auc: [], f1: [], flagged: [] });

function record(acc: Acc, scores: number[] | null, flaggedIdx: number[], labels: number[], truth: Set<number>) {
  if (scores) {
    acc.vus.push(vusPr(scores, labels, 5));
    acc.auc.push(aucPr(scores, labels));
  }
  acc.f1.push(pointF1(flaggedIdx, truth, TOLERANCE).f1);
  acc.flagged.push(flaggedIdx.length);
}

function meanStd(xs: number[]): { m: number; s: number } {
  if (xs.length === 0) return { m: NaN, s: 0 };
  const m = xs.reduce((a, b) => a + b, 0) / xs.length;
  const s = Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / xs.length);
  return { m, s };
}

function fmt({ m, s }: { m: number; s: number }, places = 3): string {
  if (Number.isNaN(m)) return "  —  ".padEnd(13);
  return `${m.toFixed(places)}±${s.toFixed(places)}`;
}

function printRow(name: string, acc: Acc) {
  const vus = acc.vus.length ? `VUS-PR=${fmt(meanStd(acc.vus))}` : `VUS-PR=${"—".padEnd(11)}`;
  const auc = acc.auc.length ? `AUC-PR=${fmt(meanStd(acc.auc))}` : `AUC-PR=${"—".padEnd(11)}`;
  const f1 = fmt(meanStd(acc.f1), 2);
  const flagged = meanStd(acc.flagged);
  console.log(`  ${name.padEnd(22)} ${vus}  ${auc}  | point-F1=${f1} (flagged ${flagged.m.toFixed(1)})`);
}

async function main() {
  if (!(await mlServiceHealthy())) {
    console.error("\n  ✗ ML service not reachable. Start it first:  npm run ml\n");
    process.exit(1);
  }

  const accStat = newAcc();
  const accVae = newAcc();
  const accEns = newAcc(); // ensemble mean-score, intersect (≥2) flags
  const accUnion = newAcc(); // union (≥1) flags, F1 only
  const accInter = newAcc(); // intersect (≥2) flags, F1 only

  const norm = (xs: number[]) => {
    const mx = Math.max(...xs) || 1;
    return xs.map((v) => v / mx);
  };

  let contamSum = 0;
  let vaeRuns = 0;
  let lastLoss = 0;
  let lastTrainMs = 0;

  for (let s = 0; s < SEEDS; s++) {
    const { series, labels, truthIdx } = makeLabeledSeries(1000 + s);
    contamSum += truthIdx.size / DAYS;
    const dateToIdx = new Map(series.map((p, i) => [p.date, i]));

    // statistical — continuous |z| scores + EVT-thresholded flags
    const statScores = scoreAnomalies(series).map((x) => x.score);
    const statFlags = detectAnomalies(series, { method: "evt" }).map((a) => dateToIdx.get(a.date)!);
    record(accStat, statScores, statFlags, labels, truthIdx);

    // Donut-style VAE (skip its rows for this seed if the service hiccups)
    const vae = await detectAnomaliesML(series, { model: "vae", threshold: "evt" });
    if (!vae) continue;
    vaeRuns++;
    lastLoss = vae.finalLoss;
    lastTrainMs = vae.trainMs;
    const vaeFlags = vae.anomalies.map((a) => a.index);
    record(accVae, vae.scores, vaeFlags, labels, truthIdx);

    // ensemble — mean of min-max-normalized scores + vote-based flag sets
    const sset = new Set(statFlags);
    const vset = new Set(vaeFlags);
    const votes = (i: number) => (sset.has(i) ? 1 : 0) + (vset.has(i) ? 1 : 0);
    const ns = norm(statScores);
    const nv = norm(vae.scores);
    const ensScores = series.map((_, i) => (ns[i] + nv[i]) / 2);
    const union = series.map((_, i) => i).filter((i) => votes(i) >= 1); // recall-oriented
    const inter = series.map((_, i) => i).filter((i) => votes(i) >= 2); // precision-oriented

    record(accEns, ensScores, inter, labels, truthIdx);
    record(accUnion, null, union, labels, truthIdx);
    record(accInter, null, inter, labels, truthIdx);
  }

  const avgContam = ((contamSum / SEEDS) * 100).toFixed(1);
  console.log(`\n  Detector benchmark — ${DAYS} days × ${SEEDS} seeds (mean ± std)`);
  console.log(`  ~${avgContam}% contamination · AR(1) Laplace noise · point + collective + level-shift anomalies`);
  console.log(`  Metrics: VUS-PR / AUC-PR (threshold-free, TSB-AD family) + point-F1 (±${TOLERANCE}d)`);
  console.log("  " + "─".repeat(78));

  printRow("statistical (EVT)", accStat);
  if (vaeRuns) {
    printRow("donut_vae", accVae);
    printRow("ensemble (mean score)", accEns);
    printRow("  └ union (≥1 vote)", accUnion);
    printRow("  └ intersect (≥2)", accInter);
  }

  console.log("  " + "─".repeat(78));
  console.log(
    `  VAE trained on ${vaeRuns}/${SEEDS} seeds · last loss ${lastLoss} (${lastTrainMs}ms)\n`,
  );
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
