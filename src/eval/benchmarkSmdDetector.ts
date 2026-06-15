/**
 * Benchmark of Prima's ACTUAL anomaly-detection path on real SMD labels.
 *
 * The app does not score per-channel Donut directly (that's ml-service/bench_smd.py).
 * It analyzes a single 1-D series — the entity *health score* = per-timestep max
 * robust z across the 38 channels (src/lib/smdWarehouse.computeHealthSeries) — and
 * flags anomalies with a self-calibrating EVT/POT threshold
 * (src/lib/stats.detectScoreAnomalies). This benchmarks THAT path against the
 * ground-truth test_label, so the number that ships is the number we report.
 *
 * Reports three metrics per entity (mle-practices: strict AND lenient AND deployed):
 *   AUC-PR      — strict, point-wise, threshold-free (ranking quality of the score)
 *   best-F1*    — lenient: point-adjusted best-F1 under an oracle threshold
 *   EVT-F1*     — deployed: point-adjusted F1 at the threshold the app actually picks
 *
 * Run:  npm run bench:smd
 */
import Database from "better-sqlite3";
import { DB_PATH } from "@/lib/db";
import { ensureSmdDb } from "@/lib/smdWarehouse";
import { detectScoreAnomalies, type Point } from "@/lib/stats";

/** Strict point-wise AUC-PR (trapezoid from (recall 0, precision 1)). */
function aucPr(scores: number[], labels: number[]): number {
  const order = scores.map((s, i) => i).sort((a, b) => scores[b] - scores[a]);
  const totalPos = labels.reduce((a, b) => a + b, 0);
  if (totalPos === 0) return NaN;
  let tp = 0;
  let fp = 0;
  let prevR = 0;
  let prevP = 1;
  let area = 0;
  for (const i of order) {
    if (labels[i] === 1) tp++;
    else fp++;
    const p = tp / (tp + fp);
    const r = tp / totalPos;
    area += (r - prevR) * (p + prevP) / 2;
    prevR = r;
    prevP = p;
  }
  return area;
}

/** Contiguous runs of label===1, as [start,end] inclusive index pairs. */
function segments(labels: number[]): [number, number][] {
  const segs: [number, number][] = [];
  let i = 0;
  const n = labels.length;
  while (i < n) {
    if (labels[i] === 1) {
      let j = i;
      while (j + 1 < n && labels[j + 1] === 1) j++;
      segs.push([i, j]);
      i = j + 1;
    } else i++;
  }
  return segs;
}

/**
 * Point-adjusted best-F1 (Donut paper §4.2): a labeled segment counts fully
 * detected if ANY point in it scores above the threshold; points outside
 * segments are scored normally. Sweep segment-max thresholds, take the best F1.
 */
function bestF1Adjusted(scores: number[], labels: number[]): number {
  const segs = segments(labels);
  const P = segs.reduce((a, [s, e]) => a + (e - s + 1), 0);
  if (P === 0) return NaN;
  const segMax = segs.map(([s, e]) => Math.max(...scores.slice(s, e + 1)));
  const segLen = segs.map(([s, e]) => e - s + 1);
  const normal = scores.filter((_, i) => labels[i] === 0).sort((a, b) => a - b);
  const order = segMax.map((_, i) => i).sort((a, b) => segMax[b] - segMax[a]);
  let tp = 0;
  let best = 0;
  for (const k of order) {
    tp += segLen[k];
    // false positives = normal points scoring >= this segment-max threshold
    const thr = segMax[k];
    let lo = 0;
    let hi = normal.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (normal[mid] < thr) lo = mid + 1;
      else hi = mid;
    }
    const fp = normal.length - lo;
    const f1 = (2 * tp) / (2 * tp + fp + (P - tp));
    if (f1 > best) best = f1;
  }
  return best;
}

/** Point-adjusted F1 at a concrete operating point (set of flagged indices). */
function pointAdjustedF1(
  flagged: Set<number>,
  labels: number[],
): { f1: number; precision: number; recall: number } {
  const segs = segments(labels);
  let tp = 0;
  let fn = 0;
  for (const [s, e] of segs) {
    const detected = [...Array(e - s + 1)].some((_, k) => flagged.has(s + k));
    if (detected) tp += e - s + 1;
    else fn += e - s + 1;
  }
  let fp = 0;
  for (const t of flagged) if (labels[t] === 0) fp++;
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall = tp + fn ? tp / (tp + fn) : 0;
  const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
  return { f1, precision, recall };
}

function main() {
  ensureSmdDb(DB_PATH);
  const db = new Database(DB_PATH, { readonly: true, fileMustExist: true });
  const entities = (db.prepare("SELECT DISTINCT entity AS e FROM health ORDER BY entity").all() as { e: string }[]).map(
    (r) => r.e,
  );

  console.log("SMD detector benchmark — Prima's actual path (health score → EVT/POT)");
  console.log("  AUC-PR = strict point-wise · best-F1* = point-adjusted oracle · EVT-F1* = deployed operating point");
  console.log(
    `${"entity".padStart(14)}  ${"pts".padStart(6)}  ${"anom".padStart(6)}  ${"AUC-PR".padStart(7)}  ${"best-F1*".padStart(8)}  ${"EVT-F1*".padStart(8)}  ${"EVT-P".padStart(6)}  ${"EVT-R".padStart(6)}`,
  );

  const auc: number[] = [];
  const bf1: number[] = [];
  const ef1: number[] = [];
  for (const entity of entities) {
    const rows = db
      .prepare("SELECT t, value, label FROM health WHERE entity = ? ORDER BY t")
      .all(entity) as { t: number; value: number; label: number }[];
    const scores = rows.map((r) => r.value);
    const labels = rows.map((r) => r.label);
    const nAnom = labels.reduce((a, b) => a + b, 0);
    if (nAnom === 0) {
      console.log(`${entity.padStart(14)}  (no labeled anomalies — skipped)`);
      continue;
    }
    const points: Point[] = rows.map((r) => ({ date: String(r.t), value: r.value }));
    const flagged = new Set(detectScoreAnomalies(points).map((a) => parseInt(a.date, 10)));

    const a = aucPr(scores, labels);
    const b = bestF1Adjusted(scores, labels);
    const evt = pointAdjustedF1(flagged, labels);
    auc.push(a);
    bf1.push(b);
    ef1.push(evt.f1);
    console.log(
      `${entity.padStart(14)}  ${String(rows.length).padStart(6)}  ${String(nAnom).padStart(6)}  ` +
        `${a.toFixed(3).padStart(7)}  ${b.toFixed(3).padStart(8)}  ${evt.f1.toFixed(3).padStart(8)}  ` +
        `${evt.precision.toFixed(2).padStart(6)}  ${evt.recall.toFixed(2).padStart(6)}`,
    );
  }
  db.close();

  const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN);
  console.log("-".repeat(78));
  console.log(
    `${"MACRO MEAN".padStart(14)}  ${"".padStart(6)}  ${"".padStart(6)}  ` +
      `${mean(auc).toFixed(3).padStart(7)}  ${mean(bf1).toFixed(3).padStart(8)}  ${mean(ef1).toFixed(3).padStart(8)}` +
      `   (n=${auc.length})`,
  );
}

main();
