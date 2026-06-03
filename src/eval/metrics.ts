/**
 * Threshold-free, range-aware evaluation metrics for time-series anomaly
 * detection — the family the TSB-AD / "Elephant in the Room" (NeurIPS 2024)
 * work recommends over the discredited point-adjusted F1.
 *
 *  - AUC-PR  : area under the precision-recall curve over all score thresholds.
 *  - VUS-PR  : Volume Under the Surface — AUC-PR averaged over a range of label
 *              buffer widths, so a detection a day early/late still counts.
 *              Robust to the temporal-offset ambiguity point metrics punish.
 */

/** Precision-recall AUC for continuous scores against a binary label vector. */
export function aucPr(scores: number[], labels: number[]): number {
  const n = scores.length;
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => scores[b] - scores[a]);
  const totalPos = labels.reduce((a, b) => a + b, 0);
  if (totalPos === 0) return 0;

  let tp = 0;
  let fp = 0;
  let prevRecall = 0;
  let area = 0;
  let prevPrecision = 1;
  for (const idx of order) {
    if (labels[idx] === 1) tp++;
    else fp++;
    const precision = tp / (tp + fp);
    const recall = tp / totalPos;
    // trapezoidal area as recall advances
    area += ((precision + prevPrecision) / 2) * (recall - prevRecall);
    prevRecall = recall;
    prevPrecision = precision;
  }
  return +area.toFixed(4);
}

/** Dilate binary labels by ±buffer (a point near a true anomaly becomes positive). */
function dilate(labels: number[], buffer: number): number[] {
  if (buffer <= 0) return labels.slice();
  const n = labels.length;
  const out = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    if (labels[i] === 1) {
      for (let j = Math.max(0, i - buffer); j <= Math.min(n - 1, i + buffer); j++) out[j] = 1;
    }
  }
  return out;
}

/**
 * VUS-PR: average AUC-PR over buffer widths 0..maxBuffer. Approximates the
 * "volume under the surface" by integrating the range-aware PR-AUC across the
 * temporal-tolerance dimension.
 */
export function vusPr(scores: number[], labels: number[], maxBuffer = 5): number {
  let sum = 0;
  for (let b = 0; b <= maxBuffer; b++) sum += aucPr(scores, dilate(labels, b));
  return +(sum / (maxBuffer + 1)).toFixed(4);
}

/** Point-wise precision/recall/F1 at a fixed prediction set (the "old" way, for contrast). */
export function pointF1(predictedIdx: number[], truthIdx: Set<number>, tolerance = 1) {
  const truth = [...truthIdx];
  const matched = new Set<number>();
  let tp = 0;
  let fp = 0;
  for (const p of predictedIdx) {
    const hit = truth.find((t) => Math.abs(t - p) <= tolerance && !matched.has(t));
    if (hit !== undefined) {
      tp++;
      matched.add(hit);
    } else fp++;
  }
  const fn = truth.length - matched.size;
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall = tp + fn ? tp / (tp + fn) : 0;
  const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
  return { tp, fp, fn, precision: +precision.toFixed(3), recall: +recall.toFixed(3), f1: +f1.toFixed(3) };
}
