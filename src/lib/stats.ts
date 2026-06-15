/**
 * Lightweight statistical-modeling toolkit (no native deps).
 *
 * Covers the "statistical modeling for predictive analytics" preferred
 * qualification: seasonal-trend decomposition, robust anomaly detection
 * (median/MAD z-scores), and Holt-Winters additive forecasting.
 */

import { potThreshold } from "./evt";

export interface Point {
  date: string;
  value: number;
}

export interface Anomaly {
  date: string;
  value: number;
  expected: number;
  zscore: number;
  deviationPct: number;
  direction: "drop" | "spike";
  severity: "low" | "medium" | "high";
}

export interface ForecastPoint {
  date: string;
  forecast: number;
  lower: number;
  upper: number;
}

export interface DecompositionResult {
  trend: number[];
  seasonal: number[];
  residual: number[];
}

function median(xs: number[]): number {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

/** Centered moving average; falls back to one-sided near the edges. */
function centeredMA(values: number[], window: number): number[] {
  const half = Math.floor(window / 2);
  return values.map((_, i) => {
    const lo = Math.max(0, i - half);
    const hi = Math.min(values.length - 1, i + half);
    let sum = 0;
    for (let j = lo; j <= hi; j++) sum += values[j];
    return sum / (hi - lo + 1);
  });
}

/**
 * Additive seasonal-trend decomposition with a weekly period.
 * trend = centered MA; seasonal = per-day-of-week mean of detrended series.
 */
export function decompose(points: Point[], period = 7): DecompositionResult {
  const values = points.map((p) => p.value);
  const trend = centeredMA(values, period);
  const detrended = values.map((v, i) => v - trend[i]);

  // average detrended value per phase (day-of-week index)
  const phaseSums = new Array(period).fill(0);
  const phaseCounts = new Array(period).fill(0);
  detrended.forEach((v, i) => {
    phaseSums[i % period] += v;
    phaseCounts[i % period] += 1;
  });
  const phaseMeans = phaseSums.map((s, i) => (phaseCounts[i] ? s / phaseCounts[i] : 0));
  const meanOfMeans = phaseMeans.reduce((a, b) => a + b, 0) / period;
  const normalizedSeasonal = phaseMeans.map((m) => m - meanOfMeans); // sum ~ 0

  const seasonal = values.map((_, i) => normalizedSeasonal[i % period]);
  const residual = values.map((v, i) => v - trend[i] - seasonal[i]);
  return { trend, seasonal, residual };
}

export interface ScoredPoint {
  date: string;
  value: number;
  expected: number;
  zscore: number;
  /** anomaly score = |robust z|; 0 during the warm-up period */
  score: number;
  direction: "drop" | "spike";
  deviationPct: number;
}

/**
 * Per-point anomaly *scores* against a seasonal-naive baseline: each day is
 * compared to the median of the same weekday over the prior `lookback` weeks,
 * and the residual is turned into a robust median/MAD z-score.
 *
 * Resistant to *sustained* level shifts a centered-MA decomposition would
 * absorb into its own trend. Returns a continuous score per point (needed for
 * threshold-free metrics like VUS-PR); thresholding is a separate step.
 */
export function scoreAnomalies(points: Point[], period = 7, lookback = 4): ScoredPoint[] {
  const n = points.length;
  const vals = points.map((p) => p.value);

  const expected: (number | null)[] = new Array(n).fill(null);
  for (let i = 0; i < n; i++) {
    const refs: number[] = [];
    for (let k = 1; k <= lookback; k++) {
      const j = i - k * period;
      if (j >= 0) refs.push(vals[j]);
    }
    if (refs.length >= 2) expected[i] = median(refs);
  }

  const resid = expected.map((e, i) => (e === null ? null : vals[i] - e));
  const present = resid.filter((r): r is number => r !== null);
  const med = present.length ? median(present) : 0;
  const mad = (present.length ? median(present.map((r) => Math.abs(r - med))) : 0) || 1e-9;
  const robustSigma = 1.4826 * mad;

  return points.map((p, i) => {
    const e = expected[i];
    const r = resid[i];
    if (e === null || r === null) {
      return { date: p.date, value: Math.round(p.value), expected: Math.round(p.value), zscore: 0, score: 0, direction: "spike", deviationPct: 0 };
    }
    const z = (r - med) / robustSigma;
    return {
      date: p.date,
      value: Math.round(p.value),
      expected: Math.round(e),
      zscore: +z.toFixed(2),
      score: +Math.abs(z).toFixed(3),
      direction: z < 0 ? "drop" : "spike",
      deviationPct: e ? +(((p.value - e) / e) * 100).toFixed(1) : 0,
    };
  });
}

export interface DetectOpts {
  /** thresholding rule: "evt" = self-calibrating POT, "mad" = fixed |z| > k */
  method?: "evt" | "mad";
  /** fixed cutoff for method="mad" */
  k?: number;
  /** target false-alarm rate for method="evt" */
  q?: number;
  period?: number;
  lookback?: number;
}

/**
 * Statistical anomaly detector. Scores every point against the seasonal-naive
 * baseline, then thresholds either with a fixed robust-z cutoff ("mad") or with
 * a self-calibrating EVT/POT threshold ("evt", the default — see lib/evt.ts).
 */
export function detectAnomalies(points: Point[], opts: DetectOpts = {}): Anomaly[] {
  const { method = "evt", k = 3.0, q = 0.02, period = 7, lookback = 4 } = opts;
  if (points.length < period * 2) return [];
  const scored = scoreAnomalies(points, period, lookback);

  // warm-up points (expected===value, score 0) are excluded from thresholding
  const active = scored.filter((s, i) => i >= period * 2 && s.score > 0);
  if (active.length < period) return [];

  let cutoff: number;
  if (method === "evt") {
    cutoff = potThreshold(active.map((s) => s.score), 0.92, q).threshold;
  } else {
    cutoff = k;
  }

  const out: Anomaly[] = [];
  scored.forEach((s, i) => {
    if (i < period * 2 || s.score < cutoff) return;
    out.push({
      date: s.date,
      value: s.value,
      expected: s.expected,
      zscore: s.zscore,
      deviationPct: s.deviationPct,
      direction: s.direction,
      severity: s.score > 6 ? "high" : s.score > 4 ? "medium" : "low",
    });
  });
  return out;
}

/**
 * Detector for a series whose value is *already* an anomaly score (e.g. the SMD
 * entity health score = max robust-z across channels). There is no seasonal
 * baseline to subtract — anomalies are simply the upper tail. We set the alert
 * threshold with the same self-calibrating EVT/POT rule used elsewhere (SPOT),
 * then flag every point above it.
 */
export function detectScoreAnomalies(points: Point[], q = 0.02): Anomaly[] {
  const n = points.length;
  if (n < 50) return [];
  const scores = points.map((p) => p.value);
  const baseline = median(scores);
  const positive = scores.filter((s) => s > 0);
  if (positive.length < 20) return [];

  const cutoff = potThreshold(positive, 0.92, q).threshold;
  const out: Anomaly[] = [];
  points.forEach((p) => {
    const s = p.value;
    if (s < cutoff) return;
    out.push({
      date: p.date,
      value: +s.toFixed(2),
      expected: +baseline.toFixed(2),
      zscore: +s.toFixed(2), // the score is itself a robust z
      deviationPct: baseline ? +(((s - baseline) / baseline) * 100).toFixed(1) : 0,
      direction: "spike",
      severity: s > 6 ? "high" : s > 4 ? "medium" : "low",
    });
  });
  return out;
}

function addDays(date: string, n: number): string {
  // SMD series are indexed by integer timestep, not calendar date — advance the
  // index directly. Calendar series (ISO dates) advance by days.
  if (/^\d+$/.test(date)) return String(Number(date) + n);
  const d = new Date(date + "T00:00:00Z");
  d.setUTCDate(d.getUTCDate() + n);
  return d.toISOString().slice(0, 10);
}

/**
 * Holt-Winters additive forecast (level + trend + weekly seasonality).
 * Returns `horizon` future points with a ±z·σ prediction band.
 */
export function forecast(
  points: Point[],
  horizon = 14,
  period = 7,
  opts: { alpha?: number; beta?: number; gamma?: number } = {},
): ForecastPoint[] {
  const { alpha = 0.4, beta = 0.08, gamma = 0.3 } = opts;
  const values = points.map((p) => p.value);
  if (values.length < period * 2) return [];

  // seasonal init from first two periods
  const seasonals: number[] = new Array(period).fill(0);
  const firstAvg = values.slice(0, period).reduce((a, b) => a + b, 0) / period;
  for (let i = 0; i < period; i++) seasonals[i] = values[i] - firstAvg;

  let level = firstAvg;
  let trend = (values.slice(period, period * 2).reduce((a, b) => a + b, 0) / period - firstAvg) / period;

  const residuals: number[] = [];
  for (let i = 0; i < values.length; i++) {
    const s = seasonals[i % period];
    const predicted = level + trend + s;
    residuals.push(values[i] - predicted);
    const lastLevel = level;
    level = alpha * (values[i] - s) + (1 - alpha) * (level + trend);
    trend = beta * (level - lastLevel) + (1 - beta) * trend;
    seasonals[i % period] = gamma * (values[i] - level) + (1 - gamma) * s;
  }

  // residual std for the prediction band
  const resMean = residuals.reduce((a, b) => a + b, 0) / residuals.length;
  const resStd = Math.sqrt(
    residuals.reduce((a, b) => a + (b - resMean) ** 2, 0) / Math.max(1, residuals.length - 1),
  );

  const lastDate = points[points.length - 1].date;
  const n = values.length;
  const out: ForecastPoint[] = [];
  for (let h = 1; h <= horizon; h++) {
    const s = seasonals[(n + h - 1) % period];
    const yhat = level + h * trend + s;
    const band = 1.96 * resStd * Math.sqrt(h); // widening band with horizon
    out.push({
      date: addDays(lastDate, h),
      forecast: Math.round(yhat),
      lower: Math.round(Math.max(0, yhat - band)),
      upper: Math.round(yhat + band),
    });
  }
  return out;
}

/** Simple linear-regression slope as a % of mean — a quick trend read. */
export function trendSlopePct(points: Point[]): number {
  const n = points.length;
  if (n < 2) return 0;
  const xs = points.map((_, i) => i);
  const ys = points.map((p) => p.value);
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    num += (xs[i] - mx) * (ys[i] - my);
    den += (xs[i] - mx) ** 2;
  }
  const slope = den ? num / den : 0;
  return my ? +((slope / my) * 100).toFixed(2) : 0;
}
