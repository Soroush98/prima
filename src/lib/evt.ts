/**
 * Extreme Value Theory thresholding — the POT (Peaks-Over-Threshold) core of
 * SPOT (Siffer et al., KDD 2017).
 *
 * Instead of a hand-picked `k` (e.g. "flag if z > 3"), POT models the *tail* of
 * the score distribution with a Generalized Pareto Distribution (GPD) and
 * derives the threshold that corresponds to a desired false-alarm probability
 * `q`. This self-calibrates per series: a noisy metric gets a higher bar, a
 * clean one a lower bar, all from one interpretable risk knob.
 */

export interface PotResult {
  /** the calibrated anomaly threshold z_q */
  threshold: number;
  /** GPD shape (ξ) and scale (σ) fitted to the tail */
  xi: number;
  sigma: number;
  /** the initial high quantile `t` whose exceedances form the tail */
  t: number;
  nPeaks: number;
  fallback: boolean;
}

function quantile(sortedAsc: number[], p: number): number {
  if (sortedAsc.length === 0) return 0;
  const idx = Math.min(sortedAsc.length - 1, Math.max(0, Math.floor(p * (sortedAsc.length - 1))));
  return sortedAsc[idx];
}

/**
 * Compute a POT/EVT threshold for a 1-D score array.
 *
 * @param scores       per-point anomaly scores (e.g. |robust z| or recon error)
 * @param initQuantile high quantile defining the tail to fit (default 0.92)
 * @param q            target tail probability / false-alarm rate (default 1e-3)
 */
export function potThreshold(scores: number[], initQuantile = 0.92, q = 0.02): PotResult {
  const n = scores.length;
  const sorted = [...scores].sort((a, b) => a - b);
  const t = quantile(sorted, initQuantile);
  const peaks = scores.filter((s) => s > t).map((s) => s - t);
  const Nt = peaks.length;

  // not enough tail samples to fit a GPD → fall back to an empirical high quantile
  if (Nt < 10) {
    return { threshold: quantile(sorted, 1 - q), xi: 0, sigma: 0, t, nPeaks: Nt, fallback: true };
  }

  const mean = peaks.reduce((a, b) => a + b, 0) / Nt;
  const variance = peaks.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, Nt - 1);

  // Method-of-moments GPD fit: ξ = ½(1 − m²/v),  σ = m(1 − ξ)
  const xi = 0.5 * (1 - (mean * mean) / (variance || 1e-9));
  const sigma = mean * (1 - xi);
  if (!isFinite(xi) || !isFinite(sigma) || sigma <= 0) {
    return { threshold: quantile(sorted, 1 - q), xi: 0, sigma: 0, t, nPeaks: Nt, fallback: true };
  }

  // z_q = t + (σ/ξ)·[ (q·n / N_t)^(−ξ) − 1 ],  with the ξ→0 (Gumbel) limit
  const ratio = (q * n) / Nt;
  let zq: number;
  if (Math.abs(xi) < 1e-6) {
    zq = t - sigma * Math.log(ratio);
  } else {
    zq = t + (sigma / xi) * (Math.pow(ratio, -xi) - 1);
  }
  if (!isFinite(zq) || zq < t) {
    return { threshold: quantile(sorted, 1 - q), xi, sigma, t, nPeaks: Nt, fallback: true };
  }
  return { threshold: zq, xi: +xi.toFixed(4), sigma: +sigma.toFixed(4), t, nPeaks: Nt, fallback: false };
}
