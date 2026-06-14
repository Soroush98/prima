import type { ForecastPoint, Point } from "@/lib/stats";

/**
 * Client for the Python Donut-VAE anomaly service (ml-service/).
 *
 * The deep-learning detector runs as a separate Python/FastAPI microservice
 * (it owns PyTorch); the agent graph reaches it over HTTP. If the service is
 * not running the call returns null and the graph proceeds with the
 * statistical detector alone — the ML detector augments, never blocks.
 */

const ML_URL = process.env.ML_SERVICE_URL || "http://127.0.0.1:8000";

export interface MlAnomaly {
  date: string;
  index: number;
  value: number;
  score: number;
  zscore: number;
  severity: "low" | "medium" | "high";
}

export interface MlDetectResult {
  anomalies: MlAnomaly[];
  scores: number[];
  threshold: number;
  model: string;
  thresholding: string;
  trainMs: number;
  finalLoss: number;
}

export async function detectAnomaliesML(
  series: Point[],
  opts: { model?: "vae"; threshold?: "evt" | "mad"; window?: number; q?: number; epochs?: number } = {},
): Promise<MlDetectResult | null> {
  try {
    const res = await fetch(`${ML_URL}/detect`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        series,
        model: opts.model ?? "vae",
        threshold: opts.threshold ?? "evt",
        window: opts.window ?? 28, // ~4 weekly cycles — see benchmark window sweep
        q: opts.q ?? 0.02,
        epochs: opts.epochs ?? 150,
      }),
      // the first call trains the model (~1s); give it room
      signal: AbortSignal.timeout(20_000),
    });
    if (!res.ok) return null;
    const data = (await res.json()) as {
      anomalies?: MlAnomaly[];
      scores?: number[];
      threshold?: number;
      model?: string;
      thresholding?: string;
      train_ms?: number;
      final_loss?: number;
      error?: string;
    };
    if (data.error || !data.anomalies) return null;
    return {
      anomalies: data.anomalies,
      scores: data.scores ?? [],
      threshold: data.threshold ?? 0,
      model: data.model ?? "vae",
      thresholding: data.thresholding ?? "evt",
      trainMs: data.train_ms ?? 0,
      finalLoss: data.final_loss ?? 0,
    };
  } catch {
    return null; // service down / unreachable
  }
}

function addDays(date: string, n: number): string {
  const d = new Date(date + "T00:00:00Z");
  d.setUTCDate(d.getUTCDate() + n);
  return d.toISOString().slice(0, 10);
}

export interface TsfmForecast {
  points: ForecastPoint[];
  model: string;
  inferMs: number;
}

/**
 * Zero-shot forecast from the Chronos-Bolt foundation model in the ML service.
 * Returns null if the service is offline → caller falls back to Holt-Winters.
 */
export async function forecastTSFM(series: Point[], horizon = 14): Promise<TsfmForecast | null> {
  try {
    const res = await fetch(`${ML_URL}/forecast`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ series, horizon }),
      signal: AbortSignal.timeout(30_000), // first call may load the model (~8s)
    });
    if (!res.ok) return null;
    const data = (await res.json()) as {
      lower?: number[];
      median?: number[];
      upper?: number[];
      model?: string;
      infer_ms?: number;
      error?: string;
    };
    if (data.error || !data.median) return null;
    const lastDate = series[series.length - 1]?.date;
    if (!lastDate) return null;
    const points: ForecastPoint[] = data.median.map((m, i) => ({
      date: addDays(lastDate, i + 1),
      forecast: Math.round(m),
      lower: Math.round(data.lower?.[i] ?? m),
      upper: Math.round(data.upper?.[i] ?? m),
    }));
    return { points, model: data.model ?? "chronos-bolt", inferMs: data.infer_ms ?? 0 };
  } catch {
    return null;
  }
}

export async function mlServiceHealthy(): Promise<boolean> {
  try {
    const res = await fetch(`${ML_URL}/health`, { signal: AbortSignal.timeout(1500) });
    return res.ok;
  } catch {
    return false;
  }
}
