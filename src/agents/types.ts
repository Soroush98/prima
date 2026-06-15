import type { Anomaly, ForecastPoint, Point } from "@/lib/stats";

/**
 * The series the fleet analyzes. For SMD this is the per-timestep entity
 * "health" anomaly score (max robust-z across the 38 metric channels).
 */
export type Metric = "health";

export interface DimensionFilter {
  /** SMD entity under analysis, e.g. "machine-1-1". */
  entity?: string;
}

export interface TraceEntry {
  agent: string;
  status: "ok" | "warn" | "error";
  ms: number;
  mode?: "live";
  detail: string;
  data?: Record<string, unknown>;
}

/** A channel the agent attributes an anomaly to, with its deviation at that timestep. */
export interface ChannelAttribution {
  channel: string;
  /** robust z of this channel at the anomaly timestep */
  deviation: number;
}

export interface RootCause {
  headline: string;
  /** timestep of the worst anomaly diagnosed */
  timestep: number;
  /** channels the agent ranked as most responsible (its attribution) */
  topChannels: ChannelAttribution[];
  /** ground-truth culprit channels from SMD interpretation_label (if this t is labeled) */
  groundTruthChannels: string[];
  /** how well the agent's attribution matched ground truth (null if t isn't a labeled segment) */
  grade: { precision: number; recall: number; overlap: string[] } | null;
  hypotheses: string[];
}

export interface PrimaResult {
  question: string;
  metric: Metric;
  filter: DimensionFilter;
  sql: string;
  series: Point[];
  anomalies: Anomaly[];
  forecast: ForecastPoint[];
  rootCause: RootCause | null;
  summary: string;
  trace: TraceEntry[];
  telemetry: {
    totalMs: number;
    tokensIn: number;
    tokensOut: number;
    costUsd: number;
    llmMode: "live";
  };
}
