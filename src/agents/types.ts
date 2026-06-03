import type { Anomaly, ForecastPoint, Point } from "@/lib/stats";

export type Metric = "DAU" | "WAU";

export interface DimensionFilter {
  region?: string;
  platform?: string;
  feature?: string;
}

export interface TraceEntry {
  agent: string;
  status: "ok" | "warn" | "error";
  ms: number;
  mode?: "live";
  detail: string;
  data?: Record<string, unknown>;
}

export interface RootCause {
  headline: string;
  suspectedDeploys: { deploy_date: string; version: string; service: string; notes: string }[];
  worstSegments: { segment: string; changePct: number }[];
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
