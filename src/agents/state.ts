import { Annotation } from "@langchain/langgraph";
import type { Anomaly, ForecastPoint, Point } from "@/lib/stats";
import type { DimensionFilter, Metric, RootCause, TraceEntry } from "./types";

/**
 * Shared LangGraph state. Every agent node reads what it needs and returns a
 * partial update; `trace` and token counters accumulate via reducers so the
 * observability layer sees the full run.
 */
export const PrimaState = Annotation.Root({
  question: Annotation<string>,
  metric: Annotation<Metric>,
  filter: Annotation<DimensionFilter>,
  sql: Annotation<string>,
  series: Annotation<Point[]>,
  anomalies: Annotation<Anomaly[]>,
  forecast: Annotation<ForecastPoint[]>,
  rootCause: Annotation<RootCause | null>,
  summary: Annotation<string>,
  trace: Annotation<TraceEntry[]>({
    reducer: (a, b) => a.concat(b),
    default: () => [],
  }),
  tokensIn: Annotation<number>({ reducer: (a, b) => a + b, default: () => 0 }),
  tokensOut: Annotation<number>({ reducer: (a, b) => a + b, default: () => 0 }),
  costUsd: Annotation<number>({ reducer: (a, b) => a + b, default: () => 0 }),
});

export type PrimaStateType = typeof PrimaState.State;
