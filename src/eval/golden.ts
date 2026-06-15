/**
 * Golden dataset for evaluating the agent fleet against the real SMD warehouse.
 *
 * Each case pins a natural-language question to the entity the orchestrator
 * should resolve and asserts on properties of the resulting analysis (anomaly
 * presence, and whether root-cause attribution lands on a ground-truth-labeled
 * segment from SMD interpretation_label).
 */
import type { Metric } from "@/agents/types";

export interface GoldenCase {
  id: string;
  question: string;
  expect: {
    metric: Metric;
    entity?: string;
    /** at least one anomaly expected in the series */
    hasAnomaly?: boolean;
    /** the worst anomaly should fall in a labeled segment (root cause is gradeable) */
    rootCauseGraded?: boolean;
  };
}

export const GOLDEN: GoldenCase[] = [
  {
    id: "m1-1-health",
    question: "Give me a health check on machine-1-1.",
    expect: { metric: "health", entity: "machine-1-1", hasAnomaly: true, rootCauseGraded: true },
  },
  {
    id: "m2-1-rootcause",
    question: "What is the worst anomaly on machine-2-1 and its root cause?",
    expect: { metric: "health", entity: "machine-2-1", hasAnomaly: true },
  },
  {
    id: "m1-6-diagnose",
    question: "Diagnose the anomalies on machine-1-6.",
    expect: { metric: "health", entity: "machine-1-6", hasAnomaly: true },
  },
  {
    id: "m3-7-trend",
    question: "Is machine-3-7 trending toward instability?",
    expect: { metric: "health", entity: "machine-3-7" },
  },
  {
    id: "m3-11-forecast",
    question: "Forecast the health score for machine-3-11.",
    expect: { metric: "health", entity: "machine-3-11" },
  },
];
