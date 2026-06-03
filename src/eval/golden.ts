/**
 * Golden dataset for evaluating the agent fleet.
 *
 * Each case pins a natural-language question to the metric/scope the
 * orchestrator should resolve and asserts on properties of the resulting
 * analysis (anomaly presence, root-cause deploy attribution, forecast shape).
 */
import type { Metric } from "@/agents/types";

export interface GoldenCase {
  id: string;
  question: string;
  expect: {
    metric: Metric;
    region?: string;
    platform?: string;
    feature?: string;
    /** at least one anomaly expected in the series */
    hasAnomaly?: boolean;
    /** root cause should cite a deploy whose version contains this string */
    rootCauseVersionIncludes?: string;
  };
}

export const GOLDEN: GoldenCase[] = [
  {
    id: "global-dau-health",
    question: "Give me a health check on global DAU.",
    expect: { metric: "DAU", hasAnomaly: true },
  },
  {
    id: "eu-mobile-drop",
    question: "Why did DAU drop for EU mobile users?",
    expect: {
      metric: "DAU",
      region: "EU",
      platform: "mobile",
      hasAnomaly: true,
      rootCauseVersionIncludes: "v4.2",
    },
  },
  {
    id: "wau-trend",
    question: "What is the WAU trend and 14-day forecast?",
    expect: { metric: "WAU" },
  },
  {
    id: "ai-assistant-usage",
    question: "How is ai_assistant feature usage trending?",
    expect: { metric: "DAU", feature: "ai_assistant" },
  },
  {
    id: "export-decay",
    question: "Is the export feature losing users?",
    expect: { metric: "DAU", feature: "export" },
  },
];
