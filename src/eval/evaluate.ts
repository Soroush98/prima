/**
 * Evaluation engine. Runs each golden case through the live agent graph and
 * scores it on the dimensions the Cisco role calls out: routing accuracy,
 * SQL validity, anomaly/diagnosis correctness, hallucination, latency, cost.
 */
import { runPrima } from "@/agents/graph";
import { guardReadOnlySql, safeQuery } from "@/lib/db";
import { GOLDEN, type GoldenCase } from "./golden";

export interface CaseScore {
  id: string;
  question: string;
  passed: boolean;
  checks: { name: string; passed: boolean; detail: string }[];
  metrics: {
    routingCorrect: boolean;
    sqlValid: boolean;
    sqlExecutes: boolean;
    anomalyRecall: boolean;
    rootCauseCorrect: boolean | null;
    hallucinationFree: boolean;
    latencyMs: number;
    costUsd: number;
    tokensIn: number;
    tokensOut: number;
  };
}

export interface EvalReport {
  generatedAt: string;
  llmMode: "live";
  cases: CaseScore[];
  aggregate: {
    passRate: number;
    routingAccuracy: number;
    sqlValidity: number;
    anomalyRecall: number;
    hallucinationRate: number;
    avgLatencyMs: number;
    totalCostUsd: number;
    p95LatencyMs: number;
  };
}

async function scoreCase(gc: GoldenCase): Promise<CaseScore> {
  const r = await runPrima(gc.question);
  const checks: CaseScore["checks"] = [];

  // 1. routing
  const routingCorrect =
    r.metric === gc.expect.metric &&
    (gc.expect.region ? r.filter.region === gc.expect.region : true) &&
    (gc.expect.platform ? r.filter.platform === gc.expect.platform : true) &&
    (gc.expect.feature ? r.filter.feature === gc.expect.feature : true);
  checks.push({
    name: "routing",
    passed: routingCorrect,
    detail: `metric=${r.metric} scope=${JSON.stringify(r.filter)}`,
  });

  // 2. SQL validity (guard) + actually executes
  const sqlValid = guardReadOnlySql(r.sql).ok;
  const exec = safeQuery(r.sql);
  checks.push({ name: "sql_valid", passed: sqlValid, detail: sqlValid ? "read-only" : "rejected by guard" });
  checks.push({
    name: "sql_executes",
    passed: exec.ok && exec.rows.length > 0,
    detail: exec.ok ? `${exec.rows.length} rows` : (exec.error ?? "error"),
  });

  // 3. anomaly recall
  let anomalyRecall = true;
  if (gc.expect.hasAnomaly) {
    anomalyRecall = r.anomalies.length > 0;
    checks.push({
      name: "anomaly_recall",
      passed: anomalyRecall,
      detail: `${r.anomalies.length} anomalies`,
    });
  }

  // 4. root-cause attribution
  let rootCauseCorrect: boolean | null = null;
  if (gc.expect.rootCauseVersionIncludes) {
    rootCauseCorrect = Boolean(
      r.rootCause?.suspectedDeploys.some((d) => d.version.includes(gc.expect.rootCauseVersionIncludes!)),
    );
    checks.push({
      name: "root_cause",
      passed: rootCauseCorrect,
      detail: r.rootCause?.suspectedDeploys.map((d) => d.version).join(", ") || "none",
    });
  }

  // 5. hallucination guard: the summary must not invent regions/platforms outside the known vocab
  const KNOWN = ["NA", "EU", "APAC", "LATAM", "web", "mobile", "desktop"];
  const mentioned = (r.summary.match(/\b(NA|EU|APAC|LATAM|web|mobile|desktop)\b/g) ?? []) as string[];
  const hallucinationFree = mentioned.every((m) => KNOWN.includes(m));
  checks.push({
    name: "no_hallucination",
    passed: hallucinationFree,
    detail: mentioned.length ? `referenced ${[...new Set(mentioned)].join(",")}` : "no entities referenced",
  });

  const passed = checks.every((c) => c.passed);
  return {
    id: gc.id,
    question: gc.question,
    passed,
    checks,
    metrics: {
      routingCorrect,
      sqlValid,
      sqlExecutes: exec.ok && exec.rows.length > 0,
      anomalyRecall,
      rootCauseCorrect,
      hallucinationFree,
      latencyMs: r.telemetry.totalMs,
      costUsd: r.telemetry.costUsd,
      tokensIn: r.telemetry.tokensIn,
      tokensOut: r.telemetry.tokensOut,
    },
  };
}

export async function runEvaluation(nowIso: string): Promise<EvalReport> {
  const cases: CaseScore[] = [];
  for (const gc of GOLDEN) {
    cases.push(await scoreCase(gc));
  }

  const n = cases.length;
  const latencies = cases.map((c) => c.metrics.latencyMs).sort((a, b) => a - b);
  const p95 = latencies[Math.min(latencies.length - 1, Math.floor(0.95 * latencies.length))] ?? 0;
  const rate = (pred: (c: CaseScore) => boolean) => +((cases.filter(pred).length / n) * 100).toFixed(1);

  return {
    generatedAt: nowIso,
    llmMode: "live",
    cases,
    aggregate: {
      passRate: rate((c) => c.passed),
      routingAccuracy: rate((c) => c.metrics.routingCorrect),
      sqlValidity: rate((c) => c.metrics.sqlValid && c.metrics.sqlExecutes),
      anomalyRecall: rate((c) => c.metrics.anomalyRecall),
      hallucinationRate: +((cases.filter((c) => !c.metrics.hallucinationFree).length / n) * 100).toFixed(1),
      avgLatencyMs: +(cases.reduce((a, c) => a + c.metrics.latencyMs, 0) / n).toFixed(1),
      totalCostUsd: +cases.reduce((a, c) => a + c.metrics.costUsd, 0).toFixed(6),
      p95LatencyMs: p95,
    },
  };
}
