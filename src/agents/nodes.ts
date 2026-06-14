import { safeQuery, guardReadOnlySql, getDimensionVocab } from "@/lib/db";
import { callLLM, estimateCostUsd } from "@/lib/llm";
import { detectAnomalies, forecast as runForecast, trendSlopePct, type Point } from "@/lib/stats";
import { forecastTSFM } from "@/lib/mlClient";
import type { PrimaStateType } from "./state";
import type { RootCause, TraceEntry } from "./types";
import { buildMetricSql, filterLabel, parseIntent, SCHEMA_DOC } from "./util";

type Update = Partial<PrimaStateType>;

/* ------------------------------------------------------------------ */
/* 1. Orchestrator — classifies the question into metric + dimensions  */
/* ------------------------------------------------------------------ */
export async function orchestrator(state: PrimaStateType): Promise<Update> {
  const start = performance.now();
  const { metric, filter } = parseIntent(state.question, getDimensionVocab());
  const trace: TraceEntry = {
    agent: "orchestrator",
    status: "ok",
    ms: +(performance.now() - start).toFixed(1),
    detail: `Routed to ${metric} analysis, scope: ${filterLabel(filter)}.`,
    data: { metric, filter },
  };
  return { metric, filter, trace: [trace] };
}

/* ------------------------------------------------------------------ */
/* 2. SQL Analyst — autonomously writes & executes SQL for the metric  */
/* ------------------------------------------------------------------ */
export async function sqlAnalyst(state: PrimaStateType): Promise<Update> {
  const start = performance.now();
  const fallback = buildMetricSql(state.metric, state.filter);

  const llm = await callLLM(
    `You are a senior analytics engineer. Write ONE read-only SQLite query (SELECT/WITH only) that returns columns exactly named "date" and "value", one row per day ordered ascending.\n\n${SCHEMA_DOC}\n\nReturn ONLY the SQL, no prose, no markdown fences.`,
    `Question: ${state.question}\nMetric: ${state.metric}\nFilter: ${JSON.stringify(state.filter)}`,
  );

  // Sanitize any markdown fences the model may add, then guard.
  let sql = llm.text.replace(/```sql|```/gi, "").trim();
  let params: unknown[] = [];
  let usedFallback = false;
  let fallbackReason: string | undefined;

  const guard = guardReadOnlySql(sql);
  if (!guard.ok) {
    sql = fallback.sql;
    params = fallback.params;
    usedFallback = true;
    fallbackReason = `guard rejected LLM SQL: ${guard.reason}`;
  }

  let result = safeQuery(sql, params);
  if (!result.ok) {
    fallbackReason = usedFallback
      ? `fallback query failed: ${result.error}`
      : `LLM SQL failed: ${result.error}`;
    sql = fallback.sql;
    params = fallback.params;
    usedFallback = true;
    result = safeQuery(sql, params);
  }

  const series: Point[] = result.rows.map((r) => ({
    date: String(r.date),
    value: Number(r.value) || 0,
  }));

  const costUsd = estimateCostUsd(llm.model, llm.tokensIn, llm.tokensOut);
  const trace: TraceEntry = {
    agent: "sql_analyst",
    status: result.ok ? "ok" : "error",
    ms: +(performance.now() - start).toFixed(1),
    mode: llm.mode,
    detail: usedFallback
      ? `Used validated template SQL (${series.length} days, ${result.ms}ms query).`
      : `Generated & executed SQL (${series.length} days, ${result.ms}ms query).`,
    data: { sql, rows: series.length, queryMs: result.ms, fallbackReason, error: result.error },
  };

  return {
    sql,
    series,
    trace: [trace],
    tokensIn: llm.tokensIn,
    tokensOut: llm.tokensOut,
    costUsd,
  };
}

/* ------------------------------------------------------------------ */
/* 3. Anomaly Detector — seasonal-naive z-score with EVT/POT threshold  */
/* ------------------------------------------------------------------ */
export async function anomalyDetector(state: PrimaStateType): Promise<Update> {
  const start = performance.now();

  // Robust seasonal-naive z-score with a self-calibrating EVT/POT threshold
  // (SPOT). For Prima's single daily KPI this matches or beats the deep Donut
  // detector (see ml-service benchmarks), so the agent runs statistical-only —
  // no ensemble, no agreement vote. The Donut VAE remains as a benchmark.
  const anomalies = detectAnomalies(state.series, { method: "evt" });

  const worst = [...anomalies].sort((x, y) => Math.abs(y.zscore) - Math.abs(x.zscore))[0];
  const trace: TraceEntry = {
    agent: "anomaly_detector",
    status: anomalies.length ? "warn" : "ok",
    ms: +(performance.now() - start).toFixed(1),
    detail: anomalies.length
      ? `EVT flagged ${anomalies.length}; worst ${worst.direction} ${worst.deviationPct}% on ${worst.date} (z=${worst.zscore}).`
      : "No statistically significant anomalies detected.",
    data: { count: anomalies.length },
  };
  return { anomalies, trace: [trace] };
}

/* ------------------------------------------------------------------ */
/* 4. Forecaster — Chronos-Bolt (zero-shot TSFM) → Holt-Winters fallback */
/* ------------------------------------------------------------------ */
export async function forecaster(state: PrimaStateType): Promise<Update> {
  const start = performance.now();

  // Prefer the zero-shot foundation model; fall back to Holt-Winters offline.
  const tsfm = await forecastTSFM(state.series, 14);
  const forecast = tsfm ? tsfm.points : runForecast(state.series, 14);
  const model = tsfm ? `${tsfm.model} (zero-shot)` : "holt-winters";

  const slope = trendSlopePct(state.series.slice(-28));
  const last = state.series[state.series.length - 1]?.value ?? 0;
  const projected = forecast[forecast.length - 1]?.forecast ?? last;
  const deltaPct = last ? +(((projected - last) / last) * 100).toFixed(1) : 0;
  const trace: TraceEntry = {
    agent: "forecaster",
    status: "ok",
    ms: +(performance.now() - start).toFixed(1),
    detail: `[${model}] 14-day projection ${deltaPct >= 0 ? "+" : ""}${deltaPct}% vs today; 28-day trend slope ${slope}%/day.`,
    data: { projected, deltaPct, slope, model },
  };
  return { forecast, trace: [trace] };
}

/* ------------------------------------------------------------------ */
/* 5. Root-Cause — correlates worst anomaly with deploys + segments    */
/* ------------------------------------------------------------------ */
export async function rootCause(state: PrimaStateType): Promise<Update> {
  const start = performance.now();
  const drops = state.anomalies.filter((a) => a.direction === "drop");
  if (drops.length === 0) {
    const trace: TraceEntry = {
      agent: "root_cause",
      status: "ok",
      ms: +(performance.now() - start).toFixed(1),
      detail: "No drops to diagnose.",
    };
    return { rootCause: null, trace: [trace] };
  }

  const worst = [...drops].sort((a, b) => a.zscore - b.zscore)[0];

  // deploys in the days leading up to the drop (a cause precedes its effect),
  // plus a small forward window for same-day rollouts.
  const deployRows = safeQuery(
    `SELECT deploy_date, version, service, COALESCE(notes,'') AS notes
       FROM deploys
      WHERE deploy_date BETWEEN date(?, '-7 day') AND date(?, '+1 day')
      ORDER BY deploy_date DESC`,
    [worst.date, worst.date],
  ).rows as { deploy_date: string; version: string; service: string; notes: string }[];

  // which region·platform segment moved most vs the prior week
  const segRows = safeQuery(
    `WITH win AS (
        SELECT region, platform, COUNT(DISTINCT user_id) AS cur
          FROM user_activity
         WHERE activity_date BETWEEN date(?, '-2 day') AND ?
         GROUP BY region, platform),
      base AS (
        SELECT region, platform, COUNT(DISTINCT user_id) AS prev
          FROM user_activity
         WHERE activity_date BETWEEN date(?, '-9 day') AND date(?, '-7 day')
         GROUP BY region, platform)
     SELECT win.region, win.platform, win.cur, base.prev,
            ROUND((win.cur - base.prev) * 100.0 / base.prev, 1) AS change_pct
       FROM win JOIN base ON win.region = base.region AND win.platform = base.platform
      ORDER BY change_pct ASC
      LIMIT 3`,
    [worst.date, worst.date, worst.date, worst.date],
  ).rows as { region: string; platform: string; change_pct: number }[];

  const worstSegments = segRows.map((s) => ({
    segment: `${s.region} · ${s.platform}`,
    changePct: s.change_pct,
  }));

  const factsForLlm = JSON.stringify({ anomaly: worst, deploys: deployRows, worstSegments }, null, 2);
  const llm = await callLLM(
    "You are an incident analyst. Given an anomaly, nearby deploys, and the worst-hit segments, produce 2-3 concise, evidence-grounded root-cause hypotheses. Return a plain numbered list, no preamble.",
    factsForLlm,
  );

  const hypotheses = llm.text
    .split(/\n+/)
    .map((l) => l.replace(/^\s*\d+[.)]\s*/, "").trim())
    .filter(Boolean);

  const rc: RootCause = {
    headline: `${worst.deviationPct}% ${worst.direction} on ${worst.date} — concentrated in ${
      worstSegments[0]?.segment ?? "no clear segment"
    }.`,
    suspectedDeploys: deployRows,
    worstSegments,
    hypotheses,
  };

  const costUsd = estimateCostUsd(llm.model, llm.tokensIn, llm.tokensOut);
  const trace: TraceEntry = {
    agent: "root_cause",
    status: "warn",
    ms: +(performance.now() - start).toFixed(1),
    mode: llm.mode,
    detail: `Diagnosed ${worst.date}: ${deployRows.length} candidate deploy(s), worst segment ${
      worstSegments[0]?.segment ?? "n/a"
    }.`,
    data: { worst: worst.date },
  };

  return {
    rootCause: rc,
    trace: [trace],
    tokensIn: llm.tokensIn,
    tokensOut: llm.tokensOut,
    costUsd,
  };
}

/* ------------------------------------------------------------------ */
/* 6. Narrator — executive summary synthesizing the whole run          */
/* ------------------------------------------------------------------ */
export async function narrator(state: PrimaStateType): Promise<Update> {
  const start = performance.now();
  const last = state.series[state.series.length - 1];
  const projected = state.forecast[state.forecast.length - 1];
  const deltaPct =
    last && projected ? +(((projected.forecast - last.value) / last.value) * 100).toFixed(1) : 0;

  const context = {
    metric: state.metric,
    scope: filterLabel(state.filter),
    latest: last?.value,
    projected14d: projected?.forecast,
    deltaPct,
    anomalies: state.anomalies.length,
    rootCause: state.rootCause,
  };

  const llm = await callLLM(
    "You are a product-analytics lead briefing executives. In 3-4 sentences, summarize the metric's current state, notable anomalies, the 14-day outlook, and a prescriptive recommendation. Be specific and quantitative. No markdown headers.",
    JSON.stringify(context, null, 2),
  );

  const costUsd = estimateCostUsd(llm.model, llm.tokensIn, llm.tokensOut);
  const trace: TraceEntry = {
    agent: "narrator",
    status: "ok",
    ms: +(performance.now() - start).toFixed(1),
    mode: llm.mode,
    detail: "Synthesized executive briefing.",
  };

  return {
    summary: llm.text.trim(),
    trace: [trace],
    tokensIn: llm.tokensIn,
    tokensOut: llm.tokensOut,
    costUsd,
  };
}
