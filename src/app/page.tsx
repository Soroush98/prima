"use client";

import { useEffect, useState } from "react";
import MetricChart from "./components/MetricChart";
import FeatureUsageChart from "./components/FeatureUsageChart";
import Architecture from "./components/Architecture";
import KpiViewer from "./components/KpiViewer";
import type { PrimaResult } from "@/agents/types";

interface DatasetOpt { id: string; label: string; description: string; kind: "agent" | "viewer" }

// Tuned for the real Online Retail dataset (`npm run seed:github`).
// For the synthetic dataset (`npm run seed`) try: "Why did DAU drop for EU mobile users?"
const EXAMPLES = [
  "Give me a health check on global DAU.",
  "How is DAU trending in the United Kingdom?",
  "Did DAU drop in Germany?",
  "What is the WAU trend and 14-day forecast?",
  "How is the seasonal category trending?",
];

interface EvalReport {
  llmMode: "live";
  aggregate: {
    passRate: number;
    routingAccuracy: number;
    sqlValidity: number;
    anomalyRecall: number;
    hallucinationRate: number;
    avgLatencyMs: number;
    p95LatencyMs: number;
    totalCostUsd: number;
  };
  cases: { id: string; passed: boolean; metrics: { latencyMs: number } }[];
}

export default function Home() {
  const [tab, setTab] = useState<"dashboard" | "architecture">("dashboard");
  const [question, setQuestion] = useState(EXAMPLES[1]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PrimaResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [features, setFeatures] = useState<{ feature: string; users: number; events: number }[]>([]);
  const [evalReport, setEvalReport] = useState<EvalReport | null>(null);
  const [datasets, setDatasets] = useState<DatasetOpt[]>([
    { id: "real", label: "Online Retail (real)", description: "", kind: "agent" },
    { id: "synthetic", label: "Synthetic storyline", description: "", kind: "agent" },
    { id: "aiops", label: "AIOps KPI (real)", description: "", kind: "viewer" },
  ]);
  const [dataset, setDataset] = useState<string>("real");

  async function analyze(q: string, ds: string = dataset) {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, dataset: ds }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Request failed");
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  function loadFeatures(ds: string) {
    fetch(`/api/feature-usage?dataset=${ds}`)
      .then((r) => r.json())
      .then((d) => setFeatures(d.features ?? []))
      .catch(() => setFeatures([]));
  }

  function onDataset(ds: string) {
    setDataset(ds);
    const kind = datasets.find((d) => d.id === ds)?.kind ?? "agent";
    if (kind === "agent") {
      setResult(null);
      analyze(question, ds);
      loadFeatures(ds);
    }
  }

  useEffect(() => {
    // Intentional: kick off the initial analysis + dashboard fetches once on mount.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    analyze(question);
    loadFeatures("real");
    fetch("/api/datasets")
      .then((r) => r.json())
      .then((d) => d.datasets && setDatasets(d.datasets))
      .catch(() => {});
    fetch("/api/eval")
      .then((r) => r.json())
      .then((d) => !d.error && setEvalReport(d))
      .catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const last = result?.series[result.series.length - 1];
  const proj = result?.forecast[result.forecast.length - 1];
  const deltaPct = last && proj ? ((proj.forecast - last.value) / last.value) * 100 : 0;
  const mode = result?.telemetry.llmMode ?? "live";
  const currentKind = datasets.find((d) => d.id === dataset)?.kind ?? "agent";
  const datasetDesc = datasets.find((d) => d.id === dataset)?.description ?? "";

  return (
    <div className="wrap">
      <div className="header">
        <div>
          <div className="title">
            Prima<span className="dot">.</span> <span className="muted" style={{ fontSize: 16 }}>Agentic DAU/WAU Intelligence</span>
          </div>
        </div>
        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          <select
            className="dataset-select"
            value={dataset}
            onChange={(e) => onDataset(e.target.value)}
            title="Dataset"
          >
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {d.label}
              </option>
            ))}
          </select>
          <span className={`badge ${mode}`}>● Live Claude agents</span>
        </div>
      </div>
      <div className="subtitle">
        A LangGraph agent fleet that autonomously writes SQL, detects anomalies, forecasts engagement, and
        diagnoses root cause — replacing static BI dashboards with prescriptive intelligence.
      </div>

      {/* Tabs */}
      <div className="tabs">
        <button className={`tab ${tab === "dashboard" ? "active" : ""}`} onClick={() => setTab("dashboard")}>
          Dashboard
        </button>
        <button className={`tab ${tab === "architecture" ? "active" : ""}`} onClick={() => setTab("architecture")}>
          Architecture
        </button>
      </div>

      {tab === "architecture" && <Architecture />}

      {tab === "dashboard" && currentKind === "viewer" && (
        <>
          {datasetDesc && <div className="subtitle" style={{ marginTop: 4 }}>{datasetDesc}</div>}
          <KpiViewer />
        </>
      )}

      {tab === "dashboard" && currentKind === "agent" && (
      <>
      {/* Ask bar */}
      <div className="ask">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && analyze(question)}
          placeholder="Ask about DAU / WAU, a region, platform, or feature…"
        />
        <button onClick={() => analyze(question)} disabled={loading}>
          {loading ? <span className="spinner" /> : "Analyze"}
        </button>
      </div>
      <div className="chips">
        {EXAMPLES.map((ex) => (
          <button
            type="button"
            key={ex}
            className="chip"
            onClick={() => {
              setQuestion(ex);
              analyze(ex);
            }}
          >
            {ex}
          </button>
        ))}
      </div>

      {error && (
        <div className="card" style={{ borderColor: "#5e1e1e", marginBottom: 16 }}>
          <strong style={{ color: "var(--bad)" }}>Error:</strong> {error}
          <div className="muted" style={{ marginTop: 6 }}>
            If the database is missing, run <code>npm run seed</code> first.
          </div>
        </div>
      )}

      {result && (
        <>
          {/* KPI row */}
          <div className="grid cols-4" style={{ marginBottom: 16 }}>
            <div className="card">
              <h3>{result.metric} · {label(result.filter)}</h3>
              <div className="kpi">{last?.value.toLocaleString()}</div>
              <div className="muted" style={{ fontSize: 13 }}>latest day</div>
            </div>
            <div className="card">
              <h3>14-day forecast</h3>
              <div className={`kpi ${deltaPct >= 0 ? "delta-up" : "delta-down"}`}>
                {deltaPct >= 0 ? "▲" : "▼"} {Math.abs(deltaPct).toFixed(1)}%
              </div>
              <div className="muted" style={{ fontSize: 13 }}>to {proj?.forecast.toLocaleString()}</div>
            </div>
            <div className="card">
              <h3>Anomalies</h3>
              <div className="kpi" style={{ color: result.anomalies.length ? "var(--warn)" : "var(--good)" }}>
                {result.anomalies.length}
              </div>
              <div className="muted" style={{ fontSize: 13 }}>flagged in window</div>
            </div>
            <div className="card">
              <h3>Run cost / latency</h3>
              <div className="kpi">
                {result.telemetry.totalMs}<small> ms</small>
              </div>
              <div className="muted" style={{ fontSize: 13 }}>
                ${result.telemetry.costUsd} · {result.telemetry.tokensOut} out-tok
              </div>
            </div>
          </div>

          {/* Chart + summary */}
          <div className="grid cols-2" style={{ marginBottom: 16, gridTemplateColumns: "1.6fr 1fr" }}>
            <div className="card">
              <h3>{result.metric} — actual, anomalies & {`{forecast}`}</h3>
              <MetricChart
                series={result.series}
                anomalies={result.anomalies}
                forecast={result.forecast}
                metric={result.metric}
              />
            </div>
            <div className="card">
              <h3>Executive briefing (Narrator agent)</h3>
              <p className="summary">{result.summary}</p>
              {result.rootCause && (
                <>
                  <h3 style={{ marginTop: 18 }}>Root cause</h3>
                  <div style={{ fontSize: 14, marginBottom: 10 }}>{result.rootCause.headline}</div>
                  {result.rootCause.suspectedDeploys.length > 0 && (
                    <div className="muted" style={{ fontSize: 13, marginBottom: 10 }}>
                      Suspect deploys:{" "}
                      {result.rootCause.suspectedDeploys.map((d) => `${d.version} (${d.deploy_date})`).join(", ")}
                    </div>
                  )}
                  <ul className="hyp">
                    {result.rootCause.hypotheses.map((h) => (
                      <li key={h}>{h}</li>
                    ))}
                  </ul>
                </>
              )}
            </div>
          </div>

          {/* Agent trace + SQL */}
          <div className="grid cols-2" style={{ marginBottom: 16 }}>
            <div className="card">
              <h3>Agent execution trace (LangGraph)</h3>
              <div className="trace">
                {result.trace.map((t) => (
                  <div key={t.agent} className={`trace-row ${t.status}`}>
                    <span className={`dot-status ${t.status}`} />
                    <span className="trace-agent">
                      {t.agent}
                      {t.mode && <span className="muted" style={{ fontWeight: 400 }}> · {t.mode}</span>}
                    </span>
                    <span className="trace-detail">{t.detail}</span>
                    <span className="trace-ms">{t.ms}ms</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="card">
              <h3>SQL the analyst agent executed</h3>
              <pre className="sql">{result.sql}</pre>
              {result.anomalies.length > 0 && (
                <div style={{ marginTop: 12 }}>
                  <h3>Top anomalies</h3>
                  {result.anomalies.slice(0, 5).map((a) => (
                    <div key={a.date} style={{ fontSize: 13, marginBottom: 4 }}>
                      <span className={`tag ${a.severity}`}>{a.severity}</span>
                      {a.date}: {a.direction} {a.deviationPct}% (z={a.zscore})
                    </div>
                  ))}
                  <div className="muted" style={{ fontSize: 11, marginTop: 6 }}>
                    Seasonal-naive z-score with a self-calibrating EVT/POT threshold (SPOT).
                  </div>
                </div>
              )}
            </div>
          </div>
        </>
      )}

      {/* Feature usage + Eval */}
      <div className="section-title">Fleet observability</div>
      <div className="grid cols-2">
        <div className="card">
          <h3>Feature usage — active users (last 30 days)</h3>
          {features.length ? (
            <FeatureUsageChart features={features} />
          ) : (
            <div className="muted">Loading…</div>
          )}
        </div>
        <div className="card">
          <h3>LLM evaluation scorecard {evalReport && <span className={`badge ${evalReport.llmMode}`}>{evalReport.llmMode}</span>}</h3>
          {evalReport ? (
            <div>
              <div className="eval-metric">
                <span className="muted">Pass rate</span>
                <span className="v" style={{ color: scoreColor(evalReport.aggregate.passRate) }}>
                  {evalReport.aggregate.passRate}%
                </span>
              </div>
              <div className="eval-metric">
                <span className="muted">Routing accuracy</span>
                <span className="v">{evalReport.aggregate.routingAccuracy}%</span>
              </div>
              <div className="eval-metric">
                <span className="muted">SQL validity & execution</span>
                <span className="v">{evalReport.aggregate.sqlValidity}%</span>
              </div>
              <div className="eval-metric">
                <span className="muted">Anomaly recall</span>
                <span className="v">{evalReport.aggregate.anomalyRecall}%</span>
              </div>
              <div className="eval-metric">
                <span className="muted">Hallucination rate</span>
                <span className="v" style={{ color: evalReport.aggregate.hallucinationRate ? "var(--bad)" : "var(--good)" }}>
                  {evalReport.aggregate.hallucinationRate}%
                </span>
              </div>
              <div className="eval-metric">
                <span className="muted">Avg latency (p95)</span>
                <span className="v">
                  {evalReport.aggregate.avgLatencyMs}ms ({evalReport.aggregate.p95LatencyMs}ms)
                </span>
              </div>
              <div className="eval-metric">
                <span className="muted">Total eval cost</span>
                <span className="v">${evalReport.aggregate.totalCostUsd}</span>
              </div>
              <div className="muted" style={{ fontSize: 12, marginTop: 10 }}>
                {evalReport.cases.length} golden cases · regenerate with <code>npm run eval</code>
              </div>
            </div>
          ) : (
            <div className="muted">
              No eval report yet. Run <code>npm run eval</code> to generate the scorecard.
            </div>
          )}
        </div>
      </div>
      </>
      )}
    </div>
  );
}

function label(filter: { region?: string; platform?: string; feature?: string }): string {
  const p = [filter.region, filter.platform, filter.feature].filter(Boolean);
  return p.length ? p.join(" · ") : "global";
}
function scoreColor(pct: number): string {
  if (pct >= 90) return "var(--good)";
  if (pct >= 70) return "var(--warn)";
  return "var(--bad)";
}
