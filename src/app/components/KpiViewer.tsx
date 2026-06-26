"use client";

import { useEffect, useMemo, useState } from "react";
import {
  ComposedChart,
  Line,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";

interface Kpi {
  id: string;
  caption: string;
  intervalSec: number;
  startTs: number;
  values: number[];
  anomalies: number[];
}

/**
 * Read-only viewer for the real 2018 AIOps Challenge KPI sample. These series
 * don't fit the DAU/WAU agent schema (per-minute value/label, no users/regions),
 * so we just plot the real signal with its ground-truth anomaly labels.
 */
export default function KpiViewer() {
  const [kpis, setKpis] = useState<Kpi[]>([]);
  const [idx, setIdx] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/kpi")
      .then((r) => r.json())
      .then((d) => (d.error ? setError(d.error) : setKpis(d.kpis ?? [])))
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  const kpi = kpis[Math.min(idx, Math.max(0, kpis.length - 1))];

  const rows = useMemo(() => {
    if (!kpi) return [];
    const anom = new Set(kpi.anomalies);
    return kpi.values.map((v, i) => ({ i, value: v, anomaly: anom.has(i) ? v : null }));
  }, [kpi]);

  if (error) {
    return (
      <div className="card" style={{ borderColor: "#5e1e1e" }}>
        <strong style={{ color: "var(--bad)" }}>Could not load AIOps sample:</strong> {error}
        <div className="muted" style={{ marginTop: 6 }}>
          Generate it with <code>ml-service/.venv/bin/python ml-service/make_aiops_sample.py</code>.
        </div>
      </div>
    );
  }
  if (!kpi) return <div className="card">Loading AIOps KPI sample…</div>;

  const pct = ((kpi.anomalies.length / kpi.values.length) * 100).toFixed(1);
  const mins = (kpi.intervalSec / 60).toFixed(kpi.intervalSec % 60 ? 1 : 0);

  return (
    <>
      <div className="ask">
        <select value={idx} onChange={(e) => setIdx(Number(e.target.value))} style={{ flex: 1 }}>
          {kpis.map((k, i) => (
            <option key={k.id} value={i}>
              {k.id.slice(0, 10)}… — {k.caption}
            </option>
          ))}
        </select>
      </div>

      <div className="grid cols-4" style={{ marginBottom: 16 }}>
        <div className="card">
          <h3>Sampling</h3>
          <div className="kpi">{mins}<small> min</small></div>
          <div className="muted" style={{ fontSize: 13 }}>{kpi.intervalSec}s interval</div>
        </div>
        <div className="card">
          <h3>Window</h3>
          <div className="kpi">{kpi.values.length.toLocaleString()}</div>
          <div className="muted" style={{ fontSize: 13 }}>points shown</div>
        </div>
        <div className="card">
          <h3>Labeled anomalies</h3>
          <div className="kpi" style={{ color: "var(--warn)" }}>{kpi.anomalies.length}</div>
          <div className="muted" style={{ fontSize: 13 }}>ground truth</div>
        </div>
        <div className="card">
          <h3>Anomaly rate</h3>
          <div className="kpi">{pct}<small>%</small></div>
          <div className="muted" style={{ fontSize: 13 }}>of window</div>
        </div>
      </div>

      <div className="card">
        <h3>{kpi.id.slice(0, 12)}… — {kpi.caption}</h3>
        <ResponsiveContainer width="100%" height={360}>
          <ComposedChart data={rows} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="#243049" strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="i" tick={{ fill: "#8b97b3", fontSize: 11 }} minTickGap={50} />
            <YAxis tick={{ fill: "#8b97b3", fontSize: 11 }} width={52} />
            <Tooltip
              contentStyle={{
                background: "#121829",
                border: "1px solid #243049",
                borderRadius: 8,
                color: "#e6ebf5",
                fontSize: 13,
              }}
            />
            <Legend wrapperStyle={{ fontSize: 12, color: "#8b97b3" }} />
            <Line type="monotone" dataKey="value" stroke="#4f9cf9" strokeWidth={1.4} dot={false} name="KPI value" connectNulls />
            <Scatter dataKey="anomaly" fill="#ff6b6b" name="labeled anomaly" />
          </ComposedChart>
        </ResponsiveContainer>
        <div className="muted" style={{ fontSize: 11, marginTop: 6 }}>
          Real 2018 AIOps Challenge KPI · ground-truth anomalies in red · read-only (the DAU/WAU agent
          doesn&apos;t run on this schema). Prima&apos;s deep detector is <b>OmniAnomaly</b> — see
          <code>ml-service/README.md</code>.
        </div>
      </div>
    </>
  );
}
