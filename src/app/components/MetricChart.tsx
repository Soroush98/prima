"use client";

import {
  ComposedChart,
  Area,
  Line,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import type { Anomaly, ForecastPoint, Point } from "@/lib/stats";

interface Props {
  series: Point[];
  anomalies: Anomaly[];
  forecast: ForecastPoint[];
  metric: string;
}

export default function MetricChart({ series, anomalies, forecast, metric }: Props) {
  const anomalyByDate = new Map(anomalies.map((a) => [a.date, a]));
  // show the trailing ~90 days for readability
  const tail = series.slice(-90);

  const rows: Record<string, number | string | null>[] = tail.map((p) => ({
    date: p.date,
    actual: p.value,
    anomaly: anomalyByDate.has(p.date) ? p.value : null,
    forecast: null,
    band: null,
    lower: null,
  }));

  const lastActual = tail[tail.length - 1];
  if (lastActual && forecast.length) {
    // anchor the forecast line to the last actual point so it connects
    rows[rows.length - 1].forecast = lastActual.value;
    for (const f of forecast) {
      rows.push({
        date: f.date,
        actual: null,
        anomaly: null,
        forecast: f.forecast,
        lower: f.lower,
        band: f.upper - f.lower,
      });
    }
  }

  return (
    <ResponsiveContainer width="100%" height={320}>
      <ComposedChart data={rows} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
        <CartesianGrid stroke="#243049" strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey="date"
          tick={{ fill: "#8b97b3", fontSize: 11 }}
          tickFormatter={(d: string) => d.slice(5)}
          minTickGap={40}
        />
        <YAxis tick={{ fill: "#8b97b3", fontSize: 11 }} width={48} />
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
        {/* prediction band: invisible base + visible upper-minus-lower */}
        <Area dataKey="lower" stackId="band" stroke="none" fill="transparent" legendType="none" name="lower" />
        <Area
          dataKey="band"
          stackId="band"
          stroke="none"
          fill="#7c5cff"
          fillOpacity={0.15}
          name="forecast band"
        />
        <Line
          type="monotone"
          dataKey="actual"
          stroke="#4f9cf9"
          strokeWidth={2}
          dot={false}
          name={`${metric} (actual)`}
          connectNulls
        />
        <Line
          type="monotone"
          dataKey="forecast"
          stroke="#7c5cff"
          strokeWidth={2}
          strokeDasharray="5 4"
          dot={false}
          name="forecast"
          connectNulls
        />
        <Scatter dataKey="anomaly" fill="#ff6b6b" name="anomaly" />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
