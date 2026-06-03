"use client";

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";

interface FeatureRow {
  feature: string;
  users: number;
  events: number;
}

const COLORS = ["#4f9cf9", "#7c5cff", "#3ddc97", "#ffb454", "#ff6b6b", "#46c5d8", "#b18cff"];

export default function FeatureUsageChart({ features }: { features: FeatureRow[] }) {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={features} layout="vertical" margin={{ top: 4, right: 16, left: 8, bottom: 0 }}>
        <CartesianGrid stroke="#243049" strokeDasharray="3 3" horizontal={false} />
        <XAxis type="number" tick={{ fill: "#8b97b3", fontSize: 11 }} />
        <YAxis
          type="category"
          dataKey="feature"
          tick={{ fill: "#8b97b3", fontSize: 12 }}
          width={92}
        />
        <Tooltip
          contentStyle={{
            background: "#121829",
            border: "1px solid #243049",
            borderRadius: 8,
            color: "#e6ebf5",
            fontSize: 13,
          }}
          formatter={(v: number) => [v.toLocaleString(), "active users (30d)"]}
        />
        <Bar dataKey="users" radius={[0, 5, 5, 0]}>
          {features.map((_, i) => (
            <Cell key={i} fill={COLORS[i % COLORS.length]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
