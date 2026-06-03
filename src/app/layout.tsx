import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Prima — Agentic DAU/WAU Intelligence",
  description: "LangGraph-powered autonomous analytics: anomaly detection, forecasting, and prescriptive insight.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
