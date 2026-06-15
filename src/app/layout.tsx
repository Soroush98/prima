import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Prima — Agentic Server-Health Intelligence",
  description: "LangGraph-powered autonomous analytics on real server telemetry: anomaly detection, forecasting, and root-cause attribution.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
