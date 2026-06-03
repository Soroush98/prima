/**
 * CLI entrypoint: `npm run eval`
 * Runs the golden-set evaluation and prints a scorecard to the terminal,
 * also persisting the JSON report the dashboard reads.
 */
import fs from "node:fs";
import path from "node:path";
import { buildSyntheticDb } from "@/lib/synthetic";
import { DB_PATH } from "@/lib/db";
import { runEvaluation } from "./evaluate";

async function main() {
  // Build the synthetic, ground-truth dataset into whatever DB_PATH resolves to
  // (the npm script points PRIMA_DB_PATH at an isolated eval DB).
  const { rows, days } = buildSyntheticDb(DB_PATH);
  console.log(`  Built synthetic eval dataset: ${rows.toLocaleString()} rows / ${days} days → ${DB_PATH}`);

  const report = await runEvaluation(new Date().toISOString());

  console.log(`\n  Prima LLM Evaluation  (mode: ${report.llmMode})`);
  console.log("  " + "─".repeat(58));
  for (const c of report.cases) {
    const mark = c.passed ? "✅" : "❌";
    console.log(`  ${mark}  ${c.id.padEnd(22)} ${c.metrics.latencyMs}ms`);
    for (const ch of c.checks.filter((x) => !x.passed)) {
      console.log(`        ↳ failed ${ch.name}: ${ch.detail}`);
    }
  }
  const a = report.aggregate;
  console.log("  " + "─".repeat(58));
  console.log(`  Pass rate          ${a.passRate}%`);
  console.log(`  Routing accuracy   ${a.routingAccuracy}%`);
  console.log(`  SQL validity       ${a.sqlValidity}%`);
  console.log(`  Anomaly recall     ${a.anomalyRecall}%`);
  console.log(`  Hallucination rate ${a.hallucinationRate}%`);
  console.log(`  Avg latency        ${a.avgLatencyMs}ms   (p95 ${a.p95LatencyMs}ms)`);
  console.log(`  Total cost         $${a.totalCostUsd}`);
  console.log("");

  const out = path.join(process.cwd(), "data", "eval-report.json");
  fs.writeFileSync(out, JSON.stringify(report, null, 2));
  console.log(`  Report written to ${out}\n`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
