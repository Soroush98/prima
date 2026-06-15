import { NextResponse } from "next/server";
import fs from "node:fs";
import path from "node:path";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const REPORT_PATH = path.join(process.cwd(), "data", "eval-report.json");

/**
 * GET → the LLM evaluation scorecard.
 *
 * The golden set runs against an isolated copy of the real SMD warehouse and
 * asserts on its ground-truth labels (anomaly presence, root-cause attribution
 * grading). The report ships pre-generated; regenerate any time with
 * `npm run eval` (which builds the SMD eval warehouse in isolation).
 */
export async function GET() {
  try {
    if (fs.existsSync(REPORT_PATH)) {
      const cached = JSON.parse(fs.readFileSync(REPORT_PATH, "utf-8"));
      return NextResponse.json({ ...cached, cached: true });
    }
    return NextResponse.json({
      available: false,
      note: "No eval report yet — run `npm run eval` to generate the synthetic-ground-truth scorecard.",
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
