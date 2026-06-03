import { NextResponse } from "next/server";
import fs from "node:fs";
import path from "node:path";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const REPORT_PATH = path.join(process.cwd(), "data", "eval-report.json");

/**
 * GET → the LLM evaluation scorecard.
 *
 * Evaluation is intentionally decoupled from the live (real-data) warehouse: the
 * golden set asserts against *injected* ground-truth that only exists in the
 * synthetic dataset. The report ships pre-generated; regenerate it any time with
 * `npm run eval` (which builds the synthetic dataset in isolation).
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
