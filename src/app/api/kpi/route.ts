import { NextResponse } from "next/server";
import fs from "node:fs";
import path from "node:path";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/** GET → the bundled AIOps KPI sample (real series + ground-truth anomaly labels). */
export function GET() {
  try {
    const p = path.join(process.cwd(), "data", "aiops-sample.json");
    if (!fs.existsSync(p)) {
      return NextResponse.json({ error: "AIOps sample not bundled.", kpis: [] }, { status: 404 });
    }
    const data = JSON.parse(fs.readFileSync(p, "utf-8"));
    return NextResponse.json(data);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ error: message, kpis: [] }, { status: 500 });
  }
}
