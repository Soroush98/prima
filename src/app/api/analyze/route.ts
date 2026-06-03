import { NextResponse } from "next/server";
import { runPrima } from "@/agents/graph";
import { ensureDatabase } from "@/lib/bootstrap";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/** POST { question } → full Prima analysis (series, anomalies, forecast, root cause, trace). */
export async function POST(req: Request) {
  try {
    const body = (await req.json()) as { question?: string };
    const question = (body.question ?? "").trim();
    if (!question) {
      return NextResponse.json({ error: "Missing 'question'." }, { status: 400 });
    }
    await ensureDatabase();
    const result = await runPrima(question);
    return NextResponse.json(result);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
