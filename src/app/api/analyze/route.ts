import { NextResponse } from "next/server";
import { runPrima } from "@/agents/graph";
import { runWithDbPath } from "@/lib/datasetContext";
import { datasetMeta, dbPathFor, ensureDataset, parseDatasetId } from "@/lib/datasets";
import { TokenBucket } from "@/lib/rateLimit";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Cap the question at the boundary: it is spliced into every agent's prompt, so an
// unbounded input is a cost/abuse vector. Bound it before any tokens are spent.
const MAX_QUESTION_CHARS = 500;

// Per-caller quota, checked BEFORE invoking the agent graph so a throttled request
// costs zero model tokens. Burst of PRIMA_RATE_BURST, refilling to PRIMA_RATE_PER_MIN.
const RATE_BURST = Number(process.env.PRIMA_RATE_BURST) || 5;
const RATE_PER_MIN = Number(process.env.PRIMA_RATE_PER_MIN) || 20;
const limiter = new TokenBucket(RATE_BURST, RATE_PER_MIN / 60);

/** Best-effort caller identity for rate limiting (Fly sets fly-client-ip). */
function callerKey(req: Request): string {
  return (
    req.headers.get("fly-client-ip") ||
    req.headers.get("x-forwarded-for")?.split(",")[0].trim() ||
    "anon"
  );
}

/** POST { question, dataset? } → full Prima analysis against the chosen warehouse. */
export async function POST(req: Request) {
  try {
    const gate = limiter.take(callerKey(req));
    if (!gate.ok) {
      return NextResponse.json(
        { error: "Rate limit exceeded. Please slow down." },
        { status: 429, headers: { "Retry-After": String(gate.retryAfterSec) } },
      );
    }

    const body = (await req.json()) as { question?: string; dataset?: string };
    const question = (body.question ?? "").trim();
    if (!question) {
      return NextResponse.json({ error: "Missing 'question'." }, { status: 400 });
    }
    if (question.length > MAX_QUESTION_CHARS) {
      return NextResponse.json(
        { error: `Question too long (max ${MAX_QUESTION_CHARS} characters).` },
        { status: 400 },
      );
    }
    const dataset = parseDatasetId(body.dataset);
    if (datasetMeta(dataset).kind !== "agent") {
      return NextResponse.json(
        { error: `Dataset '${dataset}' is read-only and can't be analyzed by the agent.` },
        { status: 400 },
      );
    }
    await ensureDataset(dataset);
    const result = await runWithDbPath(dbPathFor(dataset), () => runPrima(question));
    return NextResponse.json(result);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
