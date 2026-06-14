import { NextResponse } from "next/server";
import { runPrima } from "@/agents/graph";
import { runWithDbPath } from "@/lib/datasetContext";
import { datasetMeta, dbPathFor, ensureDataset, parseDatasetId } from "@/lib/datasets";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/** POST { question, dataset? } → full Prima analysis against the chosen warehouse. */
export async function POST(req: Request) {
  try {
    const body = (await req.json()) as { question?: string; dataset?: string };
    const question = (body.question ?? "").trim();
    if (!question) {
      return NextResponse.json({ error: "Missing 'question'." }, { status: 400 });
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
