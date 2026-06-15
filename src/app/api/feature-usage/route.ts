import { NextResponse } from "next/server";
import { safeQuery } from "@/lib/db";
import { runWithDbPath } from "@/lib/datasetContext";
import { datasetMeta, dbPathFor, ensureDataset, parseDatasetId } from "@/lib/datasets";
import { channelName } from "@/lib/smdWarehouse";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * GET ?dataset= → the metric channels most often implicated as the ground-truth
 * root cause across anomaly segments (from SMD interpretation_label). Reuses the
 * usage-panel chart shape ({ feature, users, events }) so the UI stays generic.
 */
export async function GET(req: Request) {
  try {
    const dataset = parseDatasetId(new URL(req.url).searchParams.get("dataset"));
    if (datasetMeta(dataset).kind !== "agent") {
      return NextResponse.json({ features: [] });
    }
    await ensureDataset(dataset);
    const rows = runWithDbPath(dbPathFor(dataset), () =>
      safeQuery(
        `SELECT channel,
                COUNT(*) AS segments,
                COUNT(DISTINCT entity) AS entities
           FROM interp
          GROUP BY channel
          ORDER BY segments DESC
          LIMIT 12`,
      ).rows,
    ) as { channel: number; segments: number; entities: number }[];
    const features = rows.map((r) => ({
      feature: channelName(r.channel),
      users: r.segments, // # anomaly segments implicating this channel
      events: r.entities, // # distinct entities affected
    }));
    return NextResponse.json({ features });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
