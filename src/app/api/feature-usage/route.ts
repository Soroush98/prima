import { NextResponse } from "next/server";
import { safeQuery } from "@/lib/db";
import { runWithDbPath } from "@/lib/datasetContext";
import { datasetMeta, dbPathFor, ensureDataset, parseDatasetId } from "@/lib/datasets";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/** GET ?dataset= → feature-usage breakdown over the last 30 days (usage panel). */
export async function GET(req: Request) {
  try {
    const dataset = parseDatasetId(new URL(req.url).searchParams.get("dataset"));
    if (datasetMeta(dataset).kind !== "agent") {
      return NextResponse.json({ features: [] });
    }
    await ensureDataset(dataset);
    const rows = runWithDbPath(dbPathFor(dataset), () =>
      safeQuery(
        `WITH last_day AS (SELECT MAX(activity_date) AS d FROM user_activity)
         SELECT feature,
                COUNT(*) AS events,
                COUNT(DISTINCT user_id) AS users
           FROM user_activity, last_day
          WHERE activity_date > date(last_day.d, '-30 day')
          GROUP BY feature
          ORDER BY users DESC`,
      ).rows,
    );
    return NextResponse.json({ features: rows });
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
