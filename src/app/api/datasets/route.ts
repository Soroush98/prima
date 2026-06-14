import { NextResponse } from "next/server";
import { DATASETS } from "@/lib/datasets";

export const runtime = "nodejs";

/** GET → the selectable datasets for the UI picker. */
export function GET() {
  return NextResponse.json({ datasets: DATASETS });
}
