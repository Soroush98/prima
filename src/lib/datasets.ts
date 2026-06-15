import { DB_PATH } from "./db";
import { ensureSmdDb } from "./smdWarehouse";

/**
 * Dataset registry — the selectable warehouses behind the UI dataset picker.
 *
 *  - `smd` is the **agent** dataset: real Server Machine Dataset telemetry with
 *    a `health`/`metrics`/`interp` schema, so the full LangGraph fleet runs
 *    against it (detection + root-cause attribution graded vs ground truth).
 *  - `aiops` is a **viewer** dataset: the real 2018 AIOps KPI series, read-only,
 *    served from a bundled sample (a univariate detector showcase).
 */
export type DatasetId = "smd" | "aiops";

export interface DatasetMeta {
  id: DatasetId;
  label: string;
  description: string;
  kind: "agent" | "viewer";
}

export const DATASETS: DatasetMeta[] = [
  {
    id: "smd",
    label: "Server Machine Dataset (real)",
    description:
      "Real labeled server telemetry — 38 metric channels per entity, with ground-truth root-cause channels.",
    kind: "agent",
  },
  {
    id: "aiops",
    label: "AIOps KPI (real)",
    description: "2018 AIOps Challenge web-service KPIs — read-only viewer with labeled anomalies.",
    kind: "viewer",
  },
];

export function datasetMeta(id: DatasetId): DatasetMeta {
  return DATASETS.find((d) => d.id === id) ?? DATASETS[0];
}

/** Warehouse file backing an agent dataset (only `smd` today → the default DB_PATH). */
export function dbPathFor(_id: DatasetId): string {
  return DB_PATH;
}

export function parseDatasetId(v: unknown): DatasetId {
  return v === "aiops" ? "aiops" : "smd";
}

/** Build the warehouse for an agent dataset if it isn't there yet. */
export async function ensureDataset(id: DatasetId): Promise<void> {
  if (id === "smd") await ensureSmdDb();
  // "aiops" is a read-only viewer — no warehouse needed.
}
