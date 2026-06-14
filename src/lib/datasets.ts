import path from "node:path";
import fs from "node:fs";
import Database from "better-sqlite3";
import { DB_PATH } from "./db";
import { ensureDatabase } from "./bootstrap";
import { buildSyntheticDb } from "./synthetic";

/**
 * Dataset registry — the selectable warehouses behind the UI dataset picker.
 *
 *  - `real` / `synthetic` are **agent** datasets: same `user_activity` / `deploys`
 *    schema, so the full LangGraph fleet runs against them.
 *  - `aiops` is a **viewer** dataset: the real 2018 AIOps KPI series have a
 *    different shape (per-minute value/label, no users/regions/deploys), so it's
 *    read-only — served from a bundled sample, not analyzed by the agent.
 */
export type DatasetId = "real" | "synthetic" | "aiops";

export interface DatasetMeta {
  id: DatasetId;
  label: string;
  description: string;
  kind: "agent" | "viewer";
}

export const DATASETS: DatasetMeta[] = [
  {
    id: "real",
    label: "Online Retail (real)",
    description: "UCI Online Retail — 305 days of genuine e-commerce DAU/WAU.",
    kind: "agent",
  },
  {
    id: "synthetic",
    label: "Synthetic storyline",
    description: "Seeded telemetry with a known EU/mobile deploy incident — best for root-cause.",
    kind: "agent",
  },
  {
    id: "aiops",
    label: "AIOps KPI (real)",
    description: "2018 AIOps Challenge web-service KPIs — read-only viewer with labeled anomalies.",
    kind: "viewer",
  },
];

const DATA_DIR = path.dirname(DB_PATH);
const SYNTH_PATH = path.join(DATA_DIR, "prima-synthetic.db");

export function datasetMeta(id: DatasetId): DatasetMeta {
  return DATASETS.find((d) => d.id === id) ?? DATASETS[0];
}

/** Warehouse file backing an agent dataset (`real` → the default `DB_PATH`). */
export function dbPathFor(id: DatasetId): string {
  return id === "synthetic" ? SYNTH_PATH : DB_PATH;
}

export function parseDatasetId(v: unknown): DatasetId {
  return v === "synthetic" || v === "aiops" ? v : "real";
}

function dbReady(p: string): boolean {
  if (!fs.existsSync(p)) return false;
  try {
    const db = new Database(p, { readonly: true, fileMustExist: true });
    const n = (db.prepare("SELECT COUNT(*) AS n FROM user_activity").get() as { n: number }).n;
    db.close();
    return n > 0;
  } catch {
    return false;
  }
}

/** Build the warehouse for an agent dataset if it isn't there yet. */
export async function ensureDataset(id: DatasetId): Promise<void> {
  if (id === "synthetic") {
    if (!dbReady(SYNTH_PATH)) buildSyntheticDb(SYNTH_PATH);
    return;
  }
  // "real" → the auto-bootstrapped Online Retail warehouse; "aiops" needs no DB.
  if (id === "real") await ensureDatabase();
}
