import Database from "better-sqlite3";
import path from "node:path";
import fs from "node:fs";
import { activeDbPath } from "./datasetContext";

/**
 * SQLite connections, one per warehouse file. Stands in for the BigQuery
 * warehouse the Cisco role targets — same SQL surface (CTEs, window functions,
 * COUNT DISTINCT), just local and free to demo. The *active* warehouse is set
 * per request by the dataset picker (see datasetContext); outside a request it
 * defaults to `DB_PATH`.
 */
const _dbs = new Map<string, Database.Database>();

// Overridable so the eval harness can target an isolated synthetic DB without
// disturbing the runtime warehouse.
export const DB_PATH =
  process.env.PRIMA_DB_PATH || path.join(process.cwd(), "data", "prima.db");

/** Resolve the warehouse the current request targets (ALS), else the default. */
function currentDbPath(): string {
  return activeDbPath() ?? DB_PATH;
}

export function getDb(): Database.Database {
  const p = currentDbPath();
  const existing = _dbs.get(p);
  if (existing) return existing;
  if (!fs.existsSync(p)) {
    throw new Error(
      `Database not found at ${p}. The SMD warehouse is built on first request (see ensureSmdDb).`,
    );
  }
  const db = new Database(p, { readonly: true, fileMustExist: true });
  db.pragma("journal_mode = WAL");
  _dbs.set(p, db);
  return db;
}

export interface DimensionVocab {
  /** SMD entity ids present in the active warehouse (e.g. "machine-1-1"). */
  entities: string[];
}

const _vocabs = new Map<string, DimensionVocab>();

/** Entity ids actually present in the active warehouse (cached). */
export function getDimensionVocab(): DimensionVocab {
  const p = currentDbPath();
  const cached = _vocabs.get(p);
  if (cached) return cached;
  const db = getDb();
  const entities = (
    db.prepare("SELECT DISTINCT entity AS v FROM health ORDER BY entity").all() as { v: string }[]
  )
    .map((r) => r.v)
    .filter(Boolean);
  const vocab = { entities };
  _vocabs.set(p, vocab);
  return vocab;
}

export interface SqlGuardResult {
  ok: boolean;
  reason?: string;
}

/**
 * Defense-in-depth for the autonomous SQL agent: only a single read-only
 * SELECT/WITH statement is ever allowed to touch the warehouse.
 */
export function guardReadOnlySql(sql: string): SqlGuardResult {
  // Strip SQL comments first so a banned keyword can't be smuggled past the
  // denylist via `/* */` or `--`. The real boundary is the readonly connection
  // (see getDb); this denylist is belt-and-suspenders on top of that.
  const stripped = sql
    .replace(/\/\*[\s\S]*?\*\//g, " ")
    .replace(/--[^\n]*/g, " ");
  const trimmed = stripped.trim().replace(/;+\s*$/, "");
  if (trimmed.includes(";")) {
    return { ok: false, reason: "Multiple statements are not allowed." };
  }
  if (!/^(select|with)\b/i.test(trimmed)) {
    return { ok: false, reason: "Only SELECT/WITH read queries are permitted." };
  }
  const banned = /\b(insert|update|delete|drop|alter|create|attach|pragma|replace|vacuum)\b/i;
  if (banned.test(trimmed)) {
    return { ok: false, reason: "Mutating or DDL keywords are not permitted." };
  }
  return { ok: true };
}

export interface SafeQueryResult {
  ok: boolean;
  rows: Record<string, unknown>[];
  error?: string;
  ms: number;
}

/** Run a guarded, time-bounded read query and return rows + telemetry. */
export function safeQuery(sql: string, params: unknown[] = []): SafeQueryResult {
  const guard = guardReadOnlySql(sql);
  const start = performance.now();
  if (!guard.ok) {
    return { ok: false, rows: [], error: guard.reason, ms: 0 };
  }
  try {
    const db = getDb();
    const stmt = db.prepare(sql);
    const rows = stmt.all(...(params as never[])) as Record<string, unknown>[];
    return { ok: true, rows, ms: +(performance.now() - start).toFixed(1) };
  } catch (err) {
    return {
      ok: false,
      rows: [],
      error: err instanceof Error ? err.message : String(err),
      ms: +(performance.now() - start).toFixed(1),
    };
  }
}
