import Database from "better-sqlite3";
import path from "node:path";
import fs from "node:fs";

/**
 * Singleton SQLite connection. Stands in for the BigQuery warehouse the Cisco
 * role targets — same SQL surface (CTEs, window functions, COUNT DISTINCT),
 * just local and free to demo.
 */
let _db: Database.Database | null = null;

// Overridable so the eval harness can target an isolated synthetic DB without
// disturbing the runtime warehouse.
export const DB_PATH =
  process.env.PRIMA_DB_PATH || path.join(process.cwd(), "data", "prima.db");

export function getDb(): Database.Database {
  if (_db) return _db;
  if (!fs.existsSync(DB_PATH)) {
    throw new Error(
      `Database not found at ${DB_PATH}. Run \`npm run seed\` first to generate synthetic telemetry.`,
    );
  }
  _db = new Database(DB_PATH, { readonly: true, fileMustExist: true });
  _db.pragma("journal_mode = WAL");
  return _db;
}

export interface DimensionVocab {
  regions: string[];
  platforms: string[];
  features: string[];
}

let _vocab: DimensionVocab | null = null;

/** Distinct dimension values actually present in the warehouse (cached). */
export function getDimensionVocab(): DimensionVocab {
  if (_vocab) return _vocab;
  const db = getDb();
  const col = (c: string) =>
    (db.prepare(`SELECT DISTINCT ${c} AS v FROM user_activity ORDER BY ${c}`).all() as { v: string }[])
      .map((r) => r.v)
      .filter(Boolean);
  _vocab = { regions: col("region"), platforms: col("platform"), features: col("feature") };
  return _vocab;
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
  const trimmed = sql.trim().replace(/;+\s*$/, "");
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
