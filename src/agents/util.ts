import type { DimensionFilter, Metric } from "./types";

export const SCHEMA_DOC = `Real SMD (Server Machine Dataset) server-telemetry warehouse.

Table: health (one row per entity, per timestep) — the analysis series
  entity TEXT     -- server id, e.g. 'machine-1-1'
  t      INTEGER  -- timestep index (per-minute samples), ascending from 0
  value  REAL     -- entity anomaly score = max robust z-score across all 38 metric channels
  label  INTEGER  -- ground-truth: 1 if this timestep is a known anomaly, else 0

Table: metrics (raw channels, for root-cause attribution)
  entity TEXT, t INTEGER, m01 … m38 REAL   -- the 38 metric channels, normalized [0,1]

Table: interp (ground-truth root cause per anomaly segment)
  entity TEXT, seg_start INTEGER, seg_end INTEGER, channel INTEGER  -- culprit channel (1..38)

The analysis series for an entity is its health score over time:
  SELECT t AS date, value FROM health WHERE entity = '<id>' ORDER BY t`;

export interface DimensionVocab {
  entities: string[];
}

/** Match a question against the known entity ids (case/separator-insensitive). */
function matchEntity(q: string, entities: string[]): string | undefined {
  const norm = (s: string) => s.toLowerCase().replace(/[\s_-]+/g, "");
  const nq = norm(q);
  let best: string | undefined;
  for (const e of entities) {
    if (nq.includes(norm(e))) {
      if (!best || e.length > best.length) best = e; // longest id wins
    }
  }
  return best;
}

/**
 * Heuristic parse of the natural-language question into the analysis series.
 * The metric is always the entity health score; the only dimension is *which*
 * server entity to analyze. Falls back to the first available entity.
 */
export function parseIntent(
  question: string,
  vocab: DimensionVocab,
): { metric: Metric; filter: DimensionFilter } {
  const entity = matchEntity(question, vocab.entities) ?? vocab.entities[0];
  return { metric: "health", filter: { entity } };
}

/**
 * Deterministic SQL builder used as the safe fallback if the agent's generated
 * SQL is rejected. The entity is bound as a parameter (never interpolated).
 */
export function buildMetricSql(
  _metric: Metric,
  filter: DimensionFilter,
): { sql: string; params: string[] } {
  const entity = filter.entity ?? "";
  const sql = `SELECT t AS date, value FROM health WHERE entity = ? ORDER BY t`;
  return { sql, params: [entity] };
}

export function filterLabel(filter: DimensionFilter): string {
  return filter.entity ?? "all entities";
}
