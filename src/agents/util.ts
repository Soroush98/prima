import type { DimensionFilter, Metric } from "./types";

export const SCHEMA_DOC = `Table: user_activity (one row per user, per day, per feature used)
  activity_date TEXT  -- 'YYYY-MM-DD'
  user_id       INTEGER
  region        TEXT  -- one of: NA, EU, APAC, LATAM
  platform      TEXT  -- one of: web, mobile, desktop
  feature       TEXT  -- one of: dashboard, search, export, ai_assistant, alerts, reports, admin

Table: deploys (release events to correlate against anomalies)
  deploy_date TEXT, version TEXT, service TEXT, region TEXT, platform TEXT, notes TEXT

DAU(day)  = COUNT(DISTINCT user_id) for that activity_date.
WAU(day)  = COUNT(DISTINCT user_id) over the trailing 7 days ending on that day.`;

export interface DimensionVocab {
  regions: string[];
  platforms: string[];
  features: string[];
}

/** Match a question against a list of known dimension values (case-insensitive,
 *  underscores treated as spaces). Returns the longest match so "United Kingdom"
 *  wins over a stray "United". */
function matchValue(q: string, values: string[]): string | undefined {
  let best: string | undefined;
  for (const v of values) {
    const needle = v.toLowerCase().replace(/_/g, " ");
    if (q.includes(needle) || q.includes(v.toLowerCase())) {
      if (!best || v.length > best.length) best = v;
    }
  }
  return best;
}

/**
 * Heuristic parse of the natural-language question into metric + dimension
 * filter, using the dataset's *actual* dimension vocabulary so it works on the
 * synthetic data (NA/EU, mobile, ai_assistant) and the real Online Retail data
 * (United Kingdom, Germany, seasonal, …) alike.
 */
export function parseIntent(
  question: string,
  vocab: DimensionVocab,
): { metric: Metric; filter: DimensionFilter } {
  const q = question.toLowerCase();
  const metric: Metric = /\bwau\b|weekly active/.test(q) ? "WAU" : "DAU";
  const filter: DimensionFilter = {};
  const region = matchValue(q, vocab.regions);
  const platform = matchValue(q, vocab.platforms);
  const feature = matchValue(q, vocab.features);
  if (region) filter.region = region;
  // ignore a trivial single-value platform vocab (e.g. real data is all 'web')
  if (platform && vocab.platforms.length > 1) filter.platform = platform;
  if (feature) filter.feature = feature;
  return { metric, filter };
}

/** Deterministic SQL builder used as the safe fallback if the agent's generated SQL is rejected. */
export function buildMetricSql(metric: Metric, filter: DimensionFilter): string {
  const conds: string[] = [];
  if (filter.region) conds.push(`region = '${filter.region}'`);
  if (filter.platform) conds.push(`platform = '${filter.platform}'`);
  if (filter.feature) conds.push(`feature = '${filter.feature}'`);
  const where = conds.length ? `WHERE ${conds.join(" AND ")}` : "";

  if (metric === "WAU") {
    return `WITH days AS (SELECT DISTINCT activity_date AS d FROM user_activity)
SELECT days.d AS date,
       (SELECT COUNT(DISTINCT ua.user_id)
          FROM user_activity ua
         WHERE ua.activity_date > date(days.d, '-7 day')
           AND ua.activity_date <= days.d
           ${conds.length ? "AND " + conds.join(" AND ").replace(/region|platform|feature/g, (m) => "ua." + m) : ""}
       ) AS value
  FROM days
 ORDER BY days.d`;
  }

  return `SELECT activity_date AS date, COUNT(DISTINCT user_id) AS value
  FROM user_activity
  ${where}
 GROUP BY activity_date
 ORDER BY activity_date`;
}

export function filterLabel(filter: DimensionFilter): string {
  const parts: string[] = [];
  if (filter.region) parts.push(filter.region);
  if (filter.platform) parts.push(filter.platform);
  if (filter.feature) parts.push(filter.feature);
  return parts.length ? parts.join(" · ") : "global";
}
