/**
 * SMD (Server Machine Dataset) warehouse builder.
 *
 * Turns the raw OmniAnomaly SMD text files in `data/smd/` into the SQLite
 * warehouse the agent fleet queries. SMD is real labeled server telemetry — 28
 * machines, 38 metric channels each, with a `test` split, per-point anomaly
 * `test_label`s, and `interpretation_label`s that name the channels responsible
 * for each anomaly segment (the real root-cause ground truth synthetic faked).
 *
 * Tables built (test split only — that's where labels live):
 *   health  (entity, t, value, label)   -- per-timestep entity anomaly score
 *                                           = max causal robust-z across channels.
 *                                           This is the 1-D series the agents analyze.
 *   metrics (entity, t, m01..m38)        -- raw channels, for root-cause attribution
 *   interp  (entity, seg_start, seg_end, channel)  -- ground-truth culprit channels
 *
 * The `health` score uses the same rolling-baseline robust z-score the AIOps
 * detector benchmark uses, so the series the agent sees is a faithful anomaly
 * signal, not a black box.
 */
import Database from "better-sqlite3";
import fs from "node:fs";
import path from "node:path";
import { DB_PATH } from "./db";

export const SMD_DIR = path.join(process.cwd(), "data", "smd");
export const N_CHANNELS = 38;
const ROLL = 100; // causal baseline window for the per-channel z-score

/** "metric_01" … "metric_38" — generic names (SMD ships no official metric labels). */
export function channelName(oneIndexed: number): string {
  return `metric_${String(oneIndexed).padStart(2, "0")}`;
}
const COLS = Array.from({ length: N_CHANNELS }, (_, i) => `m${String(i + 1).padStart(2, "0")}`);

/**
 * Default entity subset — 8 machines spanning all three SMD groups. Keeps the
 * warehouse build fast and the DB modest (~85MB) while staying multi-entity.
 * Override with SMD_ENTITIES="all" or a csv of ids.
 */
const DEFAULT_ENTITIES = [
  "machine-1-1", "machine-1-4", "machine-1-6", "machine-2-1",
  "machine-2-3", "machine-3-2", "machine-3-7", "machine-3-11",
];

export function smdEntities(): string[] {
  const env = process.env.SMD_ENTITIES?.trim();
  const all = fs
    .readdirSync(path.join(SMD_DIR, "test"))
    .filter((f) => f.endsWith(".txt"))
    .map((f) => f.replace(/\.txt$/, ""))
    .sort();
  if (!env) return DEFAULT_ENTITIES.filter((e) => all.includes(e));
  if (env === "all") return all;
  return env.split(",").map((s) => s.trim()).filter((e) => all.includes(e));
}

function readMatrix(file: string): number[][] {
  return fs
    .readFileSync(file, "utf-8")
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => line.split(",").map(Number));
}

function readLabels(file: string): number[] {
  return fs
    .readFileSync(file, "utf-8")
    .split(/\r?\n/)
    .filter(Boolean)
    .map((v) => (Number(v) >= 0.5 ? 1 : 0));
}

/** "15849-16368:1,9,10" lines → {start,end,channels[]}. */
function readInterp(file: string): { start: number; end: number; channels: number[] }[] {
  if (!fs.existsSync(file)) return [];
  return fs
    .readFileSync(file, "utf-8")
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => {
      const [range, dims] = line.split(":");
      const [start, end] = range.split("-").map(Number);
      const channels = (dims ?? "").split(",").map(Number).filter((n) => Number.isFinite(n));
      return { start, end, channels };
    });
}

function median(sorted: number[]): number {
  const m = sorted.length;
  if (m === 0) return 0;
  return m % 2 ? sorted[(m - 1) >> 1] : (sorted[m / 2 - 1] + sorted[m / 2]) / 2;
}

/**
 * Causal rolling median of the prior `win` values (matches bench_smd.py's
 * baseline: expected[i] = median(values[i-win .. i-1])). Maintains a sorted
 * window with binary-search insert/remove.
 */
function causalRollingMedian(values: number[], win: number): number[] {
  const n = values.length;
  const out = new Array(n).fill(NaN);
  const window: number[] = [];
  const bisect = (x: number): number => {
    let lo = 0;
    let hi = window.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (window[mid] < x) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  };
  for (let i = 0; i < n; i++) {
    if (i >= win) out[i] = median(window); // window holds values[i-win .. i-1]
    window.splice(bisect(values[i]), 0, values[i]);
    if (window.length > win) window.splice(bisect(values[i - win]), 1);
  }
  return out;
}

/**
 * Per-channel robust z-score: residual from a causal rolling median,
 * MAD-standardized. Same definition as the bench_smd.py baseline so the
 * warehouse health score and the detector benchmark agree.
 */
function channelZ(values: number[]): number[] {
  const n = values.length;
  const z = new Array(n).fill(0);
  const expected = causalRollingMedian(values, ROLL);
  const resid: number[] = [];
  for (let i = 0; i < n; i++) if (!Number.isNaN(expected[i])) resid.push(values[i] - expected[i]);
  if (resid.length === 0) return z;
  const med = median([...resid].sort((a, b) => a - b));
  const mad = median(resid.map((r) => Math.abs(r - med)).sort((a, b) => a - b)) || 1e-9;
  const sigma = 1.4826 * mad;
  for (let i = 0; i < n; i++) if (!Number.isNaN(expected[i])) z[i] = Math.abs((values[i] - expected[i] - med) / sigma);
  return z;
}

/**
 * The entity health series the agents analyze: per-timestep max robust z-score
 * across all metric channels. Single source of truth — used by the warehouse
 * builder and the detector benchmark, so the benchmarked path is the app path.
 */
export function computeHealthSeries(matrix: number[][]): number[] {
  const n = matrix.length;
  const nc = matrix[0]?.length ?? 0;
  const health = new Array(n).fill(0);
  for (let c = 0; c < nc; c++) {
    const z = channelZ(matrix.map((row) => row[c]));
    for (let t = 0; t < n; t++) if (z[t] > health[t]) health[t] = z[t];
  }
  return health;
}

export interface SmdBuildStats {
  entities: number;
  rows: number;
  anomalies: number;
}

/** Build the SMD warehouse at `target`. Atomic: builds to a temp file, renames. */
export function buildSmdDb(target: string = DB_PATH): SmdBuildStats {
  const entities = smdEntities();
  if (entities.length === 0) {
    throw new Error(`No SMD entities found under ${SMD_DIR}/test`);
  }

  const tmp = target + ".building";
  for (const ext of ["", "-wal", "-shm"]) if (fs.existsSync(tmp + ext)) fs.rmSync(tmp + ext);
  fs.mkdirSync(path.dirname(target), { recursive: true });

  const db = new Database(tmp);
  db.pragma("journal_mode = WAL");
  db.exec(`
    CREATE TABLE health  (entity TEXT NOT NULL, t INTEGER NOT NULL, value REAL NOT NULL, label INTEGER NOT NULL);
    CREATE TABLE metrics (entity TEXT NOT NULL, t INTEGER NOT NULL, ${COLS.map((c) => `${c} REAL`).join(", ")});
    CREATE TABLE interp  (entity TEXT NOT NULL, seg_start INTEGER NOT NULL, seg_end INTEGER NOT NULL, channel INTEGER NOT NULL);
  `);

  const insHealth = db.prepare("INSERT INTO health (entity, t, value, label) VALUES (?, ?, ?, ?)");
  const insMetric = db.prepare(
    `INSERT INTO metrics (entity, t, ${COLS.join(", ")}) VALUES (?, ?, ${COLS.map(() => "?").join(", ")})`,
  );
  const insInterp = db.prepare("INSERT INTO interp (entity, seg_start, seg_end, channel) VALUES (?, ?, ?, ?)");

  let totalRows = 0;
  let totalAnoms = 0;

  const build = db.transaction(() => {
    for (const entity of entities) {
      const test = readMatrix(path.join(SMD_DIR, "test", `${entity}.txt`));
      const labels = readLabels(path.join(SMD_DIR, "test_label", `${entity}.txt`));
      const interp = readInterp(path.join(SMD_DIR, "interpretation_label", `${entity}.txt`));
      const n = test.length;

      // entity health series = per-timestep max robust z across all channels
      const health = computeHealthSeries(test);

      for (let t = 0; t < n; t++) {
        const label = labels[t] ?? 0;
        insHealth.run(entity, t, +health[t].toFixed(4), label);
        insMetric.run(entity, t, ...test[t].slice(0, N_CHANNELS).map((v) => +v.toFixed(6)));
        totalRows++;
        totalAnoms += label;
      }
      for (const seg of interp) {
        for (const ch of seg.channels) insInterp.run(entity, seg.start, seg.end, ch);
      }
    }
  });
  build();

  db.exec("CREATE INDEX idx_health_entity ON health(entity, t)");
  db.exec("CREATE INDEX idx_metrics_entity ON metrics(entity, t)");
  db.exec("CREATE INDEX idx_interp_entity ON interp(entity, seg_start, seg_end)");
  db.close();

  for (const ext of ["", "-wal", "-shm"]) if (fs.existsSync(target + ext)) fs.rmSync(target + ext);
  fs.renameSync(tmp, target);
  return { entities: entities.length, rows: totalRows, anomalies: totalAnoms };
}

function dbIsReady(target: string): boolean {
  if (!fs.existsSync(target)) return false;
  try {
    const db = new Database(target, { readonly: true, fileMustExist: true });
    const n = (db.prepare("SELECT COUNT(*) AS n FROM health").get() as { n: number }).n;
    db.close();
    return n > 0;
  } catch {
    return false;
  }
}

let inFlight: Promise<void> | null = null;

/** Ensure the SMD warehouse exists at `target` (default DB_PATH). Build-once, concurrency-safe. */
export function ensureSmdDb(target: string = DB_PATH): Promise<void> {
  if (dbIsReady(target)) return Promise.resolve();
  if (inFlight) return inFlight;
  inFlight = (async () => {
    const stats = buildSmdDb(target);
    console.log(
      `[smd] Built warehouse: ${stats.entities} entities, ${stats.rows.toLocaleString()} timesteps, ` +
        `${stats.anomalies.toLocaleString()} labeled anomalies → ${target}`,
    );
  })().finally(() => {
    inFlight = null;
  });
  return inFlight;
}
