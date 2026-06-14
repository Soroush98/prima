/**
 * Zero-setup data bootstrap.
 *
 * The app needs no manual `npm run seed`. On the first query, if the warehouse
 * is missing, this downloads the real UCI "Online Retail" dataset from GitHub
 * and builds `prima.db` automatically. Subsequent runs reuse the cached DB.
 *
 * Source (full original CSV, 541,909 rows, Dec 2010 – Dec 2011):
 *   https://github.com/amir-hojjati/Data-Analysis-Online-Retail-Transactions
 *   (UCI Machine Learning Repository, "Online Retail", Daqing Chen et al.)
 */
import Database from "better-sqlite3";
import fs from "node:fs";
import path from "node:path";
import { DB_PATH } from "./db";

const RAW_URL =
  "https://raw.githubusercontent.com/amir-hojjati/Data-Analysis-Online-Retail-Transactions/master/Original-Dataset/Online%20Retail.csv";
const CSV_CACHE = path.join(process.cwd(), "data", "online-retail.csv");

/** Minimal RFC-4180-ish CSV line parser (handles quoted fields with commas). */
function parseLine(line: string): string[] {
  const out: string[] = [];
  let cur = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (inQuotes) {
      if (c === '"') {
        if (line[i + 1] === '"') {
          cur += '"';
          i++;
        } else inQuotes = false;
      } else cur += c;
    } else if (c === '"') inQuotes = true;
    else if (c === ",") {
      out.push(cur);
      cur = "";
    } else cur += c;
  }
  out.push(cur);
  return out;
}

/** "12/1/2010 8:26" (M/D/YYYY) → "2010-12-01". Returns null if unparseable. */
function toIsoDate(s: string): string | null {
  const datePart = s.trim().split(" ")[0];
  const m = datePart.split("/");
  if (m.length !== 3) return null;
  const [mm, dd, yyyy] = m;
  if (!yyyy || yyyy.length !== 4) return null;
  return `${yyyy}-${mm.padStart(2, "0")}-${dd.padStart(2, "0")}`;
}

/** Coarse product category from the item description (light feature-engineering). */
function categorize(desc: string): string {
  const d = desc.toUpperCase();
  if (/CHRISTMAS|ADVENT|SANTA|REINDEER|SNOW/.test(d)) return "seasonal";
  if (/BAG|POUCH|SATCHEL/.test(d)) return "bags";
  if (/HEART|LOVE|GIFT|VINTAGE/.test(d)) return "gifts";
  if (/MUG|CUP|BOTTLE|JUG|TEA|KITCHEN|BAKING/.test(d)) return "kitchen";
  if (/LIGHT|LANTERN|CANDLE|HOLDER|FRAME|DECORATION/.test(d)) return "home_decor";
  if (/CARD|PAPER|WRAP|NOTEBOOK|PEN|CHALK/.test(d)) return "stationery";
  return "other";
}

const DOWNLOAD_TIMEOUT_MS = 120_000; // ~46MB from a third-party raw URL
const DOWNLOAD_RETRIES = 3;

async function fetchCsv(): Promise<string> {
  if (fs.existsSync(CSV_CACHE) && fs.statSync(CSV_CACHE).size > 40_000_000) {
    return fs.readFileSync(CSV_CACHE, "utf-8");
  }
  let lastErr: unknown;
  for (let attempt = 1; attempt <= DOWNLOAD_RETRIES; attempt++) {
    try {
      const res = await fetch(RAW_URL, { signal: AbortSignal.timeout(DOWNLOAD_TIMEOUT_MS) });
      if (!res.ok) {
        // 5xx is transient and worth retrying; 4xx is not.
        if (res.status >= 500 && attempt < DOWNLOAD_RETRIES) {
          lastErr = new Error(`HTTP ${res.status}`);
        } else {
          throw new Error(`Dataset download failed: HTTP ${res.status}`);
        }
      } else {
        const text = await res.text();
        fs.mkdirSync(path.dirname(CSV_CACHE), { recursive: true });
        fs.writeFileSync(CSV_CACHE, text);
        return text;
      }
    } catch (err) {
      lastErr = err;
      if (attempt >= DOWNLOAD_RETRIES) break;
    }
    // capped exponential backoff before the next attempt
    await new Promise((r) => setTimeout(r, 500 * 2 ** (attempt - 1)));
  }
  throw new Error(`Dataset download failed after ${DOWNLOAD_RETRIES} attempts: ${String(lastErr)}`);
}

function buildDb(csv: string): { rows: number; days: number; customers: number } {
  const lines = csv.replace(/^﻿/, "").split(/\r?\n/);
  const header = parseLine(lines[0]).map((h) => h.trim());
  const idx = (name: string) => header.indexOf(name);
  const iDate = idx("InvoiceDate");
  const iCust = idx("CustomerID");
  const iCountry = idx("Country");
  const iDesc = idx("Description");
  if (iDate < 0 || iCust < 0 || iCountry < 0) {
    throw new Error(`Unexpected dataset header: ${header.join(",")}`);
  }

  // build into a temp file, then atomically rename → no half-built DB is ever read
  const tmp = DB_PATH + ".building";
  if (fs.existsSync(tmp)) fs.rmSync(tmp);
  fs.mkdirSync(path.dirname(DB_PATH), { recursive: true });

  const db = new Database(tmp);
  db.pragma("journal_mode = WAL");
  db.exec(`
    CREATE TABLE user_activity (
      activity_date TEXT NOT NULL, user_id INTEGER NOT NULL,
      region TEXT NOT NULL, platform TEXT NOT NULL, feature TEXT NOT NULL
    );
    CREATE TABLE deploys (
      deploy_date TEXT NOT NULL, version TEXT NOT NULL, service TEXT NOT NULL,
      region TEXT, platform TEXT, notes TEXT
    );
  `);
  const insert = db.prepare(
    "INSERT INTO user_activity (activity_date, user_id, region, platform, feature) VALUES (?, ?, ?, ?, ?)",
  );
  let kept = 0;
  const txn = db.transaction(() => {
    for (let i = 1; i < lines.length; i++) {
      if (!lines[i]) continue;
      const f = parseLine(lines[i]);
      const cust = (f[iCust] ?? "").trim();
      const date = toIsoDate(f[iDate] ?? "");
      if (!cust || !date) continue;
      insert.run(date, parseInt(cust, 10), (f[iCountry] ?? "Unknown").trim(), "web", categorize(f[iDesc] ?? ""));
      kept++;
    }
  });
  txn();
  db.exec("CREATE INDEX idx_activity_date ON user_activity(activity_date)");
  db.exec("CREATE INDEX idx_activity_dims ON user_activity(region, platform, feature)");
  const days = (db.prepare("SELECT COUNT(DISTINCT activity_date) AS n FROM user_activity").get() as { n: number }).n;
  const customers = (db.prepare("SELECT COUNT(DISTINCT user_id) AS n FROM user_activity").get() as { n: number }).n;
  db.close();

  // clear any stale WAL sidecars then promote the temp DB
  for (const ext of ["", "-wal", "-shm"]) {
    if (fs.existsSync(DB_PATH + ext)) fs.rmSync(DB_PATH + ext);
  }
  fs.renameSync(tmp, DB_PATH);
  return { rows: kept, days, customers };
}

function dbIsReady(): boolean {
  if (!fs.existsSync(DB_PATH)) return false;
  try {
    const db = new Database(DB_PATH, { readonly: true, fileMustExist: true });
    const row = db.prepare("SELECT COUNT(*) AS n FROM user_activity").get() as { n: number };
    db.close();
    return row.n > 0;
  } catch {
    return false;
  }
}

let inFlight: Promise<void> | null = null;

/**
 * Ensure the warehouse exists and is populated. Concurrency-safe: parallel
 * first-requests share a single build. Pass `{ force: true }` to rebuild.
 */
export function ensureDatabase(opts: { force?: boolean } = {}): Promise<void> {
  if (!opts.force && dbIsReady()) return Promise.resolve();
  if (inFlight) return inFlight;
  inFlight = (async () => {
    const csv = await fetchCsv();
    const stats = buildDb(csv);
    console.log(
      `[bootstrap] Built warehouse from real Online Retail data: ${stats.rows.toLocaleString()} rows, ` +
        `${stats.days} days, ${stats.customers} customers.`,
    );
  })().finally(() => {
    inFlight = null;
  });
  return inFlight;
}
