/**
 * Synthetic product-telemetry generator (used only by the evaluation harness
 * and the optional `npm run seed`). It bakes in known ground-truth anomalies so
 * the LLM eval has something to assert against:
 *   1. A bad EU/mobile deploy (v4.2.0) causing a sharp DAU drop ~day 150.
 *   2. An ai_assistant launch spike ~day 90.
 *   3. A slow decay of the legacy `export` feature.
 */
import Database from "better-sqlite3";
import fs from "node:fs";
import path from "node:path";

const DAYS = 180;
const END_DATE = "2026-05-31"; // fixed anchor → reproducible
const USER_POOL = 24000;

function mulberry32(seed: number) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const REGIONS = [
  { name: "NA", weight: 0.4 },
  { name: "EU", weight: 0.3 },
  { name: "APAC", weight: 0.2 },
  { name: "LATAM", weight: 0.1 },
];
const PLATFORMS = [
  { name: "web", weight: 0.5 },
  { name: "mobile", weight: 0.35 },
  { name: "desktop", weight: 0.15 },
];
const FEATURES = ["dashboard", "search", "export", "ai_assistant", "alerts", "reports", "admin"];

export function buildSyntheticDb(targetPath: string): { rows: number; days: number } {
  const rand = mulberry32(42);
  const randn = () => {
    const u = 1 - rand();
    const v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  const pick = <T>(arr: T[]) => arr[Math.floor(rand() * arr.length)];
  const weightedPick = <T extends { name: string; weight: number }>(items: T[]): string => {
    let r = rand();
    for (const it of items) {
      if (r < it.weight) return it.name;
      r -= it.weight;
    }
    return items[items.length - 1].name;
  };
  const isoDate = (daysBeforeEnd: number): string => {
    const end = new Date(END_DATE + "T00:00:00Z");
    end.setUTCDate(end.getUTCDate() - daysBeforeEnd);
    return end.toISOString().slice(0, 10);
  };

  for (const ext of ["", "-wal", "-shm"]) if (fs.existsSync(targetPath + ext)) fs.rmSync(targetPath + ext);
  fs.mkdirSync(path.dirname(targetPath), { recursive: true });

  const db = new Database(targetPath);
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
  const insertActivity = db.prepare(
    "INSERT INTO user_activity (activity_date, user_id, region, platform, feature) VALUES (?, ?, ?, ?, ?)",
  );
  const insertDeploy = db.prepare(
    "INSERT INTO deploys (deploy_date, version, service, region, platform, notes) VALUES (?, ?, ?, ?, ?, ?)",
  );

  const aiLaunchDay = 90;
  const badDeployDay = 150;
  insertDeploy.run(isoDate(DAYS - aiLaunchDay), "v3.9.0", "ai-service", null, null, "AI Assistant GA launch");
  insertDeploy.run(
    isoDate(DAYS - badDeployDay),
    "v4.2.0",
    "mobile-gateway",
    "EU",
    "mobile",
    "EU mobile auth refactor — later flagged as the cause of the DAU drop",
  );
  insertDeploy.run(isoDate(DAYS - 30), "v4.3.1", "web-app", null, null, "Routine dependency bump");

  let count = 0;
  const txn = db.transaction(() => {
    for (let d = 0; d < DAYS; d++) {
      const date = isoDate(DAYS - 1 - d);
      const dow = new Date(date + "T00:00:00Z").getUTCDay();
      const weekend = dow === 0 || dow === 6;

      let base = 2200 + d * 4;
      base *= weekend ? 0.72 : 1.0;
      base *= 1 + 0.03 * Math.sin((d / 7) * 2 * Math.PI);
      base += randn() * 45;
      const dau = Math.max(300, Math.round(base));

      const featureBoost: Record<string, number> = {};
      featureBoost["ai_assistant"] = d < aiLaunchDay ? 0.05 : Math.min(0.6, 0.05 + (d - aiLaunchDay) * 0.012);
      featureBoost["export"] = Math.max(0.04, 0.22 - d * 0.0009);

      for (let i = 0; i < dau; i++) {
        const region = weightedPick(REGIONS);
        const platform = weightedPick(PLATFORMS);
        if (d >= badDeployDay && d < badDeployDay + 20 && region === "EU" && platform === "mobile") {
          const keep = d < badDeployDay + 5 ? 0.18 : 0.55;
          if (rand() > keep) continue;
        }
        const userId = 1 + Math.floor(rand() * USER_POOL);
        const nFeatures = 1 + (rand() < 0.5 ? 0 : rand() < 0.7 ? 1 : 2);
        const used = new Set<string>();
        for (let f = 0; f < nFeatures; f++) {
          let feature = pick(FEATURES);
          const boost = featureBoost[feature] ?? 0.14;
          if (rand() > boost + 0.5) feature = "dashboard";
          if (used.has(feature)) continue;
          used.add(feature);
          insertActivity.run(date, userId, region, platform, feature);
          count++;
        }
      }
    }
  });
  txn();

  db.exec("CREATE INDEX idx_activity_date ON user_activity(activity_date)");
  db.exec("CREATE INDEX idx_activity_dims ON user_activity(region, platform, feature)");
  const days = (db.prepare("SELECT COUNT(DISTINCT activity_date) AS n FROM user_activity").get() as { n: number }).n;
  db.close();
  return { rows: count, days };
}
