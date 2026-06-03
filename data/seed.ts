/**
 * Optional CLI: build the *synthetic* warehouse into prima.db.
 *
 * The app does NOT require this — it auto-bootstraps the real Online Retail
 * dataset on first request (see src/lib/bootstrap.ts). Use this only if you
 * want to run the dashboard against the synthetic, storyline-rich dataset.
 */
import path from "node:path";
import { buildSyntheticDb } from "../src/lib/synthetic";

const target = process.env.PRIMA_DB_PATH || path.join(process.cwd(), "data", "prima.db");
const { rows, days } = buildSyntheticDb(target);
console.log(`✅ Seeded synthetic warehouse: ${rows.toLocaleString()} rows across ${days} days → ${target}`);
console.log("   Storylines: AI-assistant launch (~day 90), bad EU/mobile deploy (~day 150), export decay.");
