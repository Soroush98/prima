/**
 * Build the SMD warehouse (data/prima.db) from data/smd at IMAGE-BUILD time, so
 * the deployed app ships a ready, read-only DB and never needs the raw SMD files
 * or a runtime build on a volume. Invoked by `npm run build:warehouse` in the
 * Dockerfile build stage (where data/smd is present in the build context).
 *
 * Relative import (not the @/ alias) so it resolves without tsconfig-path setup.
 */
import { buildSmdDb } from "../src/lib/smdWarehouse";

const stats = buildSmdDb();
console.log(
  `[build:warehouse] ${stats.entities} entities, ${stats.rows.toLocaleString()} timesteps, ` +
    `${stats.anomalies.toLocaleString()} labeled anomalies`,
);
