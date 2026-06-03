/**
 * Optional CLI: force-(re)build the real-data warehouse from GitHub now,
 * instead of waiting for the app to auto-bootstrap it on first request.
 */
import { ensureDatabase } from "../src/lib/bootstrap";

await ensureDatabase({ force: true });
console.log("✅ Real Online Retail warehouse ready at data/prima.db");
