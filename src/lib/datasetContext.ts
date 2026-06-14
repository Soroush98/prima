import { AsyncLocalStorage } from "node:async_hooks";

/**
 * Request-scoped "active dataset" so the agent nodes can stay dataset-agnostic.
 * The API route sets the active warehouse path for the duration of a request;
 * `getDb()` / `safeQuery()` read it without every node having to thread a param.
 * Outside any `runWithDbPath` scope (e.g. the eval CLI) it's undefined and the
 * db layer falls back to the default `DB_PATH`.
 */
const als = new AsyncLocalStorage<string>();

export function runWithDbPath<T>(dbPath: string, fn: () => T): T {
  return als.run(dbPath, fn);
}

export function activeDbPath(): string | undefined {
  return als.getStore();
}
