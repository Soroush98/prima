/**
 * Rate / quota controls for the LLM path (see the `llm-service-practices` skill):
 *
 *  - `TokenBucket` — a per-caller quota gate. The API boundary checks it BEFORE
 *    invoking the agent graph, so a throttled request costs zero model tokens.
 *  - `Semaphore` — a global concurrency limiter. Every model call passes through
 *    one shared gate sized to the provider quota, so total in-flight Claude calls
 *    stay bounded regardless of how many requests arrive at once.
 *
 * Both are in-memory and therefore **per-process** — correct for Prima's single
 * Fly machine. A multi-instance deploy would back these with a shared store
 * (e.g. Redis) so the limits hold across the fleet.
 */

export interface RateLimitResult {
  ok: boolean;
  /** Whole seconds the caller should wait before retrying (only when `!ok`). */
  retryAfterSec: number;
  /** Approximate requests still available in the caller's bucket. */
  remaining: number;
}

interface Bucket {
  tokens: number;
  updatedMs: number;
}

/**
 * Token bucket keyed by caller id. `capacity` is the burst allowance; `refillPerSec`
 * is the sustained rate. Stale buckets are swept lazily so memory stays bounded.
 */
export class TokenBucket {
  private readonly buckets = new Map<string, Bucket>();
  private lastSweepMs = 0;

  constructor(
    private readonly capacity: number,
    private readonly refillPerSec: number,
  ) {}

  take(key: string, nowMs: number = Date.now()): RateLimitResult {
    this.maybeSweep(nowMs);

    const b = this.buckets.get(key) ?? { tokens: this.capacity, updatedMs: nowMs };
    const elapsedSec = Math.max(0, (nowMs - b.updatedMs) / 1000);
    b.tokens = Math.min(this.capacity, b.tokens + elapsedSec * this.refillPerSec);
    b.updatedMs = nowMs;

    if (b.tokens < 1) {
      this.buckets.set(key, b);
      return {
        ok: false,
        retryAfterSec: Math.max(1, Math.ceil((1 - b.tokens) / this.refillPerSec)),
        remaining: 0,
      };
    }

    b.tokens -= 1;
    this.buckets.set(key, b);
    return { ok: true, retryAfterSec: 0, remaining: Math.floor(b.tokens) };
  }

  /** Drop buckets that have fully refilled (no longer carry state worth keeping). */
  private maybeSweep(nowMs: number): void {
    if (nowMs - this.lastSweepMs < 60_000) return;
    this.lastSweepMs = nowMs;
    const fullAfterSec = this.capacity / this.refillPerSec;
    for (const [key, b] of this.buckets) {
      if ((nowMs - b.updatedMs) / 1000 > fullAfterSec) this.buckets.delete(key);
    }
  }
}

/**
 * Minimal async counting semaphore. `acquire()` resolves immediately while slots
 * are free, otherwise queues FIFO until a `release()` hands a slot over.
 */
export class Semaphore {
  private active = 0;
  private readonly waiters: Array<() => void> = [];

  constructor(private readonly max: number) {}

  async acquire(): Promise<void> {
    if (this.active < this.max) {
      this.active += 1;
      return;
    }
    await new Promise<void>((resolve) => this.waiters.push(resolve));
    // Resumed by release(), which reserved the slot for us (active unchanged).
  }

  release(): void {
    const next = this.waiters.shift();
    if (next) {
      next(); // hand the slot directly to the next waiter
    } else {
      this.active -= 1;
    }
  }

  /** Run `fn` while holding a slot, releasing even if it throws. */
  async run<T>(fn: () => Promise<T>): Promise<T> {
    await this.acquire();
    try {
      return await fn();
    } finally {
      this.release();
    }
  }
}
