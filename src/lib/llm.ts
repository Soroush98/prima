import Anthropic from "@anthropic-ai/sdk";

/**
 * LLM access layer — live Claude only.
 *
 * The SQL-Analyst / Root-Cause / Narrator agents call a live Claude model via
 * the Anthropic SDK directly (precise control over sampling params). An
 * ANTHROPIC_API_KEY is REQUIRED; the agents always call a real Claude model.
 */

export const MODEL = process.env.PRIMA_MODEL || "claude-sonnet-4-6";

let _client: Anthropic | null = null;
function getClient(): Anthropic {
  if (!process.env.ANTHROPIC_API_KEY) {
    throw new Error(
      "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key — Prima runs live Claude agents only.",
    );
  }
  if (!_client) {
    // reads ANTHROPIC_API_KEY from env. A per-request deadline plus capped
    // retries (with the SDK's exponential backoff + Retry-After handling) keep a
    // hung connection or a transient 429/503 from stalling the whole agent graph.
    _client = new Anthropic({ timeout: LLM_TIMEOUT_MS, maxRetries: 3 });
  }
  return _client;
}

/** Hard deadline for a single Claude call; the agent graph awaits these serially. */
const LLM_TIMEOUT_MS = 45_000;

export interface LLMResult {
  text: string;
  mode: "live";
  ms: number;
  tokensIn: number;
  tokensOut: number;
  model: string;
}

const estTokens = (s: string) => Math.ceil(s.length / 4);

/** Single-shot completion against a live Claude model. */
export async function callLLM(system: string, user: string): Promise<LLMResult> {
  const start = performance.now();
  const res = await getClient().messages.create({
    model: MODEL,
    max_tokens: 1024,
    temperature: 0,
    system,
    messages: [{ role: "user", content: user }],
  });
  const text = res.content
    .filter((b): b is Anthropic.TextBlock => b.type === "text")
    .map((b) => b.text)
    .join("");
  return {
    text,
    mode: "live",
    ms: +(performance.now() - start).toFixed(1),
    tokensIn: res.usage?.input_tokens ?? estTokens(system + user),
    tokensOut: res.usage?.output_tokens ?? estTokens(text),
    model: MODEL,
  };
}

/** Anthropic public per-MTok pricing (USD) for the cost line in the dashboard. */
const PRICE: Record<string, { in: number; out: number }> = {
  "claude-sonnet-4-6": { in: 3, out: 15 },
  "claude-opus-4-8": { in: 15, out: 75 },
  "claude-haiku-4-5-20251001": { in: 1, out: 5 },
};

export function estimateCostUsd(model: string, tokensIn: number, tokensOut: number): number {
  const p = PRICE[model];
  if (!p) {
    // Don't present a fabricated figure as exact telemetry — flag the gap and
    // fall back to sonnet pricing only as a rough estimate.
    console.warn(`[llm] no price entry for model "${model}"; cost is an estimate at sonnet rates`);
  }
  const rate = p ?? PRICE["claude-sonnet-4-6"];
  return +(((tokensIn * rate.in) / 1e6 + (tokensOut * rate.out) / 1e6)).toFixed(6);
}
