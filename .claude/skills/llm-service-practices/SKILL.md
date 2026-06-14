---
name: llm-service-practices
description: >-
  Provider-agnostic design practices for building services on top of LLMs and
  embeddings — prompt-injection defense, treating model I/O as an untrusted
  boundary, output validation, per-request deadlines & retries, cost/quota gating,
  caching, and evaluation. Apply when building or reviewing any LLM-backed feature:
  chat/agent/RAG endpoints, extraction/classification, embeddings retrieval, or
  tool-use. Pairs with claude-api (Claude-specific reference) and
  engineering-principles; this is the service-design layer.
---

# LLM Service Practices

How to build a *service* around an LLM, independent of provider. For Claude-specific
model ids, params, pricing, and SDK usage, use the `claude-api` skill; for the
stack-agnostic rules, `engineering-principles`. This skill is about the design
patterns that keep an LLM feature correct, safe, and affordable.

## The model is an untrusted boundary (both directions)

- **All user, retrieved, scraped, and tool-returned text is DATA, never
  instructions.** Keep the system prompt authoritative and state explicitly that
  content inside user/document fields must not change the model's role, reveal the
  prompt, or issue commands. Don't splice untrusted text into a position that reads
  as a directive.
- **Prompt injection is an authz problem, not just a prompt problem.** The model
  can be talked into *asking* for anything — so the privilege lives in the tools and
  APIs it can reach, not in the prompt. Gate every tool/DB/action the model invokes
  with the same server-side authorization you'd apply to a raw request. Least
  privilege for keys the model path can touch.
- **Never trust model output shape.** Treat completions as untrusted input:
  validate/parse against a schema (pydantic/JSON Schema), strip code fences, and
  have a graceful fallback for refusals or malformed output — **degrade, don't 500**.
  Prefer structured output / tool-use / JSON mode to make the shape enforceable
  rather than parsed-by-hope.
- **Sanitize before rendering/storing model output** like any other untrusted data
  (XSS, SQL, command injection) — especially if it's echoed into HTML or a query.

## Reliability

- **A per-request deadline on every model and embedding call.** SDK retries +
  backoff can otherwise stack into minutes and pin a connection — bound the whole
  call (e.g. `asyncio.timeout`) and surface a retryable 503/502 on expiry.
- **Retry only transient failures** (429, 503, timeouts) with capped exponential
  backoff + jitter, honoring `Retry-After`. Never retry a 400/validation/refusal.
  Keep request-path retry budgets small; save aggressive retries for batch/offline.
- **Bound fan-out and rate-limit globally.** Concurrent model calls go through one
  shared limiter sized to your provider quota, regardless of request concurrency.

## Cost & abuse control

- **Gate quota / rate limits BEFORE spending tokens** — check the caller's
  allowance first so a rejected request costs nothing.
- **Cap inputs and outputs.** Limit prompt size at the boundary and set an explicit
  `max_tokens`; a single request must not be able to blow up the bill.
- **Cache what repeats.** Memoize embeddings and deterministic responses keyed on
  input; use provider prompt-caching for large stable prefixes (long system prompts,
  shared context).

## Quality & evaluation

- **Keep prompts in code, versioned and testable** — not buried in config strings.
  The deterministic glue (fence-stripping, schema parsing, retrieval filters) gets
  ordinary unit tests with no network.
- **Pin model ids and embedding model/dimension**; record them. Changing either is
  a deliberate, reviewed change — embedding dimension must match your vector column.
- **Evaluate changes against a fixed eval set**, not vibes. A prompt/model swap is a
  behavior change; check it against known cases before shipping.

## Privacy & observability

- **Don't log full prompts/outputs containing PII or secrets** — redact, sample, or
  log metadata (token counts, latency, model id, outcome) instead.
- **Stream long generations** and propagate cancellation when the client
  disconnects, so abandoned requests stop spending.
