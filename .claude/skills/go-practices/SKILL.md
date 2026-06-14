---
name: go-practices
description: >-
  Modern Go (Golang) idioms — concurrency, error handling, iterators, API/service
  design, and testing — current to Go 1.25. Apply when writing, reviewing, or
  refactoring Go: .go files, go.mod modules, goroutines, channels, HTTP services,
  or CLIs. Complements engineering-principles (stack-agnostic rules); this is the
  Go-specific layer.
---

# Go Practices

Idiomatic, production-grade Go (current to **Go 1.25**). Defer to
`engineering-principles` for general rules; below are the Go specifics. Always match
the surrounding package's style.

## Concurrency

- **Take `ctx context.Context` as the first parameter** of any function that does
  I/O, blocks, or spawns goroutines; propagate it and honor `ctx.Done()`.
- **Bound goroutines.** Use `golang.org/x/sync/errgroup` with `SetLimit`, or a
  buffered-channel semaphore. Never `go f()` per item over an unbounded set.
- **Don't leak goroutines.** Every goroutine has a clear exit tied to ctx or a closed
  channel. Wait with `errgroup`/`sync.WaitGroup`.
- **Channels transfer ownership; mutexes protect state.** Pick one over clever
  lock-free tricks. The sender closes a channel, never the receiver.
- **`select` with `<-ctx.Done()`** in every blocking loop so it's cancellable.
- Loop variables are **per-iteration since Go 1.22** — the old `x := x` capture trick
  is no longer needed.

## Errors

- **Wrap with context:** `fmt.Errorf("doing X: %w", err)`. Define sentinel errors
  (`var ErrFoo = errors.New(...)`) or typed errors; callers use `errors.Is/As`, never
  string matching. Use `errors.Join` to combine multiple.
- **Handle errors once** — handle, or wrap-and-return; don't log-and-return.
- **Don't `panic` across package boundaries.** Return errors; `recover` only at a
  well-defined top (e.g. a request handler), if at all.
- Classify retryable vs permanent failures explicitly when the caller will retry.

## API & package design

- **Accept interfaces, return concrete types.** Keep interfaces small and define them
  in the *consumer* package, not the producer.
- **Make the zero value useful**; add constructors only when needed.
- **Inject dependencies** (stores, clocks, clients) via struct fields or params — no
  package-level globals for state.
- **Generics** only for genuinely type-parametric code, not to dodge small
  duplication. Use the `slices`, `maps`, and `cmp` stdlib packages and the `min`,
  `max`, `clear` builtins instead of hand-rolling.
- **Expose sequences as range-over-func iterators** (`iter.Seq`/`iter.Seq2`,
  Go 1.23+) rather than returning slices or taking callbacks, where lazy iteration
  helps. Keep exported surface minimal.

## Testing

- **Table-driven tests** with subtests (`t.Run`), `t.Helper()` in helpers,
  `t.Cleanup` for teardown, and `t.Context()` (Go 1.24+) for cancellation.
- **`go test -race`** on anything concurrent; hand out copies, not pointers to shared
  mutable state.
- **`testing/synctest`** (GA in Go 1.25) for deterministic concurrency tests: it runs
  goroutines in an isolated bubble with a virtual clock, so time-based code is fast
  and flake-free — prefer it over real `time.Sleep` in tests.
- **`httptest`** for HTTP handlers/clients; fake the interface for stores/external
  clients rather than hitting the network. Golden files for generated/serialized
  output.

## Tooling & operations

- **Always run** `gofmt`/`goimports`, `go vet`, and **`golangci-lint` (v2)**; `go test
  -race -cover`. Keep `go.mod` tidy (`go mod tidy`).
- **Standard library first**; add a dependency only for real value.
- `context.WithTimeout` on every outbound call; `defer cancel()` immediately. `defer`
  cleanup (Close/Unlock) right after acquisition.
- **`log/slog`** for structured logging (not `log`/`fmt.Println`) in services.
- In containers, set `GOMAXPROCS` appropriately — Go 1.25 is container-aware, but
  verify it matches your CPU limits.
