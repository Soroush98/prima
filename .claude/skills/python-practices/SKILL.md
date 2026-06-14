---
name: python-practices
description: >-
  Modern Python idioms (3.12–3.14 era) — typing, project layout, testing, async,
  concurrency/free-threading, and packaging. Apply when writing, reviewing, or
  refactoring Python: .py files, pyproject.toml, scripts, services, or data/ML code.
  Complements engineering-principles (stack-agnostic rules); this is the Python layer.
---

# Python Practices

Production-grade, modern Python (target **3.12+**, current is 3.14). Defer to
`engineering-principles` for general rules; below are the Python specifics. Match the
project's existing style.

## Typing

- **Type hints on all public functions** (params and returns). Run a type checker in
  CI (mypy or pyright are established; Astral's `ty` is emerging) and treat type
  errors as build failures.
- Use modern syntax: `list[str]`, `X | None`, and **PEP 695 generics** —
  `def first[T](xs: list[T]) -> T` and `type Alias = ...` — over the old `TypeVar`/
  `Generic` boilerplate.
- `typing.Protocol` for structural interfaces (testable with fakes);
  `dataclass`/`pydantic`/`TypedDict` for structured data.

## Structure & style

- **`src/` layout** with `pyproject.toml` as the single source of truth. Manage with
  **uv** (Astral) — it replaces pip/venv/pyenv/pip-tools and usually poetry; commit
  the lockfile.
- **Ruff** for both linting and formatting (replaces black + isort + flake8 +
  pyupgrade + pylint). Keep it clean before done.
- **No mutable default arguments** (`def f(x=[])` is a bug) — use `None` + init.
- **Context managers** (`with`) for every resource (files, connections, locks);
  author your own with `contextlib.contextmanager`.
- **`logging` (or structlog), never `print`,** in library/service code. `pathlib`
  over `os.path`; f-strings over `%`/`.format`.

## Errors

- **Define specific exception types** (subclass a project base); catch narrowly,
  never bare `except:`.
- Re-raise with `raise ... from err` to preserve the chain; don't swallow. Use
  `ExceptionGroup`/`except*` when handling concurrent failures from a TaskGroup.

## Concurrency (the GIL story changed)

- **I/O-bound → `asyncio`.** Bound it with `asyncio.Semaphore` or
  **`asyncio.TaskGroup`** (3.11+) rather than unbounded `gather`; use
  `asyncio.timeout()` for deadlines and propagate cancellation.
- **CPU-bound → it depends on the build.** On a **free-threaded (no-GIL) build**
  (officially supported in 3.14, optional since 3.13) real threads now parallelize
  CPU work — but the code must be thread-safe and there's a small single-thread
  penalty. On a standard GIL build, still use `multiprocessing`/`ProcessPoolExecutor`
  for CPU work; threads only for blocking I/O.
- Don't assume the interpreter build — check `sys._is_gil_enabled()` if behavior
  depends on it.

## Testing

- **pytest** with fixtures and `@pytest.mark.parametrize` for table-style cases.
- Mock at the boundary (`unittest.mock`, `responses`/`respx` for HTTP); keep a few
  real integration tests. `pytest-asyncio` for async, `pytest-cov` for coverage.
- A regression test per bug fix.

## Data / ML specifics (when relevant)

- Notebooks for exploration; move reusable logic into typed, tested modules.
- Pin data/model versions and seeds for reproducibility.
- Validate inputs/schemas at boundaries (pydantic / pandera) before processing.
- Prefer **Polars** over pandas for new large-data pipelines (faster, lazy,
  multi-threaded); pandas is still fine where it's already in use.
