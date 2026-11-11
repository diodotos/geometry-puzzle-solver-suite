# 0006. CI checks correctness; benchmarks run on dedicated machines

**Status:** Accepted

## Context

Shared CI runners vary run-to-run by tens of percent and have limited memory; full BFS
over the 2x2x2 and future 3x3x3 pattern-database builds are too heavy or too noisy for
per-commit timing.

## Decision

GitHub Actions runs on every push and PR: `cargo fmt --check`, `cargo clippy -D
warnings`, `cargo test --release`, and the conformance corpus against the Rust CLI with a
fast optimal solver (IDA* + dual heuristic, all 233 cases, a few seconds), plus the legacy
C++ build and its edge cases. Timing and memory benchmarks are run on a quiet dedicated
machine and committed as reports under `docs/benchmarks/` together with the machine
description.

## Consequences

- CI is fast and deterministic; a red CI always means a correctness problem.
- Performance regressions are caught by re-running the benchmark recipe, not by CI. If
  this proves insufficient, a scheduled workflow on a self-hosted runner can be added.
