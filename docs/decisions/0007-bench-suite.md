# 0007. One benchmark suite, driven by the conformance corpus, with counter baselines in CI

**Status:** Accepted (refines 0006)

## Context

ADR 0006 keeps timing out of CI. The process-level harness measures whole-process time
and RSS, which hides sub-millisecond solves and table construction inside process
start-up, and nothing caught a change in search behaviour that leaves answers correct
but explores many more nodes. Several puzzles (3x3x3) and solver families (two-phase,
learned) are being added in parallel and need the same measurements.

## Decision

- A `gpss-bench` crate with a small custom harness (no criterion: we need per-case,
  per-depth-bucket statistics over a corpus and a committed report format, not per-function
  regression detection) measures in-process latency, per-node cost, peak heap (counting
  global allocator), setup cost, PDB build time and primitive costs.
- Cases come from `conformance/corpus/<puzzle>.jsonl`; there is one source of truth for
  optimal lengths and depth tags.
- Puzzles register a `Workload` and `SolverSpec`s through a `Registry` that only uses
  `gpss-core` traits.
- The search counters (`length`, `expanded`, `generated`, `peak_stored`, `iterations`) are
  deterministic, so they are compared exactly against committed baselines in CI.
  Wall-clock and memory results are produced locally and committed as reports.

## Consequences

- Any change to search behaviour shows up in review as a baseline diff.
- Exact comparison means intentional changes need `gpss-bench check --update`; cheap,
  and it forces the author to state why counters moved.
- Nondeterministic solvers opt out of counter baselines but are still correctness-checked.
- CI time grows with the corpus; slow solvers are capped with `ci_max_depth`.
