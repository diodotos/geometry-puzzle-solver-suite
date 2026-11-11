# 0003. Prove the migration with a black-box conformance corpus

**Status:** Accepted

## Context

A clean-design port (ADR 0001) cannot be checked by comparing code structure. We need an
oracle that (a) works on any implementation in any language, (b) does not share code or
assumptions with the solvers it checks, and (c) knows the *exact* optimal answer, so that
"optimal" solvers can be held to it.

## Decision

- An **independent reference model** (`conformance/reference/cube_model.py`): a
  sticker-level NxNxN cube defined by 3D coordinates and rotation matrices. It shares
  nothing with the cubie representation used by the solvers.
- An **exhaustively verified corpus** (`conformance/corpus/cube2.jsonl`): the generator
  BFSes the whole 2x2x2 space with the reference model, checks the depth histogram against
  the published distribution before writing anything, and emits cases at every exact
  optimal depth 0-11 (including antipodes, the top of the range), random scrambles,
  edge-case notation, and invalid inputs that must be rejected.
- A **process-level harness** (`conformance/harness.py`) that runs any CLI as a black box,
  applies the returned solution to the reference model, checks it solves the cube (up to
  whole-cube rotation), checks exact optimality for optimal solvers, and records wall time,
  solver-reported time and peak RSS (via `wait4`).
- **Documented deviations** (`conformance/known_deviations.json`) instead of silent skips.

## Consequences

- The same corpus scores C++, Rust, and any future implementation (including learned
  solvers, which are scored on validity and optimality gap rather than exact length).
- Python stdlib only, so it runs anywhere CI does.
- The harness measures whole-process cost, including table construction; in-process
  micro-benchmarks complement it (see `docs/testing.md`).
