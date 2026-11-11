# 0001. Port to Rust directly into a clean design

**Status:** Accepted

## Context

The original `CubeSolver` is a C++17 2x2x2 solver written for a course project. Its
solvers and heuristics are hard-wired to `Cube2Pieces`, search state (`prevMove`) lives on
the cube object, moves are a fixed 18-value enum, heuristic tables are global statics
loaded from hard-coded paths, and every search node is a `shared_ptr`. The project's goal
has changed from "memorialise a final project" to "general solver engine for geometry
puzzles", and we want to migrate languages.

Two strategies were considered:

1. **Faithful port, then refactor.** Translate the C++ 1:1 into Rust, prove parity, then
   restructure. Cleaner as an experiment in "migration capability", but the clean design
   would be written twice, and a 1:1 port fights the borrow checker on the
   `shared_ptr` node graph.
2. **Port straight into the target design.** Write the Rust in its final shape, and prove
   behavioural parity with a black-box corpus rather than structural similarity.

## Decision

Option 2. The original repository stays untouched as a record of the project; this
repository imports its history and keeps the C++ under `legacy/cpp/` as the executable
baseline.

## Consequences

- Parity is *behavioural*: the conformance harness (ADR 0003) is the only thing that can
  say the migration is correct, so it has to be independent and thorough.
- The C++ remains buildable so the harness can always compare both implementations.
- Some legacy behaviours (crashes on malformed input, see `known_deviations.json`) are
  deliberately not ported.
