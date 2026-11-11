# AGENTS.md

Guidance for AI coding agents (and humans) working in this repository. Read this file
first, then [`docs/architecture.md`](docs/architecture.md). Everything here is meant to be
actionable; the "why" lives in [`docs/decisions/`](docs/decisions/).

## What this project is

The Geometry Puzzle Solver Suite (`gpss`) is a Rust workspace for solving
permutation puzzles: Rubik's-type cubes today, and arbitrary 3D geometry puzzles over time.
It started as [`CubeSolver`](https://github.com/diodotos/CubeSolver), a C++ 2x2x2 solver
written as a university final project, which is preserved unchanged in
[`legacy/cpp/`](legacy/cpp/) as the migration baseline.

## Repository map

| Path | What it is |
|------|------------|
| `crates/gpss-core/` | Puzzle-agnostic traits (`Puzzle`, `Heuristic`, `Coordinate`, `Solver`) and search (BFS, A*, weighted A*, IDA*) plus pattern databases. **Must never depend on a concrete puzzle.** |
| `crates/gpss-puzzles/` | Concrete puzzles. `cube` = shared face-turn notation; `cube2` = 2x2x2 state, coordinates and heuristics; `cube3` = 3x3x3 cubie model, Kociemba coordinates (`cube3::coord`) and the two-phase solver (`cube3::twophase`), Korf's optimal heuristic (`cube3::korf`); `cuboid` = generic `AxBxC` sticker model (any cube or cuboid, single-layer turns). |
| `crates/gpss-cli/` | The `gpss` binary. Thin: parses arguments, builds puzzle + solver, prints text or JSON. |
| `conformance/` | Language-agnostic black-box test corpus and harness. Independent Python reference model. |
| `legacy/cpp/` | The original C++ solver. Do not modify except to keep it building or to drop generated artefacts it regenerates itself. |
| `docs/` | Architecture, decisions (ADRs), theory, testing, migration, roadmap, references. |

## Commands

```bash
cargo build --release                       # build everything
cargo test --release                        # unit + property tests (~3 s; release is much faster)
cargo fmt --all --check                     # formatting (CI enforces)
cargo clippy --all-targets -- -D warnings   # lints (CI enforces)

# Black-box conformance against the 233-case 2x2x2 corpus
python3 conformance/harness.py --impl gpss --solver idastar --heuristic dual
python3 conformance/harness.py --impl gpss --solver bfs --tags short,edge
python3 conformance/harness.py --impl legacy-cpp --solver astardual   # builds legacy/cpp with make

# Regenerate the corpus (deterministic; ~25 s, pure Python)
python3 conformance/generate_cube2.py
python3 conformance/generate_cube3.py   # ~2 s
python3 conformance/generate_cuboid.py  # <1 s
python3 conformance/harness.py --impl gpss --solver twophase --corpus conformance/corpus/cube3.jsonl

# CLI
./target/release/gpss solve --scramble "R U2 F' R2 U F2 R' U' F R2 U" --solver idastar --heuristic dual
./target/release/gpss solve --scramble "R U R'" --format json
./target/release/gpss pdb --coordinate perfect
```

The harness needs only the Python 3.10+ standard library.

## Invariants you must preserve

1. **Search knows nothing about puzzles.** `gpss-core` code may only use the `Puzzle`
   trait. If you need puzzle-specific behaviour in search, add a trait method with a
   sensible default, not a downcast or a special case.
2. **States hold no search bookkeeping.** No `prev_move`, parent pointers or costs on
   puzzle states; those belong to the search node (see ADR 0002).
3. **Optimal means proven optimal.** A solver may set `Solution::optimal = true` only if
   it is guaranteed (BFS; A*/IDA* with weight 1 and an admissible heuristic). Learned or
   weighted solvers must report `false`.
4. **Heuristics declare admissibility honestly.** `Heuristic::is_admissible` gates the
   optimality claim. A neural network heuristic is never admissible.
5. **The conformance oracle stays independent.** `conformance/reference/` must not import
   or mirror solver code. It is a sticker-level model precisely so that a bug in the cubie
   model cannot hide itself. Never "fix" a failing case by editing the corpus unless the
   generator itself is shown to be wrong.
6. **Known deviations are documented, not hidden.** If an implementation intentionally or
   historically differs, record it in `conformance/known_deviations.json` with a reason.
7. **2x2x2 search runs in a fixed frame.** `Cube2` searches with `U R F` only, after
   `Cube2::regrip` puts the `DBL` corner home. Solutions are translated back with
   `Regrip::to_original`. Don't add `D L B` to the search moves; it multiplies the
   branching factor for nothing (see `docs/theory/cube2.md`).

## Conventions

- Rust 2021, MSRV in `Cargo.toml`. `cargo fmt` defaults. Clippy clean with `-D warnings`.
- Errors: `thiserror` enums in libraries; the CLI maps them to exit codes
  (0 ok, 2 invalid input, 3 no solution within limits).
- Public items get doc comments that say *what* and, where non-obvious, *why*.
  Long-form reasoning goes in `docs/`, not in code comments.
- Tests live next to the code (`mod tests`) and use `proptest` for algebraic properties
  (inverse, order, invariants). Anything testing the whole space or a large table must
  stay under a few seconds in `--release`.
- When you add a puzzle: implement `Puzzle`, add coordinates/heuristics in the puzzle
  module, add a corpus generator under `conformance/`, extend the harness size map, and
  write `docs/theory/<puzzle>.md`.
- When you add a solver: implement `Solver<P>` in `gpss-core/src/search/`, report
  `SearchStats` faithfully (expanded / generated / peak stored / iterations), respect all
  `SearchLimits`, and add it to the table in `search/mod.rs` and `docs/theory/search.md`.
- When you cite an algorithm, add it to [`docs/references.md`](docs/references.md).

## Where things are going

See [`docs/roadmap.md`](docs/roadmap.md). In short: 3x3x3 (Korf IDA* with pattern
databases + Kociemba two-phase), generic NxNxN / MxNxK geometry, a DeepCubeA-style learned
heuristic (PyTorch training -> ONNX -> Rust batch-weighted A*), and AWS/GCP training
launchers. Design the code you touch so these remain possible: puzzle-defined moves,
batch heuristic evaluation, fixed-length state encodings.

## Things that are easy to get wrong

- The face-turn pruning rule (`FaceTurn::can_follow`) forbids the same face twice and
  the "wrong" order of opposite faces. With only `U R F` there are no opposite pairs, so
  only the first rule matters for `Cube2`.
- Coordinates are only valid in the fixed frame; `debug_assert!`s guard this.
- Timing: CI machines are noisy. Use the harness or local benchmarks on a quiet machine
  for performance claims and record the machine in the report.
- The legacy C++ returns no solution for an empty scramble and aborts (SIGABRT) on
  malformed notation; see `conformance/known_deviations.json`. Don't port those behaviours.
