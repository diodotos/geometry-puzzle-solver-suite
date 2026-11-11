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
| `crates/gpss-core/` | Puzzle-agnostic traits (`Puzzle`, `Heuristic`, `Coordinate`, `Solver`, optional `Encodable`) and search (BFS, A*, weighted A*, IDA*, batch weighted A*) plus pattern databases. **Must never depend on a concrete puzzle.** |
| `crates/gpss-puzzles/` | Concrete puzzles. `cube` = shared face-turn notation; `cube2` = 2x2x2 state, coordinates and heuristics; `cube3` = 3x3x3 cubie model, Kociemba coordinates (`cube3::coord`) and the two-phase solver (`cube3::twophase`), Korf's optimal heuristic (`cube3::korf`), the 48 cube symmetries (`cube3::sym`) and symmetry-reduced two-phase tables (`cube3::symcoord`); `cuboid` = generic `AxBxC` sticker model (any cube or cuboid, single-layer turns) and its admissible `CornerHeuristic` (corner projection onto the `cube2` table). |
| `crates/gpss-learn/` | Learned heuristics: training-data generator (`gpss scramble-data`), weights manifests, `OnnxHeuristic` (feature `onnx`, ONNX Runtime loaded at run time), exact evaluation. |
| `training/` | Python/PyTorch DAVI trainer, ONNX export, `local`/`aws`/`gcp` launchers (`python -m gpss_train`, wrapped by `gpss train`). See `docs/training.md`. |
| `weights/` | Shipped networks: `<puzzle>/<name>.onnx` + `.json` manifest (encoding spec, checksum, training config, evaluation). |
| `crates/gpss-viz/` | Renderer-independent viewer model for cuboids: sticker geometry, turn animation, picking, drag-to-turn, playback timeline, input `Controller`, solver hand-off. No GPU code (ADR 0010). |
| `crates/gpss-viewer/` | eframe/wgpu desktop viewer and headless scenario recorder (`gpss-viewer record`); needs Rust 1.95. Input mapping and scenarios: [`docs/viewer.md`](docs/viewer.md). |
| `crates/gpss-cli/` | The `gpss` binary. Thin: parses arguments, builds puzzle + solver, prints text or JSON. |
| `crates/gpss-bench/` | Benchmark suite over the conformance corpus: correctness, counter baselines (`baselines/<puzzle>.json`, checked in CI), local latency/memory reports. Puzzle registrations live in `src/targets/`. Guide: [`docs/benchmarking.md`](docs/benchmarking.md). |
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

# Benchmark suite (docs/benchmarking.md)
./target/release/gpss-bench check            # correctness + exact counters vs baselines (CI; ~15 s)
./target/release/gpss-bench check --update   # accept intended counter changes; commit the JSON
./target/release/gpss-bench run --puzzle cube2 \
  --markdown docs/benchmarks/cube2-gpss-bench.md --json docs/benchmarks/data/cube2-gpss-bench.json

# CLI
./target/release/gpss solve --scramble "R U2 F' R2 U F2 R' U' F R2 U" --solver idastar --heuristic dual
./target/release/gpss solve --scramble "R U R'" --format json
./target/release/gpss pdb --coordinate perfect

# Learned solver (DeepCubeA-style); see docs/training.md
pip install -r training/requirements.txt                 # torch, onnx, onnxruntime, pytest
cargo build --release --features gpss-cli/learn          # gpss with ONNX inference
./target/release/gpss solve --scramble "R U2 F'" --solver bwas --weights weights/cube2/cube2-davi-v1.json
./target/release/gpss evaluate --weights weights/cube2/cube2-davi-v1.json   # exact, all 3.67M states
cargo clippy --all-targets --features gpss-cli/learn -- -D warnings
(cd training && python3 -m pytest -q)                    # trainer + launcher tests (mocked cloud)
python3 -m gpss_train launch aws --bucket s3://b/p --budget-usd 5   # dry run from training/
```

The harness needs only the Python 3.10+ standard library. `training/` needs the packages in
`training/requirements.txt`.

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
- When you add a puzzle or solver, also register it in `crates/gpss-bench/src/targets/`
  (a `Workload` plus `SolverSpec`s; see [`docs/benchmarking.md`](docs/benchmarking.md)),
  run `gpss-bench check --update`, and commit the baseline. Mark slow solvers with
  `ci_max_depth`, non-optimal ones with `max_ratio` when a bound exists, and solvers with
  run-to-run varying node counts with `nondeterministic()`.
- When you make a puzzle trainable: implement `Encodable` (Rust is the only definition of
  the encoding; Python never re-implements puzzles, ADR 0008), add it to the CLI's
  `scramble-data`/`solve`/`evaluate`, and add a preset in `training/gpss_train/davi.py`.
  Changing what `encode` outputs requires bumping `EncodingSpec::version`.
- When you cite an algorithm, add it to [`docs/references.md`](docs/references.md).

## Where things are going

See [`docs/roadmap.md`](docs/roadmap.md). In short: 3x3x3 (Korf IDA* with pattern
databases + Kociemba two-phase), generic NxNxN / MxNxK geometry, a DeepCubeA-style learned
heuristic (done for the 2x2x2: PyTorch training -> ONNX -> Rust batch weighted A*;
AWS/GCP launchers built, not yet run for real; 3x3x3 weights pending a GPU run). Design the code you touch so these remain possible: puzzle-defined moves,
batch heuristic evaluation, fixed-length state encodings.

## Things that are easy to get wrong

- Learned weights are pinned to `EncodingSpec` + the ONNX SHA-256 by their manifest. If you
  re-export, ship the `.onnx` and `.json` together; never hand-edit either.
- Cloud launchers (`training/gpss_train/launch/`) cost money. Never run them with
  `--execute` without the repository owner's explicit go-ahead; tests use a mock runner.
  Credentials come from the provider CLIs' own configuration and must never be committed.
- The `onnx`/`learn` features need Rust >= 1.92 (`ort` 2.0) and an ONNX Runtime >= 1.22
  shared library at run time (`pip install onnxruntime` or `ORT_DYLIB_PATH`). The default
  build has neither requirement.

- The face-turn pruning rule (`FaceTurn::can_follow`) forbids the same face twice and
  the "wrong" order of opposite faces. With only `U R F` there are no opposite pairs, so
  only the first rule matters for `Cube2`.
- Coordinates are only valid in the fixed frame; `debug_assert!`s guard this.
- Timing: CI machines are noisy. Use the harness or `gpss-bench run` on a quiet machine
  for performance claims and record the machine in the report.
- A `gpss-bench` counter diff in CI is a behaviour change, not flakiness: counters are
  deterministic. Don't `--update` the baseline without understanding why the counters moved,
  and say why in the PR.
- `gpss-bench` peak heap is requested bytes from a counting allocator, not RSS; RSS
  comparisons come from `conformance/harness.py`.
- The legacy C++ returns no solution for an empty scramble and aborts (SIGABRT) on
  malformed notation; see `conformance/known_deviations.json`. Don't port those behaviours.
