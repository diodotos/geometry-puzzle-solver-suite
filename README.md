# Geometry Puzzle Solver Suite

[![CI](https://github.com/diodotos/geometry-puzzle-solver-suite/actions/workflows/ci.yml/badge.svg)](https://github.com/diodotos/geometry-puzzle-solver-suite/actions/workflows/ci.yml)

A Rust toolkit for solving permutation puzzles - Rubik's-type cubes today, arbitrary 3D
geometry puzzles tomorrow - with interchangeable **puzzles**, **heuristics** and
**search engines**, and a language-agnostic conformance suite that proves the solvers are
right.

```console
$ gpss solve --scramble "R U2 F' R2 U F2 R' U' F R2 U" --solver idastar --heuristic dual
Solution: U' R2 F' U F2 U R2 F' U2
Length: 9 (optimal)
Solver: idastar(dual)
Expanded: 1217  Generated: 7285  Peak stored: 9  Iterations: 4
```

## Why this exists

This project began as [CubeSolver](https://github.com/diodotos/CubeSolver), a C++ optimal
2x2x2 solver written for a university course on computational intelligence for games. It
is now a place to explore two questions:

1. **Can a codebase be migrated across languages, by an AI agent, with proof that nothing
   broke?** The original C++ lives on in [`legacy/cpp/`](legacy/cpp/). Before any Rust was
   written, its behaviour was frozen into an independent, exhaustively verified test corpus;
   the Rust rewrite is held to the same oracle. See [`docs/migration.md`](docs/migration.md).
2. **What does a general interface for "puzzles that solve themselves" look like?** A
   puzzle only has to say what its states and moves are. Solvers then find solutions either
   by *complete search* (BFS, A*, IDA* with pattern databases) or, soon, by *self-play*:
   learning a cost-to-go function from nothing but the move rules, as in DeepCubeA.
   See [`docs/roadmap.md`](docs/roadmap.md).

## Status

| | |
|-|-|
| 2x2x2 | Done. BFS, A*, weighted A*, IDA*; orientation / permutation / dual / perfect pattern databases; 233/233 conformance cases with exact optimal lengths |
| 3x3x3 | Cubie model and Kociemba two-phase solver (solutions of 20 moves or fewer; median about 0.07 s on random scrambles); 117/117 conformance cases. Korf optimal IDA* (corner + two 6-edge pattern databases, 87 MB, built in about 5 s and cached on disk): optimal up to about 15 moves in under a minute |
| NxNxN, MxNxK | Generic sticker model (`--puzzle 4x4x4`, `--puzzle 3x3x2`): single-layer turns, half-turn-only axes, exact BFS / IDA* for short scrambles; no heuristics yet |
| Learned search | Planned: DeepCubeA-style DAVI training (PyTorch -> ONNX), batch weighted A*, shipped 2x2x2 weights, AWS/GCP training launchers |

## Quick start

```bash
cargo build --release
./target/release/gpss solve --scramble "F2 U R' U F U' F2 R2 F'"          # IDA* + dual heuristic
./target/release/gpss solve --scramble "R U R'" --solver bfs --format json
./target/release/gpss solve --scramble "R U2 F'" --solver astar --weight 2  # weighted A*, not optimal
./target/release/gpss solve --puzzle cube3 --solver twophase --scramble "R U R' U' F2 D L2 B"  # 3x3x3, Kociemba
./target/release/gpss solve --puzzle 3x3x2 --scramble "U R2 F2 D'"   # generic cuboid (domino)
./target/release/gpss scramble --length 20 --seed 1
./target/release/gpss pdb --coordinate perfect                             # full 2x2x2 distance table
```

Solvers: `bfs`, `astar` (`--weight` for weighted A*), `idastar`. Heuristics: `zero`,
`orientation`, `permutation`, `dual`, `perfect`. Limits: `--max-depth`, `--max-expanded`,
`--timeout`. Exit codes: `0` solved, `2` invalid input, `3` no solution within limits.

As a library:

```rust
use gpss_core::search::{IdaStar, SearchLimits};
use gpss_puzzles::cube::parse_sequence;
use gpss_puzzles::cube2::{Cube2, Cube2Heuristic};

let cube = Cube2::new();
let solver = IdaStar::new(Cube2Heuristic::Dual.build(&cube));
let scramble = parse_sequence("R U2 F' R2 U F2 R' U' F R2 U")?;
let solution = cube.solve_scramble(&scramble, &solver, &SearchLimits::default())?;
assert!(solution.optimal);
```

## Design in one paragraph

`gpss-core` defines three traits and knows nothing else: a [`Puzzle`](crates/gpss-core/src/puzzle.rs)
is an implicit graph (states, puzzle-defined moves, `apply`, `inverse`, optional move
pruning); a [`Heuristic<P>`](crates/gpss-core/src/heuristic.rs) estimates distance to the
goal, one state or a whole batch at a time (so a neural network fits the same slot as a
lookup table); a [`Solver<P>`](crates/gpss-core/src/search/mod.rs) returns moves, search
statistics and whether the result is provably optimal. Pattern databases turn any
`Coordinate` of a puzzle into an admissible heuristic. Concrete puzzles live in
`gpss-puzzles`. Details: [`docs/architecture.md`](docs/architecture.md) and the
[decision records](docs/decisions/).

## Performance (2x2x2, median search time per solve)

| optimal depth | legacy C++ A* (dual) | Rust A* (dual) | Rust IDA* (dual) |
|---:|---:|---:|---:|
| 8  | 12.0 ms  | 0.3 ms  | 0.1 ms |
| 11 (God's number) | 730 ms | 20.6 ms | 4.6 ms |

The legacy build is unoptimised, as in the original project. Methodology, memory numbers
and caveats: [`docs/benchmarks/cube2-baseline.md`](docs/benchmarks/cube2-baseline.md).

## Testing

```bash
cargo test --release                                                   # unit + property tests
python3 conformance/harness.py --impl gpss --solver idastar --heuristic dual   # black-box corpus
python3 conformance/harness.py --impl legacy-cpp --solver astardual            # the original C++
```

The conformance corpus is generated by an exhaustive search of all 3,674,160 2x2x2
positions using an independent sticker-level model, and checked against the published
depth distribution. See [`docs/testing.md`](docs/testing.md).

## Documentation

| | |
|-|-|
| [`AGENTS.md`](AGENTS.md) | Start here if you are an AI agent (or a human who likes checklists): commands, invariants, conventions |
| [`docs/architecture.md`](docs/architecture.md) | Crates, traits, request flow, extension points |
| [`docs/decisions/`](docs/decisions/) | Why things are the way they are (ADRs) |
| [`docs/theory/`](docs/theory/) | Search algorithms, pattern databases, the 2x2x2, Korf, Kociemba, DeepCubeA |
| [`docs/testing.md`](docs/testing.md) | Test layers, corpus tags, CI vs benchmarks |
| [`docs/migration.md`](docs/migration.md) | The C++ -> Rust migration and how it was verified |
| [`docs/roadmap.md`](docs/roadmap.md) | What's next and the long-term vision |
| [`docs/references.md`](docs/references.md) | Academic work this builds on |

## Acknowledgements and references

This project stands on published research; full citations are in
[`docs/references.md`](docs/references.md).

- **DeepCubeA** - F. Agostinelli, S. McAleer, A. Shmakov, P. Baldi. *Solving the Rubik's
  cube with deep reinforcement learning and search.* Nature Machine Intelligence 1,
  356-363 (2019). The blueprint for the learned solver.
- **IDA\*** - R. E. Korf. *Depth-first iterative-deepening: An optimal admissible tree
  search.* Artificial Intelligence 27(1), 97-109 (1985).
- **Optimal Rubik's Cube solving with pattern databases** - R. E. Korf. *Finding Optimal
  Solutions to Rubik's Cube Using Pattern Databases.* AAAI-97 (1997).
- **Pattern databases** - J. C. Culberson, J. Schaeffer. *Pattern Databases.*
  Computational Intelligence 14(3), 318-334 (1998).
- **Two-phase algorithm** - H. Kociemba, [kociemba.org](http://kociemba.org/cube.htm);
  also the cubie and coordinate conventions used here.
- **A\*** - Hart, Nilsson, Raphael (1968); **weighted A\*** - Pohl (1970).

## License

MIT - see [`LICENSE`](LICENSE). `legacy/cpp/argparse.h` is by p-ranav (MIT).
