# The C++ -> Rust migration

This repository doubles as a case study in migrating a codebase across languages with an
AI agent, and in proving that the migration preserved behaviour.

## Starting point

`legacy/cpp/` is the original CubeSolver (May 2024), an optimal 2x2x2 solver with BFS and
A* using orientation, permutation, "dual" (max of both) and perfect heuristics. Its history
is preserved in this repository's git log.

Two generated files from the original repository, `perfectLookup.txt` (55 MB) and
`heuristic_evaluation.txt` (77 MB), are no longer in the tree (they remain in git history).
`CubeSolver` rebuilds `perfectLookup.txt` the first time `--solver astarperf` runs (about a
minute), and `heuristic_evaluation.txt` is the output of its `heuristic` mode. Pattern
databases are treated as build artefacts throughout the suite.

## Method

1. **Freeze the old behaviour as data, not code.** Before writing any Rust, build an
   oracle that is independent of both implementations (ADR 0003): a sticker-level reference
   model and an exhaustive 2x2x2 corpus with exact optimal lengths.
2. **Run the legacy implementation against it.** This found 8 real deviations in the C++
   (empty scramble returns no solution; repeated whitespace and malformed tokens crash
   with SIGABRT). They are recorded in `conformance/known_deviations.json` rather than
   silently tolerated.
3. **Port straight into the target design** (ADR 0001): generic traits instead of class
   hierarchies (ADR 0002), a fixed-frame 2x2x2 instead of per-node normalisation
   (ADR 0004).
4. **Hold the new implementation to the same oracle** with no deviations allowed.

## Concept mapping

| C++ | Rust |
|-----|------|
| `AbstractCube`, `Cube2Pieces` | `Puzzle` trait, `Cube2` + `Cube2State` |
| `Move` enum (19 values incl. `None`) | `FaceTurn { face, turn }`, puzzle-defined move list |
| `prevMove` on the cube + `getNextMoves` | `Puzzle::can_follow`, previous move kept in search nodes |
| `normalize()` in the hash | `Cube2::regrip` once, then `U R F`-only search |
| `cubeHash`, `permutationHash`, `orientationHash` | `FullCoord`, `PermutationCoord`, `OrientationCoord` |
| global lookup maps loaded from `*.txt` | `PatternDatabase` built in memory in < 1 s |
| `BFSSolver`, `AStarSolver` over `shared_ptr` nodes | generic `Bfs`, `AStar`, plus `IdaStar` |
| text output parsed by scripts | `--format json`, structured `SearchStats` |
| `abort()` on bad input | `ParseMoveError` with token position, exit code 2 |

## Result

All Rust configurations pass 233/233 cases with exact optimal lengths; see
[`benchmarks/cube2-baseline.md`](benchmarks/cube2-baseline.md) for timing and memory.
