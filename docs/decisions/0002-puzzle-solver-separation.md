# 0002. Separate puzzles, heuristics and solvers with traits

**Status:** Accepted

## Context

The C++ design had an `AbstractCube` base class, but `BFSSolver`, `AStarSolver` and every
heuristic took a `Cube2Pieces` directly, and the cube stored the previous move for
pruning. Adding a 3x3x3 would have meant copying every solver. The roadmap needs many
puzzles (2x2x2, 3x3x3, NxNxN, MxNxK cuboids, non-cubic puzzles) and several solver
families (exhaustive, heuristic, iterative deepening, two-phase, learned).

## Decision

Three orthogonal abstractions in `gpss-core`:

- **`Puzzle`**: an implicit graph. Associated `State` (a plain value type, `Clone + Eq +
  Hash`) and `Move` types; `solved`, `is_solved`, `moves`, `apply`, `inverse`, and an
  optional `can_follow(prev, next)` for move-sequence pruning. Moves are puzzle-defined,
  so a 4x4x4 can have wide turns and a 2x3x3 cuboid can restrict some faces to half turns.
- **`Heuristic<P>`**: cost-to-go estimate with `estimate`, `estimate_batch` (default
  loops; neural networks override it to evaluate a whole frontier at once) and
  `is_admissible`.
- **`Solver<P>`**: `solve(puzzle, start, limits) -> Result<Solution, SearchError>`,
  generic over any puzzle, returning moves plus `SearchStats` and an optimality flag.

`Coordinate<P>` (a perfect hash of an abstraction of the state) plus `PatternDatabase`
generically turn any coordinate into an admissible heuristic.

Search bookkeeping (parent, depth, last move) lives in solver-owned nodes, never in states.

## Consequences

- Every solver works on every puzzle, and every heuristic composes with every compatible
  solver. Adding a puzzle is "implement `Puzzle` + coordinates".
- `gpss-core` has no dependency on any concrete puzzle; this is an enforced invariant.
- Trait objects (`Box<dyn Heuristic<P>>`, `Box<dyn Solver<P>>`) let the CLI choose at
  runtime; generic code paths stay monomorphised for speed where it matters.
- Puzzles with symmetry tricks (the 2x2x2 fixed frame, ADR 0004) keep them inside the
  puzzle module and expose plain methods (`regrip`, `solve_scramble`).
