# Architecture

```text
                +-------------------------------------------------+
                |                  gpss-cli (bin)                 |
                |  argument parsing, text/JSON output, exit codes |
                +---------------------------+---------------------+
                                            |
               +----------------------------+----------------------------+
               |                                                         |
   +-----------v-------------+                         +-----------------v---------------+
   |      gpss-puzzles       |  implements traits of   |            gpss-core            |
   |  cube   (notation)      | ----------------------> |  Puzzle      Heuristic<P>       |
   |  cube2  (state, coords, |                         |  Coordinate<P>  PatternDatabase |
   |          heuristics)    |                         |  Solver<P>: Bfs AStar IdaStar   |
   |             BatchWeightedAStar  |
   |  Encodable (optional)           |
   +-------------------------+                         +---------------------------------+

   conformance/  (Python, stdlib)  -- runs any CLI as a black box, validates with an
                                      independent sticker model
   legacy/cpp/   (C++17)           -- the original CubeSolver, kept as the baseline
```

`gpss-learn` (data generation, manifests, ONNX inference) sits beside `gpss-puzzles` and
depends only on `gpss-core`. Dependencies point one way: `cli -> {puzzles, learn} -> core`. `gpss-core` never names a concrete
puzzle (see `AGENTS.md`, invariant 1).

## The three traits

### `Puzzle` (`gpss-core/src/puzzle.rs`)

```rust
pub trait Puzzle {
    type State: Clone + Eq + Hash + Debug;
    type Move: Copy + Eq + Hash + Debug + Display;
    fn name(&self) -> &str;
    fn solved(&self) -> Self::State;
    fn is_solved(&self, state: &Self::State) -> bool;
    fn moves(&self) -> &[Self::Move];
    fn apply(&self, state: &Self::State, mv: Self::Move) -> Self::State;
    fn inverse(&self, mv: Self::Move) -> Self::Move;
    fn can_follow(&self, prev: Self::Move, next: Self::Move) -> bool { true } // optional pruning
}
```

The puzzle *value* (not just the type) defines the graph, so one type can describe a
family (`Cube::new(4, 4, 4)`, a cuboid with restricted moves, a puzzle loaded from a
geometry spec). States are plain values; solvers own their nodes.

### `Heuristic<P>` (`gpss-core/src/heuristic.rs`)

`estimate(&state) -> u32`, `estimate_batch(&[state], &mut Vec<u32>)` (override for
neural networks), `estimate_batch_real(&[state], &mut Vec<f32>)` (fractional estimates,
used by BWAS; defaults to `estimate_batch`), `is_admissible()`. Combinators: `ZeroHeuristic`, `MaxHeuristic`;
`PdbHeuristic<C>` turns any `Coordinate<P>` into a heuristic via a `PatternDatabase`.

### `Solver<P>` (`gpss-core/src/search/`)

`solve(&puzzle, &start, &limits) -> Result<Solution<P::Move>, SearchError>`. A `Solution`
carries the moves, `SearchStats` and whether optimality is guaranteed. `SearchError`
distinguishes "no solution exists" from "a limit was hit" and carries stats either way.

## Request flow (`gpss solve --scramble "R U2 F'" --solver idastar --heuristic dual`)

1. CLI parses the scramble with `cube::parse_sequence` (rejects bad tokens with position).
2. `Cube2Heuristic::Dual.build(&puzzle)` builds two small PDBs (~1 ms).
3. `Cube2::solve_scramble` computes the scrambled state, regrips it into the fixed frame
   (`docs/theory/cube2.md`), calls the generic `IdaStar`, and maps moves back to the user's
   grip.
4. CLI prints the solution and stats (text or JSON).

## Extension points

| want to add | do |
|-------------|----|
| a puzzle | new module in `gpss-puzzles` implementing `Puzzle`; coordinates + heuristics alongside; corpus generator + reference model support in `conformance/` |
| a heuristic | implement `Heuristic<P>` (or a `Coordinate<P>` and reuse `PdbHeuristic`) |
| a solver | implement `Solver<P>` in `gpss-core/src/search/` |
| a CLI command | a subcommand in `gpss-cli`; keep logic in the libraries |

### `Encodable` (`gpss-core/src/encode.rs`, optional)

`encoding() -> EncodingSpec` and `encode(&state, &mut [u8])`: a fixed-length categorical
state encoding for learned heuristics. Rust is the only definition of it (ADR 0008).

## Learned solver

```text
 gpss scramble-data (Rust)  --encoded states + successors-->  training/ (PyTorch DAVI)
                                                                     | export
 gpss solve --solver bwas --weights m.json  <--  weights/<puzzle>/m.{onnx,json}
   OnnxHeuristic<P>: Heuristic<P>  (gpss-learn, feature onnx; checks encoding + sha256)
   BatchWeightedAStar<H>: Solver<P> (gpss-core)
```

See [`training.md`](training.md) and [`theory/deepcubea.md`](theory/deepcubea.md).

## Crates

- `gpss-learn` - data generator, weights manifests, exact evaluation; `OnnxHeuristic`
  behind the `onnx` feature (ONNX Runtime loaded at run time). `gpss-cli`'s `learn`
  feature enables it.
- `training/` (Python) - PyTorch DAVI trainer, ONNX export, `local`/AWS/GCP launchers.

## Planned crates

- `gpss-bench` - corpus-driven benchmark suite: CI counter baselines and the local reports in `docs/benchmarks/` (see [benchmarking.md](benchmarking.md)).
