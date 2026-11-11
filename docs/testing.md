# Testing and benchmarking strategy

Three layers, from fastest to most independent, plus the benchmark suite that spans
correctness, regression counters and performance.

## 1. Unit and property tests (`cargo test --release`)

Next to the code. They check algebra and invariants that any correct implementation has:

- every face turn has order 4; `(R U R' U')` has order 6; scramble + inverse = identity
  (property test over random lengths and seeds);
- twist sum is 0 mod 3; opposite-face pairs `R L'` are whole-cube rotations;
- regripping always puts `DBL` home, and solutions mapped back solve the original cube;
- the full 2x2x2 distance histogram equals the published distribution;
- sub-problem tables are complete (729 / 5,040 entries; diameters 6 / 7);
- `dual <= perfect` on random states, and `dual == max(orientation, permutation)`;
- BFS, A*, IDA* with several heuristics all return the perfect-table length;
- weighted A* is not flagged optimal; node and depth limits are enforced;
- notation parsing: round trip, whitespace, empty input, and a list of malformed tokens.

The whole suite runs in about 3 s in release mode.

## 2. Black-box conformance (`conformance/`)

See [`conformance/README.md`](../conformance/README.md) and ADR 0003. An independent Python
sticker model and an exhaustively verified corpus check any CLI: solution validity, exact
optimality, rejection of invalid input, timing and peak memory. Run it for every solver
configuration you touch.

| tag | meaning |
|-----|---------|
| `depth`, `dN` | state at exact optimal distance N (0-11) |
| `short` / `mid` / `top` | depths 1-5 / 6-9 / 10-11 (top-of-range includes antipodes) |
| `random` | random scrambles of random length |
| `edge` | identity sequences, whitespace, 200-move scramble, all 18 moves |
| `invalid` | malformed notation that must be rejected with a non-zero, non-crash exit |

## 3. Benchmarks

`crates/gpss-bench`, documented in [benchmarking.md](benchmarking.md) (ADR 0007). It
reads the same corpus and runs every registered (puzzle, solver) pair in-process.

| part | where it runs | what it asserts or measures |
|------|---------------|-----------------------------|
| correctness | CI and local | rejects invalid input; returned moves solve the state; optimal solvers return exactly the proven optimum; nothing is shorter than the optimum; weighted solvers within their ratio; `Solution::optimal` matches `Solver::is_optimal` |
| counter baselines | CI (`cargo test --release`, `gpss-bench check`) | per case: `length`, `expanded`, `generated`, `peak_stored`, `iterations` equal `crates/gpss-bench/baselines/<puzzle>.json` |
| performance report | local, committed to [`docs/benchmarks/`](benchmarks/) | latency (median / p95 / max) per depth bucket (trivial / short / mid / top) and per exact depth, ns per expanded node, peak heap per solve, solver setup, PDB build time, move and heuristic-lookup cost, with the machine description |

Process-level numbers (fresh process per case, peak RSS, the legacy C++) still come from
the conformance harness ([`benchmarks/cube2-baseline.md`](benchmarks/cube2-baseline.md)).

Still to come as the puzzles land: 3x3x3 buckets by optimal depth (up to 20 [Rok13]) and
two-phase length/time trade-offs; learned solvers' solution length vs optimal, nodes
expanded, and batch size vs latency.

## CI

`.github/workflows/ci.yml`: fmt, clippy (`-D warnings`), tests (including the
`gpss-bench` counter baselines), `gpss-bench check` for a readable per-bucket summary, the
full corpus against the Rust CLI (IDA* and A* with the dual heuristic; BFS up to depth 8),
and a build plus short-range run of the legacy C++.

If CI fails in `gpss-bench`: a correctness failure is a bug; a counter difference means
the search explores differently. If that is intended, run
`cargo run --release -p gpss-bench -- check --update` and commit the baseline with an
explanation (see [benchmarking.md](benchmarking.md#baselines-and-the-ci-gate)).

## 3x3x3 corpus

`conformance/corpus/cube3.jsonl` (117 cases, from `conformance/generate_cube3.py`) is
built from the same independent reference model. The 3x3x3 cannot be enumerated, so
exact optimal distances (1..7) come from meet-in-the-middle: a radius-4 ball around
solved (checked against the published counts 1, 18, 243, 3240, 43239) meets a radius-3
ball around the scramble. Superflip is included with its published optimum of 20.
Long random scrambles have `optimal_length: null`; for those the harness checks only
that the solution is correct. CI runs two-phase on the whole corpus and generic IDA*
on cases of depth 5 or less through the CLI harness; `gpss-bench` additionally baselines
two-phase on the whole corpus and Korf IDA* on every exact case (depth 7 or less).

## Cuboid corpus

`conformance/corpus/cuboid.jsonl` (126 cases, from `conformance/generate_cuboid.py`) covers
2x2x2, 3x3x2, 2x2x3, 2x3x4, 3x3x3 and 4x4x4 under `--puzzle AxBxC`. Its exact lengths are
single-layer-metric distances up to rotation, taken from a BFS ball in the reference model.
CI runs IDA* on the whole file through the CLI harness; `gpss-bench` baselines IDA* on the
2x2x2 and 3x3x2 cases of depth 4 or less. See `docs/theory/cuboid.md`.
