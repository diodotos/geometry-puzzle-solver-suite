# Testing and benchmarking strategy

Three layers, from fastest to most independent.

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
| `short` / `mid` / `top` | depths 1-4 / 5-8 / 9-11 (top-of-range includes antipodes) |
| `random` | random scrambles of random length |
| `edge` | identity sequences, whitespace, 200-move scramble, all 18 moves |
| `invalid` | malformed notation that must be rejected with a non-zero, non-crash exit |

## 3. Benchmarks

Machine-dependent numbers are produced on a quiet machine and committed as reports in
[`docs/benchmarks/`](benchmarks/) with the machine description (ADR 0006). The harness
already buckets by optimal depth (median, p95, max solve time; peak RSS). Planned:

- `gpss-bench`: in-process timing for sub-millisecond solves (many repetitions per case),
  table build times, and per-node cost;
- machine-independent counters (`expanded`, `generated`, `peak_stored`) tracked per depth,
  which can safely run in CI as regression checks;
- 3x3x3 buckets by optimal depth (up to 20 [Rok13]) and two-phase length/time trade-offs;
- learned solvers: solution length vs optimal, nodes expanded, batch size vs latency.

## CI

`.github/workflows/ci.yml`: fmt, clippy (`-D warnings`), tests, the full corpus against
the Rust CLI (IDA* and A* with the dual heuristic; BFS up to depth 8), and a build plus
short-range run of the legacy C++.
