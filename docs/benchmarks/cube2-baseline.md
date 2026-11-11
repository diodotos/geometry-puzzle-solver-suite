# 2x2x2 baseline: legacy C++ vs Rust

Whole-process measurements from `conformance/harness.py` on the 233-case corpus (159 exact-
depth cases, 15 per depth 2-11 plus all 9 depth-1 states, 60 random scrambles, 14 edge
cases). Each case is a fresh process, so peak RSS includes table construction.

- Machine: 8 vCPU Intel Xeon (cloud VM), 31 GB RAM, Linux, no GPU.
- Rust: `cargo build --release` (rustc 1.9x). C++: `g++ -g -Wall --std=c++17` (the legacy
  Makefile, **no `-O` flag**, so the C++ numbers are unoptimised - this is how the
  original project was built and benchmarked).
- "solve s" is the time reported by the program for the search itself (excludes building
  heuristic tables); medians per optimal depth.

| depth | legacy `astardual` | Rust `astar` dual | Rust `idastar` dual | Rust `bfs` | Rust `idastar` perfect |
|------:|-------------------:|------------------:|--------------------:|-----------:|-----------------------:|
| 6  | 0.0006 | 0.0000 | 0.0000 | 0.0041 | 0.0000 |
| 7  | 0.0021 | 0.0000 | 0.0000 | 0.0227 | 0.0000 |
| 8  | 0.0120 | 0.0003 | 0.0001 | 0.1409 | 0.0000 |
| 9  | 0.0593 | 0.0011 | 0.0003 | 0.5027 | 0.0000 |
| 10 | 0.1701 | 0.0038 | 0.0010 | 1.0611 | 0.0000 |
| 11 | 0.7298 | 0.0206 | 0.0046 | 2.5145 | 0.0000 |

Peak RSS at depth 11 (max over cases): legacy `astardual` 63.6 MB, Rust `astar` dual
15.2 MB, Rust `idastar` dual 11.3 MB, Rust `bfs` 306.7 MB, Rust `idastar` perfect 38.2 MB
(the 3.7 MB table plus process overhead; the table is rebuilt in ~0.85 s per process).

Correctness: every Rust configuration passed 233/233, with exact optimal lengths. Legacy
C++ passed 225 and failed 8 known-deviation edge cases (empty scramble, repeated
whitespace, malformed notation; see `conformance/known_deviations.json`).

## Caveats

- The C++ was not compiled with optimisation; an `-O2` build would narrow the gap. The
  comparison that matters for the migration is correctness plus the machine-independent
  `expanded`/`generated` counters, which the Rust JSON output reports.
- Times under 1 ms are at the resolution of process-level measurement. In-process
  micro-benchmarks in [`cube2-gpss-bench.md`](cube2-gpss-bench.md) give precise numbers
  for short solves.

## Reproduce

```bash
cargo build --release
for cfg in "astar --heuristic dual" "idastar --heuristic dual" "bfs" "idastar --heuristic perfect"; do
  python3 conformance/harness.py --impl gpss --no-build --solver $cfg
done
python3 conformance/harness.py --impl legacy-cpp --solver astardual
```
