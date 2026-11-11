# Benchmarking guide (`gpss-bench`)

`crates/gpss-bench` is the in-process benchmark and regression suite. It answers three
questions for every registered (puzzle, solver) pair, using the cases in
`conformance/corpus/<puzzle>.jsonl`:

1. **Is it correct?** Every case: invalid input rejected, the returned moves solve the
   start state, optimal solvers hit the corpus's proven optimum exactly, nothing beats the
   optimum, bounded-suboptimal solvers stay within their ratio, and `Solution::optimal`
   agrees with `Solver::is_optimal`.
2. **Did the search change?** Machine-independent counters (`length`, `expanded`,
   `generated`, `peak_stored`, `iterations`) per case, compared exactly with a committed
   baseline. Cheap and deterministic, so CI runs it.
3. **How fast and how big?** Wall-clock latency per depth bucket and per exact depth,
   per-node cost, peak heap per solve, solver setup cost, pattern-database build time, and
   primitive costs (one move, one heuristic lookup). Noisy, so it runs locally and the
   report is committed under [`docs/benchmarks/`](benchmarks/) with the machine
   description (ADR 0006, ADR 0007).

## Quick reference

```bash
cargo build --release -p gpss-bench
./target/release/gpss-bench list                     # registered puzzles and targets
./target/release/gpss-bench check                    # correctness + counters vs baseline (CI)
./target/release/gpss-bench check --update           # accept new counters (after review!)
./target/release/gpss-bench check --target idastar   # filter by substring of the target id
./target/release/gpss-bench run \                    # full local report (~5 min for cube2, mostly BFS)
  --puzzle cube2 \
  --title "2x2x2 in-process benchmark (gpss-bench)" \
  --markdown docs/benchmarks/cube2-gpss-bench.md \
  --json docs/benchmarks/data/cube2-gpss-bench.json
```

`check` exits non-zero on any correctness failure or counter difference and prints a
per-bucket summary (cases, mean length, mean/max expanded, mean generated, max
`peak_stored`). `run` also exits non-zero if any case fails, so a report can never be
produced from a broken build. Always use `--release`; debug builds print a warning and
the numbers are meaningless.

## Depth buckets

Latency for a 1-move solve and an antipode differ by orders of magnitude, so every number
is reported per bucket of the case's proven optimal length:

| bucket | cube2 depths | meaning |
|--------|--------------|---------|
| `trivial` | 0 | already solved (identity sequences, empty scramble) |
| `short` | 1-5 | short-range solves |
| `mid` | 6-9 | |
| `top` | 10-11 | top of range; 11 is the 2x2x2 diameter (antipodes) |
| `invalid` | - | `expect: reject` cases (timed as part of correctness only) |
| `unbucketed` | - | valid cases without a known optimum |

Thresholds are per puzzle (`BucketScheme::new(short_max, mid_max)`) and match the corpus
generator's `short` / `mid` / `top` tags; a corpus test asserts they agree. The Markdown
report also has a per-exact-depth latency table for each target.

## What each number means

| number | how it is measured | caveats |
|--------|-------------------|---------|
| median / p95 / max latency | each case is solved repeatedly in-process for up to `--budget-ms` (default 200 ms); fast solves are batched so one sample lasts at least `--min-sample-us` (default 50 µs); the per-case statistic is the median sample divided by batch size; bucket p95/max are over per-case medians | excludes process start-up and table construction; includes `SearchStats` bookkeeping |
| ns / expanded node | total of per-case median times / total expanded nodes in the bucket | compare only within one solver: IDA* re-expands across iterations, A* pays for a heap and a closed set |
| peak heap / solve | `CountingAlloc` (a `GlobalAlloc` wrapper around the system allocator) tracks live requested bytes; peak during `solve` minus the bytes live before it | requested bytes, not RSS: no allocator overhead, stack, code or page granularity. Tables built before the solve are not counted (see setup) |
| setup | building the solver once: time and peak/retained heap | this is where the 3.7 MB perfect table shows up |
| PDB build | each registered table is rebuilt until `--table-budget-ms` is spent; median and min | |
| primitives | ns per `Puzzle::apply` (all moves, many states) and per `Heuristic::estimate` | the floor under per-node cost |

Process-level measurements (fresh process per case, peak RSS, legacy C++ comparison)
remain the job of `conformance/harness.py`; see
[`benchmarks/cube2-baseline.md`](benchmarks/cube2-baseline.md).

## Baselines and the CI gate

Baselines live in `crates/gpss-bench/baselines/<puzzle>.json`: for every deterministic
target, the counters of every case in CI scope, plus the corpus fingerprint (FNV-1a of
the JSONL bytes). `cargo test --release` runs `tests/baselines.rs`, and the CI `rust` job
additionally runs `gpss-bench check` for a readable log.

A counter difference is not necessarily a bug, but it is always a behaviour change of the
search, so it must be deliberate:

- **Expected** (you changed move ordering, a heuristic, pruning, tie-breaking, the
  corpus): run `gpss-bench check --update`, look at the per-bucket summary it prints, and
  commit the new baseline in the same PR, saying in the description why the counters
  moved and in which direction. `--update` refuses to write if any correctness check
  fails.
- **Unexpected**: the summary names the cases and the old/new counters; treat it like a
  failing test.

Scope rules that keep the gate fast (~6 s in release for cube2) and deterministic:

- `SolverSpec::ci_max_depth(d)` keeps cases with optimal length > d out of CI and out of
  the baseline (BFS on cube2 is capped at 8; it still runs on everything locally).
- `SolverSpec::nondeterministic()` keeps a target's counters out of the baseline while
  still checking correctness, for solvers whose node counts depend on threads, random
  seeds or floating-point batch order (e.g. a GPU-batched learned heuristic).
- A corpus edit changes the fingerprint; the check reports it and the cases that appeared
  or disappeared. Regenerate the baseline with `--update`.

## Adding a puzzle or solver

Everything puzzle-specific lives in `crates/gpss-bench/src/targets/`. The drivers only
know `gpss_core::{Puzzle, Solver, Heuristic}`.

1. Make sure there is a corpus `conformance/corpus/<name>.jsonl` (fields used: `id`,
   `scramble`, `expect` = `solve` / `reject`, `optimal_length` when proven, `tags`).
   Without proven optimal lengths the cases go to `unbucketed` and only validity and
   "never shorter than a known optimum" can be checked.
2. Add `targets/<name>.rs` with a `register(&mut Registry)` function and call it from
   `targets::register_all`:

   ```rust
   pub const BUCKETS: BucketScheme = BucketScheme::new(SHORT_MAX, MID_MAX);

   pub fn register(registry: &mut Registry) {
       let workload = Workload::new(MyPuzzle::new(), "<corpus name>", BUCKETS, |p, scramble| {
           // parse + apply + any normalisation the solvers expect (e.g. Cube2::regrip)
       })
       .heuristic("dual", |p| Box::new(MyHeuristic::new(p)))   // optional: primitive costs
       .table("corners", |p| PatternDatabase::build(p, &CornerCoord)); // optional: PDB build time
       registry.add(workload, vec![
           SolverSpec::new("idastar-dual", |p| Box::new(IdaStar::new(MyHeuristic::new(p)))),
           SolverSpec::new("two-phase", |p| Box::new(TwoPhase::new(p))).max_ratio(1.5),
           SolverSpec::new("bfs", |_| Box::new(Bfs)).ci_max_depth(6),
       ]);
   }
   ```

3. Run `gpss-bench check --update` and commit the new baseline file.
4. Run `gpss-bench run --puzzle <name> --markdown docs/benchmarks/<name>-gpss-bench.md
   --json docs/benchmarks/data/<name>-gpss-bench.json` on a quiet machine and commit the
   report.

`tests/plugin.rs` is a complete, runnable example with a toy puzzle; it also shows that a
solver which claims optimality but returns padded solutions is caught.

Guidance for the planned puzzles and solvers:

- **3x3x3 (Korf IDA\*, Kociemba two-phase).** Corpus optimal lengths for deep states come
  from an optimal solver, so mark only proven lengths as `optimal_length`. Korf's PDBs
  take long to build: register them with `.table(...)` so the build time is reported, and
  keep CI on short-range cases with `ci_max_depth`. Two-phase is not optimal; give it
  `max_ratio` and report length vs optimal from the latency tables (mean length is in
  the counter summary).
- **Learned solvers (`gpss-learn`).** They can register from their own crate without this
  crate depending on them: add a `register` function there and call it from a
  feature-gated line in `targets::register_all`, or build a `Registry` in a test or a
  small binary. Mark them `nondeterministic()` if batching makes node counts vary, and
  give them `max_ratio` only if the bound is real; otherwise they are checked for
  validity and "never shorter than optimal" only.

## Writing a report

- Close other workloads; note the load in the PR if you can't.
- Run from a clean commit: the report records `git describe` and marks a dirty tree.
- Commit both the Markdown (for humans) and the JSON (for diffs and plots).
- If you change defaults (`--budget-ms`, ...), the command line is recorded in the report.
- Compare reports only across the same machine description.
