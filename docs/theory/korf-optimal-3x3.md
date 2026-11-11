# Optimal 3x3x3 solving (Korf)

Status: **implemented** (`crates/gpss-puzzles/src/cube3/korf.rs`). Sources: Korf, *Finding
Optimal Solutions to Rubik's Cube Using Pattern Databases*, AAAI-97 [Korf97]; IDA* from
[Korf85]; pattern databases from Culberson and Schaeffer [CS98].

## Idea

IDA* with an admissible heuristic returns optimal solutions. Korf's heuristic takes the
maximum of three pattern databases, each giving the exact distance in a quotient of the cube:

| Table | Tracks | Entries | Max distance |
|---|---|---|---|
| `corners` | permutation and orientation of all 8 corners | 8! x 3^7 = 88,179,840 | 11 |
| `edges_a` | positions and orientations of `UR UF UL UB DR DF` | 12!/6! x 2^6 = 42,577,920 | 10 |
| `edges_b` | positions and orientations of `DL DB FR FL BL BR` | 42,577,920 | 10 |

The maximum of admissible heuristics is admissible. Korf used exactly these three tables.

## Implementation

- **Packing**: distances fit in 4 bits, so `NibbleTable` stores two entries per byte, and
  the three tables take 86.7 MB together.
- **Build**: breadth-first search runs over coordinate indices, not cube states.
  - Corners: `cperm x 2187 + twist`, using a cperm move table and a twist move table.
  - Edges: `rank(placement of the 6 edges) x 64 + orientation bits`. One move table of
    665,280 x 18 entries packs the new placement rank and an orientation XOR mask into a
    `u32`. Both edge tables share it, since only the starting index differs.
  - Once more than half the entries are known, each layer is filled backwards: an unseen
    entry is at depth d+1 if any neighbour is at depth d. The move set is closed under
    inverses, so this is exact.
  - All three tables build in about 5 s on one core.
- **Cache**: tables are saved to `$GPSS_CACHE_DIR`, or `$XDG_CACHE_HOME/gpss`, or
  `~/.cache/gpss`, as `korf-v1.bin`. The file has a magic number, a format version, table
  sizes and an FNV-1a checksum, and is written through a temporary file plus rename.
  Loading takes about 0.5 s. A corrupt or stale file is rebuilt, and setting
  `GPSS_CACHE_DIR=""` disables the cache.
- **Symmetry and inverse lookups** (`korf::Lookup`, after Korf [Korf97] and the dual
  lookups of [Fel05]): a position, its inverse and its conjugates by any of the 48
  symmetries ([`cube3-symmetry.md`](cube3-symmetry.md)) are all the same distance from
  solved, so the edge tables may be read on any of them and the maximum kept. The default
  `KorfHeuristic::new()` adds three lookups to the plain one: the inverse, and the
  conjugates by `URF3` and `URF3^2` (which move the U-D axis to R-L and F-B, so the two
  6-edge sets are read "from the other sides"). The corner table is not re-read: the
  corner distance is invariant under conjugation and inversion of the whole 8-corner
  pattern. The inverse's edge index is read straight from the state
  (`edges6_index_inverse`), and conjugates only conjugate the 12 edges.
  - The extra lookups cost about as much as the plain one, so they only pay when they
    are skipped most of the time. `Heuristic::estimate_within(state, budget)` (a
    `gpss-core` trait method whose default is `estimate`) lets IDA* say how much it
    needs: the heuristic returns as soon as the running maximum exceeds `budget`, which
    is where IDA* would prune anyway. Most generated children are pruned by the plain
    lookup alone. Without that early exit, the lookups made the search *slower* (fewer
    nodes, more time per node).
  - `KorfHeuristic::plain()` is the old heuristic, and `with_lookups` takes a custom
    list (the timing example reads it from `KORF_LOOKUPS`, e.g. `0i,16,32`).
- **Search**: the generic `IdaStar` from `gpss-core` with `KorfHeuristic`, so no 3x3-specific
  search code is involved. On the CLI:
  `gpss solve --puzzle cube3 --solver idastar --heuristic korf` (`korf` is the default
  heuristic for cube3 A* and IDA*).

## Validation

- The corner table's distance distribution is
  `1, 18, 243, 2874, 28000, 205416, 1168516, 5402628, 20776176, 45391616, 15139616, 64736`.
  It sums to 8! x 3^7, and its maximum of 11 matches [Korf97].
- The edge tables sum to 12!/6! x 2^6. Their distributions differ from each other: the two
  edge sets are not related by a symmetry that preserves the edge-orientation convention.
- `korf_tables_and_optimal_solves` is an ignored test that CI runs with `--ignored`. It
  checks these distributions and solves random scrambles optimally.
- Conformance: every case in the 3x3x3 corpus with a known distance (up to 7) is solved
  at exactly that length.

## Performance (measured on 8-core cloud VM, single thread)

| Optimal length | Typical expanded nodes | Typical time |
|---|---|---|
| 12 | 10^4 - 5x10^4 | 0.2 - 0.4 s (including 0.15 s of table loading) |
| 13 | 10^5 - 2.5x10^5 | 0.5 - 1.2 s |
| 14 | 0.7 - 1.1x10^6 | 1.5 - 2.8 s |
| 15 | 0.3 - 2x10^7 | 7 - 46 s |

Each extra move multiplies the work by about 10-13, roughly the effective branching factor.
Random positions are usually 17-18 moves from solved [Rok13], so solving them optimally takes
minutes to hours with these tables. For fast solutions use the two-phase solver.

### Before and after symmetry lookups

Measured with `cargo run --release -p gpss-puzzles --example cube3_timing -- korf 10 11 12 13`
and `... korf 14` (8 random scrambles per length, seed 2, warm cache, one thread, 8-core
Intel Xeon cloud VM). Optimal lengths are identical before and after.

| Scramble length | Before: total / median / max | After: total / median / max | Mean expanded nodes, before -> after |
|---|---|---|---|
| 10 | 0.01 s / 0.000 s / 0.012 s | 0.01 s / 0.000 s / 0.005 s | 1,313 -> 441 |
| 11 | 0.03 s / 0.003 s / 0.009 s | 0.02 s / 0.002 s / 0.005 s | 2,374 -> 1,170 |
| 12 | 0.14 s / 0.023 s / 0.038 s | 0.10 s / 0.015 s / 0.029 s | 11,923 -> 6,532 |
| 13 | 2.14 s / 0.287 s / 1.060 s | 0.99 s / 0.111 s / 0.374 s | 114,608 -> 64,184 |
| 14 | 26.6 s / 1.18 s / 9.81 s | 15.9 s / 1.04 s / 5.85 s | 1,420,797 -> 815,320 |

Choice of lookups, on another seed (lengths 12 and 13, 8 scrambles each, total time):

| Lookups | Length 13 total | Mean expanded |
|---|---|---|
| plain | 4.12 s | 220,683 |
| inverse | 2.80 s | 157,810 |
| inverse + `URF3` + `URF3^2` (default) | 2.63 s | 123,655 |
| the above + their inverses | 2.71 s | 109,098 |
| inverse, without `estimate_within` | 6.41 s | 157,810 |

More lookups keep cutting nodes but no longer cut time; the default is the knee.

## Not implemented yet (ordered by expected payoff)

1. Larger edge tables: two 7-edge tables, 510M entries each, about 255 MB each at 4 bits.
2. Symmetry-reduced tables (one entry per symmetry class), which would make room for the
   larger edge tables in the same memory.
3. Coordinate-level IDA*: carry the coordinates incrementally through move tables instead
   of recomputing them from cubie states at every node. The node rate is currently about
   0.45 M nodes/s.
4. Multi-threaded IDA* that splits the root moves across threads.
