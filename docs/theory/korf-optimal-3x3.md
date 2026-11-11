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

## Not implemented yet (ordered by expected payoff)

1. Larger edge tables: two 7-edge tables, 510M entries each, about 255 MB each at 4 bits.
2. Symmetry and anti-symmetry: look up the position conjugated by the 48 cube symmetries,
   and the inverse position, and take the maximum. Symmetry reduction would also shrink
   the tables.
3. Coordinate-level IDA*: carry the coordinates incrementally through move tables instead
   of recomputing them from cubie states at every node. The node rate is currently about
   0.45 M nodes/s.
4. Multi-threaded IDA* that splits the root moves across threads.
