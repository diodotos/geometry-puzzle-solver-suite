# Pattern databases

A pattern database (PDB) [CS98] stores, for every state of an *abstraction* of the puzzle,
the exact distance to the abstract goal. Because every real solution projects to an
abstract solution of at most the same length, the stored value is an admissible
heuristic. Taking the maximum of several admissible PDBs is still admissible [Korf97].

## In this codebase

A `Coordinate<P>` is a function from puzzle states to `0..size` such that states with the
same coordinate are "the same" for some abstraction, and the abstraction is consistent
with the moves (applying a move to two states with equal coordinates gives equal
coordinates). `PatternDatabase::build(puzzle, coord)` runs a BFS from the solved state
over *full* states but records distances by coordinate, keeping one representative per
coordinate value. Entries are `u8` distances with `u8::MAX` meaning unreached.

`PdbHeuristic<C>` wraps a database as a `Heuristic`; `MaxHeuristic` combines several.

```rust
let ori  = PdbHeuristic::new(&puzzle, OrientationCoord);   // 729 entries
let perm = PdbHeuristic::new(&puzzle, PermutationCoord);   // 5,040 entries
let dual = MaxHeuristic::new("dual", vec![Box::new(ori), Box::new(perm)]);
```

## The 2x2x2 databases

| name | abstraction | entries | max value | build time |
|------|-------------|---------|-----------|------------|
| `orientation` | twist of the 7 free corners | 3^6 = 729 | 6 | < 1 ms |
| `permutation` | positions of the 7 free corners | 7! = 5,040 | 7 | ~ 1 ms |
| `dual` | max of the two | - | - | - |
| `perfect` | the whole state | 3,674,160 | 11 | ~ 0.85 s |

`dual` is the heuristic of the original CubeSolver project, which defined
`h(C) = max(min moves to orient C, min moves to permute C)`. The perfect database is the
complete distance table: with it IDA* expands exactly `d` nodes and never backtracks.

The diameters 6 and 7 of the sub-problems were computed both by the Rust PDB builder and
by an independent Python BFS.

## 3x3x3

Korf [Korf97] used three databases: the 8 corners (88 million entries, 4 bits each) and
two sets of 6 edges, combined by `max`, with IDA*. `cube3::korf` implements them with its
own 4-bit packing and on-disk cache, and reads the edge tables on the inverse and two
conjugates of the position as well ([Fel05]; see [`korf-optimal-3x3.md`](korf-optimal-3x3.md)).

The two-phase solver uses *symmetry-reduced* tables (one row per class of a coordinate
under the 16 symmetries fixing the U-D axis) storing distances modulo 3, after Kociemba
[KocImp]; see [`kociemba-two-phase.md`](kociemba-two-phase.md). The generic
`PatternDatabase` has neither feature yet.
