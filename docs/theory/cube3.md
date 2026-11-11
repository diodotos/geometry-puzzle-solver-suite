# The 3x3x3 Rubik's Cube

Implementation: `crates/gpss-puzzles/src/cube3/`. Solver theory: [kociemba-two-phase.md](kociemba-two-phase.md),
[korf-optimal-3x3.md](korf-optimal-3x3.md). Symmetries: [cube3-symmetry.md](cube3-symmetry.md).

## State space

The cube has 8 corners and 12 edges; the 6 centres never move under face turns, so the
centres fix the orientation of the whole cube and no regripping is needed (unlike the
2x2x2, see [cube2.md](cube2.md)). Reachable positions:

\[ \frac{8! \cdot 3^8 \cdot 12! \cdot 2^{12}}{3 \cdot 2 \cdot 2} = 43{,}252{,}003{,}274{,}489{,}856{,}000 \approx 4.3 \times 10^{19} \]

The three divisors are the invariants that every face turn preserves:

- **Twist**: the sum of corner orientations is 0 mod 3.
- **Flip**: the sum of edge orientations is 0 mod 2.
- **Parity**: corner and edge permutations have equal parity (a quarter turn is a
  4-cycle on both, so each quarter turn flips both parities).

`Cube3State::is_valid` checks exactly these, plus that `cp`/`ep` are permutations and
that each orientation is in range. Every position needs at most 20 moves in the
half-turn metric; this is "God's number" [Rok13]. Superflip (every edge flipped in place) is a
classical distance-20 position.

## Representation

This is Kociemba's cubie model [Koc]. Positions are numbered `URF UFL ULB UBR DFR DLF DBL DRB` for
corners and `UR UF UL UB DR DF DL DB FR FL BL BR` for edges. `cp[i]` names the corner that sits in
position `i`, and `co[i]` is its twist (0 means its U/D sticker faces U/D). The edge arrays `ep` and `eo`
work the same way, with Kociemba's edge-orientation convention, under which only F and B quarter turns flip edges.
Composition uses the "replaced-by" convention:

```text
(a then b).cp[i] = a.cp[b.cp[i]]
(a then b).co[i] = a.co[b.cp[i]] + b.co[i]   (mod 3)
```

The corner part is identical to the 2x2x2 model; a unit test checks this move by move.

## Coordinates

`cube3::coord` maps parts of a position to dense integers for table-driven search:

| Coordinate | Range | Meaning |
|---|---|---|
| `twist` | 3^7 = 2187 | corner orientations (the 8th is implied) |
| `flip` | 2^11 = 2048 | edge orientations (the 12th is implied) |
| `slice` | C(12,4) = 495 | which positions hold the UD-slice edges |
| `cperm` | 8! = 40320 | corner permutation |
| `udeperm` | 8! = 40320 | U/D-layer edge permutation (phase 2 only) |
| `sperm` | 4! = 24 | slice-edge permutation (phase 2 only) |

## Move set

The move set is the 18 face turns (`U R F D L B` x quarter, half, inverse). The generic `can_follow`
pruning (no same face twice; opposite faces in a fixed order) cuts the branching factor
from 18 to about 13.35.

## References

See [references.md](../references.md): [Koc] for the cubie model and coordinates, and [Rok13] for God's number.
