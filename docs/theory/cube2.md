# The 2x2x2 ("pocket cube")

## State representation

Only corners. Following Kociemba [Koc], positions and pieces are both numbered

```text
0   1   2   3   4   5   6   7
URF UFL ULB UBR DFR DLF DBL DRB
```

and a state is `cp[i]` (piece at position `i`) and `co[i]` (its clockwise twist, 0-2,
measured on the U/D sticker). A move is itself a state in "replaced-by" form, and applying
move `m` to state `s` is the product

```text
(s * m).cp[i] = s.cp[m.cp[i]]
(s * m).co[i] = (s.co[m.cp[i]] + m.co[i]) mod 3
```

which is `Cube2State::then`. The six quarter turns are the only hand-written data; half
and counter-clockwise turns are derived by composition.

Invariant: the total twist `sum(co)` is 0 mod 3 (tested with random scrambles).

## Why only `U`, `R`, `F`

A 2x2x2 has no centres, so any of the 24 whole-cube rotations of a solved cube is solved.
`L` is the same as `R` followed by a whole-cube rotation (`x'`), and likewise `D` ~ `U`,
`B` ~ `F`. So:

1. Rotate the scrambled cube (24 candidates, generated from `x = R L'`, `y = U D'`,
   `z = F B'`) until the `DBL` piece sits at `DBL` untwisted. Exactly one rotation does.
2. `U`, `R`, `F` never move position `DBL`, so the frame stays fixed during search; every
   position with `DBL` home is reachable with them; the goal is the identity.
3. Each rotation `rho` relabels faces: `m' after rho = rho after face_map(m')`. The map is
   derived empirically at start-up by checking which face turn commutes correctly on a
   probe state, then used to rewrite the solution in the user's original grip.

The search space is then exactly `7! * 3^6 = 3,674,160` states, and it has a perfect index
`permutation_rank * 729 + orientation_index`.

## Coordinates

- **Orientation** - twists of positions `0..5` in base 3 (`co[7]` is implied because the
  twist sum is 0 and `co[6] = 0`). 729 values.
- **Permutation** - Lehmer-code rank of the pieces at the seven free positions
  (`0 1 2 3 4 5 7`, with piece 7 relabelled 6). 5,040 values.
- **Full** - `permutation * 729 + orientation`. 3,674,160 values, a bijection.

## Distance distribution (half-turn metric)

| depth | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|-------|---|---|---|---|---|---|---|---|---|---|----|----|
| states | 1 | 9 | 54 | 321 | 1,847 | 9,992 | 50,136 | 227,536 | 870,072 | 1,887,748 | 623,800 | 2,644 |

God's number is 11, and there are 2,644 antipodes. This table is recomputed by the test
suite (`full_state_space_matches_published_distribution`) and by the corpus generator,
independently.

## Move pruning

With the 9 search moves the only rule is "don't turn the same face twice in a row", so
after the first move the branching factor is 6. `FaceTurn::can_follow` also encodes the
opposite-face ordering rule needed for 18-move puzzles (3x3x3).
