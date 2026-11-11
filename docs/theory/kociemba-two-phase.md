# Kociemba's two-phase algorithm - planned

Status: **design**. Sources: Kociemba's own descriptions [Koc], [KocCE].

1. **Phase 1** - reach the subgroup `G1 = <U, D, R2, L2, F2, B2>`: all corner and edge
   orientations are 0 and the four UD-slice edges are in the slice. Coordinates: corner
   orientation (2,187), edge orientation (2,048), UD-slice position (495). IDA* with
   pruning tables on pairs of these.
2. **Phase 2** - solve within G1 using only G1 moves. Coordinates: corner permutation
   (40,320), UD-edge permutation (40,320), slice permutation (24).
3. **Iterate** - keep enumerating longer phase-1 solutions and re-solving phase 2; stop at
   a target length or time budget. Kociemba notes phase 1 needs at most 12 moves and phase
   2 at most 18, so the first solution has at most 30 moves, and continued search quickly
   finds solutions of 20 moves or fewer [Koc].

In gpss terms: each phase is a `Puzzle` view over the same cubie state (phase-specific move
sets and goal tests) searched by the generic `IdaStar`, and the two-phase driver is a
`Solver<Cube3>` that reports `optimal = false`. Move tables and coordinates follow
Kociemba's conventions, which the 2x2x2 corner model already uses.
