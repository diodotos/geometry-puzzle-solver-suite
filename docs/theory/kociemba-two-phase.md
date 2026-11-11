# Kociemba's two-phase algorithm

Status: **implemented** (`crates/gpss-puzzles/src/cube3/twophase.rs`). Sources: Kociemba's own descriptions [Koc], [KocCE].

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

## Implementation in gpss

- **Coordinates** (`cube3::coord`): twist (3^7), flip (2^11), UD-slice combination
  (C(12,4) = 495, ranked with the combinatorial number system), corner permutation and
  U/D-edge permutation (Lehmer rank, 8!), slice permutation (4!). Each has a `set_*`
  inverse; move tables are built by set, apply move, get.
- **Pruning tables**: phase 1 takes the maximum of twist x slice, flip x slice and
  twist x flip (1.1 M, 1.0 M and 4.5 M one-byte entries); phase 2 takes the maximum of
  corner-perm x slice-perm and edge-perm x slice-perm (0.97 M each). All tables are
  built by BFS in about 0.4 s at first use (`TwoPhase::warm_up`) and kept in memory
  (about 48 MB peak RSS for the CLI).
- **Search**: phase-1 depths are tried in increasing order with no fixed cap (longer
  phase-1 prefixes can shorten the total). Every phase-1 solution that does not end in
  a G1 move (those duplicate a shorter prefix) is completed by an IDA* phase 2 bounded
  by the best total found so far. The solver stops at the first solution of at most
  `target_length` moves (CLI `--target-length`, default 20), or at the search limits,
  where it returns the best solution found so far.
- **Not implemented yet**: Kociemba's symmetry-reduced phase-1 tables (flip x slice
  classes x twist) and his search from three orientations and their inverses. These
  are why gpss is slower than Kociemba's own program on hard positions: superflip takes
  about 2.3 s, while the median random scramble takes about 0.07 s.
- **Consequence of the target-length rule**: short scrambles are not necessarily
  solved optimally. For example `R U F` can return a 10-move solution, because that
  is the first one found within the target. Use `--target-length` with a smaller
  value, or an optimal solver, when that matters.
