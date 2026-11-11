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
- **Symmetry** (`cube3::sym`, see [`cube3-symmetry.md`](cube3-symmetry.md)): the 48
  cube symmetries, built from a geometric model, with conjugation of cubie states and of
  face turns. The 16 symmetries that fix the U-D axis (`D4h`, indices `0..16`) map `G1`
  to itself, so they preserve both the phase-1 and the phase-2 distance.
- **Pruning tables** (`cube3::symcoord`), as in Kociemba's own program [KocImp]:
  - Phase 1 uses one table over `(flip x slice symmetry class) x twist`:
    64,430 classes (out of 1,013,760 raw flip x slice values) x 2,187 twists =
    140.9 M entries. Phase 2 uses `(corner-permutation class) x (U/D-edge permutation)`:
    2,768 x 40,320 = 111.6 M entries. Both are maximised with the two small
    slice-permutation tables of the old design (corner perm x slice perm and U/D-edge
    perm x slice perm, 0.97 M one-byte entries each), which are also used as a cheap
    filter before phase 2 starts.
  - A state is looked up by conjugating it with the symmetry `s` that takes its flip x
    slice (or corner permutation) to the class representative, and conjugating the
    partner coordinate (twist, U/D-edge permutation) by the same `s`. Conjugation of
    the partner coordinate is a precomputed table (`raw x 16`). When a representative
    has a non-trivial stabiliser, the builder writes every equivalent partner value, so
    the choice of `s` never matters.
  - Entries hold the distance modulo 3 in 2 bits (35 MB + 28 MB). Because neighbouring
    states differ in distance by at most one, the search, which knows the exact distance
    of the current node, recovers each child's exact distance from its residue
    (`child_depth`), and the root's exact distance is found by walking down to the goal
    (`descend`). Phase 1 therefore prunes with the **exact** phase-1 distance.
  - Building takes about 3 s on 8 cores (`sym_bfs`: each BFS layer is split across all
    threads with atomic writes, and filled backwards, from unseen entries, once fewer
    entries are unseen than are on the frontier; the result does not depend on thread
    scheduling). The two
    tables are cached as `twophase-v1.bin` (63 MB) next to `korf-v1.bin`, with the same
    magic/version/FNV-1a/atomic-rename format (`cube3::cache`); the file also stores a
    fingerprint of the symmetry classes, so that a change in symmetry numbering or class
    order forces a rebuild. Loading takes about 0.26 s. `GPSS_CACHE_DIR=""` disables the cache.
- **Search**:
  - Phase 1 carries five coordinates through move tables: twist, flip, slice, corner
    permutation and `slice_sorted` (slice-edge positions *and* order, 495 x 24). When a
    phase-1 solution reaches `G1`, the corner and slice permutations are therefore known
    without rebuilding the cube; only the U/D-edge permutation needs the cubie state.
  - All 18 children's table entries are read before recursing, so their cache misses
    overlap.
  - Kociemba's rule: when the node is already in `G1` with fewer than five phase-1 moves
    to go, phase-2 moves are skipped, since the rest would be phase-2 moves that phase 2
    finds anyway [KocImp]. Phase-1 solutions ending in a phase-2 move are skipped for the
    same reason.
  - A phase-1 solution is first filtered with the small slice-permutation tables, then
    phase 2 computes the exact corner x U/D-edge distance with a cut-off at the current
    bound (`descend` gives up once the distance is known to be too large) before running
    its IDA*.
  - **Six directions** (`TwoPhase::six_directions`, on by default): like Kociemba, the
    solver searches the position conjugated by `URF3^0, URF3^1, URF3^2` (so phase 1 aims
    at the U-D, R-L and F-B subgroups) and the inverses of those, interleaving the six
    searches by phase-1 depth. A solution of a variant is translated back by conjugating
    its moves with the inverse symmetry and, for an inverse, reversing and inverting them.
    Some direction is usually much easier than the others.
  - Phase-1 depths are tried in increasing order with no fixed cap. Every phase-1
    solution is completed by an IDA* phase 2 bounded by the best total found so far. The
    solver stops at the first solution of at most `target_length` moves (CLI
    `--target-length`, default 20), or at the search limits, where it returns the best
    solution found so far.
- **Consequence of the target-length rule**: short scrambles are not necessarily
  solved optimally. For example `R U F` can return a 10-move solution, because that
  is the first one found within the target. Use `--target-length` with a smaller
  value, or an optimal solver, when that matters.

## Performance before and after symmetry (8-core Intel Xeon cloud VM)

Measured with `cargo run --release -p gpss-puzzles --example cube3_timing -- twophase`
(the superflip, then 200 random 40-move scrambles with seed 1; target length 20; single
thread except for the table build). "Before" is the plain-table solver this replaced.

| | Before | After |
|---|---|---|
| Table setup, cold (build) | 0.39 s | 3.06 s (all cores; once, then cached) |
| Table setup, warm (cache load) | n/a (always built) | 0.26 s |
| Superflip | 2.43 s, 19,199,625 nodes | 1.64 s, 5,983,387 nodes |
| Random: median | 33.5 ms | 0.27 ms |
| Random: mean | 451 ms | 3.1 ms |
| Random: p90 | 800 ms | 4.8 ms |
| Random: max | 39,356 ms | 159 ms |
| Random: mean / max length | 19.86 / 20 | 19.88 / 20 |
| Random: mean expanded nodes | 2,364,823 | 9,409 |
| CLI peak RSS (conformance harness) | about 48 MB | about 82 MB |

The superflip is the worst case for the six-direction trick: it is symmetric under all
48 symmetries and its own inverse, so all six searches are the same search. Its gain
comes from the exact phase-1 distance alone. Random positions gain from both.

Not implemented yet: running the six directions on separate threads (Kociemba's program
does), and an optimal mode that keeps searching after the first solution within
`target_length`.

