# Generic cubes and cuboids (`AxBxC`)

`gpss_puzzles::cuboid` models every cube and cuboid with at least two layers on each
axis at the sticker level: 2x2x2, 3x3x3, 4x4x4, ..., the 3x3x2 "domino", 2x2x3, 2x3x4,
and so on. It is the first step from hand-written puzzle models toward the project's goal
of puzzles described by geometry. See `docs/roadmap.md`.

## Why a sticker model

The specialised `cube2` and `cube3` models describe a state by cubie permutation and
orientation. That is compact and fast, but each piece type (corners, edges, centres,
wings, ...) needs its own invariants and coordinates. A sticker (facelet) model needs none
of this. A state is one colour byte per sticker, and every move is a precomputed sticker
permutation derived from 3D geometry. The same ~300 lines of code then cover every size and
shape. The cost is speed and memory: a 3x3x3 state is 54 bytes here, against 40 in `cube3`,
and the only pattern database so far is the corner projection described below.

## Conventions

* **Dimensions.** `AxBxC` means `A` layers on the R–L axis, `B` on F–B and `C` on U–D.
  Height comes last, as in common usage, so `3x3x2` is the domino. Internally
  `Dims([x, y, z])` is indexed by `Axis::{X, Y, Z}` (positive faces R, U, F).
* **Geometry.** A cubie centre has coordinates in `{-(n-1), -(n-3), ..., n-1}` per axis.
  A sticker is a (cubie centre, outward normal) pair. Layer `k` of an axis (0 = outer
  layer at the positive face) holds the cubies with coordinate `(n-1) - 2k`.
* **Turns.** `LayerTurn { axis, layer, turns }` rotates one layer by `turns` clockwise
  quarter turns, seen from the positive face. A quarter turn maps the cuboid onto itself
  only when the two perpendicular dimensions are equal. Otherwise that axis has half
  turns only. On the domino, U and D quarter turns are legal and R, L, F, B are half turns
  only.
* **Notation.** `X` turns face `X`'s outer layer. `kX` turns only the `k`-th layer from
  face `X` in `X`'s direction (SiGN single-slice notation). Suffixes are `2` and `'`. A
  negative face maps to the positive axis: on an `n`-layer axis, `kL` = layer `n-k` with
  the turn reversed. Search output uses only R/U/F-based tokens, e.g. `U 3R2 F2 U'`.
  Wide turns (`Rw`, `3Rw`) are not supported yet.

## Goal and metric

Every single-layer turn costs 1, whether quarter or half, outer or inner. A state is
**solved when every face shows one colour**. The orientation of the whole puzzle is free,
because turning every layer of an axis is a rotation, not progress.

This means the search uses *all* layers, including the ones a fixed-frame model drops.
`cube2` fixes the DBL corner and searches with U, R, F only. For a 2x2x2 that is
equivalent, since `L ≡ R` + rotation. But on a 3x3x3, dropping `D` turns a 1-move solution
into `U' 2U'` (2 moves), so fixing a corner would change distances. Keeping all layers
and testing "every face is uniform" keeps distances exact. The state space is then
larger by the order of the cuboid's rotation group (24 for cubes, 8 or 4 for cuboids).
The next section covers how rotations are factored out where that pays off.

## Whole-puzzle rotations (`cuboid/symmetry.rs`)

`Cuboid::new` enumerates every proper rotation (a signed permutation matrix with
determinant +1) that maps each axis onto an axis with the same layer count. That gives 24
for a cube, 8 for a cuboid with two equal sides (3x3x2, 2x2x3), and 4 when all three
sides differ (2x3x4). A rotation `g` acts on a state by **conjugation**: stickers move
with `g`, and colours are renamed by `g`'s face map, so the solved state is fixed
(`Cuboid::rotate`). Conjugation preserves distance to the goal: the goal set (every face
one colour) is invariant, and a layer turn conjugated by `g` is again a legal layer turn,
because `g` preserves the dimensions and so which axes allow quarter turns.
`Cuboid::canonical` picks the lexicographically smallest conjugate.

`RotationReduced` wraps a `Cuboid` as a `Puzzle` over canonical states:
`apply = canonical ∘ apply`. Moves found in that quotient are expressed in each canonical
parent's frame. `RotationReduced::lift` replays them from the real start: at each step it
picks the real layer turn whose canonical result matches. Solution lengths are unchanged.
Cuboid heuristics adapt through `ReducedHeuristic<H>`; this is only sound for heuristics
that are rotation-invariant, such as `CornerHeuristic`, which is a property test.

**Where it pays off (measured).** The quotient collapses states that are conjugates of
each other. Near a symmetric state such as solved, most states are. BFS levels from
solved on a 3x3x3 are `[1, 27, 564, 11542]` raw versus `[1, 5, 40, 548]` reduced, about
21x fewer. Near a generic scrambled start, two different move paths almost never reach
conjugate states, so a **forward solve from a scramble gains essentially nothing**. It
also loses the same-axis `can_follow` pruning, because frames change between moves. On
3x3x3 4-move scrambles, reduced BFS stored about as many states as raw BFS or more. Raw
search already accepts all 24 solved orientations as goals. The CLI therefore keeps
forward solves on the raw `Cuboid`.

The reduction is infrastructure for searches that start from the goal:
- generating pattern databases and other tables, which is where the 24x memory saving
  lands;
- exhaustive distance distributions for small cuboids;
- the backward half of bidirectional search;
- Korf-style symmetric heuristic lookups, i.e. taking the max of a PDB over the
  conjugates of a state, for PDBs whose piece subset is not rotation-invariant.

Resulting metrics:

| puzzle | metric |
|---|---|
| 2x2x2 | half-turn metric (HTM), same lengths as `cube2` (a unit test checks this) |
| 3x3x3 | slice-turn metric (STM): face turns plus middle-slice turns |
| NxNxN, N ≥ 4 | single-slice turn metric: every inner slice is one move |
| cuboids | the same, with half-turn-only axes |

## Search pruning

Turns on the same axis commute. `can_follow` therefore only allows a turn on the same
axis as the previous one if its layer index is strictly larger. That rule forbids turning
the same layer twice in a row and fixes one order for commuting layers. It never
disconnects the search, because any pruned pair has an accepted equally short reordering.

## Validation

* **Unit tests** (`crates/gpss-puzzles/src/cuboid/tests.rs`) cover:
  - move orders (2 or 4) and inverses on several shapes;
  - notation round-trips, negative-face mapping and rejection of illegal tokens;
  - whole-puzzle rotations counting as solved;
  - agreement with the `cube3` model (a two-phase solution also solves the generic
    3x3x3);
  - exact agreement with `cube2` optimal lengths;
  - BFS and IDA* agreeing on random cuboid scrambles.
* **Conformance:** `conformance/corpus/cuboid.jsonl` comes from
  `conformance/generate_cuboid.py`. Exact distances come from a BFS ball around solved,
  built with the independent Python reference model, which supports `AxBxC` and `kX`
  slice tokens. Because moves conjugated by a rotation are again moves, a state's distance
  up to rotation is the minimum of the ball distance over all rotations of the state. The
  corpus covers 2x2x2, 3x3x2, 2x2x3, 2x3x4, 3x3x3 and 4x4x4.

## Heuristic: corner projection

`CornerHeuristic` (`--heuristic corners`, the default for A* and IDA* on cuboids) is
admissible and consistent for every `AxBxC`. The argument has three steps:

1. **Every cuboid has exactly eight corner cubies.** Inner layers never touch them. An
   outer-layer turn moves them exactly like the 2x2x2 face turn with the same name.
2. **Projection.** `Cuboid::corner_state` reads the three corner stickers of each corner
   position and returns a `cube2::Cube2State` in Kociemba's convention. The U/D facelet
   comes first and the others follow clockwise. Every cuboid move maps to a 2x2x2 move or
   to the identity. Half-turn-only axes simply have fewer moves. So the projected
   distance, measured in the 2x2x2's half-turn metric, never exceeds the cuboid distance.
3. **Goal.** A solved cuboid (every face one colour) has its corners solved up to a
   whole-cube rotation. The heuristic therefore regrips the projection
   (`Cube2::regrip`) and looks it up in the full 3.7 MB `cube2` distance table
   (`cube2::FullCoord`). It is 0 on every solved orientation.

The value is the exact distance of a relaxed problem, so it is also consistent: one move
changes it by at most 1 (`prop_corner_heuristic_is_consistent`). On a 2x2x2 it is the
perfect heuristic. The table is generated once per process (about 0.8 s, shared through a
`OnceLock`), which dominates short CLI solves.

Effect on IDA* (single runs on the 8 vCPU dev VM, `--timeout 60`):

| puzzle | scramble | length | zero | corners |
|---|---|---|---|---|
| 3x3x2 | `U R2 F2 D' 2U L2 B2 U` | 7 | 3,984,282 expanded, 1.2 s | 14,219 expanded, 0.9 s |
| 2x3x4 | `R2 U2 F2 2U2 L2 B2 3U2` | 7 | 673,868 expanded, 0.2 s | 43,030 expanded, 0.9 s |
| 3x3x3 | `R U F D L B` | 6 | timeout (> 60 s) | 491 expanded, 0.9 s |
| 4x4x4 | `2R U 3F' R2 2U' F` | 6 | timeout (> 60 s) | 470,733 expanded, 2.3 s |

Wall times include the ~0.8 s table build, which is why the blind search wins on the
easy 2x3x4 case.

The heuristic knows nothing about edges or centres, so on 3x3x3 and larger it is weak:
corners alone are solved within 11 moves, while 4x4x4 positions need far more. Stronger
generic PDBs are the next step.

## Limits and next steps

With the corner heuristic, IDA* handles roughly 7–9 move positions on small cuboids and
3x3x3. Deeper positions need more information. Planned work:

1. ~~Canonicalise states modulo whole-puzzle rotation~~: done (see "Whole-puzzle
   rotations"). It pays off for goal-rooted searches, not forward solves.
2. More generic pattern databases over sticker subsets: edges, centres and wings for
   larger cubes, combined with the corner projection by `MaxHeuristic` or additively.
3. Wide turns and other move metrics (outer block turn metric).
4. Generalise from axis-aligned layers to arbitrary cutting planes. This is the
   "arbitrary geometry" step.
