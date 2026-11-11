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
and there are no pattern databases yet.

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
Canonicalising states modulo rotation is a planned optimisation.

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

## Limits and next steps

Without a heuristic, IDA* is only practical to about depth 5–6 on small cuboids. Planned
work:

1. Canonicalise states modulo whole-puzzle rotation, to shrink the space by up to 24x.
2. Generic pattern databases over sticker subsets, e.g. corner stickers only, which every
   cuboid has.
3. Wide turns and other move metrics (outer block turn metric).
4. Generalise from axis-aligned layers to arbitrary cutting planes. This is the
   "arbitrary geometry" step.
