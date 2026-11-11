# Symmetries of the 3x3x3

The cube has 48 symmetries: 24 rotations and 24 reflections (the full octahedral group
`O_h`). They are implemented in `cube3::sym` and used by both 3x3x3 solvers. Kociemba's
symmetry pages [KocImp] are the model; the numbering below is his.

## Conjugation

Conjugating a position `C` by a symmetry `g` means looking at `C` through `g`: every
sticker at location `x` with colour `c` moves to `g(x)` and is recoloured `g(c)`. In
group terms the result is `g^-1 C g`. Because `g` maps every face turn to a face turn (a
reflection maps a clockwise quarter turn to an anticlockwise one), a solution `m1 .. mk`
of `C` conjugates to a solution `g(m1) .. g(mk)` of the conjugate, and

- the conjugate is a legal position, at exactly the same distance from solved;
- conjugation is a group action (`conj(conj(C, a), b) = conj(C, a.b)`) and an
  automorphism (`conj(A.B, g) = conj(A, g).conj(B, g)`);
- the inverse position `C^-1` is also at the same distance (reverse and invert the
  solution). Combining both gives up to 96 positions with the same distance.

`SymTables::move_conj[m][s]` is the face turn `m` conjugated by `s`; it is how solutions
found for a conjugate are translated back.

## Numbering

Symmetry `s = 16a + 8b + 2c + d` is `URF3^a . F2^b . U4^c . LR2^d`, where `URF3` is the
120-degree rotation about the URF-DBL diagonal (`U -> R -> F -> U`), `F2` the half turn
about the F-B axis, `U4` the quarter turn about the U-D axis and `LR2` the left-right
mirror. Odd `s` are reflections. `s < 16` (`a = 0`) are the 16 symmetries that keep the
U-D axis in place (`D4h`); they map the two-phase subgroup `G1 = <U, D, R2, L2, F2, B2>` to
itself. `sym::urf3()` returns the index of `URF3` (16).

## Implementation

Nothing but the four generators is typed in by hand. Faces are unit normal vectors, and a
symmetry is stored as the permutation of the six faces it induces (`FaceMap`); the 48 are
the products of the generators, and composition and inverses are computed on face maps. A
cubie position is the set of faces it touches, so a symmetry also permutes positions,
and comparing the facelet order of a position with that of its image gives the
orientation change. From that, `SymTables` stores per symmetry, per position and per
cubie a recipe `(position, cubie, orientation) -> (position', cubie', orientation')`, and
`conjugate` is 20 table lookups. `conjugate_edges` does only the 12 edge lookups for
callers that read edges only (the Korf edge tables).

## Uses

- `cube3::symcoord` / `cube3::twophase`: symmetry classes of flip x slice and of the
  corner permutation under `D4h`, giving Kociemba's reduced pruning tables, and the
  "six directions" search over `URF3` conjugates and inverses
  ([`kociemba-two-phase.md`](kociemba-two-phase.md)).
- `cube3::korf`: extra edge-table lookups on the inverse and the `URF3` conjugates
  ([`korf-optimal-3x3.md`](korf-optimal-3x3.md)).

## Tests

`cube3/tests.rs` (module `symmetry`):

- the geometric model reproduces the hand-written face-turn tables (sticker level);
- the multiplication and inverse tables form a group of order 48 whose first 16
  elements are a subgroup fixing U-D;
- conjugates of face turns are face turns, reflections reverse quarter turns;
- property tests: group action, automorphism over composition, conjugate of a scramble
  equals the scramble of the conjugated moves (so legality is preserved), two-sided
  inverse, consistency of every two-phase coordinate under `D4h` (the coordinate of a
  conjugate depends only on the coordinate), and invariance of the phase-1 and phase-2
  table distances under `D4h`.
