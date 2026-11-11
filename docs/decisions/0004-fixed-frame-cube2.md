# 0004. Search the 2x2x2 in a fixed frame with `U R F` moves

**Status:** Accepted

## Context

The 2x2x2 has no fixed centres: a cube that is solved but rotated is solved. The legacy
solver handled this by normalising states (canonicalising whole-cube rotations) inside its
hash functions, and searched with all 18 face turns.

## Decision

`Cube2::regrip` rotates the scrambled cube so the `DBL` corner is home and untwisted. From
there only `U`, `R`, `F` turns are searched (9 moves, branching factor 6 after pruning
instead of 15), and the goal is the exact solved state. Solutions are mapped back to the
user's grip with `Regrip::to_original`, which relabels faces according to the rotation.

## Consequences

- The search space is exactly 7! x 3^6 = 3,674,160 states with a perfect, cheap index;
  the full distance table is 3.7 MB and builds in under a second.
- No per-node normalisation cost.
- Returned solutions only use `U R F` (expressed in the original grip, so they may use any
  face letters) and leave the cube solved *up to rotation*, which is what the conformance
  harness checks.
- This trick is specific to puzzles without fixed reference pieces and lives entirely in
  `cube2.rs`; the generic search code knows nothing about it.
