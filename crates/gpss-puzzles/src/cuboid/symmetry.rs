//! Whole-puzzle rotations of a cuboid and search modulo rotation.
//!
//! A rotation that maps the cuboid onto itself (every axis onto an axis with the same layer
//! count) is applied as a *conjugation*: stickers move with the rotation and colours are
//! renamed by the rotation's face map. A conjugate has the same distance to the goal as the
//! original, because the goal (every face one colour) and the set of layer turns are both
//! invariant under these rotations. [`RotationReduced`] searches over one canonical
//! representative per class, which shrinks the space that BFS and A* store by up to 24x on
//! a cube (8x on a 3x3x2, 4x on a 2x3x4). See `docs/theory/cuboid.md`.

use std::collections::HashMap;

use gpss_core::search::{SearchError, SearchLimits, Solution, Solver};
use gpss_core::{Heuristic, Puzzle};

use super::{Cuboid, CuboidState, Dims, LayerTurn, Sticker, FACES};

/// One whole-puzzle rotation: `new[j] = colours[old[src[j]]]`.
#[derive(Clone, Debug)]
pub(super) struct Rotation {
    src: Box<[u16]>,
    colours: [u8; 6],
}

fn mul(m: &[[i32; 3]; 3], v: [i32; 3]) -> [i32; 3] {
    [0, 1, 2].map(|r| (0..3).map(|c| m[r][c] * v[c]).sum())
}

fn det(m: &[[i32; 3]; 3]) -> i32 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn face_normal(f: usize) -> [i32; 3] {
    let (_, axis, sign) = FACES[f];
    let mut n = [0; 3];
    n[axis as usize] = sign;
    n
}

/// Every proper rotation (signed permutation matrix, determinant +1) that maps `dims` onto
/// itself, identity first.
pub(super) fn rotations(
    dims: Dims,
    stickers: &[Sticker],
    index: &HashMap<([i32; 3], [i32; 3]), usize>,
) -> Vec<Rotation> {
    const PERMS: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let mut out = Vec::new();
    for p in PERMS {
        for signs in 0..8u8 {
            let mut m = [[0i32; 3]; 3];
            for (r, row) in m.iter_mut().enumerate() {
                row[p[r]] = if signs >> r & 1 == 1 { -1 } else { 1 };
            }
            // Row `r` sends input axis `p[r]` to output axis `r`.
            if det(&m) != 1 || (0..3).any(|r| dims.0[r] != dims.0[p[r]]) {
                continue;
            }
            let mut src = vec![0u16; stickers.len()];
            for (i, s) in stickers.iter().enumerate() {
                src[index[&(mul(&m, s.pos), mul(&m, s.normal))]] = i as u16;
            }
            let colours = std::array::from_fn(|f| {
                let n = mul(&m, face_normal(f));
                (0..6)
                    .find(|&g| face_normal(g) == n)
                    .expect("rotation maps faces to faces") as u8
            });
            out.push(Rotation {
                src: src.into_boxed_slice(),
                colours,
            });
        }
    }
    out
}

impl Cuboid {
    /// Number of whole-puzzle rotations mapping this cuboid onto itself (24 for a cube).
    pub fn rotation_count(&self) -> usize {
        self.rotations.len()
    }

    /// The conjugate of `state` by rotation `k` (`0` is the identity): stickers rotated,
    /// colours renamed to match. It has the same distance to the goal as `state`.
    pub fn rotate(&self, state: &CuboidState, k: usize) -> CuboidState {
        let r = &self.rotations[k];
        r.src
            .iter()
            .map(|&i| r.colours[state[i as usize] as usize])
            .collect()
    }

    /// The lexicographically smallest conjugate of `state` over all rotations.
    pub fn canonical(&self, state: &CuboidState) -> CuboidState {
        (0..self.rotations.len())
            .map(|k| self.rotate(state, k))
            .min()
            .expect("the identity rotation is always present")
    }
}

/// A [`Cuboid`] whose states are canonical modulo whole-puzzle rotation.
///
/// Moves are the cuboid's layer turns, but each is expressed in the frame of the canonical
/// parent, so a raw solution is only meaningful after [`RotationReduced::lift`] (done by
/// [`RotationReduced::solve`]). The cuboid's same-axis move pruning does not survive the
/// frame changes, so [`Puzzle::can_follow`] allows everything. That makes this wrapper a
/// win for searches with duplicate detection (BFS, A*, BWAS) and a loss for IDA*.
///
/// Contract exception: [`Puzzle::inverse`] does **not** satisfy
/// `apply(apply(s, m), inverse(m)) == s` here, because canonicalising after `m` may change
/// the frame. The quotient graph is still undirected, but an edge's reverse move depends on
/// the state, so nothing that undoes moves (backward/bidirectional search, path reversal)
/// may use this wrapper; use the raw [`Cuboid`] with [`Cuboid::canonical`] as the
/// visited-set key instead.
#[derive(Clone, Debug)]
pub struct RotationReduced {
    cuboid: Cuboid,
    name: String,
}

impl RotationReduced {
    /// The rotation quotient of `cuboid` (clones the cuboid, including its move tables).
    pub fn new(cuboid: &Cuboid) -> Self {
        Self {
            name: format!("{}/rotations", cuboid.name),
            cuboid: cuboid.clone(),
        }
    }

    /// The underlying cuboid.
    pub fn cuboid(&self) -> &Cuboid {
        &self.cuboid
    }

    /// Translate a solution found from `canonical(start)` into layer turns that solve
    /// `start` itself, in its own frame. Lengths are unchanged.
    pub fn lift(&self, start: &CuboidState, moves: &[LayerTurn]) -> Vec<LayerTurn> {
        let c = &self.cuboid;
        let mut raw = start.clone();
        let mut canon = c.canonical(start);
        moves
            .iter()
            .map(|&mv| {
                let next = self.apply(&canon, mv);
                let real = *c
                    .moves()
                    .iter()
                    .find(|&&r| c.canonical(&c.apply(&raw, r)) == next)
                    .expect("a rotated layer turn is a layer turn");
                raw = c.apply(&raw, real);
                canon = next;
                real
            })
            .collect()
    }

    /// Solve `start` modulo rotation and return moves in `start`'s own frame.
    pub fn solve<S: Solver<RotationReduced> + ?Sized>(
        &self,
        start: &CuboidState,
        solver: &S,
        limits: &SearchLimits,
    ) -> Result<Solution<LayerTurn>, SearchError> {
        let mut sol = solver.solve(self, &self.cuboid.canonical(start), limits)?;
        sol.moves = self.lift(start, &sol.moves);
        Ok(sol)
    }
}

impl Puzzle for RotationReduced {
    type State = CuboidState;
    type Move = LayerTurn;

    fn name(&self) -> &str {
        &self.name
    }

    fn solved(&self) -> CuboidState {
        self.cuboid.canonical(&self.cuboid.solved())
    }

    fn is_solved(&self, state: &CuboidState) -> bool {
        self.cuboid.is_solved(state)
    }

    fn moves(&self) -> &[LayerTurn] {
        self.cuboid.moves()
    }

    fn apply(&self, state: &CuboidState, mv: LayerTurn) -> CuboidState {
        self.cuboid.canonical(&self.cuboid.apply(state, mv))
    }

    /// The inverse in the *same* frame; see the contract exception on [`RotationReduced`].
    fn inverse(&self, mv: LayerTurn) -> LayerTurn {
        self.cuboid.inverse(mv)
    }
}

/// Adapts a [`Cuboid`] heuristic to [`RotationReduced`] states.
///
/// Only sound for heuristics that are invariant under conjugation by whole-puzzle rotations
/// (true for [`super::CornerHeuristic`], whose `cube2` table is regripped, and trivially for
/// [`gpss_core::ZeroHeuristic`]); canonical states are ordinary cuboid states.
#[derive(Clone, Debug)]
pub struct ReducedHeuristic<H>(pub H);

impl<H: Heuristic<Cuboid>> Heuristic<RotationReduced> for ReducedHeuristic<H> {
    fn name(&self) -> String {
        self.0.name()
    }

    fn estimate(&self, state: &CuboidState) -> u32 {
        self.0.estimate(state)
    }

    fn estimate_batch(&self, states: &[CuboidState], out: &mut Vec<u32>) {
        self.0.estimate_batch(states, out)
    }

    fn estimate_batch_real(&self, states: &[CuboidState], out: &mut Vec<f32>) {
        self.0.estimate_batch_real(states, out)
    }

    fn is_admissible(&self) -> bool {
        self.0.is_admissible()
    }
}
