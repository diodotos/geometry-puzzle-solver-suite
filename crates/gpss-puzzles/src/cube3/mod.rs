//! The 3x3x3 Rubik's Cube.
//!
//! # Representation
//!
//! Kociemba's cubie model [Koc]: eight corners and twelve edges, each with a position
//! and an orientation. Centres never move under face turns, so no regripping is needed
//! (contrast [`crate::cube2`]).
//!
//! * Corners: `URF UFL ULB UBR DFR DLF DBL DRB` (as in the 2x2x2).
//! * Edges: `UR UF UL UB DR DF DL DB FR FL BL BR`.
//!
//! `cp[i]`/`ep[i]` name the cubie that sits in position `i`; `co[i]` (mod 3) and `eo[i]`
//! (mod 2) are its orientation. Composition uses the same "replaced-by" convention as
//! the 2x2x2: `a.then(b)` is "do `a`, then `b`".
//!
//! # Solvers
//!
//! * [`TwoPhase`] - Kociemba's two-phase algorithm: fast, near-optimal.
//! * Any generic `gpss_core` solver works too (BFS, A*, IDA*), but without strong
//!   pattern databases only for short scrambles.
//!
//! See `docs/theory/kociemba-two-phase.md`.

mod cache;
pub mod coord;
pub mod korf;
pub mod sym;
pub mod symcoord;
#[cfg(test)]
mod tests;
pub mod twophase;

pub use korf::KorfHeuristic;
pub use twophase::TwoPhase;

use crate::cube::FaceTurn;
use gpss_core::search::{SearchError, SearchLimits, Solution, Solver};
use gpss_core::Puzzle;

/// A 3x3x3 position.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Cube3State {
    pub cp: [u8; 8],
    pub co: [u8; 8],
    pub ep: [u8; 12],
    pub eo: [u8; 12],
}

impl Cube3State {
    /// The solved cube (identity permutation, all orientations 0).
    pub const SOLVED: Cube3State = Cube3State {
        cp: [0, 1, 2, 3, 4, 5, 6, 7],
        co: [0; 8],
        ep: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        eo: [0; 12],
    };

    /// `self` followed by `t`.
    pub fn then(&self, t: &Cube3State) -> Cube3State {
        let mut out = Cube3State::SOLVED;
        for i in 0..8 {
            let src = t.cp[i] as usize;
            out.cp[i] = self.cp[src];
            out.co[i] = (self.co[src] + t.co[i]) % 3;
        }
        for i in 0..12 {
            let src = t.ep[i] as usize;
            out.ep[i] = self.ep[src];
            out.eo[i] = (self.eo[src] + t.eo[i]) % 2;
        }
        out
    }

    /// `self` followed by each move of `moves` in turn.
    pub fn then_all(&self, moves: &[FaceTurn]) -> Cube3State {
        moves.iter().fold(*self, |s, &m| s.then(transform(m)))
    }

    /// The inverse position: `s.then(&s.inverse())` is solved. An optimal solution of
    /// the inverse, reversed and inverted move by move, solves `s`, so both are the same
    /// distance from solved.
    pub fn inverse(&self) -> Cube3State {
        let mut out = Cube3State::SOLVED;
        for i in 0..8 {
            let c = self.cp[i] as usize;
            out.cp[c] = i as u8;
            out.co[c] = (3 - self.co[i]) % 3;
        }
        for i in 0..12 {
            let e = self.ep[i] as usize;
            out.ep[e] = i as u8;
            out.eo[e] = self.eo[i];
        }
        out
    }

    /// Whether the state satisfies the cube invariants (twist, flip and parity), i.e. is
    /// reachable by face turns.
    pub fn is_valid(&self) -> bool {
        let perm_ok = |p: &[u8]| {
            let mut seen = vec![false; p.len()];
            p.iter()
                .all(|&x| (x as usize) < p.len() && !std::mem::replace(&mut seen[x as usize], true))
        };
        perm_ok(&self.cp)
            && perm_ok(&self.ep)
            && self.co.iter().all(|&x| x < 3)
            && self.eo.iter().all(|&x| x < 2)
            && self.co.iter().map(|&x| x as u32).sum::<u32>() % 3 == 0
            && self.eo.iter().map(|&x| x as u32).sum::<u32>() % 2 == 0
            && parity(&self.cp) == parity(&self.ep)
    }
}

/// Permutation parity (0 even, 1 odd).
pub fn parity(p: &[u8]) -> u8 {
    let mut s = 0;
    for i in 0..p.len() {
        for j in i + 1..p.len() {
            if p[i] > p[j] {
                s ^= 1;
            }
        }
    }
    s
}

/// Clockwise quarter turns in `Face` order `U R F D L B` (Kociemba's basic moves).
const QUARTER_TURNS: [Cube3State; 6] = [
    Cube3State {
        cp: [3, 0, 1, 2, 4, 5, 6, 7],
        co: [0; 8],
        ep: [3, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11],
        eo: [0; 12],
    },
    Cube3State {
        cp: [4, 1, 2, 0, 7, 5, 6, 3],
        co: [2, 0, 0, 1, 1, 0, 0, 2],
        ep: [8, 1, 2, 3, 11, 5, 6, 7, 4, 9, 10, 0],
        eo: [0; 12],
    },
    Cube3State {
        cp: [1, 5, 2, 3, 0, 4, 6, 7],
        co: [1, 2, 0, 0, 2, 1, 0, 0],
        ep: [0, 9, 2, 3, 4, 8, 6, 7, 1, 5, 10, 11],
        eo: [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
    },
    Cube3State {
        cp: [0, 1, 2, 3, 5, 6, 7, 4],
        co: [0; 8],
        ep: [0, 1, 2, 3, 5, 6, 7, 4, 8, 9, 10, 11],
        eo: [0; 12],
    },
    Cube3State {
        cp: [0, 2, 6, 3, 4, 1, 5, 7],
        co: [0, 1, 2, 0, 0, 2, 1, 0],
        ep: [0, 1, 10, 3, 4, 5, 9, 7, 8, 2, 6, 11],
        eo: [0; 12],
    },
    Cube3State {
        cp: [0, 1, 3, 7, 4, 5, 2, 6],
        co: [0, 0, 1, 2, 0, 0, 2, 1],
        ep: [0, 1, 2, 11, 4, 5, 6, 10, 8, 9, 3, 7],
        eo: [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
    },
];

fn move_table() -> &'static [Cube3State; 18] {
    static TABLE: std::sync::OnceLock<[Cube3State; 18]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [Cube3State::SOLVED; 18];
        for (f, q) in QUARTER_TURNS.iter().enumerate() {
            let h = q.then(q);
            t[3 * f] = *q;
            t[3 * f + 1] = h;
            t[3 * f + 2] = h.then(q);
        }
        t
    })
}

/// Index of a face turn in `FaceTurn::all()` order (face-major, then `Cw Half Ccw`).
pub fn move_index(mv: FaceTurn) -> usize {
    3 * mv.face as usize + mv.turn as usize - 1
}

/// The permutation applied by a face turn.
pub fn transform(mv: FaceTurn) -> &'static Cube3State {
    &move_table()[move_index(mv)]
}

/// The 3x3x3 with the 18 half-turn-metric face turns.
pub struct Cube3 {
    moves: Vec<FaceTurn>,
}

impl Default for Cube3 {
    fn default() -> Self {
        Self::new()
    }
}

impl Cube3 {
    /// The 3x3x3 with all 18 face turns.
    pub fn new() -> Self {
        Self {
            moves: FaceTurn::all().collect(),
        }
    }

    /// The position reached from solved by applying `scramble`.
    pub fn scrambled(&self, scramble: &[FaceTurn]) -> Cube3State {
        Cube3State::SOLVED.then_all(scramble)
    }

    /// Solve the position reached by `scramble` with `solver`.
    pub fn solve_scramble<S: Solver<Cube3> + ?Sized>(
        &self,
        scramble: &[FaceTurn],
        solver: &S,
        limits: &SearchLimits,
    ) -> Result<Solution<FaceTurn>, SearchError> {
        solver.solve(self, &self.scrambled(scramble), limits)
    }
}

impl Puzzle for Cube3 {
    type State = Cube3State;
    type Move = FaceTurn;

    fn name(&self) -> &str {
        "cube3"
    }

    fn solved(&self) -> Cube3State {
        Cube3State::SOLVED
    }

    fn moves(&self) -> &[FaceTurn] {
        &self.moves
    }

    fn apply(&self, state: &Cube3State, mv: FaceTurn) -> Cube3State {
        state.then(transform(mv))
    }

    fn inverse(&self, mv: FaceTurn) -> FaceTurn {
        mv.inverse()
    }

    fn can_follow(&self, prev: FaceTurn, next: FaceTurn) -> bool {
        FaceTurn::can_follow(prev, next)
    }
}

/// Learned-heuristic encoding: the 8 corners as `3 * piece + twist` followed by the 12
/// edges as `2 * piece + flip`, 24 classes each (20 one-hot groups, 480 inputs). This
/// is the cubie form of DeepCubeA's sticker one-hot and carries the same information.
/// Bump `version` if this ever changes. See `docs/theory/deepcubea.md`.
impl gpss_core::encode::Encodable for Cube3 {
    fn encoding(&self) -> gpss_core::encode::EncodingSpec {
        gpss_core::encode::EncodingSpec {
            puzzle: "cube3".into(),
            scheme: "cubie".into(),
            version: 1,
            len: 20,
            classes: 24,
        }
    }

    #[inline]
    fn encode(&self, state: &Cube3State, out: &mut [u8]) {
        let (corners, edges) = out.split_at_mut(8);
        for (o, (&p, &t)) in corners.iter_mut().zip(state.cp.iter().zip(&state.co)) {
            *o = 3 * p + t;
        }
        for (o, (&p, &f)) in edges.iter_mut().zip(state.ep.iter().zip(&state.eo)) {
            *o = 2 * p + f;
        }
    }
}
