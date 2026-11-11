//! The 2x2x2 "pocket cube".
//!
//! ## Representation
//!
//! A 2x2x2 has only corner pieces, so a state is the *cubie* description used by
//! Kociemba: for each of the 8 corner positions, which corner piece sits there
//! (`cp`) and how it is twisted (`co`, 0..3). Positions and pieces share the order
//!
//! ```text
//! 0   1   2   3   4   5   6   7
//! URF UFL ULB UBR DFR DLF DBL DRB
//! ```
//!
//! ## Search in a fixed frame
//!
//! A 2x2x2 has no centres, so the cube "solved in any orientation" counts as solved.
//! Turning only `U`, `R` and `F` never moves the `DBL` position, and every state is
//! reachable that way, so we:
//!
//! 1. **regrip** the scrambled cube (a whole-cube rotation) until the `DBL` piece sits
//!    at `DBL` untwisted - see [`Cube2::regrip`];
//! 2. search with the 9 moves `U R F` x `{1, 2, '}` towards the exact solved state;
//! 3. translate the solution back to the original grip ([`Regrip::to_original`]).
//!
//! This makes the state space exactly 7! x 3^6 = 3,674,160 states with a perfect
//! index, which is what lets the full distance table fit in 3.7 MB. See
//! `docs/theory/cube2.md` for the proof sketch and the coordinate definitions.

use gpss_core::heuristic::{Heuristic, MaxHeuristic};
use gpss_core::pdb::{Coordinate, PdbHeuristic};
use gpss_core::search::{SearchError, SearchLimits, Solution, Solver};
use gpss_core::Puzzle;

use crate::cube::{Face, FaceTurn, Turn};

/// Index of the corner position (and piece) that stays fixed during search.
pub const DBL: usize = 6;

/// Number of states in the fixed-`DBL` frame: 7! * 3^6.
pub const STATE_COUNT: usize = 5040 * 729;

/// Published number of 2x2x2 positions at each half-turn-metric distance.
pub const KNOWN_HTM_DISTRIBUTION: [u64; 12] = [
    1, 9, 54, 321, 1847, 9992, 50136, 227536, 870072, 1887748, 623800, 2644,
];

/// A 2x2x2 configuration in cubie form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Cube2State {
    /// `cp[i]` = piece at position `i`.
    pub cp: [u8; 8],
    /// `co[i]` = clockwise twist of the piece at position `i`.
    pub co: [u8; 8],
}

impl Cube2State {
    pub const SOLVED: Cube2State = Cube2State {
        cp: [0, 1, 2, 3, 4, 5, 6, 7],
        co: [0; 8],
    };

    /// `self` followed by `t` (Kociemba's corner multiplication).
    #[inline]
    pub fn then(&self, t: &Cube2State) -> Cube2State {
        let mut out = Cube2State::SOLVED;
        for i in 0..8 {
            let src = t.cp[i] as usize;
            out.cp[i] = self.cp[src];
            out.co[i] = (self.co[src] + t.co[i]) % 3;
        }
        out
    }

    /// Apply a sequence of face turns.
    pub fn then_all(&self, moves: &[FaceTurn]) -> Cube2State {
        moves.iter().fold(*self, |s, &m| s.then(transform(m)))
    }

    /// Solved up to a whole-cube rotation.
    pub fn is_solved_up_to_rotation(&self) -> bool {
        rotations()
            .iter()
            .any(|r| self.then(&r.transform) == Cube2State::SOLVED)
    }
}

/// Quarter-turn definitions in `URFDLB` order (Kociemba's "replaced-by" form).
const QUARTER_TURNS: [Cube2State; 6] = [
    // U
    Cube2State {
        cp: [3, 0, 1, 2, 4, 5, 6, 7],
        co: [0, 0, 0, 0, 0, 0, 0, 0],
    },
    // R
    Cube2State {
        cp: [4, 1, 2, 0, 7, 5, 6, 3],
        co: [2, 0, 0, 1, 1, 0, 0, 2],
    },
    // F
    Cube2State {
        cp: [1, 5, 2, 3, 0, 4, 6, 7],
        co: [1, 2, 0, 0, 2, 1, 0, 0],
    },
    // D
    Cube2State {
        cp: [0, 1, 2, 3, 5, 6, 7, 4],
        co: [0, 0, 0, 0, 0, 0, 0, 0],
    },
    // L
    Cube2State {
        cp: [0, 2, 6, 3, 4, 1, 5, 7],
        co: [0, 1, 2, 0, 0, 2, 1, 0],
    },
    // B
    Cube2State {
        cp: [0, 1, 3, 7, 4, 5, 2, 6],
        co: [0, 0, 1, 2, 0, 0, 2, 1],
    },
];

fn move_table() -> &'static [Cube2State; 18] {
    static TABLE: std::sync::OnceLock<[Cube2State; 18]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = [Cube2State::SOLVED; 18];
        for f in 0..6 {
            let q = QUARTER_TURNS[f];
            t[3 * f] = q;
            t[3 * f + 1] = q.then(&q);
            t[3 * f + 2] = q.then(&q).then(&q);
        }
        t
    })
}

/// The transform applied by a face turn.
#[inline]
pub fn transform(mv: FaceTurn) -> &'static Cube2State {
    &move_table()[3 * mv.face as usize + mv.turn as usize - 1]
}

/// A whole-cube rotation together with how it relabels faces.
#[derive(Debug, Clone, Copy)]
pub struct Rotation {
    pub transform: Cube2State,
    /// `face_map[f]` = face in the original grip that corresponds to face `f` after
    /// the rotation.
    face_map: [Face; 6],
}

/// All 24 whole-cube rotations (on a 2x2x2, `x = R L'`, `y = U D'`, `z = F B'`).
pub fn rotations() -> &'static [Rotation; 24] {
    static ROTS: std::sync::OnceLock<[Rotation; 24]> = std::sync::OnceLock::new();
    ROTS.get_or_init(|| {
        let ft = |f, t| *transform(FaceTurn::new(f, t));
        let gens = [
            ft(Face::R, Turn::Cw).then(&ft(Face::L, Turn::Ccw)),
            ft(Face::U, Turn::Cw).then(&ft(Face::D, Turn::Ccw)),
            ft(Face::F, Turn::Cw).then(&ft(Face::B, Turn::Ccw)),
        ];
        let mut found = vec![Cube2State::SOLVED];
        let mut i = 0;
        while i < found.len() {
            for g in &gens {
                let r = found[i].then(g);
                if !found.contains(&r) {
                    found.push(r);
                }
            }
            i += 1;
        }
        assert_eq!(
            found.len(),
            24,
            "whole-cube rotation group must have order 24"
        );
        // Derive each rotation's face relabelling empirically: m after rho must equal
        // rho after face_map(m) on every state. A generic probe state suffices because
        // distinct face turns act differently on it.
        let probe = [
            (Face::R, Turn::Cw),
            (Face::U, Turn::Half),
            (Face::F, Turn::Ccw),
            (Face::R, Turn::Half),
            (Face::U, Turn::Cw),
        ]
        .iter()
        .fold(Cube2State::SOLVED, |s, &(f, t)| s.then(&ft(f, t)));
        let rots: Vec<Rotation> = found
            .into_iter()
            .map(|rho| {
                let mut face_map = [Face::U; 6];
                for f in Face::ALL {
                    let lhs = probe.then(&rho).then(&ft(f, Turn::Cw));
                    face_map[f as usize] = *Face::ALL
                        .iter()
                        .find(|&&g| probe.then(&ft(g, Turn::Cw)).then(&rho) == lhs)
                        .expect("rotation must map face turns to face turns");
                }
                Rotation {
                    transform: rho,
                    face_map,
                }
            })
            .collect();
        rots.try_into().unwrap()
    })
}

/// The 2x2x2 puzzle definition (fixed-`DBL` frame; see the module docs).
#[derive(Debug, Clone)]
pub struct Cube2 {
    search_moves: Vec<FaceTurn>,
}

impl Default for Cube2 {
    fn default() -> Self {
        Self::new()
    }
}

/// How a scrambled cube was regripped, so solutions can be mapped back.
#[derive(Debug, Clone, Copy)]
pub struct Regrip {
    rotation: Rotation,
}

impl Regrip {
    /// Translate a move made in the regripped frame into the original grip.
    pub fn to_original(&self, mv: FaceTurn) -> FaceTurn {
        FaceTurn::new(self.rotation.face_map[mv.face as usize], mv.turn)
    }
}

impl Cube2 {
    pub fn new() -> Self {
        let search_moves = [Face::U, Face::R, Face::F]
            .into_iter()
            .flat_map(|f| Turn::ALL.into_iter().map(move |t| FaceTurn::new(f, t)))
            .collect();
        Self { search_moves }
    }

    /// The state reached from solved by a scramble (any of the 18 face turns).
    pub fn scrambled(&self, scramble: &[FaceTurn]) -> Cube2State {
        scramble
            .iter()
            .fold(Cube2State::SOLVED, |s, &m| s.then(transform(m)))
    }

    /// Rotate the whole cube so the `DBL` piece is at `DBL` with no twist.
    pub fn regrip(&self, state: &Cube2State) -> (Cube2State, Regrip) {
        for rot in rotations() {
            let s = state.then(&rot.transform);
            if s.cp[DBL] as usize == DBL && s.co[DBL] == 0 {
                return (s, Regrip { rotation: *rot });
            }
        }
        unreachable!("exactly one rotation fixes the DBL piece")
    }

    /// Regrip `scramble`'s state, solve it with `solver`, and express the solution in
    /// the original grip. The returned moves solve the cube up to whole-cube rotation.
    pub fn solve_scramble<S: Solver<Cube2> + ?Sized>(
        &self,
        scramble: &[FaceTurn],
        solver: &S,
        limits: &SearchLimits,
    ) -> Result<Solution<FaceTurn>, SearchError> {
        let (start, regrip) = self.regrip(&self.scrambled(scramble));
        let mut solution = solver.solve(self, &start, limits)?;
        for m in &mut solution.moves {
            *m = regrip.to_original(*m);
        }
        Ok(solution)
    }
}

impl Puzzle for Cube2 {
    type State = Cube2State;
    type Move = FaceTurn;

    fn name(&self) -> &str {
        "cube2"
    }

    fn solved(&self) -> Cube2State {
        Cube2State::SOLVED
    }

    fn is_solved(&self, state: &Cube2State) -> bool {
        *state == Cube2State::SOLVED
    }

    fn moves(&self) -> &[FaceTurn] {
        &self.search_moves
    }

    #[inline]
    fn apply(&self, state: &Cube2State, mv: FaceTurn) -> Cube2State {
        state.then(transform(mv))
    }

    fn inverse(&self, mv: FaceTurn) -> FaceTurn {
        mv.inverse()
    }

    fn can_follow(&self, prev: FaceTurn, next: FaceTurn) -> bool {
        FaceTurn::can_follow(prev, next)
    }
}

// --------------------------------------------------------------------------------------
// Coordinates (valid in the fixed-DBL frame)
// --------------------------------------------------------------------------------------

/// Positions other than `DBL`, in index order.
const FREE: [usize; 7] = [0, 1, 2, 3, 4, 5, 7];

/// Twists of the 7 free corners. The last free twist is implied by the others (total
/// twist is 0 mod 3), so 3^6 = 729 values.
#[derive(Debug, Clone, Copy, Default)]
pub struct OrientationCoord;

impl Coordinate<Cube2> for OrientationCoord {
    fn name(&self) -> &str {
        "orientation"
    }
    fn size(&self) -> usize {
        729
    }
    fn index(&self, s: &Cube2State) -> usize {
        debug_assert!(s.cp[DBL] as usize == DBL && s.co[DBL] == 0);
        FREE[..6]
            .iter()
            .fold(0, |acc, &p| acc * 3 + s.co[p] as usize)
    }
}

/// Lehmer-code rank of the permutation of the 7 free corners: 7! = 5040 values.
#[derive(Debug, Clone, Copy, Default)]
pub struct PermutationCoord;

impl Coordinate<Cube2> for PermutationCoord {
    fn name(&self) -> &str {
        "permutation"
    }
    fn size(&self) -> usize {
        5040
    }
    fn index(&self, s: &Cube2State) -> usize {
        debug_assert!(s.cp[DBL] as usize == DBL);
        let piece = |p: usize| match s.cp[p] {
            7 => 6u8,
            x => x,
        };
        let pieces = FREE.map(piece);
        pieces.iter().enumerate().fold(0, |rank, (i, &pi)| {
            let smaller_after = pieces[i + 1..].iter().filter(|&&pj| pj < pi).count();
            rank * (7 - i) + smaller_after
        })
    }
}

/// Perfect index of the whole state: `permutation * 729 + orientation`.
#[derive(Debug, Clone, Copy, Default)]
pub struct FullCoord;

impl Coordinate<Cube2> for FullCoord {
    fn name(&self) -> &str {
        "perfect"
    }
    fn size(&self) -> usize {
        STATE_COUNT
    }
    fn index(&self, s: &Cube2State) -> usize {
        PermutationCoord.index(s) * 729 + OrientationCoord.index(s)
    }
}

/// The heuristics available for the 2x2x2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cube2Heuristic {
    /// `h = 0` (A* degenerates to uniform-cost search).
    Zero,
    /// Exact distance to orient all corners (729-entry PDB).
    Orientation,
    /// Exact distance to permute all corners (5040-entry PDB).
    Permutation,
    /// `max(orientation, permutation)` - the original CubeSolver heuristic.
    Dual,
    /// Exact distance (full 3.7M-entry table): the perfect heuristic.
    Perfect,
}

impl Cube2Heuristic {
    pub const ALL: [Cube2Heuristic; 5] = [
        Cube2Heuristic::Zero,
        Cube2Heuristic::Orientation,
        Cube2Heuristic::Permutation,
        Cube2Heuristic::Dual,
        Cube2Heuristic::Perfect,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Cube2Heuristic::Zero => "zero",
            Cube2Heuristic::Orientation => "orientation",
            Cube2Heuristic::Permutation => "permutation",
            Cube2Heuristic::Dual => "dual",
            Cube2Heuristic::Perfect => "perfect",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|h| h.name() == name)
    }

    /// Build the heuristic (generating any pattern databases it needs).
    pub fn build(self, puzzle: &Cube2) -> Box<dyn Heuristic<Cube2>> {
        match self {
            Cube2Heuristic::Zero => Box::new(gpss_core::ZeroHeuristic),
            Cube2Heuristic::Orientation => Box::new(PdbHeuristic::new(puzzle, OrientationCoord)),
            Cube2Heuristic::Permutation => Box::new(PdbHeuristic::new(puzzle, PermutationCoord)),
            Cube2Heuristic::Dual => Box::new(MaxHeuristic::new(
                "dual",
                vec![
                    Box::new(PdbHeuristic::new(puzzle, OrientationCoord)),
                    Box::new(PdbHeuristic::new(puzzle, PermutationCoord)),
                ],
            )),
            Cube2Heuristic::Perfect => Box::new(PdbHeuristic::new(puzzle, FullCoord)),
        }
    }
}

#[cfg(test)]
mod tests;
