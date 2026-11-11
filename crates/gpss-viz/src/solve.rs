//! Hand-off between the viewer and the solvers.
//!
//! The viewer always works on a [`Cuboid`] sticker state. [`solve`] picks the best solver
//! the suite has for that shape and returns plain [`LayerTurn`]s, so the solution plays back
//! through the same path as user moves:
//!
//! * **3x3x3:** the state is converted to the cubie model and solved with Kociemba's
//!   two-phase algorithm. Slice turns may have moved the centres, so stickers are first
//!   renamed by the colour of the centre they share a face with; renaming does not change
//!   which states count as solved (every face one colour), and face turns then mean the
//!   same thing in both models.
//! * **Anything else:** IDA* on the cuboid itself with the admissible orbit pattern
//!   databases, within [`SearchLimits`]. This is exact but only practical for short
//!   scrambles on larger puzzles; past the limits the viewer reports a failure.

use std::time::Duration;

use gpss_core::search::{IdaStar, SearchLimits, Solver};
use gpss_core::Puzzle;
use gpss_puzzles::cube3::{Cube3, Cube3State, TwoPhase};
use gpss_puzzles::cuboid::{Cuboid, CuboidState, Dims, LayerTurn, OrbitHeuristic};

/// A solve to run off the UI thread; hand the result back with its `id`.
#[derive(Clone, Debug)]
pub struct SolveJob {
    /// Matches the job to its answer; stale answers are dropped.
    pub id: u64,
    /// The puzzle.
    pub cuboid: Cuboid,
    /// The state to solve (the timeline's target when the job was created).
    pub state: CuboidState,
}

/// A solution in viewer moves.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Solved {
    /// Moves that take the job's state to a solved state.
    pub moves: Vec<LayerTurn>,
    /// Name of the solver used.
    pub solver: String,
    /// Whether the solver proved the solution shortest.
    pub optimal: bool,
}

/// Why no solution came back.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SolveError {
    /// The search gave up (limits) or found nothing.
    #[error("{0}")]
    Search(String),
    /// The sticker state does not correspond to a reachable 3x3x3 position.
    #[error("not a valid 3x3x3 position")]
    InvalidCube3,
}

/// Default limits for viewer solves: generous enough for the shipped solvers, short enough
/// to keep the UI responsive.
pub fn default_limits() -> SearchLimits {
    SearchLimits {
        max_depth: Some(40),
        max_expanded: None,
        timeout: Some(Duration::from_secs(10)),
    }
}

/// Run a job.
pub fn run(job: &SolveJob) -> Result<Solved, SolveError> {
    solve(&job.cuboid, &job.state, &default_limits())
}

/// Solve `state` on `cuboid`.
pub fn solve(
    cuboid: &Cuboid,
    state: &CuboidState,
    limits: &SearchLimits,
) -> Result<Solved, SolveError> {
    if cuboid.is_solved(state) {
        return Ok(Solved {
            moves: Vec::new(),
            solver: "none (already solved)".into(),
            optimal: true,
        });
    }
    if cuboid.dims() == Dims::new(3, 3, 3) {
        let cube = to_cube3(cuboid, state).ok_or(SolveError::InvalidCube3)?;
        let sol = TwoPhase::default()
            .solve(&Cube3::new(), &cube, limits)
            .map_err(|e| SolveError::Search(e.to_string()))?;
        let moves = sol
            .moves
            .iter()
            .map(|t| {
                cuboid
                    .parse_move(&t.to_string())
                    .expect("face turns are legal on a 3x3x3")
            })
            .collect();
        return Ok(Solved {
            moves,
            solver: "Kociemba two-phase".into(),
            optimal: sol.optimal,
        });
    }
    let solver = IdaStar::new(OrbitHeuristic::new(cuboid));
    let sol = solver
        .solve(cuboid, state, limits)
        .map_err(|e| SolveError::Search(e.to_string()))?;
    Ok(Solved {
        moves: sol.moves,
        solver: "IDA* + orbit pattern databases".into(),
        optimal: sol.optimal,
    })
}

const U: usize = 0;
const R: usize = 1;
const F: usize = 2;
const D: usize = 3;
const L: usize = 4;
const B: usize = 5;
/// Faces of each corner position in Kociemba's facelet order (U/D facelet first, then
/// clockwise), matching `gpss_puzzles::cube3`.
const CORNER_FACES: [[usize; 3]; 8] = [
    [U, R, F],
    [U, F, L],
    [U, L, B],
    [U, B, R],
    [D, F, R],
    [D, L, F],
    [D, B, L],
    [D, R, B],
];
/// Faces of each edge position, reference facelet first, matching `gpss_puzzles::cube3`.
const EDGE_FACES: [[usize; 2]; 12] = [
    [U, R],
    [U, F],
    [U, L],
    [U, B],
    [D, R],
    [D, F],
    [D, L],
    [D, B],
    [F, R],
    [F, L],
    [B, L],
    [B, R],
];
/// Outward normals of the faces in `U R F D L B` order (the cuboid's colour order).
const NORMALS: [[i32; 3]; 6] = [
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
    [0, -1, 0],
    [-1, 0, 0],
    [0, 0, -1],
];

/// The cubie-model position shown by a 3x3x3 sticker state, or `None` if it is not one.
pub fn to_cube3(cuboid: &Cuboid, state: &CuboidState) -> Option<Cube3State> {
    let slot = |pos: [i32; 3], face: usize| {
        cuboid
            .stickers()
            .iter()
            .position(|s| s.pos == pos && s.normal == NORMALS[face])
    };
    // Rename colours by centre so that slice turns do not matter.
    let mut rename = [u8::MAX; 6];
    for (f, n) in NORMALS.iter().enumerate() {
        let c = state[slot(n.map(|x| 2 * x), f)?] as usize;
        if rename[c] != u8::MAX {
            return None;
        }
        rename[c] = f as u8;
    }
    let colour = |pos: [i32; 3], face: usize| -> Option<usize> {
        Some(rename[state[slot(pos, face)?] as usize] as usize)
    };
    let corner_pos = |faces: &[usize]| {
        let mut p = [0; 3];
        for &f in faces {
            for (a, v) in p.iter_mut().enumerate() {
                *v += 2 * NORMALS[f][a];
            }
        }
        p
    };
    let mut cube = Cube3State::SOLVED;
    for (p, faces) in CORNER_FACES.iter().enumerate() {
        let pos = corner_pos(faces);
        let cols = [
            colour(pos, faces[0])?,
            colour(pos, faces[1])?,
            colour(pos, faces[2])?,
        ];
        let mut sorted = cols;
        sorted.sort_unstable();
        cube.cp[p] = CORNER_FACES.iter().position(|q| {
            let mut s = *q;
            s.sort_unstable();
            s == sorted
        })? as u8;
        cube.co[p] = cols.iter().position(|&c| c == U || c == D)? as u8;
    }
    for (p, faces) in EDGE_FACES.iter().enumerate() {
        let pos = corner_pos(faces);
        let cols = [colour(pos, faces[0])?, colour(pos, faces[1])?];
        let q = EDGE_FACES.iter().position(|q| {
            (q[0] == cols[0] && q[1] == cols[1]) || (q[0] == cols[1] && q[1] == cols[0])
        })?;
        cube.ep[p] = q as u8;
        cube.eo[p] = cols.iter().position(|&c| c == EDGE_FACES[q][0])? as u8;
    }
    cube.is_valid().then_some(cube)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use rand::{rngs::StdRng, SeedableRng};

    fn cube3() -> Cuboid {
        Cuboid::new(Dims::new(3, 3, 3))
    }

    proptest! {
        /// The sticker-to-cubie conversion agrees with the cubie model's own move tables.
        #[test]
        fn conversion_matches_the_cubie_model(seed in 0u64..10_000, len in 0usize..25) {
            let mut rng = StdRng::seed_from_u64(seed);
            let turns = gpss_puzzles::cube::random_scramble(&mut rng, len);
            let text = gpss_core::puzzle::format_moves(&turns);
            let c = cube3();
            let state = c.scrambled(&c.parse_sequence(&text).unwrap());
            prop_assert_eq!(to_cube3(&c, &state), Some(Cube3::new().scrambled(&turns)));
        }

        /// Solutions found through the cubie model solve the sticker state, including
        /// positions with slice turns (centres moved).
        #[test]
        fn two_phase_solutions_solve_the_viewer_state(seed in 0u64..10_000) {
            let c = cube3();
            let mut rng = StdRng::seed_from_u64(seed);
            let scramble = c.random_scramble(&mut rng, 20);
            let state = c.scrambled(&scramble);
            let solved = solve(&c, &state, &default_limits()).unwrap();
            prop_assert!(c.is_solved(&c.apply_all(&state, &solved.moves)));
        }
    }

    #[test]
    fn small_cuboids_solve_optimally() {
        for d in ["2x2x2", "3x3x2"] {
            let c = Cuboid::new(d.parse().unwrap());
            let mut rng = StdRng::seed_from_u64(7);
            let state = c.scrambled(&c.random_scramble(&mut rng, 5));
            let s = solve(&c, &state, &default_limits()).unwrap();
            assert!(s.optimal && s.moves.len() <= 5);
            assert!(c.is_solved(&c.apply_all(&state, &s.moves)));
        }
    }

    #[test]
    fn non_positions_are_rejected() {
        let c = cube3();
        let mut state = c.solved();
        state.swap(0, 1);
        state.swap(9, 30);
        assert_eq!(to_cube3(&c, &state), None);
    }
}
