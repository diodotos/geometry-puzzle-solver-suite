//! # gpss-puzzles
//!
//! Concrete puzzles implementing [`gpss_core::Puzzle`].
//!
//! * [`cube`] - notation shared by all cube-shaped puzzles (faces, turns, parsing).
//! * [`cube2`] - the 2x2x2 "pocket cube", with pattern-database heuristics.
//! * [`cube3`] - the 3x3x3 Rubik's Cube, with Kociemba's two-phase solver.
//!
//! Each puzzle module owns its state representation and its domain-specific
//! heuristics; search lives entirely in `gpss-core`.
//!
//! ```
//! use gpss_core::search::{IdaStar, SearchLimits};
//! use gpss_puzzles::cube::parse_sequence;
//! use gpss_puzzles::cube2::{Cube2, Cube2Heuristic};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let cube = Cube2::new();
//! let solver = IdaStar::new(Cube2Heuristic::Dual.build(&cube));
//! let scramble = parse_sequence("R U2 F' R2 U F2 R' U' F R2 U")?;
//! let solution = cube.solve_scramble(&scramble, &solver, &SearchLimits::default())?;
//! assert!(solution.optimal);
//! assert_eq!(solution.moves.len(), 9);
//! # Ok(())
//! # }
//! ```

pub mod cube;
pub mod cube2;
pub mod cube3;
