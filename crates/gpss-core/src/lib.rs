//! # gpss-core
//!
//! The puzzle-agnostic half of the geometry puzzle solver suite.
//!
//! The central design rule (see `docs/architecture.md` and ADR 0002) is a strict
//! separation between **what a puzzle is** and **how it is searched**:
//!
//! * [`Puzzle`] describes a puzzle as a state space: a goal state, a set of moves and a
//!   transition function. It knows nothing about search.
//! * [`Heuristic`] estimates the distance from a state to the goal. Pattern databases
//!   ([`pdb::PatternDatabase`]) are the classical, exact-but-abstract kind; learned
//!   cost-to-go networks (DeepCubeA) are another kind behind the same trait.
//! * [`search::Solver`] implementations (BFS, A*, IDA*, batch weighted A*) consume any
//!   `Puzzle` plus any `Heuristic` and return a [`search::Solution`] with search
//!   statistics.
//! * [`encode::Encodable`] is an optional trait giving states a fixed-length encoding,
//!   which is what learned heuristics (crate `gpss-learn`) need.
//!
//! Nothing in this crate mentions cubes.

pub mod encode;
pub mod heuristic;
pub mod pdb;
pub mod puzzle;
pub mod search;

pub use encode::{Encodable, EncodingSpec};
pub use heuristic::{Heuristic, MaxHeuristic, ZeroHeuristic};
pub use pdb::{Coordinate, PatternDatabase, PdbHeuristic};
pub use puzzle::Puzzle;
pub use search::{SearchError, SearchLimits, SearchStats, Solution, Solver};
