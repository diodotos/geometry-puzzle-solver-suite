//! The [`Puzzle`] trait: a puzzle as an implicit graph of states.

use std::fmt::{Debug, Display};
use std::hash::Hash;

/// A permutation puzzle described as a state space.
///
/// A `Puzzle` value is the *definition* of a puzzle (its geometry and move set), not a
/// particular configuration. Configurations are [`Puzzle::State`] values, which are plain
/// data: cheap to clone, hashable, and free of any search bookkeeping. This lets one
/// puzzle definition be parameterised at runtime (e.g. an `NxNxN` cube) while states
/// stay small and `Copy`-like.
///
/// Implementations must guarantee:
///
/// * `apply(apply(s, m), inverse(m)) == s` for every state `s` and move `m`;
/// * every move in [`Puzzle::moves`] has its inverse in [`Puzzle::moves`], so distances
///   are symmetric and backward search from the goal is valid;
/// * [`Puzzle::can_follow`] only rejects a move pair when an equally short sequence
///   that it accepts reaches the same state (it may prune, never disconnect).
pub trait Puzzle {
    /// A configuration of the puzzle.
    type State: Clone + Eq + Hash + Debug;
    /// A single move. `Display` must produce the puzzle's standard notation.
    type Move: Copy + Eq + Hash + Debug + Display;

    /// Short machine-friendly identifier, e.g. `"cube2"`.
    fn name(&self) -> &str;

    /// The (unique) goal state.
    fn solved(&self) -> Self::State;

    /// Whether `state` is the goal state.
    fn is_solved(&self, state: &Self::State) -> bool {
        *state == self.solved()
    }

    /// The moves search is allowed to use. This may be a strict subset of the moves the
    /// notation accepts, as long as it generates the same group (e.g. the 2x2x2 searches
    /// with `U`, `R`, `F` turns only; see `docs/theory/cube2.md`).
    fn moves(&self) -> &[Self::Move];

    /// Return the state reached by applying `mv` to `state`.
    fn apply(&self, state: &Self::State, mv: Self::Move) -> Self::State;

    /// The inverse of `mv`.
    fn inverse(&self, mv: Self::Move) -> Self::Move;

    /// Move-sequence pruning: may `next` directly follow `prev` in a search path?
    ///
    /// The default allows everything. Cubes reject consecutive turns of the same face
    /// (they merge into one turn) and fix an order for commuting opposite faces.
    fn can_follow(&self, _prev: Self::Move, _next: Self::Move) -> bool {
        true
    }

    /// Apply a whole sequence of moves.
    fn apply_all<'a, I>(&self, state: &Self::State, moves: I) -> Self::State
    where
        I: IntoIterator<Item = &'a Self::Move>,
        Self::Move: 'a,
    {
        moves
            .into_iter()
            .fold(state.clone(), |s, &m| self.apply(&s, m))
    }
}

/// Format a move sequence in standard space-separated notation.
pub fn format_moves<M: Display>(moves: &[M]) -> String {
    moves
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(" ")
}
