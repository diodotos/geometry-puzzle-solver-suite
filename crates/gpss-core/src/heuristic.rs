//! Distance-to-goal estimators.

use crate::puzzle::Puzzle;

/// An estimate of the number of moves needed to solve a state.
///
/// Optimal solvers ([`crate::search::AStar`] with weight 1, [`crate::search::IdaStar`])
/// return shortest solutions only if the heuristic is **admissible** (never
/// overestimates). Pattern databases are admissible by construction; learned heuristics
/// generally are not, and say so via [`Heuristic::is_admissible`].
pub trait Heuristic<P: Puzzle + ?Sized> {
    /// Human-readable name used in CLI output and benchmark reports.
    fn name(&self) -> String;

    /// Estimated distance from `state` to the goal.
    fn estimate(&self, state: &P::State) -> u32;

    /// Evaluate many states at once.
    ///
    /// Search algorithms that expand nodes in batches (batch weighted A*) call this so
    /// that heuristics backed by a neural network can amortise inference cost. The
    /// default simply loops over [`Heuristic::estimate`].
    fn estimate_batch(&self, states: &[P::State], out: &mut Vec<u32>) {
        out.clear();
        out.extend(states.iter().map(|s| self.estimate(s)));
    }

    /// Evaluate many states at once, keeping fractional estimates.
    ///
    /// Learned heuristics produce real-valued cost-to-go estimates; rounding them to
    /// integers would throw away the tie-breaking information batch weighted A* relies
    /// on. The default converts [`Heuristic::estimate_batch`].
    fn estimate_batch_real(&self, states: &[P::State], out: &mut Vec<f32>) {
        let mut ints = Vec::with_capacity(states.len());
        self.estimate_batch(states, &mut ints);
        out.clear();
        out.extend(ints.into_iter().map(|h| h as f32));
    }

    /// Whether the estimate is guaranteed never to exceed the true distance.
    fn is_admissible(&self) -> bool {
        true
    }
}

/// The trivial heuristic `h = 0`. Turns A* into uniform-cost search.
#[derive(Debug, Default, Clone, Copy)]
pub struct ZeroHeuristic;

impl<P: Puzzle + ?Sized> Heuristic<P> for ZeroHeuristic {
    fn name(&self) -> String {
        "zero".into()
    }
    fn estimate(&self, _state: &P::State) -> u32 {
        0
    }
}

/// The pointwise maximum of several heuristics.
///
/// The maximum of admissible heuristics is admissible, and at least as informed as each
/// component. The original CubeSolver "dual" heuristic is `max(orientation, permutation)`.
pub struct MaxHeuristic<'a, P: Puzzle + ?Sized> {
    parts: Vec<Box<dyn Heuristic<P> + 'a>>,
    name: String,
}

impl<'a, P: Puzzle + ?Sized> MaxHeuristic<'a, P> {
    pub fn new(name: impl Into<String>, parts: Vec<Box<dyn Heuristic<P> + 'a>>) -> Self {
        assert!(
            !parts.is_empty(),
            "MaxHeuristic needs at least one component"
        );
        Self {
            parts,
            name: name.into(),
        }
    }
}

impl<'a, P: Puzzle + ?Sized> Heuristic<P> for MaxHeuristic<'a, P> {
    fn name(&self) -> String {
        self.name.clone()
    }
    fn estimate(&self, state: &P::State) -> u32 {
        self.parts
            .iter()
            .map(|h| h.estimate(state))
            .max()
            .unwrap_or(0)
    }
    fn is_admissible(&self) -> bool {
        self.parts.iter().all(|h| h.is_admissible())
    }
}

impl<P: Puzzle + ?Sized, H: Heuristic<P> + ?Sized> Heuristic<P> for &H {
    fn name(&self) -> String {
        (**self).name()
    }
    fn estimate(&self, state: &P::State) -> u32 {
        (**self).estimate(state)
    }
    fn estimate_batch(&self, states: &[P::State], out: &mut Vec<u32>) {
        (**self).estimate_batch(states, out)
    }
    fn estimate_batch_real(&self, states: &[P::State], out: &mut Vec<f32>) {
        (**self).estimate_batch_real(states, out)
    }
    fn is_admissible(&self) -> bool {
        (**self).is_admissible()
    }
}

impl<P: Puzzle + ?Sized, H: Heuristic<P> + ?Sized> Heuristic<P> for Box<H> {
    fn name(&self) -> String {
        (**self).name()
    }
    fn estimate(&self, state: &P::State) -> u32 {
        (**self).estimate(state)
    }
    fn estimate_batch(&self, states: &[P::State], out: &mut Vec<u32>) {
        (**self).estimate_batch(states, out)
    }
    fn estimate_batch_real(&self, states: &[P::State], out: &mut Vec<f32>) {
        (**self).estimate_batch_real(states, out)
    }
    fn is_admissible(&self) -> bool {
        (**self).is_admissible()
    }
}
