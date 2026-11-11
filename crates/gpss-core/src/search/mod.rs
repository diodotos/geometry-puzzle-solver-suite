//! Search algorithms over any [`Puzzle`].
//!
//! | solver      | optimal?                       | memory          | notes                          |
//! |-------------|--------------------------------|-----------------|--------------------------------|
//! | [`Bfs`]     | yes                            | O(states seen)  | baseline, no heuristic         |
//! | [`AStar`]   | yes if weight = 1 & admissible | O(states seen)  | weight > 1 gives weighted A*   |
//! | [`IdaStar`] | yes if admissible              | O(depth)        | Korf (1985); used for 3x3x3    |

mod astar;
mod bfs;
mod idastar;

use std::time::{Duration, Instant};

pub use astar::AStar;
pub use bfs::Bfs;
pub use idastar::IdaStar;

use crate::puzzle::Puzzle;

/// Resource limits for a single search. `None` means unlimited.
#[derive(Debug, Clone, Default)]
pub struct SearchLimits {
    /// Give up on solutions longer than this many moves.
    pub max_depth: Option<u32>,
    /// Give up after expanding this many nodes.
    pub max_expanded: Option<u64>,
    /// Give up after this much wall-clock time.
    pub timeout: Option<Duration>,
}

/// Counters describing the work a search did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SearchStats {
    /// Nodes whose successors were generated.
    pub expanded: u64,
    /// Successor nodes generated (after move pruning).
    pub generated: u64,
    /// Peak number of nodes held in memory at once (open + closed / frontier / path).
    pub peak_stored: usize,
    /// Iterations of an iterative-deepening search (1 for other searches).
    pub iterations: u32,
    /// Wall-clock time spent searching.
    pub elapsed: Duration,
}

/// A solution found by a [`Solver`].
#[derive(Debug, Clone)]
pub struct Solution<M> {
    pub moves: Vec<M>,
    pub stats: SearchStats,
    /// Whether the solver guarantees `moves` is a shortest solution.
    pub optimal: bool,
}

/// Why a search did not return a solution.
#[derive(Debug, Clone, thiserror::Error)]
pub enum SearchError {
    #[error("search space exhausted without reaching the goal")]
    Exhausted(SearchStats),
    #[error("search stopped: {reason}")]
    LimitReached {
        reason: &'static str,
        stats: SearchStats,
    },
}

/// A search algorithm.
pub trait Solver<P: Puzzle + ?Sized> {
    /// Human-readable name, e.g. `"idastar(dual)"`.
    fn name(&self) -> String;

    /// Whether solutions are guaranteed to be shortest.
    fn is_optimal(&self) -> bool;

    /// Find a move sequence taking `start` to the goal.
    fn solve(
        &self,
        puzzle: &P,
        start: &P::State,
        limits: &SearchLimits,
    ) -> Result<Solution<P::Move>, SearchError>;
}

/// Shared bookkeeping for limit checks.
pub(crate) struct Budget<'a> {
    limits: &'a SearchLimits,
    started: Instant,
    pub stats: SearchStats,
}

impl<'a> Budget<'a> {
    pub fn new(limits: &'a SearchLimits) -> Self {
        Self {
            limits,
            started: Instant::now(),
            stats: SearchStats {
                iterations: 1,
                ..SearchStats::default()
            },
        }
    }

    /// Record one expansion and check the node/time limits.
    pub fn expand(&mut self) -> Result<(), SearchError> {
        self.stats.expanded += 1;
        if let Some(max) = self.limits.max_expanded {
            if self.stats.expanded > max {
                return Err(self.stop("node limit reached"));
            }
        }
        if self.stats.expanded % 4096 == 0 {
            if let Some(t) = self.limits.timeout {
                if self.started.elapsed() > t {
                    return Err(self.stop("timeout"));
                }
            }
        }
        Ok(())
    }

    pub fn max_depth(&self) -> u32 {
        self.limits.max_depth.unwrap_or(u32::MAX)
    }

    pub fn note_stored(&mut self, n: usize) {
        self.stats.peak_stored = self.stats.peak_stored.max(n);
    }

    pub fn stop(&mut self, reason: &'static str) -> SearchError {
        SearchError::LimitReached {
            reason,
            stats: self.finish(),
        }
    }

    pub fn finish(&mut self) -> SearchStats {
        self.stats.elapsed = self.started.elapsed();
        self.stats.clone()
    }
}

/// Walk parent links in an arena back to the root and return the moves in order.
pub(crate) fn reconstruct<M: Copy>(arena: &[(Option<M>, u32)], mut idx: u32) -> Vec<M> {
    let mut moves = Vec::new();
    while let (Some(m), parent) = arena[idx as usize] {
        moves.push(m);
        idx = parent;
    }
    moves.reverse();
    moves
}
