use std::cmp::Reverse;
use std::collections::hash_map::Entry;
use std::collections::{BinaryHeap, HashMap};

use super::{reconstruct, Budget, SearchError, SearchLimits, Solution, Solver};
use crate::heuristic::Heuristic;
use crate::puzzle::Puzzle;

/// A* search (Hart, Nilsson & Raphael 1968), optionally weighted.
///
/// Nodes are ordered by `f = g + weight * h`. With `weight == 1` and an admissible,
/// consistent heuristic the first goal popped is optimal. `weight > 1` trades solution
/// length for speed (Pohl 1970); DeepCubeA's batch weighted A* uses the equivalent
/// `f = lambda * g + h` form.
pub struct AStar<H> {
    heuristic: H,
    weight: f64,
}

impl<H> AStar<H> {
    /// Plain (optimal) A*.
    pub fn new(heuristic: H) -> Self {
        Self {
            heuristic,
            weight: 1.0,
        }
    }

    /// Weighted A*: `f = g + weight * h`, `weight >= 1`.
    pub fn weighted(heuristic: H, weight: f64) -> Self {
        assert!(weight >= 1.0, "weight must be >= 1");
        Self { heuristic, weight }
    }
}

struct Node<S> {
    state: S,
    g: u32,
}

impl<P: Puzzle + ?Sized, H: Heuristic<P>> Solver<P> for AStar<H> {
    fn name(&self) -> String {
        if self.weight == 1.0 {
            format!("astar({})", self.heuristic.name())
        } else {
            format!("astar({}, w={})", self.heuristic.name(), self.weight)
        }
    }

    fn is_optimal(&self) -> bool {
        self.weight == 1.0 && self.heuristic.is_admissible()
    }

    fn solve(
        &self,
        puzzle: &P,
        start: &P::State,
        limits: &SearchLimits,
    ) -> Result<Solution<P::Move>, SearchError> {
        let mut budget = Budget::new(limits);
        // Priorities are fixed-point so the heap can use integer ordering.
        let key = |g: u32, h: u32| -> u64 { ((g as f64 + self.weight * h as f64) * 1024.0) as u64 };

        let mut nodes: Vec<Node<P::State>> = vec![Node {
            state: start.clone(),
            g: 0,
        }];
        let mut links: Vec<(Option<P::Move>, u32)> = vec![(None, 0)];
        let mut best_g: HashMap<P::State, u32> = HashMap::from([(start.clone(), 0)]);
        // Min-heap on f; ties broken towards deeper nodes (larger g), which reaches the
        // goal sooner without affecting optimality.
        let mut open = BinaryHeap::from([Reverse((
            key(0, self.heuristic.estimate(start)),
            Reverse(0u32),
            0u32,
        ))]);

        while let Some(Reverse((_, _, idx))) = open.pop() {
            let (g, state) = {
                let n = &nodes[idx as usize];
                (n.g, n.state.clone())
            };
            if best_g.get(&state).is_some_and(|&b| g > b) {
                continue; // stale entry
            }
            if puzzle.is_solved(&state) {
                budget.note_stored(best_g.len() + open.len());
                return Ok(Solution {
                    moves: reconstruct(&links, idx),
                    stats: budget.finish(),
                    optimal: Solver::<P>::is_optimal(self),
                });
            }
            if g >= budget.max_depth() {
                continue;
            }
            budget.expand()?;
            let prev = links[idx as usize].0;
            for &mv in puzzle.moves() {
                if prev.is_some_and(|p| !puzzle.can_follow(p, mv)) {
                    continue;
                }
                let child = puzzle.apply(&state, mv);
                budget.stats.generated += 1;
                let cg = g + 1;
                match best_g.entry(child.clone()) {
                    Entry::Occupied(mut e) => {
                        if cg >= *e.get() {
                            continue;
                        }
                        e.insert(cg);
                    }
                    Entry::Vacant(e) => {
                        e.insert(cg);
                    }
                }
                let h = self.heuristic.estimate(&child);
                let cidx = nodes.len() as u32;
                nodes.push(Node {
                    state: child,
                    g: cg,
                });
                links.push((Some(mv), idx));
                open.push(Reverse((key(cg, h), Reverse(cg), cidx)));
            }
            budget.note_stored(best_g.len() + open.len());
        }
        if limits.max_depth.is_some() {
            Err(budget.stop("depth limit reached"))
        } else {
            Err(SearchError::Exhausted(budget.finish()))
        }
    }
}
