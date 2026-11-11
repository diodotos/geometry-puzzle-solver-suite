use std::collections::{HashSet, VecDeque};

use super::{reconstruct, Budget, SearchError, SearchLimits, Solution, Solver};
use crate::puzzle::Puzzle;

/// Breadth-first search. Optimal, heuristic-free, memory-hungry: the baseline every
/// other solver is measured against.
#[derive(Debug, Default, Clone, Copy)]
pub struct Bfs;

impl<P: Puzzle + ?Sized> Solver<P> for Bfs {
    fn name(&self) -> String {
        "bfs".into()
    }

    fn is_optimal(&self) -> bool {
        true
    }

    fn solve(
        &self,
        puzzle: &P,
        start: &P::State,
        limits: &SearchLimits,
    ) -> Result<Solution<P::Move>, SearchError> {
        let mut budget = Budget::new(limits);
        if puzzle.is_solved(start) {
            return Ok(Solution {
                moves: vec![],
                stats: budget.finish(),
                optimal: true,
            });
        }
        // arena[i] = (move that produced node i, parent index); states live in `queue`.
        let mut arena: Vec<(Option<P::Move>, u32)> = vec![(None, 0)];
        let mut depth_of: Vec<u32> = vec![0];
        let mut seen: HashSet<P::State> = HashSet::from([start.clone()]);
        let mut queue: VecDeque<(P::State, u32)> = VecDeque::from([(start.clone(), 0)]);

        while let Some((state, idx)) = queue.pop_front() {
            let depth = depth_of[idx as usize];
            if depth >= budget.max_depth() {
                continue;
            }
            budget.expand()?;
            let prev = arena[idx as usize].0;
            for &mv in puzzle.moves() {
                if prev.is_some_and(|p| !puzzle.can_follow(p, mv)) {
                    continue;
                }
                let child = puzzle.apply(&state, mv);
                budget.stats.generated += 1;
                if !seen.insert(child.clone()) {
                    continue;
                }
                let child_idx = arena.len() as u32;
                arena.push((Some(mv), idx));
                depth_of.push(depth + 1);
                if puzzle.is_solved(&child) {
                    budget.note_stored(seen.len());
                    return Ok(Solution {
                        moves: reconstruct(&arena, child_idx),
                        stats: budget.finish(),
                        optimal: true,
                    });
                }
                queue.push_back((child, child_idx));
            }
            budget.note_stored(seen.len());
        }
        if limits.max_depth.is_some() {
            Err(budget.stop("depth limit reached"))
        } else {
            Err(SearchError::Exhausted(budget.finish()))
        }
    }
}
