use super::{Budget, SearchError, SearchLimits, Solution, Solver};
use crate::heuristic::Heuristic;
use crate::puzzle::Puzzle;

/// Iterative-deepening A* (Korf 1985).
///
/// Repeated depth-first searches bounded by `f = g + h <= threshold`, raising the
/// threshold to the smallest `f` that exceeded it. Memory is proportional to the
/// solution depth, which is what makes optimal 3x3x3 solving with large pattern
/// databases feasible (Korf 1997). With an admissible heuristic the first solution found
/// is optimal.
///
/// IDA* keeps no closed set, so it relies on [`Puzzle::can_follow`] to avoid the most
/// common duplicate paths (same-face and commuting-face move pairs).
pub struct IdaStar<H> {
    heuristic: H,
}

impl<H> IdaStar<H> {
    pub fn new(heuristic: H) -> Self {
        Self { heuristic }
    }
}

enum Step {
    Found,
    Next(u32),
}

struct Ctx<'a, 'b, P: Puzzle + ?Sized, H> {
    puzzle: &'a P,
    heuristic: &'a H,
    budget: Budget<'b>,
    path: Vec<P::Move>,
}

impl<'a, 'b, P: Puzzle + ?Sized, H: Heuristic<P>> Ctx<'a, 'b, P, H> {
    fn dfs(&mut self, state: &P::State, g: u32, h: u32, bound: u32) -> Result<Step, SearchError> {
        let f = g + h;
        if f > bound {
            return Ok(Step::Next(f));
        }
        if self.puzzle.is_solved(state) {
            return Ok(Step::Found);
        }
        self.budget.expand()?;
        let mut next_bound = u32::MAX;
        let prev = self.path.last().copied();
        for &mv in self.puzzle.moves() {
            if prev.is_some_and(|p| !self.puzzle.can_follow(p, mv)) {
                continue;
            }
            let child = self.puzzle.apply(state, mv);
            self.budget.stats.generated += 1;
            let ch = self.heuristic.estimate(&child);
            self.path.push(mv);
            self.budget.note_stored(self.path.len());
            match self.dfs(&child, g + 1, ch, bound)? {
                Step::Found => return Ok(Step::Found),
                Step::Next(b) => next_bound = next_bound.min(b),
            }
            self.path.pop();
        }
        Ok(Step::Next(next_bound))
    }
}

impl<P: Puzzle + ?Sized, H: Heuristic<P>> Solver<P> for IdaStar<H> {
    fn name(&self) -> String {
        format!("idastar({})", self.heuristic.name())
    }

    fn is_optimal(&self) -> bool {
        self.heuristic.is_admissible()
    }

    fn solve(
        &self,
        puzzle: &P,
        start: &P::State,
        limits: &SearchLimits,
    ) -> Result<Solution<P::Move>, SearchError> {
        let mut ctx = Ctx {
            puzzle,
            heuristic: &self.heuristic,
            budget: Budget::new(limits),
            path: Vec::new(),
        };
        ctx.budget.stats.iterations = 0;
        let h0 = self.heuristic.estimate(start);
        let mut bound = h0;
        loop {
            if bound > ctx.budget.max_depth() {
                return Err(ctx.budget.stop("depth limit reached"));
            }
            ctx.budget.stats.iterations += 1;
            match ctx.dfs(start, 0, h0, bound)? {
                Step::Found => {
                    let stats = ctx.budget.finish();
                    return Ok(Solution {
                        moves: ctx.path,
                        stats,
                        optimal: self.heuristic.is_admissible(),
                    });
                }
                Step::Next(u32::MAX) => return Err(SearchError::Exhausted(ctx.budget.finish())),
                Step::Next(b) => bound = b,
            }
        }
    }
}
