//! Exact evaluation of heuristics and solvers against true goal distances.
//!
//! For puzzles small enough to enumerate (the 2x2x2 has 3,674,160 states in its fixed
//! frame), [`enumerate_distances`] labels every state with its exact distance, so a
//! learned heuristic can be judged on the *whole* state space and a solver's solution
//! lengths can be compared with the optimum. This is the evaluation recommended in
//! `docs/theory/deepcubea.md` before spending GPU time on bigger puzzles.

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use gpss_core::heuristic::Heuristic;
use gpss_core::puzzle::Puzzle;
use gpss_core::search::{SearchLimits, Solver};
use rand::seq::SliceRandom;
use rand::Rng;
use serde::Serialize;

/// Every state reachable from the goal with its exact distance, in BFS order.
///
/// Returns `None` if more than `limit` states exist (the puzzle is too big to enumerate).
pub fn enumerate_distances<P: Puzzle + ?Sized>(
    puzzle: &P,
    limit: usize,
) -> Option<Vec<(P::State, u8)>> {
    let start = puzzle.solved();
    let mut seen: HashMap<P::State, u8> = HashMap::from([(start.clone(), 0)]);
    let mut order = vec![(start.clone(), 0u8)];
    let mut queue = VecDeque::from([start]);
    while let Some(state) = queue.pop_front() {
        let d = seen[&state];
        for &mv in puzzle.moves() {
            let child = puzzle.apply(&state, mv);
            if !seen.contains_key(&child) {
                if seen.len() >= limit {
                    return None;
                }
                seen.insert(child.clone(), d + 1);
                order.push((child.clone(), d + 1));
                queue.push_back(child);
            }
        }
    }
    Some(order)
}

/// Up to `per_depth` states of each distance, sampled uniformly within the depth.
pub fn stratified_sample<S: Clone, R: Rng + ?Sized>(
    labeled: &[(S, u8)],
    per_depth: usize,
    rng: &mut R,
) -> Vec<(S, u8)> {
    let max = labeled.iter().map(|&(_, d)| d).max().unwrap_or(0);
    let mut out = Vec::new();
    for depth in 0..=max {
        let mut at: Vec<&(S, u8)> = labeled.iter().filter(|(_, d)| *d == depth).collect();
        at.shuffle(rng);
        out.extend(at.into_iter().take(per_depth).cloned());
    }
    out
}

/// Error statistics of a heuristic over a set of states with known distance.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ErrorStats {
    /// True distance, or `None` for the aggregate row.
    pub depth: Option<u8>,
    pub count: u64,
    pub mean_estimate: f64,
    /// Mean absolute error.
    pub mae: f64,
    pub rmse: f64,
    /// Fraction of states with `h > d` (not admissible there).
    pub overestimate_rate: f64,
    /// Largest `h - d`.
    pub max_overestimate: f64,
    /// Fraction of states where `round(h) == d`.
    pub rounded_exact_rate: f64,
}

#[derive(Default)]
struct Acc {
    n: u64,
    sum_h: f64,
    sum_abs: f64,
    sum_sq: f64,
    over: u64,
    max_over: f64,
    exact: u64,
}

impl Acc {
    fn add(&mut self, h: f64, d: f64) {
        let e = h - d;
        self.n += 1;
        self.sum_h += h;
        self.sum_abs += e.abs();
        self.sum_sq += e * e;
        self.over += (e > 1e-6) as u64;
        self.max_over = self.max_over.max(e);
        self.exact += (h.max(0.0).round() == d) as u64;
    }

    fn stats(&self, depth: Option<u8>) -> ErrorStats {
        let n = self.n.max(1) as f64;
        ErrorStats {
            depth,
            count: self.n,
            mean_estimate: self.sum_h / n,
            mae: self.sum_abs / n,
            rmse: (self.sum_sq / n).sqrt(),
            overestimate_rate: self.over as f64 / n,
            max_overestimate: if self.n == 0 { 0.0 } else { self.max_over },
            rounded_exact_rate: self.exact as f64 / n,
        }
    }
}

/// Heuristic error by true depth plus an aggregate row.
#[derive(Debug, Clone, Serialize)]
pub struct HeuristicReport {
    pub heuristic: String,
    pub admissible_claimed: bool,
    pub by_depth: Vec<ErrorStats>,
    pub overall: ErrorStats,
    pub elapsed_s: f64,
    pub states_per_s: f64,
}

/// Evaluate `h` on every labelled state, `chunk` states per batched call.
pub fn heuristic_error<P: Puzzle + ?Sized, H: Heuristic<P> + ?Sized>(
    h: &H,
    labeled: &[(P::State, u8)],
    chunk: usize,
) -> HeuristicReport {
    let started = Instant::now();
    let mut per: Vec<Acc> = Vec::new();
    let mut all = Acc::default();
    let mut states = Vec::with_capacity(chunk);
    let mut out = Vec::with_capacity(chunk);
    for part in labeled.chunks(chunk.max(1)) {
        states.clear();
        states.extend(part.iter().map(|(s, _)| s.clone()));
        h.estimate_batch_real(&states, &mut out);
        for (&est, (_, d)) in out.iter().zip(part) {
            let d = *d as usize;
            if per.len() <= d {
                per.resize_with(d + 1, Acc::default);
            }
            per[d].add(est as f64, d as f64);
            all.add(est as f64, d as f64);
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    HeuristicReport {
        heuristic: h.name(),
        admissible_claimed: h.is_admissible(),
        by_depth: per
            .iter()
            .enumerate()
            .filter(|(_, a)| a.n > 0)
            .map(|(d, a)| a.stats(Some(d as u8)))
            .collect(),
        overall: all.stats(None),
        elapsed_s: elapsed,
        states_per_s: labeled.len() as f64 / elapsed.max(1e-9),
    }
}

/// Solver results for one true depth (or all, if `depth` is `None`).
#[derive(Debug, Clone, Default, Serialize)]
pub struct SolveStats {
    pub depth: Option<u8>,
    pub count: u64,
    pub solved: u64,
    pub solved_rate: f64,
    /// Over solved instances.
    pub mean_length: f64,
    pub mean_optimal_length: f64,
    /// Fraction of solved instances whose solution is optimal.
    pub optimal_rate: f64,
    pub max_extra_moves: u32,
    pub mean_expanded: f64,
    pub median_expanded: u64,
    pub max_expanded: u64,
    pub mean_generated: f64,
    pub mean_iterations: f64,
    pub mean_time_s: f64,
}

/// The outcome of solving every state of a test set.
#[derive(Debug, Clone, Serialize)]
pub struct SolveReport {
    pub solver: String,
    pub by_depth: Vec<SolveStats>,
    pub overall: SolveStats,
}

struct Case {
    depth: u8,
    length: Option<u32>,
    expanded: u64,
    generated: u64,
    iterations: u32,
    time_s: f64,
}

fn summarise(depth: Option<u8>, cases: &[&Case]) -> SolveStats {
    let n = cases.len() as u64;
    let solved: Vec<&&Case> = cases.iter().filter(|c| c.length.is_some()).collect();
    let s = solved.len().max(1) as f64;
    let mut expanded: Vec<u64> = cases.iter().map(|c| c.expanded).collect();
    expanded.sort_unstable();
    let nf = n.max(1) as f64;
    SolveStats {
        depth,
        count: n,
        solved: solved.len() as u64,
        solved_rate: solved.len() as f64 / nf,
        mean_length: solved.iter().map(|c| c.length.unwrap() as f64).sum::<f64>() / s,
        mean_optimal_length: solved.iter().map(|c| c.depth as f64).sum::<f64>() / s,
        optimal_rate: solved
            .iter()
            .filter(|c| c.length == Some(c.depth as u32))
            .count() as f64
            / s,
        max_extra_moves: solved
            .iter()
            .map(|c| c.length.unwrap().saturating_sub(c.depth as u32))
            .max()
            .unwrap_or(0),
        mean_expanded: cases.iter().map(|c| c.expanded as f64).sum::<f64>() / nf,
        median_expanded: expanded.get(expanded.len() / 2).copied().unwrap_or(0),
        max_expanded: expanded.last().copied().unwrap_or(0),
        mean_generated: cases.iter().map(|c| c.generated as f64).sum::<f64>() / nf,
        mean_iterations: cases.iter().map(|c| c.iterations as f64).sum::<f64>() / nf,
        mean_time_s: cases.iter().map(|c| c.time_s).sum::<f64>() / nf,
    }
}

/// Solve every labelled state with `solver`, checking each solution really solves it.
///
/// # Panics
///
/// If a returned solution does not reach the goal, or is shorter than the known optimum
/// (either would be a solver bug, not a quality problem).
pub fn solve_all<P: Puzzle + ?Sized, S: Solver<P> + ?Sized>(
    puzzle: &P,
    solver: &S,
    labeled: &[(P::State, u8)],
    limits: &SearchLimits,
) -> SolveReport {
    let mut cases = Vec::with_capacity(labeled.len());
    for (state, depth) in labeled {
        let started = Instant::now();
        let case = match solver.solve(puzzle, state, limits) {
            Ok(sol) => {
                assert!(
                    puzzle.is_solved(&puzzle.apply_all(state, &sol.moves)),
                    "{} returned a non-solution",
                    solver.name()
                );
                assert!(
                    sol.moves.len() >= *depth as usize,
                    "{} beat the optimum",
                    solver.name()
                );
                Case {
                    depth: *depth,
                    length: Some(sol.moves.len() as u32),
                    expanded: sol.stats.expanded,
                    generated: sol.stats.generated,
                    iterations: sol.stats.iterations,
                    time_s: started.elapsed().as_secs_f64(),
                }
            }
            Err(e) => {
                let stats = match e {
                    gpss_core::SearchError::Exhausted(s) => s,
                    gpss_core::SearchError::LimitReached { stats, .. } => stats,
                };
                Case {
                    depth: *depth,
                    length: None,
                    expanded: stats.expanded,
                    generated: stats.generated,
                    iterations: stats.iterations,
                    time_s: started.elapsed().as_secs_f64(),
                }
            }
        };
        cases.push(case);
    }
    let max = cases.iter().map(|c| c.depth).max().unwrap_or(0);
    let by_depth = (0..=max)
        .filter_map(|d| {
            let at: Vec<&Case> = cases.iter().filter(|c| c.depth == d).collect();
            (!at.is_empty()).then(|| summarise(Some(d), &at))
        })
        .collect();
    let all: Vec<&Case> = cases.iter().collect();
    SolveReport {
        solver: solver.name(),
        by_depth,
        overall: summarise(None, &all),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpss_core::search::BatchWeightedAStar;
    use gpss_core::PdbHeuristic;
    use gpss_puzzles::cube2::{Cube2, Cube2Heuristic, FullCoord, KNOWN_HTM_DISTRIBUTION};
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn exact_evaluation_of_cube2_heuristics() {
        let p = Cube2::new();
        let all = enumerate_distances(&p, 4_000_000).unwrap();
        let mut hist = [0u64; 12];
        for (_, d) in &all {
            hist[*d as usize] += 1;
        }
        assert_eq!(hist, KNOWN_HTM_DISTRIBUTION);
        assert!(enumerate_distances(&p, 1000).is_none());

        let mut rng = StdRng::seed_from_u64(1);
        let sample = stratified_sample(&all, 50, &mut rng);
        assert_eq!(sample.len(), 1 + 9 + 50 * 10); // only 1 goal and 9 depth-1 states exist

        let perfect = PdbHeuristic::new(&p, FullCoord);
        let r = heuristic_error::<Cube2, _>(&perfect, &sample, 64);
        assert_eq!(r.overall.mae, 0.0);
        assert_eq!(r.overall.rounded_exact_rate, 1.0);
        let dual = Cube2Heuristic::Dual.build(&p);
        let r = heuristic_error::<Cube2, _>(dual.as_ref(), &sample, 64);
        assert!(r.overall.mae > 0.5);
        assert_eq!(r.overall.overestimate_rate, 0.0, "dual is admissible");

        let solver = BatchWeightedAStar::new(&dual, 1.0, 20);
        let rep = solve_all(&p, &solver, &sample[..120], &SearchLimits::default());
        assert_eq!(rep.overall.solved_rate, 1.0);
        assert_eq!(rep.overall.optimal_rate, 1.0);
    }
}
