use std::cmp::{Ordering, Reverse};
use std::collections::hash_map::Entry;
use std::collections::{BinaryHeap, HashMap};

use super::{reconstruct, Budget, SearchError, SearchLimits, Solution, Solver};
use crate::heuristic::Heuristic;
use crate::puzzle::Puzzle;

/// Batch weighted A* (BWAS), the search half of DeepCubeA [Ago19].
///
/// Nodes are ordered by `f = lambda * g + h`. Each iteration pops up to `batch_size`
/// lowest-`f` nodes, expands all of them, and evaluates the heuristic on every new child
/// in a single [`Heuristic::estimate_batch_real`] call, which is what makes a neural
/// network heuristic affordable (one inference call per iteration instead of per node).
///
/// * `lambda < 1` discounts path cost, so the search dives towards states the heuristic
///   likes; `lambda = 0` is greedy best-first search.
/// * Larger `batch_size` expands more nodes per iteration (more parallelism, usually
///   fewer iterations) at the price of expanding nodes plain A* would never touch.
///
/// The goal test happens when a node is *popped*, in priority order. Consequently, with
/// `lambda = 1` and an admissible heuristic BWAS returns optimal solutions for any
/// `batch_size` (every node still in the open list has `f` at least that of the goal),
/// and [`Solver::is_optimal`] says so. With a learned heuristic (never admissible) or
/// `lambda != 1` solutions are reported as not optimal.
///
/// Heuristic values are used as `f32`, so learned estimates keep their fractional part.
/// Negative estimates are clamped to 0 and NaN is treated as "infinitely far".
///
/// `SearchStats::iterations` counts batch iterations.
pub struct BatchWeightedAStar<H> {
    heuristic: H,
    lambda: f32,
    batch_size: usize,
}

impl<H> BatchWeightedAStar<H> {
    /// Create a BWAS solver.
    ///
    /// # Panics
    ///
    /// If `lambda` is negative or not finite, or `batch_size == 0`.
    pub fn new(heuristic: H, lambda: f32, batch_size: usize) -> Self {
        assert!(
            lambda.is_finite() && lambda >= 0.0,
            "lambda must be finite and >= 0"
        );
        assert!(batch_size >= 1, "batch_size must be >= 1");
        Self {
            heuristic,
            lambda,
            batch_size,
        }
    }

    pub fn lambda(&self) -> f32 {
        self.lambda
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn heuristic(&self) -> &H {
        &self.heuristic
    }
}

/// Open-list entry: smallest `f` first, ties towards larger `g`, then insertion order.
#[derive(Clone, Copy)]
struct Key {
    f: f32,
    g: u32,
    idx: u32,
}

impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Key {}
impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Key {
    /// "Greater" means "expand later", so the heap holds `Reverse<Key>`.
    fn cmp(&self, other: &Self) -> Ordering {
        self.f
            .total_cmp(&other.f)
            .then(other.g.cmp(&self.g))
            .then(self.idx.cmp(&other.idx))
    }
}

fn sanitize(h: f32) -> f32 {
    if h.is_nan() {
        f32::INFINITY
    } else {
        h.max(0.0)
    }
}

impl<P: Puzzle + ?Sized, H: Heuristic<P>> Solver<P> for BatchWeightedAStar<H> {
    fn name(&self) -> String {
        format!(
            "bwas({}, lambda={}, batch={})",
            self.heuristic.name(),
            self.lambda,
            self.batch_size
        )
    }

    fn is_optimal(&self) -> bool {
        self.lambda == 1.0 && self.heuristic.is_admissible()
    }

    fn solve(
        &self,
        puzzle: &P,
        start: &P::State,
        limits: &SearchLimits,
    ) -> Result<Solution<P::Move>, SearchError> {
        let mut budget = Budget::new(limits);
        budget.stats.iterations = 0;
        let max_depth = budget.max_depth();
        let lambda = self.lambda;

        let mut h_buf: Vec<f32> = Vec::new();
        self.heuristic
            .estimate_batch_real(std::slice::from_ref(start), &mut h_buf);
        let h0 = sanitize(h_buf.first().copied().unwrap_or(0.0));

        let mut states: Vec<P::State> = vec![start.clone()];
        let mut gs: Vec<u32> = vec![0];
        let mut links: Vec<(Option<P::Move>, u32)> = vec![(None, 0)];
        let mut best_g: HashMap<P::State, u32> = HashMap::from([(start.clone(), 0)]);
        let mut open = BinaryHeap::from([Reverse(Key {
            f: h0,
            g: 0,
            idx: 0,
        })]);

        let mut batch: Vec<u32> = Vec::with_capacity(self.batch_size);
        let mut new_children: Vec<u32> = Vec::new();
        let mut child_states: Vec<P::State> = Vec::new();

        while !open.is_empty() {
            budget.stats.iterations += 1;

            // 1. Pop up to `batch_size` live nodes, goal-testing each in priority order.
            batch.clear();
            while batch.len() < self.batch_size {
                let Some(Reverse(Key { g, idx, .. })) = open.pop() else {
                    break;
                };
                let state = &states[idx as usize];
                if best_g.get(state).is_some_and(|&b| g > b) {
                    continue; // stale entry: a shorter path to this state was found
                }
                if puzzle.is_solved(state) {
                    budget.note_stored(best_g.len() + open.len());
                    return Ok(Solution {
                        moves: reconstruct(&links, idx),
                        stats: budget.finish(),
                        optimal: Solver::<P>::is_optimal(self),
                    });
                }
                if g < max_depth {
                    batch.push(idx);
                }
            }

            // 2. Expand the whole batch.
            new_children.clear();
            for &idx in &batch {
                budget.expand()?;
                let parent = states[idx as usize].clone();
                let cg = gs[idx as usize] + 1;
                let prev = links[idx as usize].0;
                for &mv in puzzle.moves() {
                    if prev.is_some_and(|p| !puzzle.can_follow(p, mv)) {
                        continue;
                    }
                    let child = puzzle.apply(&parent, mv);
                    budget.stats.generated += 1;
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
                    new_children.push(states.len() as u32);
                    states.push(child);
                    gs.push(cg);
                    links.push((Some(mv), idx));
                }
            }

            // 3. One batched heuristic call for all new children.
            if !new_children.is_empty() {
                child_states.clear();
                child_states.extend(new_children.iter().map(|&i| states[i as usize].clone()));
                self.heuristic
                    .estimate_batch_real(&child_states, &mut h_buf);
                assert_eq!(
                    h_buf.len(),
                    child_states.len(),
                    "heuristic {} returned the wrong number of estimates",
                    self.heuristic.name()
                );
                for (&idx, &h) in new_children.iter().zip(&h_buf) {
                    let g = gs[idx as usize];
                    open.push(Reverse(Key {
                        f: lambda * g as f32 + sanitize(h),
                        g,
                        idx,
                    }));
                }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heuristic::ZeroHeuristic;
    use crate::search::Bfs;

    /// Sort a permutation of `0..N` by swapping adjacent entries. The distance of a
    /// state is exactly its number of inversions, which makes a perfect heuristic.
    struct Swaps;
    const N: usize = 6;

    impl Puzzle for Swaps {
        type State = [u8; N];
        type Move = SwapMove;
        fn name(&self) -> &str {
            "swaps"
        }
        fn solved(&self) -> [u8; N] {
            [0, 1, 2, 3, 4, 5]
        }
        fn moves(&self) -> &[SwapMove] {
            &[
                SwapMove(0),
                SwapMove(1),
                SwapMove(2),
                SwapMove(3),
                SwapMove(4),
            ]
        }
        fn apply(&self, s: &[u8; N], m: SwapMove) -> [u8; N] {
            let mut t = *s;
            t.swap(m.0, m.0 + 1);
            t
        }
        fn inverse(&self, m: SwapMove) -> SwapMove {
            m
        }
        fn can_follow(&self, prev: SwapMove, next: SwapMove) -> bool {
            prev != next
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    struct SwapMove(usize);
    impl std::fmt::Display for SwapMove {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "s{}", self.0)
        }
    }

    fn inversions(s: &[u8; N]) -> u32 {
        let mut n = 0;
        for i in 0..N {
            for j in i + 1..N {
                n += (s[i] > s[j]) as u32;
            }
        }
        n
    }

    struct Exact;
    impl Heuristic<Swaps> for Exact {
        fn name(&self) -> String {
            "exact".into()
        }
        fn estimate(&self, s: &[u8; N]) -> u32 {
            inversions(s)
        }
    }

    /// Inadmissible, real-valued: exact distance plus noise, served only in batches.
    struct Noisy;
    impl Heuristic<Swaps> for Noisy {
        fn name(&self) -> String {
            "noisy".into()
        }
        fn estimate(&self, _s: &[u8; N]) -> u32 {
            unreachable!("BWAS must use the batched real-valued interface")
        }
        fn estimate_batch_real(&self, states: &[[u8; N]], out: &mut Vec<f32>) {
            out.clear();
            out.extend(states.iter().map(|s| {
                let wobble = (s[0] as f32 * 0.37 + s[5] as f32 * 0.21) % 1.3;
                inversions(s) as f32 * 1.2 + wobble - 0.4
            }));
        }
        fn is_admissible(&self) -> bool {
            false
        }
    }

    fn all_states() -> Vec<[u8; N]> {
        let mut out = Vec::new();
        let mut a = [0u8, 1, 2, 3, 4, 5];
        permute(&mut a, 0, &mut out);
        out
    }

    fn permute(a: &mut [u8; N], k: usize, out: &mut Vec<[u8; N]>) {
        if k == N {
            out.push(*a);
            return;
        }
        for i in k..N {
            a.swap(k, i);
            permute(a, k + 1, out);
            a.swap(k, i);
        }
    }

    fn check_valid(start: &[u8; N], moves: &[SwapMove]) {
        assert_eq!(Swaps.apply_all(start, moves), Swaps.solved());
    }

    #[test]
    fn lambda_one_admissible_is_optimal_for_every_state_and_batch_size() {
        for batch in [1, 3, 16, 1000] {
            for h in [&ZeroHeuristic as &dyn Heuristic<Swaps>, &Exact] {
                let solver = BatchWeightedAStar::new(h, 1.0, batch);
                assert!(Solver::<Swaps>::is_optimal(&solver));
                for s in all_states() {
                    let sol = solver.solve(&Swaps, &s, &SearchLimits::default()).unwrap();
                    check_valid(&s, &sol.moves);
                    assert_eq!(
                        sol.moves.len() as u32,
                        inversions(&s),
                        "{s:?} batch={batch}"
                    );
                    assert!(sol.optimal);
                }
            }
        }
    }

    #[test]
    fn perfect_heuristic_with_batch_one_expands_only_the_path() {
        let solver = BatchWeightedAStar::new(Exact, 1.0, 1);
        let s = [5, 4, 3, 2, 1, 0];
        let sol = solver.solve(&Swaps, &s, &SearchLimits::default()).unwrap();
        assert_eq!(sol.moves.len(), 15);
        assert_eq!(sol.stats.expanded, 15);
        assert_eq!(sol.stats.iterations, 16);
    }

    #[test]
    fn inadmissible_real_valued_heuristic_finds_valid_suboptimal_flagged_solutions() {
        for lambda in [0.0, 0.5, 1.0] {
            let solver = BatchWeightedAStar::new(Noisy, lambda, 8);
            assert!(!Solver::<Swaps>::is_optimal(&solver));
            for s in all_states().into_iter().step_by(7) {
                let sol = solver.solve(&Swaps, &s, &SearchLimits::default()).unwrap();
                check_valid(&s, &sol.moves);
                assert!(sol.moves.len() as u32 >= inversions(&s));
                assert!(!sol.optimal);
            }
        }
    }

    #[test]
    fn larger_batches_use_fewer_iterations() {
        let s = [5, 4, 3, 2, 1, 0];
        let iters = |n| {
            BatchWeightedAStar::new(ZeroHeuristic, 1.0, n)
                .solve(&Swaps, &s, &SearchLimits::default())
                .unwrap()
                .stats
                .iterations
        };
        assert!(iters(64) < iters(1));
    }

    #[test]
    fn limits_are_respected() {
        let s = [5, 4, 3, 2, 1, 0];
        let solver = BatchWeightedAStar::new(ZeroHeuristic, 1.0, 4);
        let err = solver
            .solve(
                &Swaps,
                &s,
                &SearchLimits {
                    max_expanded: Some(10),
                    ..Default::default()
                },
            )
            .unwrap_err();
        match err {
            SearchError::LimitReached { stats, .. } => assert_eq!(stats.expanded, 10),
            other => panic!("unexpected {other:?}"),
        }
        let err = solver
            .solve(
                &Swaps,
                &s,
                &SearchLimits {
                    max_depth: Some(14),
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert!(matches!(err, SearchError::LimitReached { .. }));
    }

    #[test]
    fn solved_start_needs_no_expansion_and_agrees_with_bfs() {
        let solver = BatchWeightedAStar::new(ZeroHeuristic, 0.5, 4);
        let sol = solver
            .solve(&Swaps, &Swaps.solved(), &SearchLimits::default())
            .unwrap();
        assert!(sol.moves.is_empty());
        assert_eq!(sol.stats.expanded, 0);
        let s = [1, 0, 3, 2, 5, 4];
        let bfs = Bfs.solve(&Swaps, &s, &SearchLimits::default()).unwrap();
        let bwas = BatchWeightedAStar::new(Exact, 1.0, 2)
            .solve(&Swaps, &s, &SearchLimits::default())
            .unwrap();
        assert_eq!(bfs.moves.len(), bwas.moves.len());
    }

    #[test]
    #[should_panic(expected = "batch_size")]
    fn zero_batch_is_rejected() {
        let _ = BatchWeightedAStar::new(ZeroHeuristic, 1.0, 0);
    }
}
