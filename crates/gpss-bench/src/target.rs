//! The plug-in API: how a puzzle and its solvers register themselves with the suite.
//!
//! A puzzle contributes one [`Workload`] (the puzzle value, its corpus, its depth buckets,
//! a function turning corpus scrambles into search start states, and optionally the
//! heuristics and pattern databases worth micro-benchmarking) plus any number of
//! [`SolverSpec`]s. [`Registry::add`] erases the puzzle type, so the drivers in
//! [`crate::check`] and [`crate::run`] work with any `Puzzle` without knowing it.
//!
//! Everything here only uses `gpss-core` traits; puzzle-specific code lives in
//! [`crate::targets`].

use std::hint::black_box;
use std::rc::Rc;
use std::time::{Duration, Instant};

use gpss_core::pdb::PatternDatabase;
use gpss_core::search::{SearchLimits, Solver};
use gpss_core::{Heuristic, Puzzle};
use serde::{Deserialize, Serialize};

use crate::alloc::{self, HeapUsage};
use crate::corpus::{Bucket, BucketScheme, Case, Expect};
use crate::stats;

/// Turns a corpus scramble into the state a solver starts from (parse, apply, and any
/// puzzle-specific normalisation such as the 2x2x2 regrip). `Err` means "invalid input".
pub type StartFn<P> = Box<dyn Fn(&P, &str) -> Result<<P as Puzzle>::State, String>>;
/// Builds a solver; called once per target so its setup cost can be measured.
pub type SolverBuildFn<P> = Box<dyn Fn(&P) -> Box<dyn Solver<P>>>;
/// Builds a heuristic for micro-benchmarks.
pub type HeuristicBuildFn<P> = Box<dyn Fn(&P) -> Box<dyn Heuristic<P>>>;
/// Builds a pattern database for build-time benchmarks.
pub type TableBuildFn<P> = Box<dyn Fn(&P) -> PatternDatabase>;

/// Everything the suite needs to know about one puzzle.
pub struct Workload<P: Puzzle> {
    puzzle: P,
    corpus: String,
    buckets: BucketScheme,
    start: StartFn<P>,
    heuristics: Vec<(String, HeuristicBuildFn<P>)>,
    tables: Vec<(String, TableBuildFn<P>)>,
}

impl<P: Puzzle> Workload<P> {
    /// `corpus` names `conformance/corpus/<corpus>.jsonl` and is also the puzzle's id in
    /// reports and baselines.
    pub fn new(
        puzzle: P,
        corpus: impl Into<String>,
        buckets: BucketScheme,
        start: impl Fn(&P, &str) -> Result<P::State, String> + 'static,
    ) -> Self {
        Self {
            puzzle,
            corpus: corpus.into(),
            buckets,
            start: Box::new(start),
            heuristics: Vec::new(),
            tables: Vec::new(),
        }
    }

    /// Register a heuristic whose per-call cost should be reported.
    pub fn heuristic(
        mut self,
        name: impl Into<String>,
        build: impl Fn(&P) -> Box<dyn Heuristic<P>> + 'static,
    ) -> Self {
        self.heuristics.push((name.into(), Box::new(build)));
        self
    }

    /// Register a pattern database whose build time and size should be reported.
    pub fn table(
        mut self,
        name: impl Into<String>,
        build: impl Fn(&P) -> PatternDatabase + 'static,
    ) -> Self {
        self.tables.push((name.into(), Box::new(build)));
        self
    }
}

/// One solver configuration to benchmark against a [`Workload`].
pub struct SolverSpec<P: Puzzle> {
    id: String,
    build: SolverBuildFn<P>,
    ci_max_depth: Option<u32>,
    max_depth: Option<u32>,
    deterministic: bool,
    max_ratio: Option<f64>,
}

impl<P: Puzzle> SolverSpec<P> {
    /// `id` is short and stable (it keys the baseline file), e.g. `"idastar-dual"`.
    pub fn new(id: impl Into<String>, build: impl Fn(&P) -> Box<dyn Solver<P>> + 'static) -> Self {
        Self {
            id: id.into(),
            build: Box::new(build),
            ci_max_depth: None,
            max_depth: None,
            deterministic: true,
            max_ratio: None,
        }
    }

    /// In the CI counter check, skip cases whose optimal length exceeds `depth` (e.g. BFS,
    /// which is too slow at the top of the 2x2x2 range). Full runs still cover them.
    pub fn ci_max_depth(mut self, depth: u32) -> Self {
        self.ci_max_depth = Some(depth);
        self
    }

    /// Skip cases whose optimal length exceeds `depth`, or is unknown, in every scope (e.g.
    /// uninformed IDA* on the 3x3x3, which never finishes a 20-move random scramble).
    pub fn max_depth(mut self, depth: u32) -> Self {
        self.max_depth = Some(depth);
        self
    }

    /// The solver's counters vary run to run (threads, floating-point batching, wall-clock
    /// limits), so no counter baseline is kept; correctness is still checked.
    pub fn nondeterministic(mut self) -> Self {
        self.deterministic = false;
        self
    }

    /// For non-optimal solvers: fail a case if `length > ratio * optimal_length`
    /// (weighted A* with weight `w` guarantees `ratio = w`).
    pub fn max_ratio(mut self, ratio: f64) -> Self {
        self.max_ratio = Some(ratio);
        self
    }
}

/// Machine-independent work counters of one solve; the unit of the CI baselines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Counters {
    pub length: u32,
    pub expanded: u64,
    pub generated: u64,
    pub peak_stored: u64,
    pub iterations: u32,
}

/// Outcome of running one case once.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "status", rename_all = "lowercase")]
pub enum CaseResult {
    /// An invalid input was rejected, as expected.
    Rejected,
    /// A valid, correctly-sized solution was returned.
    Solved {
        counters: Counters,
        elapsed_s: f64,
        /// Peak heap of the solve call (only with [`alloc::CountingAlloc`] installed).
        peak_heap_bytes: Option<usize>,
    },
    /// Anything else: wrong length, invalid solution, crash-free error, ...
    Failed { reason: String },
}

/// One case's correctness check.
#[derive(Debug, Clone, Serialize)]
pub struct CaseCheck {
    pub id: String,
    pub bucket: Bucket,
    pub optimal_length: Option<u32>,
    pub result: CaseResult,
}

impl CaseCheck {
    pub fn passed(&self) -> bool {
        !matches!(self.result, CaseResult::Failed { .. })
    }
    pub fn counters(&self) -> Option<Counters> {
        match self.result {
            CaseResult::Solved { counters, .. } => Some(counters),
            _ => None,
        }
    }
}

/// Cost of building a target's solver (heuristic tables included).
#[derive(Debug, Clone, Serialize)]
pub struct SetupReport {
    pub target: String,
    pub solver: String,
    pub optimal: bool,
    pub build_s: f64,
    pub heap: Option<HeapUsage>,
}

/// Repeated in-process timing of one case.
#[derive(Debug, Clone, Serialize)]
pub struct CaseTiming {
    pub id: String,
    pub bucket: Bucket,
    pub optimal_length: Option<u32>,
    /// Number of timed samples; each sample may batch several solves.
    pub samples: u32,
    /// Solves per sample.
    pub batch: u32,
    pub median_s: f64,
    pub min_s: f64,
    pub expanded: u64,
    pub generated: u64,
}

/// How much time to spend timing each case.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct TimingBudget {
    /// Target wall time per case. A case slower than this is timed once (by its check run).
    pub per_case: Duration,
    /// Minimum duration of one sample; fast solves are batched to reach it, which keeps
    /// timer overhead (~20-50 ns per `Instant::now`) out of sub-microsecond results.
    pub min_sample: Duration,
    /// Upper bound on samples per case.
    pub max_samples: u32,
}

impl Default for TimingBudget {
    fn default() -> Self {
        Self {
            per_case: Duration::from_millis(200),
            min_sample: Duration::from_micros(50),
            max_samples: 101,
        }
    }
}

/// Which cases a check covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    /// Honour [`SolverSpec::ci_max_depth`]; the scope of committed baselines.
    Ci,
    /// Every case in the corpus.
    Full,
}

/// A type-erased (puzzle, solver) pair. Implemented generically for every `Puzzle` by
/// [`Registry::add`]; you should not need to implement it by hand.
pub trait BenchTarget {
    /// `"<puzzle>/<solver id>"`, e.g. `"cube2/idastar-dual"`.
    fn id(&self) -> &str;
    fn puzzle(&self) -> &str;
    fn deterministic(&self) -> bool;
    /// Whether `case` is in `scope` for this target.
    fn applies_to(&self, case: &Case, scope: Scope) -> bool;
    /// Build the solver (idempotent) and report what that cost.
    fn setup(&mut self) -> SetupReport;
    /// Run `case` once and check the answer. Calls [`BenchTarget::setup`] if needed.
    fn check(&mut self, case: &Case) -> CaseCheck;
    /// Time `case` repeatedly. `first` is the check run's elapsed time, reused as the only
    /// sample when it already exceeds the per-case budget. `None` for non-solve cases.
    fn time(&mut self, case: &Case, first: f64, budget: &TimingBudget) -> Option<CaseTiming>;
}

/// Pattern-database build cost.
#[derive(Debug, Clone, Serialize)]
pub struct TableReport {
    pub puzzle: String,
    pub table: String,
    pub entries: usize,
    pub bytes: usize,
    pub builds: u32,
    pub median_build_s: f64,
    pub min_build_s: f64,
    pub heap: Option<HeapUsage>,
}

/// Cost of a primitive operation (move application, one heuristic lookup).
#[derive(Debug, Clone, Serialize)]
pub struct PrimitiveReport {
    pub puzzle: String,
    pub op: String,
    pub ops: u64,
    pub ns_per_op: f64,
}

/// A type-erased [`Workload`].
pub trait BenchWorkload {
    fn puzzle(&self) -> &str;
    fn buckets(&self) -> BucketScheme;
    /// Build each registered table at least once and until `budget` is spent.
    fn tables(&self, budget: Duration) -> Vec<TableReport>;
    /// Time `apply` over all moves and each registered heuristic, on the start states of
    /// the solvable `cases`, for roughly `budget` each.
    fn primitives(&self, cases: &[Case], budget: Duration) -> Vec<PrimitiveReport>;
}

/// The set of registered workloads and targets.
#[derive(Default)]
pub struct Registry {
    workloads: Vec<Box<dyn BenchWorkload>>,
    targets: Vec<Box<dyn BenchTarget>>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a puzzle and the solvers to benchmark on it.
    pub fn add<P: Puzzle + 'static>(&mut self, workload: Workload<P>, solvers: Vec<SolverSpec<P>>) {
        let w = Rc::new(workload);
        for spec in solvers {
            let id = format!("{}/{}", w.corpus, spec.id);
            assert!(
                self.targets.iter().all(|t| t.id() != id),
                "duplicate bench target {id}"
            );
            self.targets.push(Box::new(Target {
                workload: Rc::clone(&w),
                spec,
                id,
                solver: None,
            }));
        }
        self.workloads.push(Box::new(WorkloadHandle(w)));
    }

    pub fn workloads(&self) -> &[Box<dyn BenchWorkload>] {
        &self.workloads
    }

    pub fn targets(&self) -> &[Box<dyn BenchTarget>] {
        &self.targets
    }

    pub fn targets_mut(&mut self) -> &mut [Box<dyn BenchTarget>] {
        &mut self.targets
    }

    /// Keep only targets whose puzzle equals `puzzle` (if given) and whose id contains
    /// `target` (if given); drop workloads left without targets.
    pub fn retain(&mut self, puzzle: Option<&str>, target: Option<&str>) {
        self.targets.retain(|t| {
            puzzle.map_or(true, |p| t.puzzle() == p) && target.map_or(true, |s| t.id().contains(s))
        });
        let targets = &self.targets;
        self.workloads
            .retain(|w| targets.iter().any(|t| t.puzzle() == w.puzzle()));
    }

    /// Distinct puzzles with at least one target, in registration order.
    pub fn puzzles(&self) -> Vec<String> {
        self.workloads
            .iter()
            .map(|w| w.puzzle().to_string())
            .collect()
    }
}

struct WorkloadHandle<P: Puzzle>(Rc<Workload<P>>);

struct Target<P: Puzzle> {
    workload: Rc<Workload<P>>,
    spec: SolverSpec<P>,
    id: String,
    solver: Option<Box<dyn Solver<P>>>,
}

impl<P: Puzzle> Target<P> {
    fn solver(&mut self) -> &dyn Solver<P> {
        if self.solver.is_none() {
            self.setup_inner();
        }
        self.solver.as_deref().unwrap()
    }

    fn setup_inner(&mut self) -> SetupReport {
        let t = Instant::now();
        let (solver, heap) = alloc::measure(|| (self.spec.build)(&self.workload.puzzle));
        let build_s = t.elapsed().as_secs_f64();
        let report = SetupReport {
            target: self.id.clone(),
            solver: solver.name(),
            optimal: solver.is_optimal(),
            build_s,
            heap,
        };
        self.solver = Some(solver);
        report
    }

    fn validate(
        &self,
        case: &Case,
        start: &P::State,
        solution: &gpss_core::Solution<P::Move>,
        optimal: bool,
    ) -> Result<(), String> {
        let w = &self.workload;
        let end = w.puzzle.apply_all(start, &solution.moves);
        if !w.puzzle.is_solved(&end) {
            return Err("returned moves do not solve the start state".into());
        }
        if solution.optimal != optimal {
            return Err(format!(
                "Solution::optimal = {} but Solver::is_optimal() = {optimal}",
                solution.optimal
            ));
        }
        let len = solution.moves.len() as u32;
        if let Some(opt) = case.optimal_length {
            if len < opt {
                return Err(format!(
                    "length {len} beats the proven optimum {opt} (solver or corpus bug)"
                ));
            }
            if optimal && len != opt {
                return Err(format!(
                    "optimal solver returned {len} moves, optimum is {opt}"
                ));
            }
            if let Some(r) = self.spec.max_ratio {
                let bound = (r * opt as f64).floor() as u32;
                if len > bound {
                    return Err(format!(
                        "length {len} exceeds {r} x optimum {opt} = {bound}"
                    ));
                }
            }
        }
        Ok(())
    }
}

fn counters_of(stats: &gpss_core::SearchStats, length: usize) -> Counters {
    Counters {
        length: length as u32,
        expanded: stats.expanded,
        generated: stats.generated,
        peak_stored: stats.peak_stored as u64,
        iterations: stats.iterations,
    }
}

impl<P: Puzzle> BenchTarget for Target<P> {
    fn id(&self) -> &str {
        &self.id
    }

    fn puzzle(&self) -> &str {
        &self.workload.corpus
    }

    fn deterministic(&self) -> bool {
        self.spec.deterministic
    }

    fn applies_to(&self, case: &Case, scope: Scope) -> bool {
        if case.expect == Expect::Unknown {
            return false;
        }
        let limit = match scope {
            Scope::Ci => match (self.spec.ci_max_depth, self.spec.max_depth) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            },
            Scope::Full => self.spec.max_depth,
        };
        match (limit, case.optimal_length) {
            (Some(max), Some(d)) => d <= max,
            (Some(_), None) => case.expect == Expect::Reject,
            (None, _) => true,
        }
    }

    fn setup(&mut self) -> SetupReport {
        self.setup_inner()
    }

    fn check(&mut self, case: &Case) -> CaseCheck {
        let bucket = self.workload.buckets.bucket(case);
        let mk = |result| CaseCheck {
            id: case.id.clone(),
            bucket,
            optimal_length: case.optimal_length,
            result,
        };
        let fail = |reason: String| mk(CaseResult::Failed { reason });
        let start = (self.workload.start)(&self.workload.puzzle, &case.scramble);
        let start = match (case.expect, start) {
            (Expect::Reject, Err(_)) => return mk(CaseResult::Rejected),
            (Expect::Reject, Ok(_)) => {
                return fail("accepted an input the corpus marks invalid".into())
            }
            (Expect::Unknown, _) => return fail("unknown expectation".into()),
            (Expect::Solve, Err(e)) => return fail(format!("rejected a valid scramble: {e}")),
            (Expect::Solve, Ok(s)) => s,
        };
        let workload = Rc::clone(&self.workload);
        let solver = self.solver();
        let optimal = solver.is_optimal();
        let t = Instant::now();
        let (outcome, heap) =
            alloc::measure(|| solver.solve(&workload.puzzle, &start, &SearchLimits::default()));
        let elapsed_s = t.elapsed().as_secs_f64();
        match outcome {
            Err(e) => fail(format!("no solution: {e}")),
            Ok(sol) => match self.validate(case, &start, &sol, optimal) {
                Err(reason) => fail(reason),
                Ok(()) => mk(CaseResult::Solved {
                    counters: counters_of(&sol.stats, sol.moves.len()),
                    elapsed_s,
                    peak_heap_bytes: heap.map(|h| h.peak_bytes),
                }),
            },
        }
    }

    fn time(&mut self, case: &Case, first: f64, budget: &TimingBudget) -> Option<CaseTiming> {
        if case.expect != Expect::Solve {
            return None;
        }
        let workload = Rc::clone(&self.workload);
        let start = (workload.start)(&workload.puzzle, &case.scramble).ok()?;
        let bucket = workload.buckets.bucket(case);
        let solver = self.solver();
        let limits = SearchLimits::default();
        let solve = || {
            solver
                .solve(black_box(&workload.puzzle), black_box(&start), &limits)
                .ok()
        };
        let sol = solve()?;
        let (expanded, generated) = (sol.stats.expanded, sol.stats.generated);
        let mut timing = CaseTiming {
            id: case.id.clone(),
            bucket,
            optimal_length: case.optimal_length,
            samples: 1,
            batch: 1,
            median_s: first,
            min_s: first,
            expanded,
            generated,
        };
        let per_case = budget.per_case.as_secs_f64();
        if first >= per_case {
            return Some(timing);
        }
        // Calibrate on a warm run, then batch fast solves up to `min_sample`.
        let t = Instant::now();
        black_box(solve());
        let one = t.elapsed().as_secs_f64().max(1e-9);
        let batch = (budget.min_sample.as_secs_f64() / one)
            .ceil()
            .clamp(1.0, 1e6) as u32;
        let samples = (per_case / (one * batch as f64))
            .floor()
            .clamp(1.0, budget.max_samples as f64) as u32;
        let mut times = Vec::with_capacity(samples as usize);
        for _ in 0..samples {
            let t = Instant::now();
            for _ in 0..batch {
                black_box(solve());
            }
            times.push(t.elapsed().as_secs_f64() / batch as f64);
        }
        let times = stats::sorted(times);
        timing.samples = samples;
        timing.batch = batch;
        timing.median_s = stats::median(&times).unwrap();
        timing.min_s = times[0];
        Some(timing)
    }
}

/// Run `f` repeatedly (in batches) for about `budget`; return (calls, ns per call).
fn time_op(budget: Duration, mut f: impl FnMut() -> u64) -> (u64, f64) {
    let t = Instant::now();
    let mut ops = 0;
    loop {
        ops += f();
        if t.elapsed() >= budget || ops >= 1 << 40 {
            break;
        }
    }
    (ops, t.elapsed().as_nanos() as f64 / ops.max(1) as f64)
}

impl<P: Puzzle> BenchWorkload for WorkloadHandle<P> {
    fn puzzle(&self) -> &str {
        &self.0.corpus
    }

    fn buckets(&self) -> BucketScheme {
        self.0.buckets
    }

    fn tables(&self, budget: Duration) -> Vec<TableReport> {
        let w = &self.0;
        w.tables
            .iter()
            .map(|(name, build)| {
                let t = Instant::now();
                let (pdb, heap) = alloc::measure(|| build(&w.puzzle));
                let mut times = vec![t.elapsed().as_secs_f64()];
                let started = Instant::now();
                while started.elapsed() < budget && times.len() < 15 {
                    let t = Instant::now();
                    black_box(build(&w.puzzle));
                    times.push(t.elapsed().as_secs_f64());
                }
                let times = stats::sorted(times);
                TableReport {
                    puzzle: w.corpus.clone(),
                    table: name.clone(),
                    entries: pdb.len(),
                    bytes: pdb.memory_bytes(),
                    builds: times.len() as u32,
                    median_build_s: stats::median(&times).unwrap(),
                    min_build_s: times[0],
                    heap,
                }
            })
            .collect()
    }

    fn primitives(&self, cases: &[Case], budget: Duration) -> Vec<PrimitiveReport> {
        let w = &self.0;
        let states: Vec<P::State> = cases
            .iter()
            .filter(|c| c.expect == Expect::Solve)
            .filter_map(|c| (w.start)(&w.puzzle, &c.scramble).ok())
            .collect();
        if states.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::new();
        let moves = w.puzzle.moves();
        let (ops, ns) = time_op(budget, || {
            for s in &states {
                for &m in moves {
                    black_box(w.puzzle.apply(black_box(s), m));
                }
            }
            (states.len() * moves.len()) as u64
        });
        out.push(PrimitiveReport {
            puzzle: w.corpus.clone(),
            op: "apply".into(),
            ops,
            ns_per_op: ns,
        });
        for (name, build) in &w.heuristics {
            let h = build(&w.puzzle);
            let (ops, ns) = time_op(budget, || {
                for s in &states {
                    black_box(h.estimate(black_box(s)));
                }
                states.len() as u64
            });
            out.push(PrimitiveReport {
                puzzle: w.corpus.clone(),
                op: format!("estimate:{name}"),
                ops,
                ns_per_op: ns,
            });
        }
        out
    }
}
