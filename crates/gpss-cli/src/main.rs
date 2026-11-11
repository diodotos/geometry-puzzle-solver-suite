//! `gpss` - command-line front end for the geometry puzzle solver suite.
//!
//! Output is human-readable by default; `--format json` emits one JSON object on stdout
//! (the format consumed by `conformance/harness.py`). Exit codes:
//!
//! | code | meaning                                           |
//! |------|---------------------------------------------------|
//! | 0    | success                                           |
//! | 2    | invalid input (bad arguments or move notation)    |
//! | 3    | no solution within the given limits               |

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand, ValueEnum};
use gpss_core::encode::Encodable;
use gpss_core::pdb::PatternDatabase;
use gpss_core::puzzle::format_moves;
use gpss_core::search::{
    AStar, BatchWeightedAStar, Bfs, IdaStar, SearchError, SearchLimits, SearchStats, Solution,
    Solver,
};
use gpss_core::{Heuristic, Puzzle, ZeroHeuristic};
use gpss_learn::data::{self, ScrambleConfig};
use gpss_learn::eval;
use gpss_puzzles::cube::{parse_sequence, random_scramble};
use gpss_puzzles::cube2::{Cube2, Cube2Heuristic, FullCoord, OrientationCoord, PermutationCoord};
use gpss_puzzles::cube3::{Cube3, KorfHeuristic, TwoPhase};
use gpss_puzzles::cuboid::{CornerHeuristic, Cuboid, Dims};
use rand::{rngs::StdRng, SeedableRng};
use serde_json::json;

#[derive(Parser)]
#[command(name = "gpss", version, about = "Geometry Puzzle Solver Suite")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Format {
    Text,
    Json,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum PuzzleKind {
    /// 2x2x2 (fixed-frame, 9 moves).
    Cube2,
    /// 3x3x3 (18 face turns).
    Cube3,
    /// Generic `AxBxC` cube or cuboid (sticker model, every layer turn costs 1).
    Cuboid(Dims),
}

fn parse_puzzle(s: &str) -> Result<PuzzleKind, String> {
    match s {
        "cube2" => Ok(PuzzleKind::Cube2),
        "cube3" => Ok(PuzzleKind::Cube3),
        _ => s
            .parse()
            .map(PuzzleKind::Cuboid)
            .map_err(|e| format!("{e}; or one of cube2, cube3")),
    }
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SolverKind {
    /// Breadth-first search (optimal, no heuristic).
    Bfs,
    /// A* (optimal with an admissible heuristic; weighted if --weight > 1).
    Astar,
    /// Iterative-deepening A* (optimal, memory O(depth)).
    Idastar,
    /// Kociemba two-phase (cube3 only; fast, near-optimal).
    Twophase,
    /// Batch weighted A* (DeepCubeA): f = lambda * g + h, N nodes expanded per batch.
    Bwas,
}

#[derive(Subcommand)]
enum Command {
    /// Solve a scrambled puzzle.
    Solve {
        /// `cube2`, `cube3`, or `AxBxC` for a generic cuboid (width x depth x height),
        /// e.g. `4x4x4` or `3x3x2`.
        #[arg(long, value_parser = parse_puzzle, default_value = "cube2")]
        puzzle: PuzzleKind,
        /// Scramble in WCA notation, e.g. "R U2 F'". Empty means already solved.
        #[arg(long, allow_hyphen_values = true)]
        scramble: String,
        #[arg(long, value_enum, default_value = "idastar")]
        solver: SolverKind,
        /// cube2: zero | orientation | permutation | dual (default) | perfect.
        /// cube3 (generic solvers): zero (default) | korf. Ignored by bfs and twophase, and
        /// when --weights is given.
        #[arg(long)]
        heuristic: Option<String>,
        /// twophase: stop at the first solution with at most this many moves.
        #[arg(long, default_value_t = 20)]
        target_length: u32,
        #[command(flatten)]
        learned: LearnedArgs,
        /// Weight on h for weighted A* (1.0 = plain A*).
        #[arg(long, default_value_t = 1.0)]
        weight: f64,
        #[arg(long)]
        max_depth: Option<u32>,
        #[arg(long)]
        max_expanded: Option<u64>,
        /// Timeout in seconds.
        #[arg(long)]
        timeout: Option<f64>,
        #[arg(long, value_enum, default_value = "text")]
        format: Format,
    },
    /// Evaluate a heuristic exactly (every state) and a solver on a stratified test set,
    /// against the perfect distance table. Prints a JSON report.
    Evaluate {
        #[arg(long, value_parser = parse_puzzle, default_value = "cube2")]
        puzzle: PuzzleKind,
        /// Classical heuristic to evaluate when --weights is not given.
        #[arg(long, default_value = "dual")]
        heuristic: String,
        #[command(flatten)]
        learned: LearnedArgs,
        /// Solve this many states of every true distance (0 skips the solver part).
        #[arg(long, default_value_t = 100)]
        per_depth: usize,
        /// Also solve this many uniformly random states.
        #[arg(long, default_value_t = 1000)]
        random: usize,
        /// Skip the whole-state-space heuristic evaluation.
        #[arg(long)]
        no_exhaustive: bool,
        #[arg(long, default_value_t = 1)]
        seed: u64,
        /// Per-solve node limit; a solve that hits it counts as unsolved.
        #[arg(long)]
        max_expanded: Option<u64>,
        /// cube3 only: label --per-depth states for each scramble length 1..=this with
        /// their exact distance (Korf IDA*, slow past ~13).
        #[arg(long, default_value_t = 10)]
        optimal_depth: u8,
    },
    /// Write DAVI training data (scrambled states, their successors, goal masks; all
    /// encoded) for `training/`. See `crates/gpss-learn/src/data.rs` for the format.
    ScrambleData {
        #[arg(long, value_parser = parse_puzzle, default_value = "cube2")]
        puzzle: PuzzleKind,
        #[arg(long, default_value_t = 10_000)]
        count: usize,
        #[arg(long, default_value_t = 1)]
        min_scramble: u8,
        #[arg(long, default_value_t = 14)]
        max_scramble: u8,
        #[arg(long)]
        seed: Option<u64>,
        /// Output file, or `-` for stdout.
        #[arg(long, default_value = "-")]
        out: String,
    },
    /// Train a learned heuristic: runs `python3 -m gpss_train <ARGS>` from `training/`
    /// (see docs/training.md), e.g. `gpss train train --puzzle cube2 --run-dir training/runs/x`.
    Train {
        /// Arguments passed to `python3 -m gpss_train`.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Print a random scramble.
    Scramble {
        #[arg(long, default_value_t = 11)]
        length: usize,
        #[arg(long)]
        seed: Option<u64>,
    },
    /// Build a pattern database and print its distance histogram.
    Pdb {
        /// orientation | permutation | perfect
        #[arg(long, default_value = "perfect")]
        coordinate: String,
        #[arg(long, value_enum, default_value = "text")]
        format: Format,
    },
}

/// Options for learned (DeepCubeA-style) heuristics and batch weighted A*.
#[derive(clap::Args, Clone)]
struct LearnedArgs {
    /// Weights manifest (`.json`) or network (`.onnx`) of a learned heuristic, e.g.
    /// weights/cube2/cube2-davi-v1.json. Needs a build with `--features learn`.
    #[arg(long)]
    weights: Option<PathBuf>,
    /// BWAS: weight on the path cost g in f = lambda * g + h (1 = A* order, 0 = greedy).
    #[arg(long, default_value_t = 0.8)]
    lambda: f32,
    /// BWAS: nodes expanded per batch (and per heuristic call).
    #[arg(long, default_value_t = 10)]
    batch_size: usize,
    /// ONNX Runtime threads (default: all cores).
    #[arg(long)]
    threads: Option<usize>,
}

impl LearnedArgs {
    fn validate(&self) -> Result<(), String> {
        if !(self.lambda.is_finite() && self.lambda >= 0.0) {
            return Err("--lambda must be a finite number >= 0".into());
        }
        if self.batch_size == 0 {
            return Err("--batch-size must be >= 1".into());
        }
        Ok(())
    }
}

/// A heuristic plus the name to report for it.
type BuiltHeuristic<P = Cube2> = (Box<dyn Heuristic<P>>, String);

/// Build the classical heuristic `name`, or the learned one from `--weights`.
fn build_heuristic(
    puzzle: &Cube2,
    name: &str,
    learned: &LearnedArgs,
) -> Result<BuiltHeuristic, (u8, &'static str, String)> {
    if let Some(weights) = &learned.weights {
        return load_learned(puzzle.clone(), weights, learned.threads);
    }
    let Some(kind) = Cube2Heuristic::from_name(name) else {
        return Err((2, "invalid_argument", format!("unknown heuristic {name:?}")));
    };
    Ok((kind.build(puzzle), kind.name().to_string()))
}

/// A learned heuristic for any `Encodable` puzzle; the manifest's encoding must match.
#[cfg(feature = "learn")]
fn load_learned<P: Encodable + 'static>(
    puzzle: P,
    weights: &Path,
    threads: Option<usize>,
) -> Result<BuiltHeuristic<P>, (u8, &'static str, String)> {
    match gpss_learn::OnnxHeuristic::load(puzzle, weights, threads) {
        Ok(h) => {
            let name = Heuristic::<P>::name(&h);
            Ok((Box::new(h), name))
        }
        Err(e) => Err((2, "invalid_weights", e.to_string())),
    }
}

#[cfg(not(feature = "learn"))]
fn load_learned<P: Encodable + 'static>(
    _puzzle: P,
    weights: &Path,
    _threads: Option<usize>,
) -> Result<BuiltHeuristic<P>, (u8, &'static str, String)> {
    // Validate the manifest anyway so mistakes surface before a rebuild.
    if let Err(e) = gpss_learn::Manifest::load(weights) {
        return Err((2, "invalid_weights", e.to_string()));
    }
    Err((
        2,
        "unsupported",
        gpss_learn::LearnError::OnnxDisabled.to_string(),
    ))
}

fn stats_json(stats: &SearchStats) -> serde_json::Value {
    json!({
        "expanded": stats.expanded,
        "generated": stats.generated,
        "peak_stored": stats.peak_stored,
        "iterations": stats.iterations,
        "elapsed_s": stats.elapsed.as_secs_f64(),
    })
}

fn fail(format: Format, code: u8, kind: &str, message: &str) -> ExitCode {
    if format == Format::Json {
        println!("{}", json!({ "error": kind, "message": message }));
    }
    eprintln!("error: {message}");
    ExitCode::from(code)
}

#[allow(clippy::too_many_arguments)]
struct Request<'a> {
    scramble: &'a str,
    solver: SolverKind,
    heuristic: Option<&'a str>,
    learned: &'a LearnedArgs,
    weight: f64,
    target_length: u32,
    limits: SearchLimits,
    format: Format,
}

fn solve(puzzle: PuzzleKind, req: Request) -> ExitCode {
    let format = req.format;
    let moves = match puzzle {
        PuzzleKind::Cuboid(_) => Vec::new(),
        _ => match parse_sequence(req.scramble) {
            Ok(m) => m,
            Err(e) => return fail(format, 2, "invalid_scramble", &e.to_string()),
        },
    };
    if let Err(msg) = req.learned.validate() {
        return fail(format, 2, "invalid_argument", &msg);
    }
    if req.learned.weights.is_some() && matches!(puzzle, PuzzleKind::Cuboid(_)) {
        return fail(
            format,
            2,
            "invalid_argument",
            "--weights: learned heuristics need --puzzle cube2 or cube3 (cuboids have no encoding yet)",
        );
    }
    if !(req.weight.is_finite() && req.weight >= 1.0) {
        return fail(
            format,
            2,
            "invalid_argument",
            "--weight must be a finite number >= 1",
        );
    }
    let unknown_heuristic = |h: &str| {
        fail(
            format,
            2,
            "invalid_argument",
            &format!("unknown heuristic {h:?}"),
        )
    };
    let setup = Instant::now();
    match puzzle {
        PuzzleKind::Cube2 => {
            let cube = Cube2::new();
            let (solver, h_name): (Box<dyn Solver<Cube2>>, Option<String>) = match req.solver {
                SolverKind::Bfs => (Box::new(Bfs), None),
                SolverKind::Twophase => {
                    return fail(
                        format,
                        2,
                        "invalid_argument",
                        "--solver twophase requires --puzzle cube3",
                    )
                }
                kind => {
                    let h = req.heuristic.unwrap_or("dual");
                    let (heuristic, name) = match build_heuristic(&cube, h, req.learned) {
                        Ok(built) => built,
                        Err((code, kind, msg)) => return fail(format, code, kind, &msg),
                    };
                    let solver: Box<dyn Solver<Cube2>> = match kind {
                        SolverKind::Astar => Box::new(AStar::weighted(heuristic, req.weight)),
                        SolverKind::Idastar => Box::new(IdaStar::new(heuristic)),
                        _ => Box::new(BatchWeightedAStar::new(
                            heuristic,
                            req.learned.lambda,
                            req.learned.batch_size,
                        )),
                    };
                    (solver, Some(name))
                }
            };
            let setup_s = setup.elapsed().as_secs_f64();
            let result = cube.solve_scramble(&moves, solver.as_ref(), &req.limits);
            report(
                "cube2",
                &solver.name(),
                h_name.as_deref(),
                &moves,
                setup_s,
                result,
                format,
            )
        }
        PuzzleKind::Cube3 => {
            let learned = match (&req.learned.weights, req.solver) {
                (Some(_), SolverKind::Bfs | SolverKind::Twophase) => {
                    return fail(
                        format,
                        2,
                        "invalid_argument",
                        "--weights needs --solver astar, idastar or bwas",
                    )
                }
                (Some(weights), _) => {
                    match load_learned(Cube3::new(), weights, req.learned.threads) {
                        Ok(built) => Some(built),
                        Err((code, kind, msg)) => return fail(format, code, kind, &msg),
                    }
                }
                (None, _) => None,
            };
            let (heuristic, h): BuiltHeuristic<Cube3> = match learned {
                Some(built) => built,
                None => {
                    let h = req.heuristic.unwrap_or("korf");
                    let built: Box<dyn Heuristic<Cube3>> = match (req.solver, h) {
                        (SolverKind::Astar | SolverKind::Idastar | SolverKind::Bwas, "korf") => {
                            Box::new(KorfHeuristic::new())
                        }
                        (_, "zero") | (SolverKind::Bfs | SolverKind::Twophase, "korf") => {
                            Box::new(ZeroHeuristic)
                        }
                        _ => return unknown_heuristic(h),
                    };
                    (built, h.to_string())
                }
            };
            let cube = Cube3::new();
            let (solver, h_name): (Box<dyn Solver<Cube3>>, Option<String>) = match req.solver {
                SolverKind::Bfs => (Box::new(Bfs), None),
                SolverKind::Astar => (Box::new(AStar::weighted(heuristic, req.weight)), Some(h)),
                SolverKind::Idastar => (Box::new(IdaStar::new(heuristic)), Some(h)),
                SolverKind::Bwas => (
                    Box::new(BatchWeightedAStar::new(
                        heuristic,
                        req.learned.lambda,
                        req.learned.batch_size,
                    )),
                    Some(h),
                ),
                SolverKind::Twophase => {
                    TwoPhase::warm_up();
                    (Box::new(TwoPhase::new(req.target_length)), None)
                }
            };
            let setup_s = setup.elapsed().as_secs_f64();
            let result = cube.solve_scramble(&moves, solver.as_ref(), &req.limits);
            report(
                "cube3",
                &solver.name(),
                h_name.as_deref(),
                &moves,
                setup_s,
                result,
                format,
            )
        }
        PuzzleKind::Cuboid(dims) => {
            let cuboid = Cuboid::new(dims);
            let moves = match cuboid.parse_sequence(req.scramble) {
                Ok(m) => m,
                Err(e) => return fail(format, 2, "invalid_scramble", &e.to_string()),
            };
            let h = req.heuristic.unwrap_or(match req.solver {
                SolverKind::Bfs => "zero",
                _ => "corners",
            });
            let heuristic: Box<dyn Heuristic<Cuboid>> = match h {
                "corners" if req.solver != SolverKind::Bfs => {
                    Box::new(CornerHeuristic::new(&cuboid))
                }
                "zero" | "corners" => Box::new(ZeroHeuristic),
                _ => return unknown_heuristic(h),
            };
            let (solver, h_name): (Box<dyn Solver<Cuboid>>, Option<&str>) = match req.solver {
                SolverKind::Bfs => (Box::new(Bfs), None),
                SolverKind::Astar => (Box::new(AStar::weighted(heuristic, req.weight)), Some(h)),
                SolverKind::Idastar => (Box::new(IdaStar::new(heuristic)), Some(h)),
                SolverKind::Bwas => (
                    Box::new(BatchWeightedAStar::new(
                        heuristic,
                        req.learned.lambda,
                        req.learned.batch_size,
                    )),
                    Some(h),
                ),
                SolverKind::Twophase => {
                    return fail(
                        format,
                        2,
                        "invalid_argument",
                        "--solver twophase requires --puzzle cube3",
                    )
                }
            };
            let setup_s = setup.elapsed().as_secs_f64();
            let result = cuboid.solve_scramble(&moves, solver.as_ref(), &req.limits);
            report(
                &dims.to_string(),
                &solver.name(),
                h_name,
                &moves,
                setup_s,
                result,
                format,
            )
        }
    }
}

fn report<M: std::fmt::Display>(
    puzzle: &str,
    solver: &str,
    h_name: Option<&str>,
    moves: &[M],
    setup_s: f64,
    result: Result<Solution<M>, SearchError>,
    format: Format,
) -> ExitCode {
    match result {
        Ok(sol) => {
            let text = format_moves(&sol.moves);
            match format {
                Format::Json => println!(
                    "{}",
                    json!({
                        "puzzle": puzzle,
                        "solver": solver,
                        "heuristic": h_name,
                        "scramble": format_moves(moves),
                        "solution": text,
                        "length": sol.moves.len(),
                        "optimal": sol.optimal,
                        "setup_s": setup_s,
                        "stats": stats_json(&sol.stats),
                    })
                ),
                Format::Text => {
                    println!("Solution: {text}");
                    println!(
                        "Length: {}{}",
                        sol.moves.len(),
                        if sol.optimal { " (optimal)" } else { "" }
                    );
                    println!("Solver: {solver}");
                    println!(
                        "Expanded: {}  Generated: {}  Peak stored: {}  Iterations: {}",
                        sol.stats.expanded,
                        sol.stats.generated,
                        sol.stats.peak_stored,
                        sol.stats.iterations
                    );
                    println!(
                        "Setup: {setup_s:.3}s  Search: {:.6}s",
                        sol.stats.elapsed.as_secs_f64()
                    );
                }
            }
            ExitCode::SUCCESS
        }
        Err(err) => {
            let (kind, stats) = match &err {
                SearchError::Exhausted(s) => ("exhausted", s),
                SearchError::LimitReached { stats, .. } => ("limit_reached", stats),
            };
            if format == Format::Json {
                println!(
                    "{}",
                    json!({ "error": kind, "message": err.to_string(), "stats": stats_json(stats) })
                );
            }
            eprintln!("error: {err}");
            ExitCode::from(3)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn evaluate(
    heuristic: &str,
    learned: &LearnedArgs,
    per_depth: usize,
    random: usize,
    no_exhaustive: bool,
    seed: u64,
    max_expanded: Option<u64>,
) -> ExitCode {
    let format = Format::Json;
    if let Err(msg) = learned.validate() {
        return fail(format, 2, "invalid_argument", &msg);
    }
    let puzzle = Cube2::new();
    let (h, h_name) = match build_heuristic(&puzzle, heuristic, learned) {
        Ok(built) => built,
        Err((code, kind, msg)) => return fail(format, code, kind, &msg),
    };
    let started = Instant::now();
    let all = eval::enumerate_distances(&puzzle, gpss_puzzles::cube2::STATE_COUNT)
        .expect("the 2x2x2 fixed frame has exactly STATE_COUNT states");
    let enumerate_s = started.elapsed().as_secs_f64();

    let heuristic_report =
        (!no_exhaustive).then(|| eval::heuristic_error::<Cube2, _>(h.as_ref(), &all, 8192));

    let mut rng = StdRng::seed_from_u64(seed);
    let stratified = eval::stratified_sample(&all, per_depth, &mut rng);
    let uniform: Vec<_> = (0..random)
        .map(|_| {
            use rand::Rng;
            all[rng.gen_range(0..all.len())]
        })
        .collect();
    let limits = SearchLimits {
        max_expanded,
        ..SearchLimits::default()
    };
    let solver = BatchWeightedAStar::new(h.as_ref(), learned.lambda, learned.batch_size);
    let solve_stratified =
        (!stratified.is_empty()).then(|| eval::solve_all(&puzzle, &solver, &stratified, &limits));
    let solve_uniform =
        (!uniform.is_empty()).then(|| eval::solve_all(&puzzle, &solver, &uniform, &limits));

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "puzzle": "cube2",
            "heuristic": h_name,
            "weights": learned.weights,
            "solver": Solver::<Cube2>::name(&solver),
            "lambda": learned.lambda,
            "batch_size": learned.batch_size,
            "seed": seed,
            "max_expanded": max_expanded,
            "states_enumerated": all.len(),
            "enumerate_s": enumerate_s,
            "heuristic_error": heuristic_report,
            "solve_stratified": { "per_depth": per_depth, "report": solve_stratified },
            "solve_uniform": { "count": random, "report": solve_uniform },
        }))
        .expect("report serialises")
    );
    ExitCode::SUCCESS
}

/// The 3x3x3 cannot be enumerated, so evaluation samples instead: for each scramble length
/// `1..=optimal_depth`, `per_depth` scrambles labelled with their exact distance by Korf
/// IDA* (heuristic error + solution quality), plus `random` states 100 random moves deep
/// whose distance is unknown (solve rate, length, effort only).
#[allow(clippy::too_many_arguments)]
fn evaluate_cube3(
    heuristic: &str,
    learned: &LearnedArgs,
    per_depth: usize,
    random: usize,
    seed: u64,
    max_expanded: Option<u64>,
    optimal_depth: u8,
) -> ExitCode {
    let format = Format::Json;
    if let Err(msg) = learned.validate() {
        return fail(format, 2, "invalid_argument", &msg);
    }
    let puzzle = Cube3::new();
    let (h, h_name): BuiltHeuristic<Cube3> = match &learned.weights {
        Some(weights) => match load_learned(Cube3::new(), weights, learned.threads) {
            Ok(built) => built,
            Err((code, kind, msg)) => return fail(format, code, kind, &msg),
        },
        None => match heuristic {
            "korf" | "dual" => (Box::new(KorfHeuristic::new()), "korf".into()),
            "zero" => (Box::new(ZeroHeuristic), "zero".into()),
            _ => {
                return fail(
                    format,
                    2,
                    "invalid_argument",
                    &format!("unknown heuristic {heuristic:?}"),
                )
            }
        },
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let started = Instant::now();
    let mut labeled = Vec::with_capacity(per_depth * optimal_depth as usize);
    // Built only when there is something to label: the Korf tables are 87 MB.
    let optimal = (per_depth > 0 && optimal_depth > 0).then(|| IdaStar::new(KorfHeuristic::new()));
    for len in 1..=optimal_depth as usize {
        let Some(optimal) = &optimal else { break };
        for _ in 0..per_depth {
            let state = puzzle.apply_all(&puzzle.solved(), &random_scramble(&mut rng, len));
            let sol = optimal
                .solve(&puzzle, &state, &SearchLimits::default())
                .expect("IDA* with an admissible heuristic always solves the 3x3x3");
            labeled.push((state, sol.moves.len() as u8));
        }
    }
    let label_s = started.elapsed().as_secs_f64();
    let heuristic_report = (!labeled.is_empty())
        .then(|| eval::heuristic_error::<Cube3, _>(h.as_ref(), &labeled, 8192));

    let limits = SearchLimits {
        max_expanded,
        ..SearchLimits::default()
    };
    let solver = BatchWeightedAStar::new(h.as_ref(), learned.lambda, learned.batch_size);
    let solve_labeled =
        (!labeled.is_empty()).then(|| eval::solve_all(&puzzle, &solver, &labeled, &limits));
    let deep: Vec<_> = (0..random)
        .map(|_| {
            (
                puzzle.apply_all(&puzzle.solved(), &random_scramble(&mut rng, 100)),
                0u8,
            )
        })
        .collect();
    // Distance 0 is a placeholder (unknown); only solve rate, length and effort are meaningful.
    let solve_random = (!deep.is_empty()).then(|| {
        let r = eval::solve_all(&puzzle, &solver, &deep, &limits).overall;
        json!({
            "count": r.count,
            "solved_rate": r.solved_rate,
            "mean_length": r.mean_length,
            "max_length": r.max_extra_moves, // label 0, so extra moves == length
            "mean_expanded": r.mean_expanded,
            "max_expanded": r.max_expanded,
            "mean_time_s": r.mean_time_s,
        })
    });

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "puzzle": "cube3",
            "heuristic": h_name,
            "weights": learned.weights,
            "solver": Solver::<Cube3>::name(&solver),
            "lambda": learned.lambda,
            "batch_size": learned.batch_size,
            "seed": seed,
            "max_expanded": max_expanded,
            "optimal_depth": optimal_depth,
            "label_s": label_s,
            "heuristic_error": heuristic_report,
            "solve_labeled": { "per_depth": per_depth, "report": solve_labeled },
            "solve_random": { "count": random, "scramble_length": 100, "report": solve_random },
        }))
        .expect("report serialises")
    );
    ExitCode::SUCCESS
}

/// `evaluate` needs an `Encodable` puzzle; cuboids have no encoding yet.
fn learned_cube2_only(format: Format) -> ExitCode {
    fail(
        format,
        2,
        "invalid_argument",
        "only --puzzle cube2 or cube3 is supported here so far (it needs an `Encodable` puzzle)",
    )
}

fn scramble_data(
    puzzle: PuzzleKind,
    count: usize,
    min_scramble: u8,
    max_scramble: u8,
    seed: Option<u64>,
    out: &str,
) -> ExitCode {
    if min_scramble > max_scramble {
        return fail(
            Format::Text,
            2,
            "invalid_argument",
            "--min-scramble must be <= --max-scramble",
        );
    }
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    let cfg = ScrambleConfig {
        count,
        min_scramble,
        max_scramble,
    };
    let batch = match puzzle {
        PuzzleKind::Cube2 => data::generate(&Cube2::new(), &cfg, &mut rng),
        PuzzleKind::Cube3 => data::generate(&Cube3::new(), &cfg, &mut rng),
        PuzzleKind::Cuboid(_) => {
            return fail(
                Format::Text,
                2,
                "invalid_argument",
                "scramble-data supports --puzzle cube2 or cube3 (cuboids have no encoding yet)",
            )
        }
    };
    let result = if out == "-" {
        batch.write(std::io::BufWriter::new(std::io::stdout().lock()))
    } else {
        std::fs::File::create(out).and_then(|f| batch.write(std::io::BufWriter::new(f)))
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => fail(Format::Text, 1, "io", &format!("{out}: {e}")),
    }
}

/// Where the Python trainer lives: `$GPSS_TRAINING_DIR`, else the source checkout.
fn training_dir() -> PathBuf {
    std::env::var_os("GPSS_TRAINING_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../training"))
}

fn train(args: &[String]) -> ExitCode {
    let dir = training_dir();
    let python = std::env::var("GPSS_PYTHON").unwrap_or_else(|_| "python3".into());
    let mut cmd = std::process::Command::new(&python);
    cmd.arg("-m")
        .arg("gpss_train")
        .args(args)
        .env("PYTHONPATH", &dir);
    if let Ok(exe) = std::env::current_exe() {
        cmd.env("GPSS_BIN", exe);
    }
    match cmd.status() {
        Ok(status) => ExitCode::from(status.code().unwrap_or(1).clamp(0, 255) as u8),
        Err(e) => fail(
            Format::Text,
            2,
            "io",
            &format!(
                "could not run {python} -m gpss_train (training dir {}): {e}",
                dir.display()
            ),
        ),
    }
}

fn pdb(coordinate: &str, format: Format) -> ExitCode {
    let puzzle = Cube2::new();
    let start = Instant::now();
    let table = match coordinate {
        "orientation" => PatternDatabase::build(&puzzle, &OrientationCoord),
        "permutation" => PatternDatabase::build(&puzzle, &PermutationCoord),
        "perfect" => PatternDatabase::build(&puzzle, &FullCoord),
        other => {
            return fail(
                format,
                2,
                "invalid_argument",
                &format!("unknown coordinate {other:?}"),
            )
        }
    };
    let build_s = start.elapsed().as_secs_f64();
    let hist = table.histogram();
    match format {
        Format::Json => println!(
            "{}",
            json!({
                "coordinate": coordinate,
                "entries": table.len(),
                "memory_bytes": table.memory_bytes(),
                "build_s": build_s,
                "histogram": hist,
            })
        ),
        Format::Text => {
            println!(
                "{coordinate}: {} entries, {} bytes, built in {build_s:.3}s",
                table.len(),
                table.memory_bytes()
            );
            for (d, n) in hist.iter().enumerate() {
                println!("{d:>3} {n:>10}");
            }
        }
    }
    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    match Cli::parse().command {
        Command::Solve {
            puzzle,
            scramble,
            solver,
            heuristic,
            target_length,
            learned,
            weight,
            max_depth,
            max_expanded,
            timeout,
            format,
        } => {
            let timeout = match timeout.map(Duration::try_from_secs_f64).transpose() {
                Ok(t) => t,
                Err(_) => {
                    return fail(
                        format,
                        2,
                        "invalid_argument",
                        "--timeout must be a finite number of seconds >= 0",
                    )
                }
            };
            let limits = SearchLimits {
                max_depth,
                max_expanded,
                timeout,
            };
            solve(
                puzzle,
                Request {
                    scramble: &scramble,
                    solver,
                    heuristic: heuristic.as_deref(),
                    learned: &learned,
                    weight,
                    target_length,
                    limits,
                    format,
                },
            )
        }
        Command::Evaluate {
            puzzle: PuzzleKind::Cube3,
            heuristic,
            learned,
            per_depth,
            random,
            seed,
            max_expanded,
            optimal_depth,
            ..
        } => evaluate_cube3(
            &heuristic,
            &learned,
            per_depth,
            random,
            seed,
            max_expanded,
            optimal_depth,
        ),
        Command::Evaluate { puzzle, .. } if puzzle != PuzzleKind::Cube2 => {
            learned_cube2_only(Format::Json)
        }
        Command::Evaluate {
            heuristic,
            learned,
            per_depth,
            random,
            no_exhaustive,
            seed,
            max_expanded,
            ..
        } => evaluate(
            &heuristic,
            &learned,
            per_depth,
            random,
            no_exhaustive,
            seed,
            max_expanded,
        ),
        Command::ScrambleData {
            puzzle,
            count,
            min_scramble,
            max_scramble,
            seed,
            out,
            ..
        } => scramble_data(puzzle, count, min_scramble, max_scramble, seed, &out),
        Command::Train { args } => train(&args),
        Command::Scramble { length, seed } => {
            let mut rng = match seed {
                Some(s) => StdRng::seed_from_u64(s),
                None => StdRng::from_entropy(),
            };
            println!("{}", format_moves(&random_scramble(&mut rng, length)));
            ExitCode::SUCCESS
        }
        Command::Pdb { coordinate, format } => pdb(&coordinate, format),
    }
}
