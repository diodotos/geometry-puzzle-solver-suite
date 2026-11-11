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

use std::process::ExitCode;
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand, ValueEnum};
use gpss_core::pdb::PatternDatabase;
use gpss_core::puzzle::format_moves;
use gpss_core::search::{
    AStar, Bfs, IdaStar, SearchError, SearchLimits, SearchStats, Solution, Solver,
};
use gpss_core::{Heuristic, ZeroHeuristic};
use gpss_puzzles::cube::{parse_sequence, random_scramble};
use gpss_puzzles::cube2::{Cube2, Cube2Heuristic, FullCoord, OrientationCoord, PermutationCoord};
use gpss_puzzles::cube3::{Cube3, KorfHeuristic, TwoPhase};
use gpss_puzzles::cuboid::{Cuboid, Dims};
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
        /// cube3 (generic solvers): zero (default). Ignored by bfs and twophase.
        #[arg(long)]
        heuristic: Option<String>,
        /// twophase: stop at the first solution with at most this many moves.
        #[arg(long, default_value_t = 20)]
        target_length: u32,
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
            let h = req.heuristic.unwrap_or("dual");
            let Some(h_kind) = Cube2Heuristic::from_name(h) else {
                return unknown_heuristic(h);
            };
            let cube = Cube2::new();
            let (solver, h_name): (Box<dyn Solver<Cube2>>, Option<&str>) = match req.solver {
                SolverKind::Bfs => (Box::new(Bfs), None),
                SolverKind::Astar => (
                    Box::new(AStar::weighted(h_kind.build(&cube), req.weight)),
                    Some(h_kind.name()),
                ),
                SolverKind::Idastar => (
                    Box::new(IdaStar::new(h_kind.build(&cube))),
                    Some(h_kind.name()),
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
            let result = cube.solve_scramble(&moves, solver.as_ref(), &req.limits);
            report(
                "cube2",
                &solver.name(),
                h_name,
                &moves,
                setup_s,
                result,
                format,
            )
        }
        PuzzleKind::Cube3 => {
            let h = req.heuristic.unwrap_or("korf");
            let heuristic: Box<dyn Heuristic<Cube3>> = match (req.solver, h) {
                (SolverKind::Astar | SolverKind::Idastar, "korf") => Box::new(KorfHeuristic::new()),
                (_, "zero") | (SolverKind::Bfs | SolverKind::Twophase, "korf") => {
                    Box::new(ZeroHeuristic)
                }
                _ => return unknown_heuristic(h),
            };
            let cube = Cube3::new();
            let (solver, h_name): (Box<dyn Solver<Cube3>>, Option<&str>) = match req.solver {
                SolverKind::Bfs => (Box::new(Bfs), None),
                SolverKind::Astar => (Box::new(AStar::weighted(heuristic, req.weight)), Some(h)),
                SolverKind::Idastar => (Box::new(IdaStar::new(heuristic)), Some(h)),
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
                h_name,
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
            let h = req.heuristic.unwrap_or("zero");
            if h != "zero" {
                return unknown_heuristic(h);
            }
            let (solver, h_name): (Box<dyn Solver<Cuboid>>, Option<&str>) = match req.solver {
                SolverKind::Bfs => (Box::new(Bfs), None),
                SolverKind::Astar => (
                    Box::new(AStar::weighted(ZeroHeuristic, req.weight)),
                    Some(h),
                ),
                SolverKind::Idastar => (Box::new(IdaStar::new(ZeroHeuristic)), Some(h)),
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
                    weight,
                    target_length,
                    limits,
                    format,
                },
            )
        }
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
