//! `gpss-bench` - benchmark and regression driver. See `docs/benchmarking.md`.
//!
//! | command | where | what |
//! |---------|-------|------|
//! | `list`  | anywhere | registered puzzles and targets |
//! | `check` | CI + local | correctness + exact counter baselines (`--update` to accept changes) |
//! | `run`   | quiet local machine | latency / memory / tables report (`--markdown`, `--json`) |
//!
//! Exit codes: 0 ok, 1 correctness failure or counter difference, 2 usage or I/O error.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use clap::{Parser, Subcommand};
use gpss_bench::alloc::CountingAlloc;
use gpss_bench::{check, report, run, TimingBudget};

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc;

#[derive(Parser)]
#[command(
    name = "gpss-bench",
    version,
    about = "GPSS benchmark and regression suite"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(clap::Args)]
struct Filter {
    /// Only this puzzle (corpus name, e.g. cube2).
    #[arg(long)]
    puzzle: Option<String>,
    /// Only targets whose id contains this string (e.g. idastar).
    #[arg(long)]
    target: Option<String>,
}

#[derive(Subcommand)]
enum Command {
    /// List registered puzzles and targets.
    List,
    /// Check correctness and compare search counters with the committed baselines.
    Check {
        #[command(flatten)]
        filter: Filter,
        /// Rewrite the baselines with the current counters (after reviewing the diff).
        #[arg(long)]
        update: bool,
        /// Also write the full results as JSON.
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Full benchmark run: correctness, latency, memory, setup, tables, primitives.
    Run {
        #[command(flatten)]
        filter: Filter,
        /// Wall-time budget per case in milliseconds.
        #[arg(long, default_value_t = 200)]
        budget_ms: u64,
        /// Minimum duration of one timing sample in microseconds (fast solves are batched).
        #[arg(long, default_value_t = 50)]
        min_sample_us: u64,
        /// Maximum timing samples per case.
        #[arg(long, default_value_t = 101)]
        max_samples: u32,
        /// Time spent rebuilding each pattern database, in milliseconds.
        #[arg(long, default_value_t = 2000)]
        table_budget_ms: u64,
        /// Write the Markdown report here (e.g. docs/benchmarks/cube2-gpss-bench.md).
        #[arg(long)]
        markdown: Option<PathBuf>,
        /// Write the raw results as JSON here.
        #[arg(long)]
        json: Option<PathBuf>,
        /// Report title.
        #[arg(long, default_value = "gpss-bench report")]
        title: String,
    },
}

fn write(path: &PathBuf, contents: &str) -> Result<(), ExitCode> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        let _ = std::fs::create_dir_all(dir);
    }
    std::fs::write(path, contents).map_err(|e| {
        eprintln!("error: {}: {e}", path.display());
        ExitCode::from(2)
    })
}

fn registry(filter: &Filter) -> gpss_bench::Registry {
    let mut r = gpss_bench::default_registry();
    r.retain(filter.puzzle.as_deref(), filter.target.as_deref());
    r
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    if cfg!(debug_assertions) {
        eprintln!("warning: debug build; use `cargo run --release -p gpss-bench` for real numbers");
    }
    match cli.command {
        Command::List => {
            let r = gpss_bench::default_registry();
            for w in r.workloads() {
                println!("{}  ({})", w.puzzle(), report::describe_scheme(w.buckets()));
                for t in r.targets().iter().filter(|t| t.puzzle() == w.puzzle()) {
                    println!(
                        "  {}{}",
                        t.id(),
                        if t.deterministic() {
                            ""
                        } else {
                            "  (nondeterministic)"
                        }
                    );
                }
            }
            ExitCode::SUCCESS
        }
        Command::Check {
            filter,
            update,
            json,
        } => {
            let mut r = registry(&filter);
            let mut ok = true;
            let mut all = Vec::new();
            for puzzle in r.puzzles() {
                match check::check_puzzle(&mut r, &puzzle, update) {
                    Ok(res) => {
                        print!("{}", res.summary());
                        ok &= res.ok();
                        all.push(res);
                    }
                    Err(e) => {
                        eprintln!("error: {e}");
                        return ExitCode::from(2);
                    }
                }
            }
            if let Some(path) = json {
                if let Err(code) = write(&path, &serde_json::to_string_pretty(&all).unwrap()) {
                    return code;
                }
            }
            if r.targets().is_empty() {
                eprintln!("error: no targets match the filter");
                return ExitCode::from(2);
            }
            if ok {
                println!("all checks passed");
                ExitCode::SUCCESS
            } else {
                println!(
                    "checks FAILED. If the counter changes are intended, review them and run \
                     `cargo run --release -p gpss-bench -- check --update`."
                );
                ExitCode::from(1)
            }
        }
        Command::Run {
            filter,
            budget_ms,
            min_sample_us,
            max_samples,
            table_budget_ms,
            markdown,
            json,
            title,
        } => {
            let mut r = registry(&filter);
            if r.targets().is_empty() {
                eprintln!("error: no targets match the filter");
                return ExitCode::from(2);
            }
            let options = run::RunOptions {
                timing: TimingBudget {
                    per_case: Duration::from_millis(budget_ms),
                    min_sample: Duration::from_micros(min_sample_us),
                    max_samples: max_samples.max(1),
                },
                table_budget: Duration::from_millis(table_budget_ms),
                ..run::RunOptions::default()
            };
            let report = match run::run(&mut r, &options) {
                Ok(rep) => rep,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
            if report.failure_count() > 0 {
                // Don't overwrite a committed report with numbers from an incorrect build.
                eprintln!(
                    "{} correctness failures; no report written (run `gpss-bench check` for \
                     details)",
                    report.failure_count()
                );
                return ExitCode::from(1);
            }
            let command = format!(
                "cargo run --release -p gpss-bench -- {}",
                std::env::args()
                    .skip(1)
                    .map(|a| shell_quote(&a))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
            let md = report::markdown(&report, &title, &command);
            match &markdown {
                Some(path) => {
                    if let Err(code) = write(path, &md) {
                        return code;
                    }
                    eprintln!("wrote {}", path.display());
                }
                None => print!("{md}"),
            }
            if let Some(path) = json {
                if let Err(code) = write(&path, &serde_json::to_string_pretty(&report).unwrap()) {
                    return code;
                }
                eprintln!("wrote {}", path.display());
            }
            ExitCode::SUCCESS
        }
    }
}

/// Quote an argument for a POSIX shell if it contains anything but safe characters.
fn shell_quote(arg: &str) -> String {
    let safe = |c: char| c.is_ascii_alphanumeric() || "-_./=:,+".contains(c);
    if !arg.is_empty() && arg.chars().all(safe) {
        arg.to_string()
    } else {
        format!("'{}'", arg.replace('\'', "'\\''"))
    }
}
