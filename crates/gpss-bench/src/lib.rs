//! # gpss-bench
//!
//! The benchmark and regression suite of the geometry puzzle solver suite. It answers two
//! different questions with two different tools (ADR 0007):
//!
//! * **Did the search change?** [`check`] runs every conformance-corpus case through every
//!   registered solver, checks the answer (valid, optimal when claimed, invalid input
//!   rejected) and compares the machine-independent search counters (`expanded`,
//!   `generated`, `peak_stored`, `iterations`, length) *exactly* against committed
//!   baselines in `crates/gpss-bench/baselines/`. Deterministic and cheap, so CI runs it
//!   (`gpss-bench check`, and the `baselines` integration test under `cargo test`).
//! * **How fast and how big is it?** [`run`] additionally times each case in-process
//!   (repeated, batched for sub-microsecond solves), aggregates latency per optimal-depth
//!   bucket (short / mid / top-of-range) and per depth, derives per-node cost, measures
//!   peak heap per solve with a counting allocator ([`alloc`]), solver setup cost, pattern
//!   database build times and primitive costs. Machine-dependent, so it runs on a quiet
//!   machine and the Markdown report is committed under `docs/benchmarks/`.
//!
//! ## Plugging in a puzzle or solver
//!
//! The suite is generic over [`gpss_core::Puzzle`] and [`gpss_core::Solver`]. A puzzle
//! registers a [`Workload`] (puzzle value, corpus name, depth [`BucketScheme`], and a
//! function from scramble text to start state) and a list of [`SolverSpec`]s with
//! [`Registry::add`]; see [`targets::cube2`] for the complete 2x2x2 registration and
//! `docs/benchmarking.md` for the walkthrough. Nothing outside [`targets`] names a
//! concrete puzzle.
//!
//! ```no_run
//! use gpss_bench::{check, Registry};
//!
//! let mut registry = gpss_bench::default_registry();
//! let result = check::check_puzzle(&mut registry, "cube2", false).unwrap();
//! print!("{}", result.summary());
//! assert!(result.ok());
//! ```

pub mod alloc;
pub mod check;
pub mod corpus;
pub mod report;
pub mod run;
pub mod stats;
pub mod target;
pub mod targets;

pub use corpus::{Bucket, BucketScheme, Case, Corpus, Expect};
pub use target::{
    BenchTarget, BenchWorkload, CaseCheck, CaseResult, Counters, Registry, Scope, SolverSpec,
    TimingBudget, Workload,
};

/// A registry with every puzzle and solver in [`targets`].
pub fn default_registry() -> Registry {
    let mut r = Registry::new();
    targets::register_all(&mut r);
    r
}
