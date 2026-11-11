//! CI gate: every registered puzzle passes correctness and matches its committed counter
//! baseline exactly. Run with `--release` (the BFS cases are slow in debug builds).
//!
//! On failure the printed summary lists each failing case and counter diff. If a counter
//! change is intended, accept it with
//! `cargo run --release -p gpss-bench -- check --update` and commit the baseline.

#[test]
fn counters_match_committed_baselines() {
    let mut registry = gpss_bench::default_registry();
    let mut failed = Vec::new();
    for puzzle in registry.puzzles() {
        let result = gpss_bench::check::check_puzzle(&mut registry, &puzzle, false)
            .unwrap_or_else(|e| panic!("{puzzle}: {e}"));
        println!("{}", result.summary());
        if !result.ok() {
            failed.push(puzzle);
        }
    }
    assert!(
        failed.is_empty(),
        "benchmark checks failed for {failed:?}; see the summary above"
    );
}
