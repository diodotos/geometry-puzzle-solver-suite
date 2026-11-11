//! The plug-in API works for a puzzle the suite has never heard of, and catches broken
//! solvers. This is the template for registering a new puzzle (see docs/benchmarking.md).

use gpss_bench::check::check_target;
use gpss_bench::corpus::{self, BucketScheme, Corpus};
use gpss_bench::{CaseResult, Registry, Scope, SolverSpec, Workload};
use gpss_core::search::{Bfs, IdaStar, SearchError, SearchLimits, Solution, Solver};
use gpss_core::{Puzzle, ZeroHeuristic};

/// A ring of `N` positions; moves step one position either way. Distance = ring distance.
struct Ring;
const N: u8 = 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Step {
    Plus,
    Minus,
}

impl std::fmt::Display for Step {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if *self == Step::Plus { "+" } else { "-" })
    }
}

impl Puzzle for Ring {
    type State = u8;
    type Move = Step;
    fn name(&self) -> &str {
        "ring"
    }
    fn solved(&self) -> u8 {
        0
    }
    fn moves(&self) -> &[Step] {
        &[Step::Plus, Step::Minus]
    }
    fn apply(&self, s: &u8, m: Step) -> u8 {
        match m {
            Step::Plus => (s + 1) % N,
            Step::Minus => (s + N - 1) % N,
        }
    }
    fn inverse(&self, m: Step) -> Step {
        if m == Step::Plus {
            Step::Minus
        } else {
            Step::Plus
        }
    }
}

fn start(ring: &Ring, scramble: &str) -> Result<u8, String> {
    scramble
        .split_whitespace()
        .try_fold(ring.solved(), |s, t| match t {
            "+" => Ok(ring.apply(&s, Step::Plus)),
            "-" => Ok(ring.apply(&s, Step::Minus)),
            other => Err(format!("bad token {other:?}")),
        })
}

/// Claims optimality but pads every solution with `+ -`.
struct Padded;
impl Solver<Ring> for Padded {
    fn name(&self) -> String {
        "padded-bfs".into()
    }
    fn is_optimal(&self) -> bool {
        true
    }
    fn solve(&self, p: &Ring, s: &u8, l: &SearchLimits) -> Result<Solution<Step>, SearchError> {
        let mut sol = Bfs.solve(p, s, l)?;
        sol.moves.extend([Step::Plus, Step::Minus]);
        Ok(sol)
    }
}

fn corpus() -> Corpus {
    let mut lines = vec![
        r#"{"id":"empty","puzzle":"ring","scramble":"","expect":"solve","optimal_length":0}"#
            .to_string(),
        r#"{"id":"bad","puzzle":"ring","scramble":"+ x","expect":"reject"}"#.to_string(),
    ];
    for k in 1..=7u32 {
        let d = k.min(N as u32 - k);
        lines.push(format!(
            r#"{{"id":"plus{k}","puzzle":"ring","scramble":"{}","expect":"solve","optimal_length":{d}}}"#,
            vec!["+"; k as usize].join(" ")
        ));
    }
    let text = lines.join("\n");
    Corpus {
        puzzle: "ring".into(),
        cases: corpus::parse(&text, "ring.jsonl".as_ref()).unwrap(),
        fingerprint: corpus::fingerprint(text.as_bytes()),
    }
}

#[test]
fn a_new_puzzle_plugs_in_and_broken_solvers_are_caught() {
    let mut registry = Registry::new();
    registry.add(
        Workload::new(Ring, "ring", BucketScheme::new(2, 4), start),
        vec![
            SolverSpec::new("bfs", |_: &Ring| Box::new(Bfs)),
            SolverSpec::new("idastar-zero", |_: &Ring| {
                Box::new(IdaStar::new(ZeroHeuristic))
            })
            .ci_max_depth(3),
            SolverSpec::new("padded", |_: &Ring| Box::new(Padded)),
        ],
    );
    assert_eq!(registry.puzzles(), vec!["ring".to_string()]);
    let corpus = corpus();
    let results: Vec<_> = registry
        .targets_mut()
        .iter_mut()
        .map(|t| check_target(t.as_mut(), &corpus, Scope::Ci))
        .collect();

    assert_eq!(results[0].target, "ring/bfs");
    assert_eq!(results[0].failures().count(), 0);
    assert_eq!(results[0].cases.len(), 9);
    assert!(matches!(results[0].cases[1].result, CaseResult::Rejected));

    // ci_max_depth(3) drops plus4..plus6 (depths 4, 5, 4) but keeps plus7 (depth 3).
    assert_eq!(results[1].failures().count(), 0);
    assert_eq!(results[1].cases.len(), 6);

    let failures: Vec<_> = results[2].failures().map(|c| c.id.as_str()).collect();
    assert_eq!(
        failures.len(),
        8,
        "every solvable case is two moves too long"
    );
    assert!(!failures.contains(&"bad"));
}
