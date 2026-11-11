//! The CI regression check: correctness on every corpus case plus exact, machine-
//! independent search counters compared against committed baselines.
//!
//! Search counters (`expanded`, `generated`, `peak_stored`, `iterations`, solution length)
//! are a pure function of the code and the input for deterministic solvers, so they can be
//! compared *exactly* on any machine, including noisy CI runners. A change in them means
//! the search did different work: a regression if it went up, an improvement if it went
//! down, and either way something a reviewer should see. Intentional changes are accepted
//! by re-running with `--update` and committing the new baseline (see
//! `docs/benchmarking.md`).

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::corpus::{self, Bucket, Corpus, CorpusError};
use crate::target::{BenchTarget, CaseCheck, Counters, Registry, Scope, SetupReport};

/// Committed counters for one puzzle: `targets[target id][case id]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Baseline {
    pub puzzle: String,
    /// [`corpus::fingerprint`] of the corpus the counters were recorded on.
    pub corpus_fingerprint: String,
    pub targets: BTreeMap<String, BTreeMap<String, Counters>>,
}

impl Baseline {
    /// Diff-friendly JSON: one case per line, keys sorted.
    pub fn to_json(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "{{");
        let _ = writeln!(
            s,
            "  \"puzzle\": {},",
            serde_json::to_string(&self.puzzle).unwrap()
        );
        let _ = writeln!(
            s,
            "  \"corpus_fingerprint\": {},",
            serde_json::to_string(&self.corpus_fingerprint).unwrap()
        );
        let _ = writeln!(s, "  \"targets\": {{");
        for (ti, (target, cases)) in self.targets.iter().enumerate() {
            let _ = writeln!(s, "    {}: {{", serde_json::to_string(target).unwrap());
            for (ci, (case, c)) in cases.iter().enumerate() {
                let comma = if ci + 1 < cases.len() { "," } else { "" };
                let _ = writeln!(
                    s,
                    "      {}: {}{comma}",
                    serde_json::to_string(case).unwrap(),
                    serde_json::to_string(c).unwrap()
                );
            }
            let comma = if ti + 1 < self.targets.len() { "," } else { "" };
            let _ = writeln!(s, "    }}{comma}");
        }
        let _ = writeln!(s, "  }}");
        let _ = writeln!(s, "}}");
        s
    }
}

/// Directory of committed baselines: `crates/gpss-bench/baselines/`.
pub fn baseline_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("baselines")
}

/// `baselines/<puzzle>.json`.
pub fn baseline_path(puzzle: &str) -> PathBuf {
    baseline_dir().join(format!("{puzzle}.json"))
}

/// Load a baseline; `Ok(None)` if the file does not exist.
pub fn load_baseline(path: &Path) -> Result<Option<Baseline>, String> {
    match std::fs::read_to_string(path) {
        Ok(text) => serde_json::from_str(&text)
            .map(Some)
            .map_err(|e| format!("{}: {e}", path.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("{}: {e}", path.display())),
    }
}

/// One difference between baseline and current counters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Diff {
    /// The target has no baseline at all.
    NoBaseline,
    /// A case is in the baseline but was not run (corpus or scope changed).
    Missing { case: String },
    /// A case was run but is not in the baseline.
    New { case: String, now: Counters },
    /// Counters differ.
    Changed {
        case: String,
        was: Counters,
        now: Counters,
    },
}

impl std::fmt::Display for Diff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let c = |c: &Counters| {
            format!(
                "len={} expanded={} generated={} peak_stored={} iterations={}",
                c.length, c.expanded, c.generated, c.peak_stored, c.iterations
            )
        };
        match self {
            Diff::NoBaseline => write!(f, "no committed baseline for this target"),
            Diff::Missing { case } => write!(f, "{case}: in baseline but not run"),
            Diff::New { case, now } => write!(f, "{case}: not in baseline ({})", c(now)),
            Diff::Changed { case, was, now } => {
                write!(f, "{case}: was {} | now {}", c(was), c(now))
            }
        }
    }
}

/// Compare a target's current counters with its baseline entry.
pub fn compare(
    baseline: Option<&BTreeMap<String, Counters>>,
    now: &BTreeMap<String, Counters>,
) -> Vec<Diff> {
    let Some(base) = baseline else {
        return vec![Diff::NoBaseline];
    };
    let mut diffs = Vec::new();
    for (case, was) in base {
        match now.get(case) {
            None => diffs.push(Diff::Missing { case: case.clone() }),
            Some(n) if n != was => diffs.push(Diff::Changed {
                case: case.clone(),
                was: *was,
                now: *n,
            }),
            Some(_) => {}
        }
    }
    for (case, n) in now {
        if !base.contains_key(case) {
            diffs.push(Diff::New {
                case: case.clone(),
                now: *n,
            });
        }
    }
    diffs
}

/// Per-bucket aggregate of counters, for human-readable summaries.
#[derive(Debug, Clone, Serialize)]
pub struct BucketCounters {
    pub bucket: Bucket,
    pub cases: usize,
    pub mean_length: f64,
    pub mean_expanded: f64,
    pub max_expanded: u64,
    pub mean_generated: f64,
    pub max_peak_stored: u64,
}

/// Aggregate the solved cases of `checks` per bucket (buckets with no solved case omitted).
pub fn bucket_counters(checks: &[CaseCheck]) -> Vec<BucketCounters> {
    Bucket::ALL
        .iter()
        .filter_map(|&b| {
            let cs: Vec<Counters> = checks
                .iter()
                .filter(|c| c.bucket == b)
                .filter_map(CaseCheck::counters)
                .collect();
            if cs.is_empty() {
                return None;
            }
            let n = cs.len() as f64;
            Some(BucketCounters {
                bucket: b,
                cases: cs.len(),
                mean_length: cs.iter().map(|c| c.length as f64).sum::<f64>() / n,
                mean_expanded: cs.iter().map(|c| c.expanded as f64).sum::<f64>() / n,
                max_expanded: cs.iter().map(|c| c.expanded).max().unwrap(),
                mean_generated: cs.iter().map(|c| c.generated as f64).sum::<f64>() / n,
                max_peak_stored: cs.iter().map(|c| c.peak_stored).max().unwrap(),
            })
        })
        .collect()
}

/// Correctness results of one target over a corpus.
#[derive(Debug, Clone, Serialize)]
pub struct TargetChecks {
    pub target: String,
    pub puzzle: String,
    pub deterministic: bool,
    pub setup: SetupReport,
    pub cases: Vec<CaseCheck>,
}

impl TargetChecks {
    pub fn failures(&self) -> impl Iterator<Item = &CaseCheck> {
        self.cases.iter().filter(|c| !c.passed())
    }

    /// Counters of solved cases, keyed by case id.
    pub fn counters(&self) -> BTreeMap<String, Counters> {
        self.cases
            .iter()
            .filter_map(|c| c.counters().map(|k| (c.id.clone(), k)))
            .collect()
    }
}

/// Set up `target` and check every corpus case in `scope`.
pub fn check_target(target: &mut dyn BenchTarget, corpus: &Corpus, scope: Scope) -> TargetChecks {
    let setup = target.setup();
    let mut cases = Vec::new();
    for c in &corpus.cases {
        if target.applies_to(c, scope) {
            cases.push(target.check(c));
        }
    }
    TargetChecks {
        target: target.id().to_string(),
        puzzle: target.puzzle().to_string(),
        deterministic: target.deterministic(),
        setup,
        cases,
    }
}

/// Result of [`check_puzzle`].
#[derive(Debug, Clone, Serialize)]
pub struct PuzzleCheck {
    pub puzzle: String,
    pub corpus_fingerprint: String,
    pub baseline_fingerprint: Option<String>,
    pub targets: Vec<TargetChecks>,
    /// Counter differences per deterministic target (empty vec = matches).
    pub diffs: BTreeMap<String, Vec<Diff>>,
    /// Whether the baseline file was rewritten (`update = true`).
    pub updated: bool,
}

impl PuzzleCheck {
    pub fn failure_count(&self) -> usize {
        self.targets.iter().map(|t| t.failures().count()).sum()
    }

    pub fn diff_count(&self) -> usize {
        self.diffs.values().map(Vec::len).sum()
    }

    /// Correct and (unless just updated) identical to the baseline.
    pub fn ok(&self) -> bool {
        self.failure_count() == 0 && (self.updated || self.diff_count() == 0)
    }

    /// A plain-text summary suitable for CI logs.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(
            s,
            "== {} (corpus {}) ==",
            self.puzzle, self.corpus_fingerprint
        );
        if let Some(b) = &self.baseline_fingerprint {
            if *b != self.corpus_fingerprint {
                let _ = writeln!(
                    s,
                    "NOTE: baseline was recorded on corpus {b}; the corpus changed, so counter \
                     diffs are expected. Re-run with --update after reviewing."
                );
            }
        }
        for t in &self.targets {
            let failed = t.failures().count();
            let diffs = self.diffs.get(&t.target).map_or(0, Vec::len);
            let status = match (failed, diffs, t.deterministic) {
                (0, _, false) => "ok (counters not tracked: nondeterministic)".to_string(),
                (0, 0, true) => "ok".to_string(),
                (0, d, true) if self.updated => format!("ok, baseline updated ({d} changes)"),
                (0, d, true) => format!("COUNTERS DIFFER ({d})"),
                (f, _, _) => format!("FAILED ({f} cases)"),
            };
            let _ = writeln!(
                s,
                "{:<28} {:<26} {:>4} cases  {status}",
                t.target,
                t.setup.solver,
                t.cases.len()
            );
            for c in t.failures() {
                if let crate::target::CaseResult::Failed { reason } = &c.result {
                    let _ = writeln!(s, "    FAIL {}: {reason}", c.id);
                }
            }
            if !self.updated {
                for d in self.diffs.get(&t.target).into_iter().flatten().take(20) {
                    let _ = writeln!(s, "    DIFF {d}");
                }
                if diffs > 20 {
                    let _ = writeln!(s, "    ... {} more", diffs - 20);
                }
            }
            for b in bucket_counters(&t.cases) {
                let _ = writeln!(
                    s,
                    "    {:<8} n={:<3} len={:>5.2} expanded mean={:>12.1} max={:>10} generated mean={:>12.1} peak_stored max={:>9}",
                    b.bucket.name(),
                    b.cases,
                    b.mean_length,
                    b.mean_expanded,
                    b.max_expanded,
                    b.mean_generated,
                    b.max_peak_stored
                );
            }
        }
        s
    }
}

/// Errors that prevent a check from running at all.
#[derive(Debug, thiserror::Error)]
pub enum CheckError {
    #[error(transparent)]
    Corpus(#[from] CorpusError),
    #[error("baseline: {0}")]
    Baseline(String),
}

/// Check every target of `puzzle` in `registry` (CI scope) against
/// `baselines/<puzzle>.json`. With `update`, rewrite the baseline for the targets that ran
/// (other targets already in the file are kept).
pub fn check_puzzle(
    registry: &mut Registry,
    puzzle: &str,
    update: bool,
) -> Result<PuzzleCheck, CheckError> {
    check_puzzle_at(registry, puzzle, &baseline_path(puzzle), update)
}

/// [`check_puzzle`] with an explicit baseline path.
pub fn check_puzzle_at(
    registry: &mut Registry,
    puzzle: &str,
    path: &Path,
    update: bool,
) -> Result<PuzzleCheck, CheckError> {
    let corpus = corpus::load(puzzle)?;
    let baseline = load_baseline(path).map_err(CheckError::Baseline)?;
    let mut targets = Vec::new();
    for t in registry.targets_mut().iter_mut() {
        if t.puzzle() == puzzle {
            targets.push(check_target(t.as_mut(), &corpus, Scope::Ci));
        }
    }
    let mut diffs = BTreeMap::new();
    for t in targets.iter().filter(|t| t.deterministic) {
        let base = baseline.as_ref().and_then(|b| b.targets.get(&t.target));
        diffs.insert(t.target.clone(), compare(base, &t.counters()));
    }
    let failures: usize = targets.iter().map(|t| t.failures().count()).sum();
    let mut updated = false;
    if update {
        if failures > 0 {
            return Err(CheckError::Baseline(format!(
                "refusing to update the baseline: {failures} cases fail correctness checks"
            )));
        }
        let mut new = baseline.clone().unwrap_or(Baseline {
            puzzle: puzzle.to_string(),
            corpus_fingerprint: String::new(),
            targets: BTreeMap::new(),
        });
        new.corpus_fingerprint = corpus.fingerprint.clone();
        for t in targets.iter().filter(|t| t.deterministic) {
            new.targets.insert(t.target.clone(), t.counters());
        }
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| CheckError::Baseline(e.to_string()))?;
        }
        std::fs::write(path, new.to_json())
            .map_err(|e| CheckError::Baseline(format!("{}: {e}", path.display())))?;
        updated = true;
    }
    Ok(PuzzleCheck {
        puzzle: puzzle.to_string(),
        corpus_fingerprint: corpus.fingerprint,
        baseline_fingerprint: baseline.map(|b| b.corpus_fingerprint),
        targets,
        diffs,
        updated,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k(expanded: u64) -> Counters {
        Counters {
            length: 3,
            expanded,
            generated: expanded * 6,
            peak_stored: 3,
            iterations: 1,
        }
    }

    #[test]
    fn compare_reports_every_kind_of_difference() {
        let base: BTreeMap<_, _> = [("a".to_string(), k(1)), ("b".to_string(), k(2))].into();
        let now: BTreeMap<_, _> = [("a".to_string(), k(1)), ("c".to_string(), k(3))].into();
        assert_eq!(compare(Some(&base), &base), vec![]);
        assert_eq!(compare(None, &now), vec![Diff::NoBaseline]);
        let d = compare(Some(&base), &now);
        assert_eq!(
            d,
            vec![
                Diff::Missing { case: "b".into() },
                Diff::New {
                    case: "c".into(),
                    now: k(3)
                }
            ]
        );
        let changed: BTreeMap<_, _> = [("a".to_string(), k(9)), ("b".to_string(), k(2))].into();
        assert!(matches!(
            compare(Some(&base), &changed).as_slice(),
            [Diff::Changed { case, .. }] if case == "a"
        ));
    }

    #[test]
    fn baseline_json_round_trips() {
        let b = Baseline {
            puzzle: "p".into(),
            corpus_fingerprint: "fnv1a64:0".into(),
            targets: [
                (
                    "p/x".to_string(),
                    [("c1".to_string(), k(1)), ("c2".to_string(), k(2))].into(),
                ),
                ("p/y".to_string(), BTreeMap::new()),
            ]
            .into(),
        };
        let json = b.to_json();
        assert_eq!(serde_json::from_str::<Baseline>(&json).unwrap(), b);
        assert!(json
            .lines()
            .any(|l| l.trim_start().starts_with("\"c1\": {")));
    }
}
