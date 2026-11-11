//! The full local benchmark run: correctness, in-process latency per depth bucket and per
//! depth, per-node cost, peak heap per solve, solver setup cost, pattern-database build
//! times and primitive costs. Wall-clock numbers are machine-dependent; run on a quiet
//! machine and commit the report with its machine description (ADR 0006, ADR 0007).

use std::collections::BTreeMap;
use std::time::Duration;

use serde::Serialize;

use crate::check::{self, BucketCounters};
use crate::corpus::{Bucket, BucketScheme, CorpusError};
use crate::report::Machine;
use crate::stats;
use crate::target::{
    CaseResult, CaseTiming, PrimitiveReport, Registry, Scope, SetupReport, TableReport,
    TimingBudget,
};

/// Knobs for [`run`].
#[derive(Debug, Clone, Serialize)]
pub struct RunOptions {
    pub timing: TimingBudget,
    /// Time spent re-building each pattern database after the first build.
    pub table_budget: Duration,
    /// Time spent on each primitive micro-benchmark.
    pub primitive_budget: Duration,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            timing: TimingBudget::default(),
            table_budget: Duration::from_secs(2),
            primitive_budget: Duration::from_millis(300),
        }
    }
}

/// Latency over the cases of one bucket. Each case contributes its median solve time.
#[derive(Debug, Clone, Serialize)]
pub struct BucketLatency {
    pub bucket: Bucket,
    pub cases: usize,
    pub median_s: f64,
    pub p95_s: f64,
    pub max_s: f64,
    pub mean_expanded: f64,
    /// Total time / total expanded nodes: the per-node cost at this depth range.
    pub ns_per_expanded: Option<f64>,
}

/// Latency at one exact optimal depth.
#[derive(Debug, Clone, Serialize)]
pub struct DepthLatency {
    pub depth: u32,
    pub cases: usize,
    pub median_s: f64,
    pub max_s: f64,
    pub mean_expanded: f64,
}

/// Peak heap per solve over the cases of one bucket.
#[derive(Debug, Clone, Serialize)]
pub struct BucketMemory {
    pub bucket: Bucket,
    pub cases: usize,
    pub median_peak_bytes: usize,
    pub max_peak_bytes: usize,
}

/// Everything measured for one target.
#[derive(Debug, Clone, Serialize)]
pub struct TargetReport {
    pub target: String,
    pub puzzle: String,
    pub setup: SetupReport,
    pub checked: usize,
    pub failures: Vec<(String, String)>,
    pub latency: Vec<BucketLatency>,
    pub by_depth: Vec<DepthLatency>,
    pub memory: Vec<BucketMemory>,
    pub counters: Vec<BucketCounters>,
    pub cases: Vec<CaseTiming>,
}

/// Puzzle-level measurements.
#[derive(Debug, Clone, Serialize)]
pub struct WorkloadReport {
    pub puzzle: String,
    pub corpus_fingerprint: String,
    pub corpus_cases: usize,
    pub buckets: BucketScheme,
    pub tables: Vec<TableReport>,
    pub primitives: Vec<PrimitiveReport>,
}

/// A complete benchmark run; serialised as the JSON report and rendered as Markdown by
/// [`crate::report::markdown`].
#[derive(Debug, Clone, Serialize)]
pub struct SuiteReport {
    pub machine: Machine,
    pub options: RunOptions,
    pub workloads: Vec<WorkloadReport>,
    pub targets: Vec<TargetReport>,
}

impl SuiteReport {
    pub fn failure_count(&self) -> usize {
        self.targets.iter().map(|t| t.failures.len()).sum()
    }
}

fn bucket_latency(timings: &[CaseTiming]) -> Vec<BucketLatency> {
    Bucket::ALL
        .iter()
        .filter_map(|&b| {
            let ts: Vec<&CaseTiming> = timings.iter().filter(|t| t.bucket == b).collect();
            if ts.is_empty() {
                return None;
            }
            let times = stats::sorted(ts.iter().map(|t| t.median_s).collect());
            let total_s: f64 = times.iter().sum();
            let expanded: u64 = ts.iter().map(|t| t.expanded).sum();
            Some(BucketLatency {
                bucket: b,
                cases: ts.len(),
                median_s: stats::median(&times).unwrap(),
                p95_s: stats::percentile(&times, 0.95).unwrap(),
                max_s: *times.last().unwrap(),
                mean_expanded: expanded as f64 / ts.len() as f64,
                ns_per_expanded: (expanded > 0).then(|| total_s * 1e9 / expanded as f64),
            })
        })
        .collect()
}

fn depth_latency(timings: &[CaseTiming]) -> Vec<DepthLatency> {
    let mut by: BTreeMap<u32, Vec<&CaseTiming>> = BTreeMap::new();
    for t in timings {
        if let Some(d) = t.optimal_length {
            by.entry(d).or_default().push(t);
        }
    }
    by.into_iter()
        .map(|(depth, ts)| {
            let times = stats::sorted(ts.iter().map(|t| t.median_s).collect());
            DepthLatency {
                depth,
                cases: ts.len(),
                median_s: stats::median(&times).unwrap(),
                max_s: *times.last().unwrap(),
                mean_expanded: ts.iter().map(|t| t.expanded as f64).sum::<f64>() / ts.len() as f64,
            }
        })
        .collect()
}

fn bucket_memory(checks: &[crate::target::CaseCheck]) -> Vec<BucketMemory> {
    Bucket::ALL
        .iter()
        .filter_map(|&b| {
            let mut peaks: Vec<usize> = checks
                .iter()
                .filter(|c| c.bucket == b)
                .filter_map(|c| match c.result {
                    CaseResult::Solved {
                        peak_heap_bytes: Some(p),
                        ..
                    } => Some(p),
                    _ => None,
                })
                .collect();
            if peaks.is_empty() {
                return None;
            }
            peaks.sort_unstable();
            Some(BucketMemory {
                bucket: b,
                cases: peaks.len(),
                median_peak_bytes: peaks[(peaks.len() - 1) / 2],
                max_peak_bytes: *peaks.last().unwrap(),
            })
        })
        .collect()
}

/// Run the full suite over every target in `registry`. Progress goes to stderr.
pub fn run(registry: &mut Registry, options: &RunOptions) -> Result<SuiteReport, CorpusError> {
    let mut corpora = BTreeMap::new();
    let mut workloads = Vec::new();
    for w in registry.workloads() {
        let corpus = w.corpus()?;
        eprintln!("[{}] pattern databases ...", w.puzzle());
        let tables = w.tables(options.table_budget);
        eprintln!("[{}] primitives ...", w.puzzle());
        let primitives = w.primitives(&corpus.cases, options.primitive_budget);
        workloads.push(WorkloadReport {
            puzzle: w.puzzle().to_string(),
            corpus_fingerprint: corpus.fingerprint.clone(),
            corpus_cases: corpus.cases.len(),
            buckets: w.buckets(),
            tables,
            primitives,
        });
        corpora.insert(w.puzzle().to_string(), corpus);
    }
    let mut targets = Vec::new();
    for t in registry.targets_mut().iter_mut() {
        let corpus = &corpora[t.puzzle()];
        eprintln!("[{}] checking {} cases ...", t.id(), corpus.cases.len());
        let checks = check::check_target(t.as_mut(), corpus, Scope::Full);
        eprintln!("[{}] timing ...", t.id());
        let mut timings = Vec::new();
        let in_scope: Vec<_> = corpus
            .cases
            .iter()
            .filter(|c| t.applies_to(c, Scope::Full))
            .collect();
        for (case, c) in in_scope.into_iter().zip(&checks.cases) {
            debug_assert_eq!(case.id, c.id);
            if let CaseResult::Solved {
                elapsed_s,
                ref counters,
                ..
            } = c.result
            {
                if let Some(timing) = t.time(case, elapsed_s, counters, &options.timing) {
                    timings.push(timing);
                }
            }
        }
        let failures = checks
            .cases
            .iter()
            .filter_map(|c| match &c.result {
                CaseResult::Failed { reason } => Some((c.id.clone(), reason.clone())),
                _ => None,
            })
            .collect();
        targets.push(TargetReport {
            target: checks.target.clone(),
            puzzle: checks.puzzle.clone(),
            setup: checks.setup.clone(),
            checked: checks.cases.len(),
            failures,
            latency: bucket_latency(&timings),
            by_depth: depth_latency(&timings),
            memory: bucket_memory(&checks.cases),
            counters: check::bucket_counters(&checks.cases),
            cases: timings,
        });
    }
    Ok(SuiteReport {
        machine: Machine::detect(),
        options: options.clone(),
        workloads,
        targets,
    })
}
