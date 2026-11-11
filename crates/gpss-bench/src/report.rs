//! Machine description and the Markdown rendering of a [`SuiteReport`].

use std::fmt::Write as _;
use std::process::Command;

use serde::Serialize;

use crate::corpus::{self, Bucket};
use crate::run::SuiteReport;

/// Where and how a benchmark ran. Every committed report carries one (ADR 0006).
#[derive(Debug, Clone, Serialize)]
pub struct Machine {
    pub cpu: String,
    pub logical_cpus: usize,
    pub memory_gib: Option<f64>,
    pub os: String,
    pub kernel: String,
    pub rustc: String,
    pub profile: String,
    pub git_commit: String,
    pub date_utc: String,
    pub heap_counting: bool,
}

fn read(path: &str) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .args(args)
        .current_dir(corpus::workspace_root())
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
fn civil(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

impl Machine {
    /// Best-effort detection; fields fall back to `"unknown"` off Linux.
    pub fn detect() -> Self {
        let unknown = || "unknown".to_string();
        let cpuinfo = read("/proc/cpuinfo").unwrap_or_default();
        let cpu = cpuinfo
            .lines()
            .find(|l| l.starts_with("model name"))
            .and_then(|l| l.split(':').nth(1))
            .map(|s| s.trim().to_string())
            .unwrap_or_else(unknown);
        let memory_gib = read("/proc/meminfo").and_then(|m| {
            let kb: f64 = m
                .lines()
                .find(|l| l.starts_with("MemTotal"))?
                .split_whitespace()
                .nth(1)?
                .parse()
                .ok()?;
            Some((kb / 1024.0 / 1024.0 * 10.0).round() / 10.0)
        });
        let os = read("/etc/os-release")
            .and_then(|s| {
                s.lines()
                    .find_map(|l| l.strip_prefix("PRETTY_NAME="))
                    .map(|v| v.trim_matches('"').to_string())
            })
            .unwrap_or_else(|| std::env::consts::OS.to_string());
        let kernel = read("/proc/sys/kernel/osrelease")
            .map(|s| s.trim().to_string())
            .unwrap_or_else(unknown);
        let git_commit = match git(&["rev-parse", "--short", "HEAD"]) {
            Some(c) => {
                let dirty = git(&["status", "--porcelain", "--untracked-files=no"])
                    .is_some_and(|s| !s.is_empty());
                if dirty {
                    format!("{c} (+ uncommitted changes)")
                } else {
                    c
                }
            }
            None => unknown(),
        };
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        let (y, m, d) = civil(secs.div_euclid(86_400));
        Self {
            cpu,
            logical_cpus: std::thread::available_parallelism().map_or(0, |n| n.get()),
            memory_gib,
            os,
            kernel,
            rustc: env!("GPSS_BENCH_RUSTC").to_string(),
            profile: env!("GPSS_BENCH_PROFILE").to_string(),
            git_commit,
            date_utc: format!("{y:04}-{m:02}-{d:02}"),
            heap_counting: crate::alloc::is_active(),
        }
    }
}

/// Human-readable duration with 3 significant figures.
pub fn fmt_s(s: f64) -> String {
    let (v, unit) = if s < 1e-6 {
        (s * 1e9, "ns")
    } else if s < 1e-3 {
        (s * 1e6, "µs")
    } else if s < 1.0 {
        (s * 1e3, "ms")
    } else {
        (s, "s")
    };
    let prec = if v >= 100.0 {
        0
    } else if v >= 10.0 {
        1
    } else {
        2
    };
    format!("{v:.prec$} {unit}")
}

/// Human-readable byte count (binary units).
pub fn fmt_bytes(b: f64) -> String {
    let units = ["B", "KiB", "MiB", "GiB"];
    let mut v = b;
    let mut i = 0;
    while v.abs() >= 1024.0 && i + 1 < units.len() {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{v:.0} B")
    } else {
        format!("{v:.1} {}", units[i])
    }
}

fn fmt_count(v: f64) -> String {
    if v >= 100.0 || v == v.trunc() {
        format!("{v:.0}")
    } else {
        format!("{v:.1}")
    }
}

/// Render `report` as Markdown. `command` is the invocation to print under "Reproduce".
pub fn markdown(report: &SuiteReport, title: &str, command: &str) -> String {
    let m = &report.machine;
    let mut s = String::new();
    let w = &mut s;
    let _ = writeln!(w, "# {title}\n");
    let _ = writeln!(
        w,
        "Generated by `gpss-bench run` (see [`docs/benchmarking.md`](../benchmarking.md)). \
         Do not edit by hand: re-run the command below and commit the output. Numbers are \
         in-process wall-clock times on the machine below and are only comparable with \
         reports from the same machine.\n"
    );
    let _ = writeln!(w, "## Machine\n");
    let _ = writeln!(w, "| | |\n|---|---|");
    let _ = writeln!(w, "| CPU | {} ({} logical CPUs) |", m.cpu, m.logical_cpus);
    if let Some(g) = m.memory_gib {
        let _ = writeln!(w, "| Memory | {g} GiB |");
    }
    let _ = writeln!(w, "| OS | {} (kernel {}) |", m.os, m.kernel);
    let _ = writeln!(w, "| Compiler | {} (profile `{}`) |", m.rustc, m.profile);
    let _ = writeln!(w, "| Commit | `{}` |", m.git_commit);
    let _ = writeln!(w, "| Date (UTC) | {} |", m.date_utc);
    let o = &report.options.timing;
    let _ = writeln!(
        w,
        "| Timing budget | {} per case, samples >= {}, <= {} samples |",
        fmt_s(o.per_case.as_secs_f64()),
        fmt_s(o.min_sample.as_secs_f64()),
        o.max_samples
    );
    let _ = writeln!(
        w,
        "| Heap counting | {} |\n",
        if m.heap_counting { "on" } else { "off" }
    );
    let _ = writeln!(w, "## Reproduce\n\n```bash\n{command}\n```\n");

    let _ = writeln!(w, "## Correctness\n");
    let _ = writeln!(
        w,
        "Every case was checked before timing: valid solution, exact optimal length for \
         optimal solvers, bounded length otherwise, rejection of invalid input.\n"
    );
    let _ = writeln!(w, "| target | solver | optimal | cases | failures |");
    let _ = writeln!(w, "|---|---|:-:|--:|--:|");
    for t in &report.targets {
        let _ = writeln!(
            w,
            "| `{}` | `{}` | {} | {} | {} |",
            t.target,
            t.setup.solver,
            if t.setup.optimal { "yes" } else { "no" },
            t.checked,
            t.failures.len()
        );
    }
    for t in report.targets.iter().filter(|t| !t.failures.is_empty()) {
        let _ = writeln!(w, "\nFailures of `{}`:\n", t.target);
        for (id, why) in &t.failures {
            let _ = writeln!(w, "- `{id}`: {why}");
        }
    }

    let _ = writeln!(w, "\n## Latency by depth bucket\n");
    let _ = writeln!(
        w,
        "Per case: median of repeated in-process solves (fast solves batched). Per bucket: \
         median / p95 / max over cases. `ns/node` = total time / total expanded nodes.\n"
    );
    for wl in &report.workloads {
        let sch = describe_scheme(wl.buckets);
        let _ = writeln!(w, "Buckets for `{}`: {sch}.\n", wl.puzzle);
    }
    let _ = writeln!(
        w,
        "| target | bucket | cases | median | p95 | max | mean expanded | ns/node |"
    );
    let _ = writeln!(w, "|---|---|--:|--:|--:|--:|--:|--:|");
    for t in &report.targets {
        for b in &t.latency {
            let _ = writeln!(
                w,
                "| `{}` | {} | {} | {} | {} | {} | {} | {} |",
                t.target,
                b.bucket,
                b.cases,
                fmt_s(b.median_s),
                fmt_s(b.p95_s),
                fmt_s(b.max_s),
                fmt_count(b.mean_expanded),
                b.ns_per_expanded.map_or("-".into(), |n| format!("{n:.1}"))
            );
        }
    }

    for wl in &report.workloads {
        let ts: Vec<_> = report
            .targets
            .iter()
            .filter(|t| t.puzzle == wl.puzzle)
            .collect();
        let depths: std::collections::BTreeSet<u32> = ts
            .iter()
            .flat_map(|t| t.by_depth.iter().map(|d| d.depth))
            .collect();
        if depths.is_empty() {
            continue;
        }
        let _ = writeln!(
            w,
            "\n## Median latency by optimal depth (`{}`)\n",
            wl.puzzle
        );
        let _ = write!(w, "| depth |");
        for t in &ts {
            let _ = write!(
                w,
                " `{}` |",
                t.target.split('/').nth(1).unwrap_or(&t.target)
            );
        }
        let _ = write!(w, "\n|--:|");
        for _ in &ts {
            let _ = write!(w, "--:|");
        }
        let _ = writeln!(w);
        for d in depths {
            let _ = write!(w, "| {d} |");
            for t in &ts {
                let cell = t
                    .by_depth
                    .iter()
                    .find(|x| x.depth == d)
                    .map_or("-".into(), |x| fmt_s(x.median_s));
                let _ = write!(w, " {cell} |");
            }
            let _ = writeln!(w);
        }
    }

    if report.machine.heap_counting {
        let _ = writeln!(w, "\n## Peak heap per solve\n");
        let _ = writeln!(
            w,
            "Bytes requested from the allocator during one `solve` call (search data \
             structures; excludes the solver's pre-built tables, listed under Setup).\n"
        );
        let _ = writeln!(w, "| target | bucket | cases | median | max |");
        let _ = writeln!(w, "|---|---|--:|--:|--:|");
        for t in &report.targets {
            for b in &t.memory {
                let _ = writeln!(
                    w,
                    "| `{}` | {} | {} | {} | {} |",
                    t.target,
                    b.bucket,
                    b.cases,
                    fmt_bytes(b.median_peak_bytes as f64),
                    fmt_bytes(b.max_peak_bytes as f64)
                );
            }
        }
    }

    let _ = writeln!(w, "\n## Solver setup\n");
    let _ = writeln!(
        w,
        "Building the solver, including all heuristic tables (measured once).\n"
    );
    let _ = writeln!(w, "| target | build time | heap retained | peak heap |");
    let _ = writeln!(w, "|---|--:|--:|--:|");
    for t in &report.targets {
        let (ret, peak) = t.setup.heap.map_or(("-".into(), "-".into()), |h| {
            (
                fmt_bytes(h.retained_bytes as f64),
                fmt_bytes(h.peak_bytes as f64),
            )
        });
        let _ = writeln!(
            w,
            "| `{}` | {} | {ret} | {peak} |",
            t.target,
            fmt_s(t.setup.build_s)
        );
    }

    let _ = writeln!(w, "\n## Pattern databases\n");
    let _ = writeln!(
        w,
        "| puzzle | table | entries | size | builds | median build | min build | peak heap |"
    );
    let _ = writeln!(w, "|---|---|--:|--:|--:|--:|--:|--:|");
    for wl in &report.workloads {
        for t in &wl.tables {
            let _ = writeln!(
                w,
                "| {} | {} | {} | {} | {} | {} | {} | {} |",
                t.puzzle,
                t.table,
                t.entries,
                fmt_bytes(t.bytes as f64),
                t.builds,
                fmt_s(t.median_build_s),
                fmt_s(t.min_build_s),
                t.heap
                    .map_or("-".into(), |h| fmt_bytes(h.peak_bytes as f64))
            );
        }
    }

    let _ = writeln!(w, "\n## Primitives\n");
    let _ = writeln!(
        w,
        "Over the start states of all solvable corpus cases (`apply`: every search move).\n"
    );
    let _ = writeln!(w, "| puzzle | operation | ns/op | ops timed |");
    let _ = writeln!(w, "|---|---|--:|--:|");
    for wl in &report.workloads {
        for p in &wl.primitives {
            let _ = writeln!(
                w,
                "| {} | `{}` | {:.2} | {} |",
                p.puzzle, p.op, p.ns_per_op, p.ops
            );
        }
    }

    let _ = writeln!(w, "\n## Search counters by bucket\n");
    let _ = writeln!(
        w,
        "Machine-independent. The CI check (`gpss-bench check`) compares these per case \
         against `crates/gpss-bench/baselines/`.\n"
    );
    let _ = writeln!(
        w,
        "| target | bucket | cases | mean length | mean expanded | max expanded | mean generated | max peak stored |"
    );
    let _ = writeln!(w, "|---|---|--:|--:|--:|--:|--:|--:|");
    for t in &report.targets {
        for b in &t.counters {
            let _ = writeln!(
                w,
                "| `{}` | {} | {} | {:.2} | {} | {} | {} | {} |",
                t.target,
                b.bucket,
                b.cases,
                b.mean_length,
                fmt_count(b.mean_expanded),
                b.max_expanded,
                fmt_count(b.mean_generated),
                b.max_peak_stored
            );
        }
    }
    s
}

/// Describe a bucket scheme, e.g. "short 1-4, mid 5-8, top >= 9".
pub fn describe_scheme(s: crate::corpus::BucketScheme) -> String {
    format!(
        "{} = 0, {} 1-{}, {} {}-{}, {} >= {}",
        Bucket::Trivial,
        Bucket::Short,
        s.short_max,
        Bucket::Mid,
        s.short_max + 1,
        s.mid_max,
        Bucket::Top,
        s.mid_max + 1
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats() {
        assert_eq!(fmt_s(0.000_000_123), "123 ns");
        assert_eq!(fmt_s(0.000_012_34), "12.3 µs");
        assert_eq!(fmt_s(0.5), "500 ms");
        assert_eq!(fmt_s(2.5), "2.50 s");
        assert_eq!(fmt_bytes(512.0), "512 B");
        assert_eq!(fmt_bytes(3.0 * 1024.0 * 1024.0), "3.0 MiB");
    }

    #[test]
    fn civil_dates() {
        assert_eq!(civil(0), (1970, 1, 1));
        assert_eq!(civil(19_723), (2024, 1, 1));
        assert_eq!(civil(20_513), (2026, 3, 1));
    }
}
