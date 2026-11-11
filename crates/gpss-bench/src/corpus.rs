//! Loading conformance-corpus cases and grouping them into depth buckets.
//!
//! The benchmark suite does not invent its own inputs: it reads the same JSONL corpus that
//! `conformance/harness.py` uses (`conformance/corpus/<puzzle>.jsonl`), so every case a
//! benchmark times has been verified by the independent reference model.

use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// What the corpus expects an implementation to do with a case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Expect {
    /// Return a valid solution (of exactly `optimal_length` moves if the solver is optimal).
    Solve,
    /// Reject the input (malformed notation).
    Reject,
    /// An expectation this version of the suite does not understand; such cases are skipped.
    #[serde(other)]
    Unknown,
}

/// One corpus case. Unknown JSON fields are ignored so corpora can grow new metadata.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Case {
    pub id: String,
    pub puzzle: String,
    pub scramble: String,
    pub expect: Expect,
    #[serde(default)]
    pub tags: Vec<String>,
    /// Proven optimal length, when the corpus knows it.
    #[serde(default)]
    pub optimal_length: Option<u32>,
}

impl Case {
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }
}

/// A coarse difficulty class used to aggregate results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Bucket {
    /// Already solved (optimal length 0): identity scrambles, empty input.
    Trivial,
    /// Short-range solves.
    Short,
    /// Mid-range solves.
    Mid,
    /// Top-of-range solves, up to the puzzle's diameter (antipodes).
    Top,
    /// Inputs that must be rejected.
    Invalid,
    /// Solvable, but the corpus gives neither an optimal length nor a range tag.
    Unbucketed,
}

impl Bucket {
    pub const ALL: [Bucket; 6] = [
        Bucket::Trivial,
        Bucket::Short,
        Bucket::Mid,
        Bucket::Top,
        Bucket::Invalid,
        Bucket::Unbucketed,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Bucket::Trivial => "trivial",
            Bucket::Short => "short",
            Bucket::Mid => "mid",
            Bucket::Top => "top",
            Bucket::Invalid => "invalid",
            Bucket::Unbucketed => "unbucketed",
        }
    }
}

impl fmt::Display for Bucket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Per-puzzle depth ranges: `1..=short_max` is short, `short_max+1..=mid_max` is mid,
/// anything deeper is top-of-range.
///
/// The ranges match the `short` / `mid` / `top` tags the corpus generator writes, but are
/// applied to `optimal_length` so that random scrambles and edge cases (which carry no
/// range tag) land in the right bucket too.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BucketScheme {
    pub short_max: u32,
    pub mid_max: u32,
}

impl BucketScheme {
    pub const fn new(short_max: u32, mid_max: u32) -> Self {
        assert!(short_max < mid_max);
        Self { short_max, mid_max }
    }

    pub fn bucket(&self, case: &Case) -> Bucket {
        match case.expect {
            Expect::Reject => return Bucket::Invalid,
            Expect::Unknown => return Bucket::Unbucketed,
            Expect::Solve => {}
        }
        match case.optimal_length {
            Some(0) => Bucket::Trivial,
            Some(d) if d <= self.short_max => Bucket::Short,
            Some(d) if d <= self.mid_max => Bucket::Mid,
            Some(_) => Bucket::Top,
            None if case.has_tag("short") => Bucket::Short,
            None if case.has_tag("mid") => Bucket::Mid,
            None if case.has_tag("top") => Bucket::Top,
            None => Bucket::Unbucketed,
        }
    }
}

/// Errors loading a corpus file.
#[derive(Debug, thiserror::Error)]
pub enum CorpusError {
    #[error("cannot read corpus {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("{path}:{line}: {source}")]
    Parse {
        path: PathBuf,
        line: usize,
        source: serde_json::Error,
    },
}

/// The workspace root (two levels above this crate's manifest).
pub fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Directory holding `<puzzle>.jsonl` corpora: `$GPSS_CORPUS_DIR` if set, otherwise
/// `conformance/corpus/` in the workspace.
pub fn corpus_dir() -> PathBuf {
    std::env::var_os("GPSS_CORPUS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root().join("conformance/corpus"))
}

/// Path of the corpus for `puzzle` (e.g. `"cube2"`).
pub fn corpus_path(puzzle: &str) -> PathBuf {
    corpus_dir().join(format!("{puzzle}.jsonl"))
}

/// Parse a JSONL corpus (blank lines allowed).
pub fn parse(text: &str, path: &Path) -> Result<Vec<Case>, CorpusError> {
    text.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| {
            serde_json::from_str(l).map_err(|source| CorpusError::Parse {
                path: path.to_path_buf(),
                line: i + 1,
                source,
            })
        })
        .collect()
}

/// A loaded corpus plus a fingerprint of its bytes (stored in baselines so that a
/// regenerated corpus is reported as such rather than as a counter regression).
#[derive(Debug, Clone)]
pub struct Corpus {
    pub puzzle: String,
    pub cases: Vec<Case>,
    pub fingerprint: String,
}

/// Load the corpus for `puzzle` from [`corpus_path`].
pub fn load(puzzle: &str) -> Result<Corpus, CorpusError> {
    let path = corpus_path(puzzle);
    let text = std::fs::read_to_string(&path).map_err(|source| CorpusError::Io {
        path: path.clone(),
        source,
    })?;
    Ok(Corpus {
        puzzle: puzzle.to_string(),
        cases: parse(&text, &path)?,
        fingerprint: fingerprint(text.as_bytes()),
    })
}

/// 64-bit FNV-1a of `bytes`, as hex. Not cryptographic; it only detects corpus changes.
pub fn fingerprint(bytes: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("fnv1a64:{h:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case(json: &str) -> Case {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn parses_corpus_lines_and_ignores_unknown_fields() {
        let text = r#"{"id":"a","puzzle":"p","scramble":"R","expect":"solve","tags":["d1"],"optimal_length":1,"extra":true}

{"id":"b","puzzle":"p","scramble":"X","expect":"reject"}
{"id":"c","puzzle":"p","scramble":"R","expect":"time-travel"}"#;
        let cases = parse(text, Path::new("t.jsonl")).unwrap();
        assert_eq!(cases.len(), 3);
        assert_eq!(cases[0].optimal_length, Some(1));
        assert_eq!(cases[1].expect, Expect::Reject);
        assert!(cases[1].tags.is_empty());
        assert_eq!(cases[2].expect, Expect::Unknown);
    }

    #[test]
    fn parse_errors_report_the_line() {
        let err = parse("{}\n", Path::new("t.jsonl")).unwrap_err();
        assert!(err.to_string().starts_with("t.jsonl:1:"), "{err}");
    }

    #[test]
    fn buckets_follow_optimal_length_then_tags() {
        let s = BucketScheme::new(4, 8);
        let mk = |len: &str, tags: &str| {
            case(&format!(
                r#"{{"id":"x","puzzle":"p","scramble":"","expect":"solve","tags":[{tags}],"optimal_length":{len}}}"#
            ))
        };
        assert_eq!(s.bucket(&mk("0", "")), Bucket::Trivial);
        assert_eq!(s.bucket(&mk("1", "")), Bucket::Short);
        assert_eq!(s.bucket(&mk("4", "")), Bucket::Short);
        assert_eq!(s.bucket(&mk("5", "")), Bucket::Mid);
        assert_eq!(s.bucket(&mk("8", "")), Bucket::Mid);
        assert_eq!(s.bucket(&mk("9", "")), Bucket::Top);
        assert_eq!(s.bucket(&mk("null", r#""top""#)), Bucket::Top);
        assert_eq!(s.bucket(&mk("null", "")), Bucket::Unbucketed);
        let reject = case(r#"{"id":"r","puzzle":"p","scramble":"X","expect":"reject"}"#);
        assert_eq!(s.bucket(&reject), Bucket::Invalid);
    }

    #[test]
    fn fingerprint_is_stable_and_sensitive() {
        assert_eq!(fingerprint(b""), "fnv1a64:cbf29ce484222325");
        assert_ne!(fingerprint(b"a"), fingerprint(b"b"));
    }

    #[test]
    fn cube2_corpus_loads_and_matches_its_own_tags() {
        let corpus = load("cube2").unwrap();
        assert_eq!(corpus.cases.len(), 233);
        let s = crate::targets::cube2::BUCKETS;
        for c in &corpus.cases {
            for (tag, b) in [
                ("short", Bucket::Short),
                ("mid", Bucket::Mid),
                ("top", Bucket::Top),
            ] {
                if c.has_tag(tag) {
                    assert_eq!(s.bucket(c), b, "{}", c.id);
                }
            }
        }
    }
}
