//! Error type for `gpss-learn`.

use std::path::PathBuf;

/// Everything that can go wrong loading or running a learned heuristic.
#[derive(Debug, thiserror::Error)]
pub enum LearnError {
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid weights manifest {path}: {message}")]
    Manifest { path: PathBuf, message: String },
    #[error("weights are for encoding {found}, but this puzzle uses {expected}; train or download weights for {expected}")]
    Incompatible { expected: String, found: String },
    #[error("{path}: SHA-256 {actual} does not match the manifest ({expected}); the network and its manifest are out of sync")]
    Checksum {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    #[error("ONNX Runtime: {0}")]
    Onnx(String),
    #[error("this build has no ONNX support; rebuild with `--features learn` (gpss-cli) or `--features onnx` (gpss-learn)")]
    OnnxDisabled,
}
