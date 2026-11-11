//! Weight manifests: the JSON file that travels with every ONNX network.
//!
//! A shipped network is a pair `<name>.onnx` + `<name>.json`. The manifest records which
//! puzzle and which exact state encoding the network expects ([`EncodingRecord`]), the
//! SHA-256 of the `.onnx` file, the tensor names, and free-form architecture, training
//! and evaluation metadata. [`Manifest::load`] refuses a manifest whose network file has
//! a different checksum, and [`Manifest::check_compatible`] refuses an encoding that is
//! not *exactly* the puzzle's, so a 2x2x2 network can never silently be used on another
//! puzzle, nor on the 2x2x2 after its encoding changed (ADR 0007).
//!
//! The manifest is written by `training/gpss_train/export.py`; see `docs/training.md`.

use std::fs;
use std::path::{Path, PathBuf};

use gpss_core::encode::EncodingSpec;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::LearnError;

/// Value of [`Manifest::format`].
pub const MANIFEST_FORMAT: &str = "gpss-weights";
/// Highest manifest version this build understands.
pub const MANIFEST_FORMAT_VERSION: u32 = 1;

/// The encoding a network was trained on (mirror of [`EncodingSpec`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EncodingRecord {
    pub puzzle: String,
    pub scheme: String,
    pub version: u32,
    pub len: usize,
    pub classes: u16,
}

impl From<&EncodingSpec> for EncodingRecord {
    fn from(s: &EncodingSpec) -> Self {
        Self {
            puzzle: s.puzzle.clone(),
            scheme: s.scheme.clone(),
            version: s.version,
            len: s.len,
            classes: s.classes,
        }
    }
}

impl EncodingRecord {
    pub fn key(&self) -> String {
        format!("{}/{}/v{}", self.puzzle, self.scheme, self.version)
    }
}

/// The network file and its interface.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRecord {
    /// File name of the `.onnx` network, relative to the manifest.
    pub file: String,
    /// Lower-case hex SHA-256 of the network file.
    pub sha256: String,
    /// Input tensor name; `int64[batch, len]` of encoding categories.
    pub input: String,
    /// Output tensor name; `float32[batch]` cost-to-go estimates.
    pub output: String,
    #[serde(default)]
    pub opset: Option<u32>,
}

/// A weights manifest (see the module docs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub format: String,
    pub format_version: u32,
    /// Short human-readable name, e.g. `cube2-davi-v1`.
    pub name: String,
    pub encoding: EncodingRecord,
    pub model: ModelRecord,
    /// Network architecture (layer sizes etc.), as written by the exporter.
    #[serde(default)]
    pub architecture: serde_json::Value,
    /// Training configuration and summary statistics.
    #[serde(default)]
    pub training: serde_json::Value,
    /// Evaluation summary, if one was run.
    #[serde(default)]
    pub evaluation: serde_json::Value,
    /// Works this network is based on; keep `[Ago19]` here.
    #[serde(default)]
    pub references: Vec<String>,
}

/// A manifest together with the resolved path of its network.
#[derive(Debug, Clone)]
pub struct LoadedManifest {
    pub manifest: Manifest,
    pub manifest_path: PathBuf,
    pub model_path: PathBuf,
}

/// Accept either the manifest (`.json`) or the network (`.onnx`) path.
pub fn manifest_path(path: &Path) -> PathBuf {
    if path.extension().is_some_and(|e| e == "json") {
        path.to_path_buf()
    } else {
        path.with_extension("json")
    }
}

/// Lower-case hex SHA-256 of a file.
pub fn sha256_file(path: &Path) -> Result<String, LearnError> {
    let bytes = fs::read(path).map_err(|source| LearnError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

impl Manifest {
    /// Read a manifest (given its own path or its network's path) and verify that the
    /// network file exists and matches the recorded checksum.
    pub fn load(path: impl AsRef<Path>) -> Result<LoadedManifest, LearnError> {
        let manifest_path = manifest_path(path.as_ref());
        let text = fs::read_to_string(&manifest_path).map_err(|source| LearnError::Io {
            path: manifest_path.clone(),
            source,
        })?;
        let bad = |message: String| LearnError::Manifest {
            path: manifest_path.clone(),
            message,
        };
        let manifest: Manifest = serde_json::from_str(&text).map_err(|e| bad(e.to_string()))?;
        if manifest.format != MANIFEST_FORMAT {
            return Err(bad(format!(
                "format is {:?}, expected {MANIFEST_FORMAT:?}",
                manifest.format
            )));
        }
        if manifest.format_version > MANIFEST_FORMAT_VERSION {
            return Err(bad(format!(
                "format_version {} is newer than this build supports ({MANIFEST_FORMAT_VERSION})",
                manifest.format_version
            )));
        }
        let dir = manifest_path.parent().unwrap_or(Path::new("."));
        let model_path = dir.join(&manifest.model.file);
        let actual = sha256_file(&model_path)?;
        if !actual.eq_ignore_ascii_case(&manifest.model.sha256) {
            return Err(LearnError::Checksum {
                path: model_path,
                expected: manifest.model.sha256.clone(),
                actual,
            });
        }
        Ok(LoadedManifest {
            manifest,
            manifest_path,
            model_path,
        })
    }

    /// Fail unless the network was trained on exactly `spec`.
    pub fn check_compatible(&self, spec: &EncodingSpec) -> Result<(), LearnError> {
        let expected = EncodingRecord::from(spec);
        if self.encoding == expected {
            Ok(())
        } else {
            Err(LearnError::Incompatible {
                expected: format!(
                    "{} (len {}, classes {})",
                    expected.key(),
                    expected.len,
                    expected.classes
                ),
                found: format!(
                    "{} (len {}, classes {})",
                    self.encoding.key(),
                    self.encoding.len,
                    self.encoding.classes
                ),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpss_core::Encodable;
    use gpss_puzzles::cube2::Cube2;

    fn write_pair(dir: &Path, encoding: EncodingRecord, tamper: bool) -> PathBuf {
        fs::create_dir_all(dir).unwrap();
        let onnx = dir.join("net.onnx");
        fs::write(&onnx, b"not really onnx").unwrap();
        let mut sha = sha256_file(&onnx).unwrap();
        if tamper {
            sha = "0".repeat(64);
        }
        let m = Manifest {
            format: MANIFEST_FORMAT.into(),
            format_version: 1,
            name: "test".into(),
            encoding,
            model: ModelRecord {
                file: "net.onnx".into(),
                sha256: sha,
                input: "state".into(),
                output: "cost_to_go".into(),
                opset: Some(17),
            },
            architecture: serde_json::Value::Null,
            training: serde_json::Value::Null,
            evaluation: serde_json::Value::Null,
            references: vec!["Ago19".into()],
        };
        let path = dir.join("net.json");
        fs::write(&path, serde_json::to_string_pretty(&m).unwrap()).unwrap();
        path
    }

    fn tmp(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("gpss-learn-{}-{name}", std::process::id()))
    }

    #[test]
    fn loads_by_either_path_and_accepts_matching_encoding() {
        let spec = Cube2::new().encoding();
        let dir = tmp("ok");
        write_pair(&dir, (&spec).into(), false);
        for p in [dir.join("net.json"), dir.join("net.onnx")] {
            let loaded = Manifest::load(&p).unwrap();
            loaded.manifest.check_compatible(&spec).unwrap();
            assert_eq!(loaded.model_path, dir.join("net.onnx"));
        }
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn rejects_other_puzzle_or_encoding_version() {
        let spec = Cube2::new().encoding();
        let mut other_version = EncodingRecord::from(&spec);
        other_version.version += 1;
        let mut other_puzzle = EncodingRecord::from(&spec);
        other_puzzle.puzzle = "cube:3x3x3".into();
        for (i, enc) in [other_version, other_puzzle].into_iter().enumerate() {
            let dir = tmp(&format!("bad{i}"));
            let loaded = Manifest::load(write_pair(&dir, enc, false)).unwrap();
            let err = loaded.manifest.check_compatible(&spec).unwrap_err();
            assert!(matches!(err, LearnError::Incompatible { .. }), "{err}");
            let _ = fs::remove_dir_all(dir);
        }
    }

    #[test]
    fn rejects_checksum_mismatch() {
        let spec = Cube2::new().encoding();
        let dir = tmp("sha");
        let err = Manifest::load(write_pair(&dir, (&spec).into(), true)).unwrap_err();
        assert!(matches!(err, LearnError::Checksum { .. }), "{err}");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn shipped_cube2_weights_match_the_current_encoding() {
        // The committed network must always be loadable by the current Cube2 encoding.
        // If you change `impl Encodable for Cube2`, bump its version and retrain.
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../weights/cube2/cube2-davi-v1.json");
        if !path.exists() {
            return; // weights not trained yet in this checkout
        }
        let loaded = Manifest::load(&path).unwrap();
        loaded
            .manifest
            .check_compatible(&Cube2::new().encoding())
            .unwrap();
        assert!(loaded
            .manifest
            .references
            .iter()
            .any(|r| r.contains("Agostinelli")));
    }
}
