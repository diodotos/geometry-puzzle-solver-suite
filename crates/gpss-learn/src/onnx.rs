//! ONNX Runtime inference: a trained network as a [`Heuristic`].

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use gpss_core::encode::{Encodable, EncodingSpec};
use gpss_core::heuristic::Heuristic;
use ort::environment::Environment;
use ort::session::Session;
use ort::value::Tensor;

use crate::error::LearnError;
use crate::manifest::{LoadedManifest, Manifest};

/// Environment variable naming the ONNX Runtime shared library (same as `ort`'s own).
pub const DYLIB_ENV: &str = "ORT_DYLIB_PATH";

/// Locate the ONNX Runtime shared library, in order:
///
/// 1. `$ORT_DYLIB_PATH`;
/// 2. the library bundled with the Python `onnxruntime` package (`python3` on `PATH`),
///    which the training environment already has;
/// 3. the platform default name (`libonnxruntime.so` etc.), resolved by the dynamic loader.
pub fn runtime_library() -> PathBuf {
    if let Some(p) = std::env::var_os(DYLIB_ENV).filter(|p| !p.is_empty()) {
        return PathBuf::from(p);
    }
    if let Some(p) = python_onnxruntime_library() {
        return p;
    }
    PathBuf::from(format!(
        "{}onnxruntime{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    ))
}

fn python_onnxruntime_library() -> Option<PathBuf> {
    let out = std::process::Command::new("python3")
        .args([
            "-c",
            "import onnxruntime, os; print(os.path.join(os.path.dirname(onnxruntime.__file__), 'capi'))",
        ])
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let dir = PathBuf::from(String::from_utf8(out.stdout).ok()?.trim());
    let prefix = format!("{}onnxruntime", std::env::consts::DLL_PREFIX);
    let mut found: Vec<PathBuf> = std::fs::read_dir(&dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                n.starts_with(&prefix)
                    && n.contains(std::env::consts::DLL_SUFFIX)
                    && !n.contains("providers")
            })
        })
        .collect();
    found.sort();
    found.pop()
}

/// Load ONNX Runtime once per process and return an environment builder.
fn init_runtime() -> Result<ort::environment::EnvironmentBuilder, LearnError> {
    static LOADED: OnceLock<Result<PathBuf, String>> = OnceLock::new();
    let loaded = LOADED.get_or_init(|| {
        let path = runtime_library();
        ort::init_from(&path)
            .map(|_| path.clone())
            .map_err(|e| {
                format!(
                    "cannot load ONNX Runtime from {}: {e}; install it (`pip install onnxruntime`) or set {DYLIB_ENV}",
                    path.display()
                )
            })
    });
    match loaded {
        Ok(_) => Ok(ort::init()),
        Err(e) => Err(LearnError::Onnx(e.clone())),
    }
}

/// States per ONNX Runtime call; larger batches are split to bound memory.
const MAX_CHUNK: usize = 1 << 16;

/// A DAVI-trained cost-to-go network used as a (never admissible) heuristic.
///
/// Built from a weights manifest ([`Manifest`]); loading fails unless the manifest's
/// encoding equals `puzzle.encoding()` and the `.onnx` checksum matches. Estimates for
/// goal states are forced to 0 (the network is never trained on `J(goal)`; DeepCubeA
/// defines it as 0). Negative outputs are left to the search, which clamps them.
pub struct OnnxHeuristic<P> {
    puzzle: P,
    spec: EncodingSpec,
    loaded: LoadedManifest,
    session: Mutex<Session>,
    _env: Environment,
}

impl<P: Encodable> OnnxHeuristic<P> {
    /// Load `weights` (the manifest `.json` or the `.onnx` next to it) for `puzzle`.
    /// `threads` caps ONNX Runtime's intra-op threads (`None` = all cores).
    pub fn load(
        puzzle: P,
        weights: impl AsRef<Path>,
        threads: Option<usize>,
    ) -> Result<Self, LearnError> {
        let loaded = Manifest::load(weights)?;
        let spec = puzzle.encoding();
        loaded.manifest.check_compatible(&spec)?;
        let onnx = |e: &dyn std::fmt::Display| LearnError::Onnx(e.to_string());
        let env = init_runtime()?
            .with_name("gpss")
            .build()
            .map_err(|e| onnx(&e))?;
        let threads = threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        });
        let session = Session::builder(&env)
            .map_err(|e| onnx(&e))?
            .with_intra_threads(threads)
            .map_err(|e| onnx(&e))?
            .commit_from_file(&loaded.model_path)
            .map_err(|e| onnx(&e))?;
        let names = |outlets: &[ort::value::Outlet]| {
            outlets
                .iter()
                .map(|o| o.name().to_string())
                .collect::<Vec<_>>()
        };
        let (inputs, outputs) = (names(session.inputs()), names(session.outputs()));
        let m = &loaded.manifest.model;
        if !inputs.contains(&m.input) || !outputs.contains(&m.output) {
            return Err(LearnError::Manifest {
                path: loaded.manifest_path.clone(),
                message: format!(
                    "network has inputs {inputs:?} / outputs {outputs:?}, manifest says {:?} / {:?}",
                    m.input, m.output
                ),
            });
        }
        Ok(Self {
            puzzle,
            spec,
            loaded,
            session: Mutex::new(session),
            _env: env,
        })
    }

    pub fn manifest(&self) -> &Manifest {
        &self.loaded.manifest
    }

    /// Run the network on `states` (no goal override), appending to `out`.
    fn infer(&self, states: &[P::State], out: &mut Vec<f32>) -> Result<(), LearnError> {
        let len = self.spec.len;
        let mut codes = vec![0u8; len];
        let mut input: Vec<i64> = Vec::with_capacity(states.len() * len);
        for s in states {
            self.puzzle.encode(s, &mut codes);
            input.extend(codes.iter().map(|&c| c as i64));
        }
        let tensor = Tensor::from_array((vec![states.len() as i64, len as i64], input))
            .map_err(|e| LearnError::Onnx(e.to_string()))?;
        let m = &self.loaded.manifest.model;
        let mut session = self.session.lock().expect("ONNX session mutex poisoned");
        let outputs = session
            .run(ort::inputs![m.input.as_str() => tensor])
            .map_err(|e| LearnError::Onnx(e.to_string()))?;
        let (_, values) = outputs[m.output.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| LearnError::Onnx(e.to_string()))?;
        if values.len() != states.len() {
            return Err(LearnError::Onnx(format!(
                "network returned {} values for {} states",
                values.len(),
                states.len()
            )));
        }
        out.extend_from_slice(values);
        Ok(())
    }

    /// Batched estimates with errors surfaced (the `Heuristic` methods panic instead).
    pub fn try_estimate_batch(
        &self,
        states: &[P::State],
        out: &mut Vec<f32>,
    ) -> Result<(), LearnError> {
        out.clear();
        out.reserve(states.len());
        for chunk in states.chunks(MAX_CHUNK) {
            self.infer(chunk, out)?;
        }
        for (h, s) in out.iter_mut().zip(states) {
            if self.puzzle.is_solved(s) {
                *h = 0.0;
            }
        }
        Ok(())
    }
}

impl<P: Encodable> Heuristic<P> for OnnxHeuristic<P> {
    fn name(&self) -> String {
        self.loaded.manifest.name.clone()
    }

    fn estimate(&self, state: &P::State) -> u32 {
        let mut out = Vec::with_capacity(1);
        Heuristic::<P>::estimate_batch_real(self, std::slice::from_ref(state), &mut out);
        out[0].max(0.0).round() as u32
    }

    fn estimate_batch(&self, states: &[P::State], out: &mut Vec<u32>) {
        let mut real = Vec::with_capacity(states.len());
        Heuristic::<P>::estimate_batch_real(self, states, &mut real);
        out.clear();
        out.extend(real.into_iter().map(|h| h.max(0.0).round() as u32));
    }

    fn estimate_batch_real(&self, states: &[P::State], out: &mut Vec<f32>) {
        if let Err(e) = self.try_estimate_batch(states, out) {
            panic!(
                "learned heuristic {} failed: {e}",
                self.loaded.manifest.name
            );
        }
    }

    fn is_admissible(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpss_core::search::{BatchWeightedAStar, SearchLimits, Solver};
    use gpss_core::Puzzle;
    use gpss_puzzles::cube2::Cube2;

    /// The shipped network; tests skip (with a message) when it is absent.
    fn shipped() -> Option<PathBuf> {
        let p =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../weights/cube2/cube2-davi-v1.json");
        if p.exists() {
            Some(p)
        } else {
            eprintln!("skipping: {} not found", p.display());
            None
        }
    }

    #[test]
    fn shipped_weights_load_and_estimate() {
        let Some(path) = shipped() else { return };
        let puzzle = Cube2::new();
        let h = OnnxHeuristic::load(puzzle.clone(), &path, Some(2)).expect("shipped weights load");
        assert!(!Heuristic::<Cube2>::is_admissible(&h));
        assert_eq!(h.manifest().encoding.key(), "cube2/corner-cubie/v1");
        let goal = puzzle.solved();
        let neighbours: Vec<_> = puzzle
            .moves()
            .iter()
            .map(|&m| puzzle.apply(&goal, m))
            .collect();
        let mut out = Vec::new();
        h.try_estimate_batch(std::slice::from_ref(&goal), &mut out)
            .unwrap();
        assert_eq!(out, vec![0.0]);
        h.try_estimate_batch(&neighbours, &mut out).unwrap();
        assert_eq!(out.len(), neighbours.len());
        for v in &out {
            assert!((v - 1.0).abs() < 0.5, "depth-1 estimate {v}");
        }
    }

    #[test]
    fn shipped_weights_solve_with_bwas() {
        let Some(path) = shipped() else { return };
        let puzzle = Cube2::new();
        let h = OnnxHeuristic::load(puzzle.clone(), &path, Some(2)).unwrap();
        let solver = BatchWeightedAStar::new(&h, 0.6, 100);
        let scramble = gpss_puzzles::cube::parse_sequence("R U2 F' R2 U F2 R' U' F R2 U").unwrap();
        let (start, _) = puzzle.regrip(&puzzle.scrambled(&scramble));
        let sol = solver
            .solve(&puzzle, &start, &SearchLimits::default())
            .unwrap();
        assert!(!sol.optimal);
        assert!(puzzle.is_solved(&puzzle.apply_all(&start, sol.moves.iter())));
    }

    #[test]
    fn runtime_library_resolves_a_path() {
        assert!(!runtime_library().as_os_str().is_empty());
    }
}
