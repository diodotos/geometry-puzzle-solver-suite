//! Fixed-length state encodings for learned heuristics.
//!
//! A learned cost-to-go function (DeepCubeA [Ago19], `docs/theory/deepcubea.md`) needs
//! every state as a fixed-length tensor. [`Encodable`] is the optional trait a puzzle
//! implements to become trainable: it maps a state to `len` small integers, each a
//! category in `0..classes`. A network one-hot encodes them (or, equivalently, uses an
//! embedding lookup), so the encoding stays compact on disk and over the Python/Rust
//! boundary.
//!
//! The [`EncodingSpec`] is the *identity* of an encoding. Training data, weight files and
//! the inference code all carry it, and a weights file is refused unless its spec is
//! exactly equal to the puzzle's (see ADR 0008). Whenever `encode` changes in any way
//! that alters its output, bump [`EncodingSpec::version`].

use crate::puzzle::Puzzle;

/// Identity and shape of a state encoding.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EncodingSpec {
    /// Puzzle specification the encoding belongs to, e.g. `"cube2"`. For parameterised
    /// puzzle families this must include the parameters (e.g. `"cube:4x4x4"`), because a
    /// network trained on one member is meaningless for another.
    pub puzzle: String,
    /// Name of the encoding scheme, e.g. `"corner-cubie"`.
    pub scheme: String,
    /// Bumped whenever `encode` changes its output for any state.
    pub version: u32,
    /// Number of categorical features per state.
    pub len: usize,
    /// Every feature is in `0..classes`.
    pub classes: u16,
}

impl EncodingSpec {
    /// A short stable key such as `cube2/corner-cubie/v1`, used in file names and logs.
    pub fn key(&self) -> String {
        format!("{}/{}/v{}", self.puzzle, self.scheme, self.version)
    }

    /// Width of the equivalent one-hot vector (`len * classes`).
    pub fn one_hot_width(&self) -> usize {
        self.len * self.classes as usize
    }
}

/// A puzzle whose states can be encoded as fixed-length categorical vectors.
///
/// Implementations must guarantee that `encode` is a deterministic function of the state,
/// writes exactly `encoding().len` values, each `< encoding().classes`, and that distinct
/// states the search can reach map to distinct encodings (otherwise the network cannot
/// tell them apart).
pub trait Encodable: Puzzle {
    /// The encoding's identity and shape.
    fn encoding(&self) -> EncodingSpec;

    /// Write the encoding of `state` into `out` (`out.len() == encoding().len`).
    fn encode(&self, state: &Self::State, out: &mut [u8]);
}
