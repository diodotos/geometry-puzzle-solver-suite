//! # gpss-learn
//!
//! The learned-heuristic half of the suite, following DeepCubeA (Agostinelli, McAleer,
//! Shmakov & Baldi, *Solving the Rubik's cube with deep reinforcement learning and
//! search*, Nature Machine Intelligence 2019, `[Ago19]` in `docs/references.md`).
//!
//! * [`data`] generates DAVI training data from any [`gpss_core::Encodable`] puzzle:
//!   random backward scrambles, every one-move successor, and goal masks, already
//!   encoded. The Python trainer (`training/`) consumes this format and never
//!   re-implements a puzzle, so Rust stays the single source of truth for both the move
//!   function and the state encoding (ADR 0007).
//! * [`manifest`] reads and validates weight manifests: a JSON file next to each `.onnx`
//!   network, pinning the puzzle, the exact [`gpss_core::EncodingSpec`] and the network's
//!   SHA-256, so a network can never be loaded for the wrong puzzle or encoding.
//! * [`onnx`] (feature `onnx`) runs a network through ONNX Runtime as a
//!   [`gpss_core::Heuristic`] with batched inference, for use with
//!   [`gpss_core::search::BatchWeightedAStar`].
//! * [`eval`] measures a heuristic and a solver against exact distances (for puzzles
//!   small enough to enumerate, like the 2x2x2).
//!
//! The search itself (BWAS) lives in `gpss-core`, because it works with any heuristic.

pub mod data;
pub mod error;
pub mod eval;
pub mod manifest;
#[cfg(feature = "onnx")]
pub mod onnx;

pub use error::LearnError;
pub use manifest::Manifest;
#[cfg(feature = "onnx")]
pub use onnx::OnnxHeuristic;

/// Whether this build can run ONNX networks (the `onnx` feature).
pub const ONNX_ENABLED: bool = cfg!(feature = "onnx");
