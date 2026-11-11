# 0007. Rust owns puzzle encodings; weights are pinned to them by a manifest

**Status:** Accepted

## Context

DAVI training (Python) and BWAS inference (Rust) must agree exactly on two things: the
puzzle's move function (to compute Bellman targets over successors) and the mapping from a
state to the network input. If Python re-implemented the puzzle, every puzzle would exist
twice and any drift would silently produce a network that is wrong in Rust while looking
fine in Python. Weight files are also easy to mix up once several puzzles and encodings
exist.

## Decision

- **One definition.** A puzzle becomes trainable by implementing the additive
  `gpss_core::Encodable` trait (`encoding() -> EncodingSpec`, `encode(state, &mut [u8])`).
  The encoding is categorical: `len` features, each in `0..classes`; the network does the
  one-hot expansion.
- **Python never sees a puzzle.** `gpss scramble-data` (Rust, `gpss_learn::data`) writes
  scrambled states, *all* their successors and goal masks, already encoded, in a small
  binary format (JSON header + raw `u8` arrays). `training/gpss_train/data.py` only parses
  bytes. A new `Encodable` puzzle is trainable with zero Python changes.
- **Versioned encodings.** `EncodingSpec { puzzle, scheme, version, len, classes }`, keyed as
  `puzzle/scheme/vN` (Cube2: `cube2/corner-cubie/v1`). Changing what `encode` produces for
  any state requires bumping `version`.
- **Manifests pin weights.** Every `.onnx` ships with a `.json` manifest
  (`gpss_learn::manifest`, format `gpss-weights` v1) recording the full `EncodingSpec`, the
  ONNX file's SHA-256, tensor names, architecture, training configuration and evaluation.
  `OnnxHeuristic::load` refuses to run unless the spec equals `puzzle.encoding()` exactly
  and the checksum matches.
- **ONNX Runtime is loaded at run time** (`ort` with `load-dynamic`, behind the `onnx` /
  `learn` features). Nothing is downloaded or linked at build time, the default build is
  unaffected, and the same binary works with any ONNX Runtime >= 1.22 (located via
  `ORT_DYLIB_PATH`, the Python `onnxruntime` package, or the system loader).

## Consequences

- Training data generation runs at Rust speed (100k 2x2x2 states with all 9 successors
  in well under a second) and is deterministic given a seed.
- Python tests can't catch an encoding bug in Rust, but nothing in Python could cause one;
  Rust unit tests cover the encoding (range, injectivity on samples) and the manifest
  check.
- Prebuilt static ONNX Runtime binaries (the `ort` default) need glibc >= 2.38; dynamic
  loading avoids that constraint, at the cost of a runtime dependency the user installs
  (`pip install onnxruntime` is enough).
