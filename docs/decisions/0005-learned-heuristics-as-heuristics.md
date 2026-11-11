# 0005. Learned models plug in as heuristics; PyTorch training, ONNX inference

**Status:** Accepted, implemented in Phase C2 (see ADR 0007 for the encoding and weights
format, ADR 0008 for launchers, `docs/training.md` for the workflow)

## Context

DeepCubeA [Ago19] is often described as a separate "deep RL solver", but structurally it
is two parts: a learned cost-to-go function `J(s)`, trained by approximate value iteration
on scrambled states, and batch weighted A* (BWAS), which expands the best `N` frontier
nodes per step with priority `f = lambda * g + J`. Training wants a mature autodiff stack
and GPUs; inference wants to run inside the Rust search loop.

## Decision

- The network is a `Heuristic<P>` implementation that overrides `estimate_batch`; it is
  never admissible.
- BWAS is a `Solver<P>` in `gpss-core`, usable with any heuristic.
- Training is Python/PyTorch (`training/`), driven by a puzzle-generic data generator that
  needs only a state encoding and random scrambles. Weights are exported to ONNX and run
  from Rust via ONNX Runtime.
- Weight files carry metadata (puzzle spec, encoding version, architecture) and are
  refused for a mismatched puzzle.
- 2x2x2 weights are shipped in the repo (and evaluated exactly against the perfect table);
  3x3x3 weights only once a credible GPU training run exists.
- Remote training goes through a pluggable launcher (`local`, then AWS and GCP backends),
  with a cost/time estimate and explicit confirmation before anything is launched.

## Consequences

- The learned solver reuses all puzzle, search, benchmark and conformance infrastructure.
- Puzzles need a fixed-length tensor encoding to be trainable; this becomes an optional
  trait (`Encodable`) rather than a requirement of `Puzzle`.
- Python is a training-time dependency only; the solver binary stays pure Rust plus the
  ONNX Runtime library when the learned feature is enabled.
