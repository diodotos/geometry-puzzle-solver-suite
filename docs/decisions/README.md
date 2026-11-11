# Architecture Decision Records

Short records of decisions that shape the code, in the
[Nygard format](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions):
context, decision, consequences. They exist so that future contributors (human or AI)
don't undo a deliberate choice by accident, and so that a decision can be revisited with
its original reasoning in view. Add a new numbered file rather than rewriting an old one;
mark superseded records as such.

| # | Decision | Status |
|---|----------|--------|
| [0001](0001-rust-clean-port.md) | Port to Rust directly into a clean design | Accepted |
| [0002](0002-puzzle-solver-separation.md) | Separate puzzles, heuristics and solvers with traits | Accepted |
| [0003](0003-black-box-conformance.md) | Prove the migration with a black-box conformance corpus | Accepted |
| [0004](0004-fixed-frame-cube2.md) | Search the 2x2x2 in a fixed frame with `U R F` moves | Accepted |
| [0005](0005-learned-heuristics-as-heuristics.md) | Learned models plug in as heuristics; PyTorch -> ONNX | Accepted |
| [0006](0006-ci-vs-local-benchmarks.md) | CI checks correctness; benchmarks run on dedicated machines | Accepted |
| [0007](0007-bench-suite.md) | One benchmark suite, driven by the conformance corpus, with counter baselines in CI | Accepted |
