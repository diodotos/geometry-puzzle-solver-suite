# DeepCubeA: learned cost-to-go + batch weighted A*

Status: **implemented** for the 2x2x2 (Phase C2). Workflow: [`../training.md`](../training.md);
results: [`../benchmarks/cube2-learned.md`](../benchmarks/cube2-learned.md). Primary source:
Agostinelli et al., *Solving the Rubik's cube with deep reinforcement learning and
search*, Nature Machine Intelligence 2019 [Ago19].

## The idea

Instead of hand-designing an admissible heuristic, learn an approximation `J_theta(s)` of
the true distance to the goal, then use it inside a search that tolerates an inadmissible
heuristic.

### Training: deep approximate value iteration (DAVI)

1. Sample training states by scrambling the goal `k ~ Uniform{1..K}` random moves
   (`K = 30` for the 3x3x3 in the paper). States near the goal are well covered, which is
   where value information starts.
2. Compute targets with a one-step Bellman lookahead using a periodically-updated copy of
   the network: `y(s) = min_a [1 + J_theta'(A(s, a))]`, and `y(goal) = 0`.
3. Regress `J_theta(s)` to `y(s)` (mean squared error). Update `theta'` from `theta` when
   the loss falls below a threshold.

Values propagate outward from the goal one step per target update, so no solved
trajectories or human data are ever needed - only the puzzle's move function.

### Network

Fully connected: one-hot sticker encoding -> two hidden layers (5000, 1000) -> four residual
blocks of width 1000 -> scalar output, batch-norm and ReLU throughout (the paper's 3x3x3
configuration). For the 2x2x2 a much smaller network is enough.

### Search: batch weighted A* (BWAS)

Weighted A* with `f(s) = lambda * g(s) + J_theta(s)`, expanding the `N` lowest-`f` nodes
per iteration so their children can be evaluated as one GPU batch. `lambda < 1` and
larger `N` trade solution length for speed. The paper reports optimal solutions on 60.3%
of a 3x3x3 test set, with a mean length close to optimal (see the paper for exact figures
and settings).

## How it maps onto this codebase (ADR 0005)

| DeepCubeA piece | gpss component |
|-----------------|----------------|
| puzzle environment | any `Puzzle` + `gpss_core::Encodable` (fixed-length categorical encoding) |
| `J_theta` | `gpss_learn::OnnxHeuristic<P>: Heuristic<P>` overriding `estimate_batch_real`, `is_admissible = false` |
| BWAS | `gpss_core::search::BatchWeightedAStar<H>: Solver<P>` |
| training data | `gpss scramble-data` (Rust, `gpss_learn::data`) consumed by `training/gpss_train/davi.py` |
| DAVI + network | `training/gpss_train/{davi,model}.py` |
| weights | `weights/<puzzle>/<name>.onnx` + `.json` manifest (encoding spec, SHA-256, arch, training config, evaluation; ADR 0008) |
| evaluation | `gpss evaluate` (`gpss_learn::eval`), exact against the full 2x2x2 distance table |

## Evaluation plan

The 2x2x2 has an exact distance table, so the learned heuristic can be evaluated
*exactly*: mean absolute error of `J_theta` by true depth, rate of over-estimation, and
BWAS solution length versus optimal on the conformance corpus. That validates the full
pipeline cheaply on CPU before spending GPU time on the 3x3x3.

## Compute

The paper trained its 3x3x3 network on GPUs for about 36 hours (see paper for hardware).
2x2x2 training is CPU-feasible (the shipped network: about two hours on 8 CPU cores, see
the benchmark report). 3x3x3 training will use the AWS/GCP launchers (ADR 0009).
