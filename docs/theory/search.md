# Search algorithms

All solvers live in `crates/gpss-core/src/search/` and work on any `Puzzle`. A puzzle is
an implicit graph: vertices are states, edges are moves, every edge costs 1 (half-turn
metric for cubes). Solving = finding a shortest (or short) path from the scrambled state to
the goal.

## Notation

- `b` - effective branching factor after move pruning (2x2x2 with `U R F`: 6; 3x3x3 with
  18 face turns: about 13.35).
- `d` - optimal solution length.
- `h(s)` - heuristic estimate of the distance from `s` to the goal; *admissible* if it
  never overestimates.

## Breadth-first search (`Bfs`)

Expands states in order of depth, remembering every state seen. Optimal; time and memory
`O(b^d)`. With `can_follow` pruning and a seen-set it visits each state once. It is the
baseline and the cheapest way to verify the others on small puzzles. On the 2x2x2 a
depth-11 (antipodal) scramble touches most of the 3.67M-state space: about 2.5 s and
300 MB per solve in this implementation.

## A* (`AStar`) [HNR68]

Best-first on `f(s) = g(s) + h(s)`. With an admissible (here also consistent) heuristic
the first time the goal is popped, its path is optimal. Memory is the open + closed set,
so A* is only as good as its heuristic lets it be small. Implementation notes:

- Priorities are fixed-point integers so the heap is deterministic and weights are exact.
- Ties on `f` are broken towards larger `g` (deeper nodes), which reaches the goal sooner
  without affecting optimality.
- Stale heap entries are skipped by comparing with the best known `g` per state.

### Weighted A* [Pohl70]

`f = g + w * h` with `w > 1`. Solutions are at most `w` times optimal for admissible `h`,
usually far fewer nodes are expanded. `AStar::weighted(h, w)` reports `optimal = false`
whenever `w != 1`. DeepCubeA's batch weighted A* [Ago19] is the same idea written as
`f = lambda * g + h`, plus expanding `N` nodes at once so a neural heuristic can be
evaluated in batches.

## Batch weighted A* (`BatchWeightedAStar`) [Ago19]

The search half of DeepCubeA. Priority `f = lambda * g + h` with real-valued `h`
(`Heuristic::estimate_batch_real`). Each iteration pops the `N` lowest-`f` open nodes,
tests each for the goal, expands them all, and evaluates every new child with **one**
heuristic call, so a neural network runs on batches instead of single states.

- `lambda = 1, N = 1` is A*; `lambda < 1` is greedier (fewer nodes, longer solutions).
- The goal is tested when a node is popped, so with `lambda = 1` and an admissible
  heuristic the first goal popped is optimal for any `N` (the batch only changes the order
  in which equal-or-better nodes are expanded). `Solution::optimal` is set only in that
  case; with a learned heuristic it is always `false`.
- Negative estimates are clamped to 0; NaN is treated as +infinity.
- Duplicates are detected with a best-`g` table, as in `AStar`.

CI runs `bwas --lambda 1` with the `dual` heuristic on the full corpus and checks optimal
lengths. Choosing `lambda`/`N` for the learned 2x2x2 net: see
[`../benchmarks/cube2-learned.md`](../benchmarks/cube2-learned.md).

## IDA* (`IdaStar`) [Korf85]

Depth-first search bounded by an `f` threshold; when a pass fails, the threshold becomes
the smallest `f` that exceeded it. Optimal with an admissible heuristic, memory `O(d)`
(just the current path), at the cost of re-expanding shallow nodes on every iteration
(a geometric series, so asymptotically cheap when `b > 1`). It does not detect
transpositions, which is why move pruning (`can_follow`) matters so much.

IDA* is the algorithm of choice for the 3x3x3, whose 4.3 x 10^19 states rule out
anything that stores the frontier: Korf's optimal solver [Korf97] is IDA* plus pattern
databases (see [`pattern-databases.md`](pattern-databases.md)).

On the 2x2x2 with the `dual` heuristic, IDA* solves antipodal scrambles in about 5 ms
using ~9 stored nodes, versus ~20 ms for A* with the same heuristic (and its larger
memory) - per-node work is tiny and IDA* has no hash table.

## Limits and statistics

Every solver honours `SearchLimits { max_depth, max_expanded, timeout }` and returns
`SearchStats { expanded, generated, peak_stored, iterations, elapsed }`:

| field | BFS | A* | IDA* | BWAS |
|-------|-----|----|------|------|
| `expanded` | states dequeued | states popped (non-stale) | nodes whose children were generated, all iterations | states popped (non-stale) |
| `generated` | children after pruning | children after pruning | children after pruning | children after pruning |
| `peak_stored` | arena size | max(open + closed) | max path length | max(open + closed) |
| `iterations` | 1 | 1 | thresholds tried | batches (= heuristic calls) |

These counters are what the benchmark reports compare across implementations, because
they are machine-independent.

## Choosing a solver

| situation | use |
|-----------|-----|
| small puzzle, verifying correctness | `bfs` |
| exact distance table fits in memory | `idastar --heuristic perfect` (expands exactly `d` nodes) |
| good admissible heuristic, optimal needed | `idastar` (memory-bounded) or `astar` (fewer re-expansions) |
| speed over optimality | `astar --weight 2` (later: two-phase) |
| learned heuristic (DeepCubeA) | `bwas --weights <manifest> --lambda 0.6 --batch-size 100` |
