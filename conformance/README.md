# Conformance suite

Language-agnostic, black-box correctness and performance checks for any solver CLI.
Python 3.10+ standard library only.

| file | purpose |
|------|---------|
| `reference/cube_model.py` | Independent sticker-level NxNxN cube (3D coordinates + rotations). Shares nothing with the solvers. |
| `generate_cube2.py` | Exhaustive BFS of the 2x2x2 space with the reference model; checks the depth histogram against the published distribution; writes the corpus. Deterministic (`--seed`). |
| `corpus/cube2.jsonl` | 233 cases: exact optimal depth 0-11, random scrambles, edge cases, invalid inputs. |
| `harness.py` | Runs an implementation per case, validates the solution with the reference model, checks optimality, measures time and peak RSS (`wait4`), prints a per-depth table, optional JSON report. |
| `known_deviations.json` | Documented, expected failures per implementation (reported as XFAIL / XPASS). |

## Case format

```json
{"id": "depth11-03", "puzzle": "cube2", "scramble": "R U2 F' ...", "expect": "solve",
 "tags": ["depth", "d11", "top"], "optimal_length": 11}
```

`expect` is `solve` or `reject`. For `solve`, a returned solution must leave the cube
solved up to whole-cube rotation; optimal solvers must match `optimal_length` exactly, and
no solver may beat it. For `reject`, the process must exit non-zero without crashing and
without printing a solution.

## Usage

```bash
python3 conformance/harness.py --impl gpss --solver idastar --heuristic dual
python3 conformance/harness.py --impl gpss --solver bfs --tags short,edge -v
python3 conformance/harness.py --impl legacy-cpp --solver astardual --max-depth 8 --report out.json
```

## Adding an implementation

Subclass `Implementation` in `harness.py`: `build()`, `command(case, args)`, and
`parse(result)` (set `result.solution` and optionally `result.reported_time_s`). The Rust
adapter expects `gpss solve --format json` to print `{"solution": "...", "stats":
{"elapsed_s": ...}}`.
