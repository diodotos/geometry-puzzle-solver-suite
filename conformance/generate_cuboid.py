#!/usr/bin/env python3
"""Generate ``corpus/cuboid.jsonl``: conformance cases for generic AxBxC cubes and cuboids.

Distances are in the single-layer metric that ``gpss`` uses for ``--puzzle AxBxC``:
every turn of one layer (outer face or inner slice, quarter or half) costs 1, and a
position counts as solved when every face is a single colour (orientation is free).

Exact optimal lengths come from a breadth-first ball of radius R around the solved
state, built with the independent reference model (``reference/cube_model.py``). Moves
conjugated by a whole-puzzle rotation are moves again, so the distance of ``s`` up to
rotation is ``min over rotations g of ball[g(s)]``. Scrambles farther than R get
``optimal_length: null`` (checked for correctness only).

Usage:  python3 conformance/generate_cuboid.py [--seed 2026] [--per-depth 3]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "reference"))
from cube_model import CubeModel  # noqa: E402

# (puzzle name, CubeModel dims (R-L, F-B, U-D), BFS radius)
PUZZLES = [
    ("2x2x2", (2, 2, 2), 4),
    ("3x3x2", (3, 3, 2), 4),
    ("2x2x3", (2, 2, 3), 4),
    ("2x3x4", (2, 3, 4), 5),
    ("3x3x3", (3, 3, 3), 3),
    ("4x4x4", (4, 4, 4), 3),
]

ROTATIONS = {"x", "y", "z"}


def layer_tokens(model: CubeModel) -> list[str]:
    return sorted(t for t in model.moves if t.rstrip("2'") not in ROTATIONS)


def ball(model: CubeModel, tokens: list[str], radius: int) -> dict[bytes, int]:
    dist = {model.solved: 0}
    frontier = [model.solved]
    for d in range(1, radius + 1):
        nxt = []
        for s in frontier:
            for t in tokens:
                c = model.apply(s, [t])
                if c not in dist:
                    dist[c] = d
                    nxt.append(c)
        frontier = nxt
    return dist


def distance(model: CubeModel, rotations, dist: dict[bytes, int], state: bytes):
    found = [dist[r] for r in (bytes(state[i] for i in perm) for _, perm in rotations) if r in dist]
    return min(found) if found else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--per-depth", type=int, default=3)
    ap.add_argument("--random", type=int, default=2, help="long scrambles per puzzle (null optimal)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "corpus" / "cuboid.jsonl")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    cases = []

    for name, dims, radius in PUZZLES:
        model = CubeModel(dims)
        tokens = layer_tokens(model)
        rotations = model.whole_cube_rotations()
        dist = ball(model, tokens, radius)
        seen = set()
        for depth in range(1, radius + 1):
            made = attempts = 0
            while made < args.per_depth and attempts < 2000:
                attempts += 1
                scramble = [rng.choice(tokens) for _ in range(depth)]
                state = model.apply(model.solved, scramble)
                d = distance(model, rotations, dist, state)
                if d != depth or state in seen:
                    continue
                seen.add(state)
                cases.append({
                    "id": f"{name}-d{depth}-{made}",
                    "puzzle": name,
                    "scramble": " ".join(scramble),
                    "expect": "solve",
                    "tags": ["cuboid", name, "depth", f"d{depth}"],
                    "optimal_length": depth,
                })
                made += 1
        for i in range(args.random):
            scramble = [rng.choice(tokens) for _ in range(radius + 3)]
            cases.append({
                "id": f"{name}-random-{i}",
                "puzzle": name,
                "scramble": " ".join(scramble),
                "expect": "solve",
                "tags": ["cuboid", name, "random"],
                "optimal_length": distance(model, rotations, dist, model.apply(model.solved, scramble)),
            })
        cases.append({"id": f"{name}-solved", "puzzle": name, "scramble": "", "expect": "solve",
                      "tags": ["cuboid", name, "edge"], "optimal_length": 0})
        # Turning every layer of one axis the same way is a whole-puzzle rotation.
        x_layers = dims[0]
        rotation = " ".join(("R2" if k == 1 else f"{k}R2") for k in range(1, x_layers + 1))
        assert model.is_solved_up_to_rotation(model.apply(model.solved, rotation))
        cases.append({"id": f"{name}-rotation", "puzzle": name, "scramble": rotation, "expect": "solve",
                      "tags": ["cuboid", name, "edge"], "optimal_length": 0})
        invalid = [f"{dims[0] + 1}R2", "0R2", "X", "R3", "Rw"]
        if "R" not in model.moves:
            invalid.append("R")
        for tok in invalid:
            cases.append({"id": f"{name}-invalid-{tok}", "puzzle": name, "scramble": tok, "expect": "reject",
                          "tags": ["cuboid", name, "edge", "invalid"]})

    args.out.write_text("".join(json.dumps(c) + "\n" for c in cases))
    print(f"wrote {len(cases)} cases to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
