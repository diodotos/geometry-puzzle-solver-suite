#!/usr/bin/env python3
"""Generate the 2x2x2 conformance corpus (``corpus/cube2.jsonl``).

The generator is independent of every solver in this repository: it uses the
sticker-level reference model in ``reference/cube_model.py`` and runs an exhaustive
breadth-first search over the whole 2x2x2 state space (3,674,160 states, modulo
whole-cube rotation) to obtain the exact optimal half-turn-metric distance of every
state. The resulting depth histogram is checked against the published distribution
before any case is written, so a bug in the reference model cannot silently produce
a wrong corpus.

Usage:  python3 conformance/generate_cube2.py [--seed 2024] [--per-depth 15]
Runtime: roughly 25 seconds and ~0.5 GB RAM (pure Python, stdlib only).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from operator import itemgetter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "reference"))
from cube_model import (  # noqa: E402
    FACE_NORMALS,
    NORMAL_TO_FACE,
    ROTATION_AXES,
    CubeModel,
    invert_sequence,
    rotate_cw,
)

# Number of 2x2x2 positions at each optimal distance in the half-turn metric
# (God's number for the 2x2x2 is 11). Source: well-established exhaustive
# enumerations, e.g. https://www.jaapsch.net/puzzles/cube2.htm
KNOWN_HTM_DISTRIBUTION = [1, 9, 54, 321, 1847, 9992, 50136, 227536, 870072, 1887748, 623800, 2644]

# Turning only U, R and F never moves the DBL corner, so these 9 moves reach every
# state exactly once modulo whole-cube rotation and preserve optimal HTM distance.
SEARCH_MOVES = [f + s for f in "URF" for s in ("", "2", "'")]


def bfs(model: CubeModel):
    getters = {m: itemgetter(*model.moves[m]) for m in SEARCH_MOVES}
    dist = {model.solved: 0}
    layers = [[model.solved]]
    while layers[-1]:
        depth = len(layers)
        nxt = []
        for state in layers[-1]:
            for g in getters.values():
                child = bytes(g(state))
                if child not in dist:
                    dist[child] = depth
                    nxt.append(child)
        layers.append(nxt)
        print(f"  depth {depth - 1}: {len(layers[-2])} states", file=sys.stderr)
    layers.pop()
    return dist, layers


class Oracle:
    """Exact optimal distance for any 2x2x2 state (orientation of the whole cube is free)."""

    def __init__(self, model: CubeModel, dist: dict):
        self.model = model
        self.dist = dist
        self.rotations = [itemgetter(*p) for _, p in model.whole_cube_rotations()]

    def canonical(self, state: bytes) -> bytes:
        for rot in self.rotations:
            s = bytes(rot(state))
            if s in self.dist:
                return s
        raise AssertionError("state not reachable; reference model is inconsistent")

    def distance(self, scramble: str) -> int:
        return self.dist[self.canonical(self.model.apply(self.model.solved, scramble))]

    def optimal_path_to_solved(self, state: bytes) -> list:
        path = []
        d = self.dist[state]
        while d:
            for mv in SEARCH_MOVES:
                child = self.model.apply(state, [mv])
                if self.dist.get(child) == d - 1:
                    path.append(mv)
                    state, d = child, d - 1
                    break
        return path


def reorient(tokens: list, rng: random.Random) -> list:
    """Rewrite a move sequence as seen from a random whole-cube orientation.

    Relabelling faces by a rotation conjugates the scramble, which leaves the optimal
    distance unchanged; it is used so the corpus exercises all 18 face turns."""
    mapping = {f: f for f in FACE_NORMALS}
    for _ in range(rng.randint(1, 4)):
        axis = FACE_NORMALS[ROTATION_AXES[rng.choice("xyz")]]
        mapping = {f: NORMAL_TO_FACE[rotate_cw(FACE_NORMALS[g], axis)] for f, g in mapping.items()}
    return [mapping[t[0]] + t[1:] for t in tokens]


def random_scramble(rng: random.Random, length: int) -> list:
    out, last = [], None
    for _ in range(length):
        face = rng.choice([f for f in "URFDLB" if f != last])
        out.append(face + rng.choice(("", "2", "'")))
        last = face
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--per-depth", type=int, default=15)
    ap.add_argument("--random", type=int, default=60, help="number of random-move scrambles")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "corpus" / "cube2.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    model = CubeModel(2)
    t0 = time.time()
    print("Enumerating the 2x2x2 state space...", file=sys.stderr)
    dist, layers = bfs(model)
    histogram = [len(layer) for layer in layers]
    if histogram != KNOWN_HTM_DISTRIBUTION:
        print(f"FATAL: histogram {histogram} != published {KNOWN_HTM_DISTRIBUTION}", file=sys.stderr)
        return 1
    print(f"State space verified ({len(dist)} states, {time.time() - t0:.1f}s)", file=sys.stderr)
    oracle = Oracle(model, dist)

    cases = []

    def add(case_id, scramble, tags, expect="solve"):
        case = {"id": case_id, "puzzle": "cube2", "scramble": scramble, "expect": expect, "tags": tags}
        if expect == "solve":
            case["optimal_length"] = oracle.distance(scramble)
        cases.append(case)

    # 1. Exact-depth cases: uniformly sampled states at every distance 1..11.
    for depth in range(1, len(layers)):
        layer = sorted(layers[depth])
        picks = layer if len(layer) <= args.per_depth else rng.sample(layer, args.per_depth)
        for i, state in enumerate(picks):
            tokens = invert_sequence(oracle.optimal_path_to_solved(state))
            if i % 2:
                tokens = reorient(tokens, rng)
            scramble = " ".join(tokens)
            band = "short" if depth <= 5 else ("mid" if depth <= 9 else "top")
            add(f"depth{depth:02d}-{i:02d}", scramble, ["depth", f"d{depth}", band])
            assert cases[-1]["optimal_length"] == depth

    # 2. Random-move scrambles of assorted lengths (WCA-style: no repeated face).
    for i in range(args.random):
        length = rng.choice([1, 2, 3, 5, 8, 11, 15, 20, 25, 30])
        add(f"random-{i:02d}", " ".join(random_scramble(rng, length)), ["random"])

    # 3. Edge cases. These pin down input handling, not search.
    add("edge-empty", "", ["edge", "identity"])
    add("edge-cancel-pair", "R R'", ["edge", "identity"])
    add("edge-quarter-x4", "U U U U", ["edge", "identity"])
    add("edge-sexy-x6", " ".join(["R U R' U'"] * 6), ["edge", "identity"])
    add("edge-whole-cube-equivalent", "R L'", ["edge", "identity"])
    add("edge-extra-whitespace", "  R   U  F'  ", ["edge", "whitespace"])
    add("edge-long-200", " ".join(random_scramble(rng, 200)), ["edge", "long"])
    add("edge-all-18-moves", "U U' U2 D D' D2 R R' R2 L L' L2 F F' F2 B B' B2", ["edge"])
    for bad in ["X", "R3", "r", "U4", "R,U", "Rw"]:
        add(f"edge-invalid-{bad.replace(',', 'comma')}", bad, ["edge", "invalid"], expect="reject")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for case in cases:
            fh.write(json.dumps(case) + "\n")
    print(f"Wrote {len(cases)} cases to {args.out} ({time.time() - t0:.1f}s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
