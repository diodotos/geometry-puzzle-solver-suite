#!/usr/bin/env python3
"""Generate the 3x3x3 conformance corpus (``corpus/cube3.jsonl``).

Like ``generate_cube2.py`` this uses only the independent sticker-level reference model.
The 3x3x3 state space (about 4.3e19 positions) cannot be enumerated, so exact optimal
distances are computed by meet-in-the-middle for short scrambles: a breadth-first ball
of radius ``R`` around the solved cube is intersected with a ball of radius ``S``
around the scrambled state, which gives exact distances up to ``R + S`` (default 4 + 3).
The ball sizes are checked against the published half-turn-metric distance counts
before any case is written.

Cases:
* ``depth``  - random positions of exact optimal distance 1..7 (tags ``d1``..``d7``).
* ``random`` - long random scrambles; the optimal length is unknown (``null``), so only
  the returned solution's correctness is checked.
* ``known``  - positions with published optimal lengths (superflip = 20, Reid 1995).
* ``edge``   - identities, whitespace, invalid input.

Usage:  python3 conformance/generate_cube3.py [--seed 2025] [--per-depth 10]
Runtime: roughly a minute and ~1 GB RAM (pure Python, stdlib only).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "reference"))
from cube_model import CubeModel, invert_sequence  # noqa: E402

# Positions at each HTM distance 0..4 (e.g. https://www.cube20.org/ and OEIS A080601).
KNOWN_HTM_COUNTS = [1, 18, 243, 3240, 43239]
FACE_TURNS = [f + s for f in "URFDLB" for s in ("", "2", "'")]
SUPERFLIP = "U R2 F B R B2 R U2 L B2 R U' D' R2 F R' L B2 U2 F2"


def ball(model: CubeModel, start: bytes, radius: int) -> dict:
    dist = {start: 0}
    frontier = [start]
    perms = [model.moves[m] for m in FACE_TURNS]
    for d in range(1, radius + 1):
        nxt = []
        for s in frontier:
            for p in perms:
                t = bytes(s[i] for i in p)
                if t not in dist:
                    dist[t] = d
                    nxt.append(t)
        frontier = nxt
    return dist


def random_scramble(rng: random.Random, length: int) -> list:
    out, last = [], None
    for _ in range(length):
        face = rng.choice([f for f in "URFDLB" if f != last])
        out.append(face + rng.choice(("", "2", "'")))
        last = face
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--per-depth", type=int, default=10)
    ap.add_argument("--random", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "corpus" / "cube3.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    model = CubeModel(3)
    t0 = time.time()
    near = ball(model, model.solved, 4)
    counts = [0] * 5
    for d in near.values():
        counts[d] += 1
    if counts != KNOWN_HTM_COUNTS:
        print(f"FATAL: ball counts {counts} != published {KNOWN_HTM_COUNTS}", file=sys.stderr)
        return 1
    print(f"Radius-4 ball verified ({len(near)} states, {time.time() - t0:.1f}s)", file=sys.stderr)

    def distance(scramble: str):
        """Exact optimal distance if <= 7, else None."""
        state = model.apply(model.solved, scramble)
        best = None
        for s, d in ball(model, state, 3).items():
            if s in near:
                total = d + near[s]
                best = total if best is None else min(best, total)
        return best if best is not None and best <= 7 else None

    cases = []

    def add(case_id, scramble, tags, expect="solve", optimal=None):
        case = {"id": case_id, "puzzle": "cube3", "scramble": scramble, "expect": expect, "tags": tags}
        if expect == "solve":
            case["optimal_length"] = optimal
        cases.append(case)

    by_depth = {d: [] for d in range(1, 8)}
    while any(len(v) < args.per_depth for v in by_depth.values()):
        tokens = random_scramble(rng, rng.randint(1, 9))
        scramble = " ".join(tokens)
        d = distance(scramble)
        if d and len(by_depth[d]) < args.per_depth and scramble not in by_depth[d]:
            by_depth[d].append(scramble)
    for d, scrambles in by_depth.items():
        for i, s in enumerate(scrambles):
            add(f"depth{d:02d}-{i:02d}", s, ["depth", f"d{d}", "short" if d <= 5 else "mid"], optimal=d)

    for i in range(args.random):
        length = rng.choice([15, 20, 25, 30, 40])
        add(f"random-{i:02d}", " ".join(random_scramble(rng, length)), ["random"])

    add("known-superflip", SUPERFLIP, ["known", "top"], optimal=20)
    add("known-superflip-inverse", " ".join(invert_sequence(SUPERFLIP.split())), ["known", "top"], optimal=20)

    add("edge-empty", "", ["edge", "identity"], optimal=0)
    add("edge-cancel-pair", "R R'", ["edge", "identity"], optimal=0)
    add("edge-quarter-x4", "U U U U", ["edge", "identity"], optimal=0)
    add("edge-sexy-x6", " ".join(["R U R' U'"] * 6), ["edge", "identity"], optimal=0)
    add("edge-commuting-faces", "R L R' L'", ["edge", "identity"], optimal=0)
    add("edge-extra-whitespace", "  R   U  F'  ", ["edge", "whitespace"], optimal=3)
    add("edge-long-200", " ".join(random_scramble(rng, 200)), ["edge", "long"])
    add("edge-all-18-moves", " ".join(FACE_TURNS), ["edge"])
    for bad in ["X", "R3", "r", "U4", "R,U", "Rw", "M"]:
        add(f"edge-invalid-{bad.replace(',', 'comma')}", bad, ["edge", "invalid"], expect="reject")

    for c in cases:
        if c["expect"] == "solve" and c["optimal_length"] is not None and "known" not in c["tags"]:
            assert distance(c["scramble"]) == c["optimal_length"], c
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(c) + "\n" for c in cases))
    print(f"Wrote {len(cases)} cases to {args.out} ({time.time() - t0:.1f}s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
