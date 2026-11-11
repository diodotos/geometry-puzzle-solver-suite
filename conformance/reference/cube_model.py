"""Independent reference model of an NxNxN twisty cube at the sticker (facelet) level.

This module is deliberately written from first principles (3D geometry), shares no
code or representation with any solver in this repository, and uses only the Python
standard library. It is the ground truth the conformance harness uses to check that a
solution produced by an implementation actually solves the scrambled puzzle.

Conventions
-----------
* Axes: +x = R (right), +y = U (up), +z = F (front).
* A cubie centre has integer coordinates in {-(N-1), -(N-3), ..., N-1}; a sticker is
  identified by (cubie centre, outward unit normal).
* The face turn ``X`` (for X in U D R L F B) rotates the outer layer of face X by 90
  degrees clockwise *as seen looking directly at that face*, i.e. by -90 degrees about
  the face's outward normal. ``X'`` is the inverse and ``X2`` the half turn
  (standard WCA outer-block notation).
* Whole-cube rotations ``x``, ``y``, ``z`` follow R, U and F respectively.
"""

from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, List, Sequence, Tuple

Vec = Tuple[int, int, int]

FACE_NORMALS: Dict[str, Vec] = {
    "U": (0, 1, 0),
    "D": (0, -1, 0),
    "R": (1, 0, 0),
    "L": (-1, 0, 0),
    "F": (0, 0, 1),
    "B": (0, 0, -1),
}
NORMAL_TO_FACE: Dict[Vec, str] = {v: k for k, v in FACE_NORMALS.items()}
FACES = "URFDLB"
ROTATION_AXES: Dict[str, str] = {"x": "R", "y": "U", "z": "F"}
SUFFIX_TURNS = {"": 1, "2": 2, "'": 3}


def _cross(a: Vec, b: Vec) -> Vec:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a: Vec, b: Vec) -> int:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def rotate_cw(v: Vec, axis: Vec) -> Vec:
    """Rotate ``v`` by -90 degrees about unit ``axis`` (Rodrigues with cos=0, sin=-1)."""
    c = _cross(axis, v)
    d = _dot(axis, v)
    return (-c[0] + axis[0] * d, -c[1] + axis[1] * d, -c[2] + axis[2] * d)


class CubeModel:
    """Sticker-level model of an NxNxN cube. States are ``bytes`` of face indices."""

    def __init__(self, n: int):
        if n < 2:
            raise ValueError("cube size must be >= 2")
        self.n = n
        coords = range(-(n - 1), n, 2)
        stickers: List[Tuple[Vec, Vec]] = []
        for face in FACES:
            normal = FACE_NORMALS[face]
            axis = [i for i in range(3) if normal[i] != 0][0]
            for c in product(coords, repeat=3):
                if c[axis] == normal[axis] * (n - 1):
                    stickers.append((c, normal))
        self.stickers = stickers
        self.index = {s: i for i, s in enumerate(stickers)}
        self.solved: bytes = bytes(FACES.index(NORMAL_TO_FACE[nrm]) for _, nrm in stickers)

        self.moves: Dict[str, Tuple[int, ...]] = {}
        for face, normal in FACE_NORMALS.items():
            quarter = self._layer_perm(normal, lambda c, nm=normal: _dot(c, nm) == n - 1)
            self._register(face, quarter)
        for rot, face in ROTATION_AXES.items():
            normal = FACE_NORMALS[face]
            self._register(rot, self._layer_perm(normal, lambda c: True))

    # -- construction helpers -------------------------------------------------
    def _layer_perm(self, axis: Vec, in_layer) -> Tuple[int, ...]:
        """Return ``src`` such that new_state[i] = state[src[i]]."""
        src = list(range(len(self.stickers)))
        for i, (c, nrm) in enumerate(self.stickers):
            if in_layer(c):
                j = self.index[(rotate_cw(c, axis), rotate_cw(nrm, axis))]
                src[j] = i
        return tuple(src)

    def _register(self, name: str, quarter: Tuple[int, ...]) -> None:
        perm = quarter
        for suffix in ("", "2", "'"):
            self.moves[name + suffix] = perm
            perm = tuple(perm[i] for i in quarter)

    # -- state operations -----------------------------------------------------
    @staticmethod
    def parse(sequence: str) -> List[str]:
        return sequence.split()

    def apply(self, state: bytes, sequence: Iterable[str] | str) -> bytes:
        tokens = self.parse(sequence) if isinstance(sequence, str) else sequence
        for tok in tokens:
            try:
                perm = self.moves[tok]
            except KeyError:
                raise ValueError(f"invalid move token: {tok!r}") from None
            state = bytes(state[i] for i in perm)
        return state

    def is_solved_up_to_rotation(self, state: bytes) -> bool:
        """True iff every face shows a single colour (orientation of the whole cube is free)."""
        per_face: Dict[Vec, set] = {}
        for (_, nrm), colour in zip(self.stickers, state):
            per_face.setdefault(nrm, set()).add(colour)
        return all(len(colours) == 1 for colours in per_face.values())

    def whole_cube_rotations(self) -> List[Tuple[str, Tuple[int, ...]]]:
        """All 24 whole-cube orientations as (name, sticker permutation)."""
        identity = tuple(range(len(self.stickers)))
        found: Dict[Tuple[int, ...], str] = {identity: ""}
        frontier = [identity]
        while frontier:
            nxt = []
            for p in frontier:
                for r in ("x", "y", "z"):
                    q = tuple(p[i] for i in self.moves[r])
                    if q not in found:
                        found[q] = (found[p] + " " + r).strip()
                        nxt.append(q)
            frontier = nxt
        assert len(found) == 24
        return [(name, perm) for perm, name in found.items()]


def invert_sequence(tokens: Sequence[str]) -> List[str]:
    out = []
    for tok in reversed(tokens):
        base, suffix = tok[0], tok[1:]
        out.append(base + {"": "'", "'": "", "2": "2"}[suffix])
    return out
