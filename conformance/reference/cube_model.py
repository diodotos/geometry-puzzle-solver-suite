"""Independent reference model of an NxNxN twisty cube (or AxBxC cuboid) at the sticker level.

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
* ``kX`` (k >= 2) turns only the k-th layer from face X, in X's direction (SiGN
  single-slice notation); ``2R`` on a 3x3x3 is the middle slice turned like R.
* Cuboids: ``CubeModel((a, b, c))`` has a layers on the R-L axis, b on F-B and c on U-D
  (height last, so (3, 3, 2) is the domino). About an axis whose cross-section is not
  square only half turns exist; the quarter-turn tokens for it are simply not defined.
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
    """Sticker-level model of an NxNxN cube or AxBxC cuboid. States are ``bytes`` of face indices."""

    def __init__(self, n):
        a, b, c = (n, n, n) if isinstance(n, int) else n
        if min(a, b, c) < 2:
            raise ValueError("every dimension must be >= 2")
        self.n = n
        # half-extent along x (R-L), y (U-D), z (F-B), in units where layers are 2 apart
        ext = (a - 1, c - 1, b - 1)
        self.ext = ext
        stickers: List[Tuple[Vec, Vec]] = []
        for face in FACES:
            normal = FACE_NORMALS[face]
            axis = [i for i in range(3) if normal[i] != 0][0]
            for pos in product(*(range(-e, e + 1, 2) for e in ext)):
                if pos[axis] == normal[axis] * ext[axis]:
                    stickers.append((pos, normal))
        self.stickers = stickers
        self.index = {s: i for i, s in enumerate(stickers)}
        self.solved: bytes = bytes(FACES.index(NORMAL_TO_FACE[nrm]) for _, nrm in stickers)

        self.moves: Dict[str, Tuple[int, ...]] = {}
        for face, normal in FACE_NORMALS.items():
            axis = [i for i in range(3) if normal[i] != 0][0]
            square = len({ext[i] for i in range(3) if i != axis}) == 1
            for k in range(1, ext[axis] + 2):
                level = ext[axis] - 2 * (k - 1)
                name = face if k == 1 else f"{k}{face}"
                self._register(name, normal, lambda c, nm=normal, lv=level: _dot(c, nm) == lv, square)
        for rot, face in ROTATION_AXES.items():
            normal = FACE_NORMALS[face]
            axis = [i for i in range(3) if normal[i] != 0][0]
            square = len({ext[i] for i in range(3) if i != axis}) == 1
            self._register(rot, normal, lambda c: True, square)

    # -- construction helpers -------------------------------------------------
    def _layer_perm(self, axis: Vec, in_layer, quarters: int) -> Tuple[int, ...]:
        """Return ``src`` such that new_state[i] = state[src[i]]."""
        src = list(range(len(self.stickers)))
        for i, (c, nrm) in enumerate(self.stickers):
            if in_layer(c):
                for _ in range(quarters):
                    c, nrm = rotate_cw(c, axis), rotate_cw(nrm, axis)
                src[self.index[(c, nrm)]] = i
        return tuple(src)

    def _register(self, name: str, axis: Vec, in_layer, square: bool) -> None:
        self.moves[name + "2"] = self._layer_perm(axis, in_layer, 2)
        if square:
            self.moves[name] = self._layer_perm(axis, in_layer, 1)
            self.moves[name + "'"] = self._layer_perm(axis, in_layer, 3)

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
                    q = tuple(p[i] for i in self.moves.get(r) or self.moves[r + "2"])
                    if q not in found:
                        found[q] = (found[p] + " " + r).strip()
                        nxt.append(q)
            frontier = nxt
        assert len(found) in (4, 8, 24)
        return [(name, perm) for perm, name in found.items()]


def invert_sequence(tokens: Sequence[str]) -> List[str]:
    out = []
    for tok in reversed(tokens):
        base = tok.rstrip("2'")
        suffix = tok[len(base):]
        out.append(base + {"": "'", "'": "", "2": "2"}[suffix])
    return out
