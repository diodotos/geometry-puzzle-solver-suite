"""Training data produced by the Rust binary (``gpss scramble-data``).

The format is defined in ``crates/gpss-learn/src/data.rs``: one JSON header line, then
raw ``u8`` arrays in header order. Rust owns the puzzle and its encoding; this module only
parses bytes, so it works unchanged for any puzzle that implements ``Encodable``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

DATA_FORMAT = "gpss-davi-data"
DATA_FORMAT_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DaviBatch:
    header: dict
    states: np.ndarray  # [N, L] uint8
    children: np.ndarray  # [N, M, L] uint8
    child_is_goal: np.ndarray  # [N, M] bool
    is_goal: np.ndarray  # [N] bool
    scramble_depth: np.ndarray  # [N] uint8

    @property
    def encoding(self) -> dict:
        return self.header["encoding"]

    @property
    def moves(self) -> List[str]:
        return self.header["moves"]


def parse(blob: bytes) -> DaviBatch:
    """Parse the bytes written by ``DaviBatch::write`` in Rust."""
    nl = blob.index(b"\n")
    header = json.loads(blob[:nl].decode("utf-8"))
    if header.get("format") != DATA_FORMAT:
        raise ValueError(f"not gpss DAVI data (format={header.get('format')!r})")
    if header.get("format_version", 0) > DATA_FORMAT_VERSION:
        raise ValueError(f"data format_version {header['format_version']} is newer than supported")
    arrays: Dict[str, np.ndarray] = {}
    offset = nl + 1
    for spec in header["arrays"]:
        if spec["dtype"] != "u8":
            raise ValueError(f"unsupported dtype {spec['dtype']}")
        size = int(np.prod(spec["shape"]))
        arrays[spec["name"]] = np.frombuffer(blob, dtype=np.uint8, count=size, offset=offset).reshape(spec["shape"])
        offset += size
    if offset != len(blob):
        raise ValueError(f"trailing or missing bytes: header describes {offset}, got {len(blob)}")
    return DaviBatch(
        header=header,
        states=arrays["states"],
        children=arrays["children"],
        child_is_goal=arrays["child_is_goal"].astype(bool),
        is_goal=arrays["is_goal"].astype(bool),
        scramble_depth=arrays["scramble_depth"],
    )


def find_gpss(explicit: Optional[str] = None) -> str:
    """Locate the ``gpss`` binary: argument, ``$GPSS_BIN``, the repo's release build, PATH."""
    for candidate in (explicit, os.environ.get("GPSS_BIN"), str(REPO_ROOT / "target" / "release" / "gpss")):
        if candidate and Path(candidate).is_file():
            return candidate
    found = shutil.which("gpss")
    if found:
        return found
    raise FileNotFoundError(
        "gpss binary not found; run `cargo build --release` or pass --gpss-bin / set GPSS_BIN"
    )


class RustDataSource:
    """Generates DAVI batches by running ``gpss scramble-data``."""

    def __init__(self, puzzle: str, min_scramble: int, max_scramble: int, gpss_bin: Optional[str] = None):
        self.puzzle = puzzle
        self.min_scramble = min_scramble
        self.max_scramble = max_scramble
        self.gpss_bin = find_gpss(gpss_bin)

    def command(self, count: int, seed: int) -> List[str]:
        return [
            self.gpss_bin, "scramble-data",
            "--puzzle", self.puzzle,
            "--count", str(count),
            "--min-scramble", str(self.min_scramble),
            "--max-scramble", str(self.max_scramble),
            "--seed", str(seed),
            "--out", "-",
        ]

    def sample(self, count: int, seed: int) -> DaviBatch:
        out = subprocess.run(self.command(count, seed), check=True, stdout=subprocess.PIPE).stdout
        return parse(out)

    def encoding(self) -> dict:
        """The puzzle's encoding spec (from a one-state sample)."""
        return self.sample(1, 0).encoding
