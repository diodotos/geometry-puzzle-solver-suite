import json
import shutil

import numpy as np
import pytest

from gpss_train import data
from gpss_train.data import DaviBatch, parse


def make_blob(n=3, m=2, length=4, version=1, extra=b""):
    states = (np.arange(n * length) % 30).astype(np.uint8).reshape(n, length)
    children = (np.arange(n * m * length) % 30).astype(np.uint8).reshape(n, m, length)
    child_is_goal = np.zeros((n, m), dtype=np.uint8)
    child_is_goal[0, 1] = 1
    is_goal = np.array([0] * (n - 1) + [1], dtype=np.uint8)
    depth = (np.arange(n) % 256).astype(np.uint8)
    arrays = [("states", states), ("children", children), ("child_is_goal", child_is_goal),
              ("is_goal", is_goal), ("scramble_depth", depth)]
    header = {
        "format": "gpss-davi-data", "format_version": version,
        "encoding": {"puzzle": "toy", "scheme": "x", "version": 1, "len": length, "classes": 30, "key": "toy/x/v1"},
        "moves": ["a", "b"],
        "arrays": [{"name": k, "dtype": "u8", "shape": list(a.shape)} for k, a in arrays],
    }
    return json.dumps(header).encode() + b"\n" + b"".join(a.tobytes() for _, a in arrays) + extra


def test_parse_roundtrip():
    b = parse(make_blob())
    assert b.states.shape == (3, 4) and b.children.shape == (3, 2, 4)
    assert b.child_is_goal[0, 1] and not b.child_is_goal[0, 0]
    assert b.is_goal.tolist() == [False, False, True]
    assert b.moves == ["a", "b"] and b.encoding["key"] == "toy/x/v1"


def test_parse_rejects_bad_input():
    with pytest.raises(ValueError, match="trailing"):
        parse(make_blob(extra=b"\0"))
    with pytest.raises(ValueError, match="newer"):
        parse(make_blob(version=99))
    with pytest.raises(ValueError, match="not gpss"):
        parse(b'{"format": "other"}\n')


def gpss_or_skip():
    try:
        return data.find_gpss()
    except FileNotFoundError:
        pytest.skip("gpss binary not built")


def test_rust_source_cube2():
    """End-to-end against the Rust generator: successors of a 1-move scramble include the goal."""
    src = data.RustDataSource("cube2", 1, 1, gpss_or_skip())
    b = src.sample(64, seed=3)
    assert b.encoding["key"] == "cube2/corner-cubie/v1"
    assert b.states.shape == (64, 8) and b.children.shape == (64, 9, 8)
    assert (b.states < 24).all() and (b.children < 24).all()
    assert b.child_is_goal.any(axis=1).all()
    assert not b.is_goal.any()
    assert (b.scramble_depth == 1).all()
