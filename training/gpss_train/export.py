"""Export a trained network to ONNX and write its weights manifest.

Produces ``<out>.onnx`` and ``<out>.json``. The manifest format is defined (and
validated on load) by ``crates/gpss-learn/src/manifest.rs``: it pins the puzzle and the
exact encoding spec the network was trained on and the SHA-256 of the ONNX file, so the
Rust side refuses to run the network on anything else.

The exported model uses the training network ``J`` (not the target ``J'``), in eval mode
(batch norm uses running statistics). Input ``state: int64[batch, len]``, output
``cost_to_go: float32[batch]``.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from . import AGOSTINELLI_2019
from .davi import build_model, load_checkpoint

MANIFEST_FORMAT = "gpss-weights"
MANIFEST_FORMAT_VERSION = 1
INPUT_NAME = "state"
OUTPUT_NAME = "cost_to_go"
OPSET = 18


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).parent).stdout.strip()
    except Exception:
        return None


def export(checkpoint: Path, out: Path, name: Optional[str] = None, evaluation: Optional[dict] = None,
           verify: bool = True) -> Path:
    """Write ``out.onnx`` + ``out.json``; returns the manifest path."""
    ck = load_checkpoint(checkpoint)
    model = build_model(ck["architecture"])
    model.load_state_dict(ck["model"])
    model.eval()
    enc = ck["encoding"]
    out = out.with_suffix("")
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx_path = out.with_suffix(".onnx")
    example = torch.zeros((2, enc["len"]), dtype=torch.long)
    batch = torch.export.Dim("batch", min=1)
    torch.onnx.export(
        model, (example,), str(onnx_path),
        input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
        dynamic_shapes=({0: batch},), opset_version=OPSET, dynamo=True, external_data=False,
    )
    if verify:
        verify_onnx(model, onnx_path, enc)

    st = ck["state"]
    last = st["history"][-1] if st["history"] else {}
    manifest = {
        "format": MANIFEST_FORMAT,
        "format_version": MANIFEST_FORMAT_VERSION,
        "name": name or out.name,
        "encoding": {k: enc[k] for k in ("puzzle", "scheme", "version", "len", "classes")},
        "model": {"file": onnx_path.name, "sha256": sha256(onnx_path), "input": INPUT_NAME,
                  "output": OUTPUT_NAME, "opset": OPSET},
        "architecture": ck["architecture"],
        "training": {
            "method": "DAVI (deep approximate value iteration) [Ago19]",
            "config": ck["config"],
            "updates": st["update"], "target_syncs": st["syncs"], "samples_seen": st["samples_seen"],
            "wall_hours": round(st["wall_s"] / 3600, 3), "final_loss": last.get("loss"),
            "hardware": "see docs/benchmarks for the machine used",
            "git_commit": _git_commit(),
            "exported": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "torch": torch.__version__,
        },
        "evaluation": evaluation or {},
        "references": [AGOSTINELLI_2019],
    }
    manifest_path = out.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def verify_onnx(model: torch.nn.Module, onnx_path: Path, enc: dict, n: int = 257) -> float:
    """Compare ONNX Runtime with PyTorch on random encodings; returns the max abs diff."""
    try:
        import onnxruntime as ort
    except ImportError:
        return float("nan")
    x = np.random.default_rng(0).integers(0, enc["classes"], size=(n, enc["len"]), dtype=np.int64)
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run([OUTPUT_NAME], {INPUT_NAME: x})[0]
    diff = float(np.abs(ref - got.reshape(ref.shape)).max())
    if diff > 1e-3:
        raise RuntimeError(f"ONNX output differs from PyTorch by {diff}")
    return diff
