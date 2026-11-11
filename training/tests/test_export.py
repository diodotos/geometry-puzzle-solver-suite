import hashlib
import json

import pytest

from gpss_train.davi import DaviConfig, train
from gpss_train.export import export

from test_davi import FakeSource


def test_export_manifest(tmp_path):
    pytest.importorskip("onnxscript")
    cfg = DaviConfig(run_dir=str(tmp_path / "run"), updates=1, states_per_update=16, iters_per_update=2,
                     batch_size=8, hidden1=16, hidden2=8, blocks=1)
    train(cfg, source=FakeSource(), log=lambda s: None)
    manifest_path = export(tmp_path / "run" / "checkpoint.pt", tmp_path / "w" / "toy-davi")
    m = json.loads(manifest_path.read_text())
    assert m["format"] == "gpss-weights" and m["format_version"] == 1
    assert m["encoding"] == {"puzzle": "toy", "scheme": "x", "version": 1, "len": 4, "classes": 30}
    onnx_file = manifest_path.parent / m["model"]["file"]
    assert hashlib.sha256(onnx_file.read_bytes()).hexdigest() == m["model"]["sha256"]
    assert m["model"]["input"] == "state" and m["model"]["output"] == "cost_to_go"
    assert any("Agostinelli" in r for r in m["references"])


def test_one_hot_linear_training_path_matches_export_path():
    import torch
    from gpss_train.model import OneHotLinear

    torch.manual_seed(0)
    layer = OneHotLinear(20, 24, 64)
    x = torch.randint(0, 24, (32, 20))
    reference = layer.embed(x.long() + layer.offsets).sum(dim=1) + layer.bias
    assert torch.allclose(layer(x), reference, atol=1e-5)
