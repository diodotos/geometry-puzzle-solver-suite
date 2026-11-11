import json
import pytest
import numpy as np
import torch

from gpss_train.data import parse
from gpss_train.davi import DaviConfig, bellman_targets, train
from gpss_train.model import Architecture, CostToGo, OneHotLinear

from test_data import make_blob


def test_one_hot_linear_matches_dense():
    torch.manual_seed(0)
    layer = OneHotLinear(4, 5, 3)
    x = torch.randint(0, 5, (7, 4))
    one_hot = torch.nn.functional.one_hot(x, 5).reshape(7, 20).float()
    dense = one_hot @ layer.embed.weight + layer.bias
    assert torch.allclose(layer(x), dense, atol=1e-6)


def test_model_shape():
    m = CostToGo(Architecture(8, 24, 32, 16, 1)).eval()
    assert m(torch.zeros(5, 8, dtype=torch.long)).shape == (5,)


class Const(torch.nn.Module):
    def __init__(self, v):
        super().__init__()
        self.v = v

    def forward(self, x):
        return torch.full((x.shape[0],), self.v)


def test_bellman_targets():
    b = parse(make_blob(n=3, m=2))
    # Before the first sync J' = 0: every non-goal target is 1, goal target 0.
    assert bellman_targets(None, b, 100).tolist() == [1.0, 1.0, 0.0]
    # J' = 5 everywhere except goal children (forced to 0): state 0 has a goal child.
    assert bellman_targets(Const(5.0), b, 100).tolist() == [1.0, 6.0, 0.0]
    # Negative network outputs are clamped to 0 before the min.
    assert bellman_targets(Const(-3.0), b, 2).tolist() == [1.0, 1.0, 0.0]


class FakeSource:
    """Data source stub: random states over a toy encoding."""

    def __init__(self):
        self.rng = np.random.default_rng(0)

    def encoding(self):
        return parse(make_blob()).encoding

    def sample(self, count, seed):
        return parse(make_blob(n=count, m=2))


def test_train_and_resume(tmp_path):
    cfg = DaviConfig(run_dir=str(tmp_path / "run"), updates=2, states_per_update=16, iters_per_update=3,
                     batch_size=8, hidden1=16, hidden2=8, blocks=1, loss_threshold=1e9)
    train(cfg, source=FakeSource(), log=lambda s: None)
    ck = torch.load(tmp_path / "run" / "checkpoint.pt", weights_only=True)
    assert ck["state"]["update"] == 2 and ck["state"]["syncs"] == 2
    cfg.updates = 3
    train(cfg, resume=True, source=FakeSource(), log=lambda s: None)
    ck = torch.load(tmp_path / "run" / "checkpoint.pt", weights_only=True)
    assert ck["state"]["update"] == 3
    assert len((tmp_path / "run" / "log.jsonl").read_text().splitlines()) == 3


def test_forced_sync_after_stall(tmp_path):
    cfg = DaviConfig(run_dir=str(tmp_path / "run"), updates=5, states_per_update=16, iters_per_update=1,
                     batch_size=8, hidden1=16, hidden2=8, blocks=1, loss_threshold=-1.0, max_updates_per_sync=2)
    train(cfg, source=FakeSource(), log=lambda s: None)
    ck = torch.load(tmp_path / "run" / "checkpoint.pt", weights_only=True)
    assert ck["state"]["syncs"] == 2 and ck["state"]["last_sync_update"] == 4
    assert [r["forced_sync"] for r in ck["state"]["history"]] == [False, True, False, True, False]


def test_fresh_sync_metric_and_held_out_loss(tmp_path):
    cfg = DaviConfig(run_dir=str(tmp_path / "run"), updates=2, states_per_update=16, iters_per_update=2,
                     batch_size=8, hidden1=16, hidden2=8, blocks=1, loss_threshold=1e9, sync_metric="fresh")
    train(cfg, source=FakeSource(), log=lambda s: None)
    hist = torch.load(tmp_path / "run" / "checkpoint.pt", weights_only=True)["state"]["history"]
    assert all(r["synced"] and not r["forced_sync"] and r["fresh_loss"] >= 0 for r in hist)
    with pytest.raises(ValueError, match="sync_metric"):
        train(DaviConfig(run_dir=str(tmp_path / "bad"), sync_metric="val"), source=FakeSource(), log=lambda s: None)


def test_init_from_copies_weights_but_starts_fresh(tmp_path):
    small = dict(states_per_update=16, iters_per_update=2, batch_size=8, hidden1=16, hidden2=8, blocks=1)
    train(DaviConfig(run_dir=str(tmp_path / "a"), updates=3, loss_threshold=1e9, **small),
          source=FakeSource(), log=lambda s: None)
    src = torch.load(tmp_path / "a" / "checkpoint.pt", weights_only=True)
    logs = []
    cfg = DaviConfig(run_dir=str(tmp_path / "b"), updates=0, lr=1e-4,
                     init_from=str(tmp_path / "a" / "checkpoint.pt"), **small)
    train(cfg, source=FakeSource(), log=logs.append)
    assert any("warm start" in line and "update 3" in line for line in logs)
    cfg.updates = 1
    train(cfg, source=FakeSource(), log=lambda s: None)
    ck = torch.load(tmp_path / "b" / "checkpoint.pt", weights_only=True)
    assert ck["state"]["update"] == 1 and ck["optimizer"]["param_groups"][0]["initial_lr"] == 1e-4
    assert ck["target"] is not None and src["target"] is not None
    assert ck["state"]["sample_offset"] == 3  # batches continue after the source run's 3
    with pytest.raises(ValueError, match="architecture"):
        train(DaviConfig(run_dir=str(tmp_path / "c"), updates=1, init_from=cfg.init_from,
                         **{**small, "hidden1": 32}), source=FakeSource(), log=lambda s: None)


def test_depth_profile(tmp_path):
    from gpss_train.profile import depth_profile, render

    cfg = DaviConfig(run_dir=str(tmp_path / "run"), updates=2, states_per_update=16, iters_per_update=2,
                     batch_size=8, hidden1=16, hidden2=8, blocks=1, loss_threshold=1e9)
    train(cfg, source=FakeSource(), log=lambda s: None)
    prof = depth_profile(tmp_path / "run" / "checkpoint.pt", states=12, source=FakeSource())
    assert prof["update"] == 2 and sum(r["n"] for r in prof["depths"]) == 12
    assert [r["depth"] for r in prof["depths"]] == list(range(12))
    assert "| scramble depth |" in render(prof)


def test_plot_progress(tmp_path):
    pytest.importorskip("matplotlib")
    from gpss_train.plot import load_log, plot
    from gpss_train.profile import depth_profile

    small = dict(states_per_update=16, iters_per_update=2, batch_size=8, hidden1=16, hidden2=8, blocks=1,
                 loss_threshold=1e9)
    train(DaviConfig(run_dir=str(tmp_path / "a"), updates=2, **small), source=FakeSource(), log=lambda s: None)
    train(DaviConfig(run_dir=str(tmp_path / "b"), updates=2, sync_metric="fresh",
                     init_from=str(tmp_path / "a" / "checkpoint.pt"), **small), source=FakeSource(), log=lambda s: None)
    (tmp_path / "p.json").write_text(json.dumps(depth_profile(tmp_path / "b" / "checkpoint.pt", 12, source=FakeSource())))
    assert [r["update"] for r in load_log(tmp_path / "a" / "log.jsonl")] == [1, 2]
    out = plot([f"a={tmp_path / 'a' / 'log.jsonl'}", str(tmp_path / "b" / "log.jsonl")], [str(tmp_path / "p.json")],
               tmp_path / "progress.png", "t")
    assert out.stat().st_size > 10_000
