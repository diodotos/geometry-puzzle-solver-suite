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
    ck = torch.load(tmp_path / "run" / "checkpoint.pt", weights_only=False)
    assert ck["state"]["update"] == 2 and ck["state"]["syncs"] == 2
    cfg.updates = 3
    train(cfg, resume=True, source=FakeSource(), log=lambda s: None)
    ck = torch.load(tmp_path / "run" / "checkpoint.pt", weights_only=False)
    assert ck["state"]["update"] == 3
    assert len((tmp_path / "run" / "log.jsonl").read_text().splitlines()) == 3


def test_forced_sync_after_stall(tmp_path):
    cfg = DaviConfig(run_dir=str(tmp_path / "run"), updates=5, states_per_update=16, iters_per_update=1,
                     batch_size=8, hidden1=16, hidden2=8, blocks=1, loss_threshold=-1.0, max_updates_per_sync=2)
    train(cfg, source=FakeSource(), log=lambda s: None)
    ck = torch.load(tmp_path / "run" / "checkpoint.pt", weights_only=False)
    assert ck["state"]["syncs"] == 2 and ck["state"]["last_sync_update"] == 4
    assert [r["forced_sync"] for r in ck["state"]["history"]] == [False, True, False, True, False]
