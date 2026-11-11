"""Deep approximate value iteration (DAVI), following Agostinelli et al. 2019 [Ago19].

Each *update*:

1. Sample ``states_per_update`` states by scrambling the goal ``k ~ U{min..max}`` moves
   (Rust: ``gpss scramble-data``), together with all one-move successors.
2. Compute Bellman targets with the frozen *target* network ``J'``:
   ``y(s) = min_a [1 + J'(A(s, a))]`` with ``J'(goal) = 0`` and ``y(goal) = 0``. Before
   the first synchronisation ``J' = 0`` (so every non-goal target is 1), as in the
   DeepCubeA reference implementation.
3. Regress ``J(s)`` to ``y(s)`` for ``iters_per_update`` minibatches (MSE, Adam).
4. If the update's loss is below ``loss_threshold`` (0.05 in the paper), copy ``J`` into
   ``J'``. ``sync_metric="train"`` uses the mean training loss; ``"fresh"`` uses the loss of
   the network on this update's new states *before* training on them, which is an honest
   held-out estimate (a large network revisiting the same states for several epochs can
   memorise them, so its training loss understates its error). As a fallback (not in the paper), also copy after
   ``max_updates_per_sync`` updates without a sync: once targets are noisy and large, a
   small network can plateau just above the threshold and propagation would stall. Distance information propagates one step outward from
   the goal per synchronisation, so a puzzle of diameter ``D`` needs at least ``D`` syncs.

Everything is checkpointed under ``run_dir`` so training can be resumed (``--resume``).
``init_from`` warm-starts a *new* run from another run's checkpoint: the network and target
weights are copied, the optimiser, LR schedule and counters start fresh (so new ``lr`` /
sync settings take effect). It is ignored when ``--resume`` finds a checkpoint.
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch import nn

from .data import DaviBatch, RustDataSource
from .model import Architecture, CostToGo, parameter_count


@dataclass
class DaviConfig:
    puzzle: str = "cube2"
    run_dir: str = "runs/cube2"
    min_scramble: int = 1
    max_scramble: int = 14
    updates: int = 400
    states_per_update: int = 100_000
    iters_per_update: int = 250
    batch_size: int = 1000
    lr: float = 1e-3
    lr_decay: float = 0.9999993
    loss_threshold: float = 0.05
    max_updates_per_sync: int = 10
    sync_metric: str = "train"
    fresh_states: int = 100_000
    hidden1: int = 512
    hidden2: int = 256
    blocks: int = 3
    seed: int = 1
    threads: int = 0
    target_batch: int = 65536
    gpss_bin: Optional[str] = None
    stop_after_syncs: int = 0
    init_from: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# Defaults per puzzle, chosen so that max_scramble exceeds the diameter (God's number).
PRESETS = {
    "cube2": dict(max_scramble=14),
    # DeepCubeA's 3x3x3 setup: scrambles up to 30, a (5000, 1000) MLP with four residual
    # blocks, batch 10,000 (Agostinelli et al. 2019, Methods).
    # After the L4 pilot (docs/training.md, "Tuning after the pilot"): one epoch per update as in
    # the reference code (the pilot's 2.5 memorised the states), the sync rule on held-out loss,
    # and a forced sync every 3 updates (waiting 10 did not lower the loss).
    "cube3": dict(max_scramble=30, hidden1=5000, hidden2=1000, blocks=4, batch_size=10_000,
                  states_per_update=1_000_000, iters_per_update=100, sync_metric="fresh",
                  max_updates_per_sync=3, run_dir="runs/cube3"),
}


@dataclass
class TrainState:
    update: int = 0
    syncs: int = 0
    last_sync_update: int = 0
    samples_seen: int = 0
    wall_s: float = 0.0
    history: list = field(default_factory=list)


@torch.no_grad()
def bellman_targets(target: Optional[nn.Module], batch: DaviBatch, chunk: int) -> np.ndarray:
    """``y(s) = min_a [1 + J'(A(s,a))]``, ``J'(goal) = 0``, ``y(goal) = 0``.

    ``target is None`` means ``J' = 0`` (before the first synchronisation).
    """
    n, m, length = batch.children.shape
    if target is None:
        child_values = np.zeros((n, m), dtype=np.float32)
    else:
        target.eval()
        device = next(target.parameters(), torch.empty(0)).device
        # Ship the compact uint8 encoding (the model widens it on device) and keep results on
        # the device until the end: 1M 3x3 states have 18M children.
        children = np.ascontiguousarray(batch.children.reshape(n * m, length))
        flat = torch.from_numpy(children if children.flags.writeable else children.copy())
        out = torch.empty(n * m, dtype=torch.float32, device=device)
        for start in range(0, n * m, chunk):
            out[start:start + chunk] = target(flat[start:start + chunk].to(device, non_blocking=True)).clamp_min(0)
        child_values = out.cpu().numpy().reshape(n, m)
    child_values[batch.child_is_goal] = 0.0
    y = (1.0 + child_values).min(axis=1)
    y[batch.is_goal] = 0.0
    return y.astype(np.float32)


@torch.no_grad()
def held_out_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor, chunk: int) -> float:
    """MSE of ``model`` (eval mode) on states it has not been trained on."""
    if len(x) == 0:
        return float("nan")
    model.eval()
    pred = torch.cat([model(x[i:i + chunk]) for i in range(0, len(x), chunk)])
    model.train()
    return float(((pred - y) ** 2).mean())


def _save(run_dir: Path, cfg: DaviConfig, model: nn.Module, target: Optional[nn.Module], opt, sched, st: TrainState, encoding: dict) -> None:
    ckpt = {
        "config": cfg.to_dict(),
        "encoding": encoding,
        "architecture": model.arch.to_dict(),
        "model": model.state_dict(),
        "target": None if target is None else target.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "state": asdict(st),
    }
    tmp = run_dir / "checkpoint.pt.tmp"
    torch.save(ckpt, tmp)
    tmp.replace(run_dir / "checkpoint.pt")


def load_checkpoint(path: Path):
    return torch.load(path, map_location="cpu", weights_only=True)


def build_model(arch_dict: dict) -> CostToGo:
    keys = ("len", "classes", "hidden1", "hidden2", "blocks")
    return CostToGo(Architecture(**{k: arch_dict[k] for k in keys}))


def train(cfg: DaviConfig, resume: bool = False, log: Callable[[str], None] = print,
          source: Optional[RustDataSource] = None) -> Path:
    """Run DAVI; returns the run directory (containing ``checkpoint.pt`` and ``log.jsonl``)."""
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    source = source or RustDataSource(cfg.puzzle, cfg.min_scramble, cfg.max_scramble, cfg.gpss_bin)
    encoding = source.encoding()
    arch = Architecture(encoding["len"], encoding["classes"], cfg.hidden1, cfg.hidden2, cfg.blocks)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TF32 matmuls on Ampere+ GPUs (e.g. L4): several times faster, ~1e-3 relative error.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = CostToGo(arch).to(device)
    target: Optional[CostToGo] = None
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=cfg.lr_decay)
    st = TrainState()
    if cfg.sync_metric not in ("train", "fresh"):
        raise ValueError(f"sync_metric must be 'train' or 'fresh', not {cfg.sync_metric!r}")

    ckpt_path = run_dir / "checkpoint.pt"
    if resume and ckpt_path.exists():
        ck = load_checkpoint(ckpt_path)
        if ck["encoding"] != encoding:
            raise ValueError(f"checkpoint encoding {ck['encoding']} != puzzle encoding {encoding}")
        model.load_state_dict(ck["model"])
        if ck["target"] is not None:
            target = copy.deepcopy(model)
            target.load_state_dict(ck["target"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        st = TrainState(**ck["state"])
        rng = np.random.default_rng([cfg.seed, st.update])
        log(f"resumed at update {st.update} ({st.syncs} syncs)")
    elif cfg.init_from:
        ck = load_checkpoint(Path(cfg.init_from))
        if ck["encoding"] != encoding or ck["architecture"] != arch.to_dict():
            raise ValueError(f"init_from {cfg.init_from}: encoding/architecture differ from this run")
        model.load_state_dict(ck["model"])
        if ck["target"] is not None:
            target = copy.deepcopy(model)
            target.load_state_dict(ck["target"])
            target.eval()
        log(f"warm start from {cfg.init_from} (update {ck['state']['update']}, {ck['state']['syncs']} syncs)")
    (run_dir / "config.json").write_text(json.dumps({**cfg.to_dict(), "encoding": encoding,
                                                     "architecture": arch.to_dict()}, indent=2))
    log(f"{encoding['key']}: {parameter_count(model):,} parameters on {device}, run dir {run_dir}")

    loss_fn = nn.MSELoss()
    with open(run_dir / "log.jsonl", "a") as logf:
        while st.update < cfg.updates:
            t0 = time.perf_counter()
            batch = source.sample(cfg.states_per_update, seed=cfg.seed * 1_000_003 + st.update)
            t_data = time.perf_counter() - t0
            y = torch.from_numpy(bellman_targets(target, batch, cfg.target_batch)).to(device)
            x = torch.from_numpy(batch.states.astype(np.int64)).to(device)
            t_target = time.perf_counter() - t0 - t_data
            fresh_loss = held_out_loss(model, x[:cfg.fresh_states], y[:cfg.fresh_states], cfg.target_batch)

            model.train()
            losses = []
            n = x.shape[0]
            for _ in range(cfg.iters_per_update):
                idx = torch.from_numpy(rng.integers(0, n, size=min(cfg.batch_size, n))).to(device)
                loss = loss_fn(model(x[idx]), y[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                losses.append(loss.item())
            mean_loss = float(np.mean(losses))
            st.update += 1
            stale = cfg.max_updates_per_sync > 0 and st.update - st.last_sync_update >= cfg.max_updates_per_sync
            sync_loss = fresh_loss if cfg.sync_metric == "fresh" else mean_loss
            synced = sync_loss < cfg.loss_threshold or stale
            if synced:
                st.last_sync_update = st.update
                target = copy.deepcopy(model)
                target.eval()
                st.syncs += 1
            st.samples_seen += n
            dt = time.perf_counter() - t0
            st.wall_s += dt
            rec = {
                "update": st.update, "loss": mean_loss, "fresh_loss": fresh_loss, "synced": synced,
                "forced_sync": synced and sync_loss >= cfg.loss_threshold, "syncs": st.syncs,
                "target_mean": float(y.mean()), "target_max": float(y.max()),
                "lr": sched.get_last_lr()[0], "data_s": t_data, "target_s": t_target, "update_s": dt,
                "wall_s": st.wall_s,
            }
            st.history.append(rec)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            log(f"update {st.update:4d}  loss {mean_loss:.4f}  fresh {fresh_loss:.4f}  syncs {st.syncs:3d}  "
                f"y_mean {rec['target_mean']:.2f}  y_max {rec['target_max']:.1f}  {dt:.1f}s")
            _save(run_dir, cfg, model, target, opt, sched, st, encoding)
            if cfg.stop_after_syncs and st.syncs >= cfg.stop_after_syncs:
                log(f"reached {st.syncs} syncs; stopping")
                break
    if not math.isfinite(st.history[-1]["loss"] if st.history else 0.0):
        raise RuntimeError("training diverged (non-finite loss)")
    return run_dir
