"""Depth profile of a checkpoint: what the network believes at each scramble depth.

For fresh states scrambled ``k`` moves from the goal it reports, per ``k``, the mean
prediction ``J(s)``, the mean Bellman target ``y(s)`` from the checkpoint's target network,
and the held-out MSE. Random-move scrambles of 20+ moves are typically 17-18 moves from solved
on the 3x3x3, so a profile whose ``J`` saturates well below that shows that value propagation
has stalled, which the aggregate loss does not show. ``y - J`` is the per-sync growth still
available: near zero means the network is self-consistent at its current level.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .data import RustDataSource
from .davi import bellman_targets, build_model, load_checkpoint


@torch.no_grad()
def depth_profile(checkpoint: Path, states: int = 30_000, seed: int = 12345, source=None,
                  chunk: int = 65536) -> dict:
    ck = load_checkpoint(checkpoint)
    cfg = ck["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(ck["architecture"]).to(device).eval()
    model.load_state_dict(ck["model"])
    target = None
    if ck["target"] is not None:
        target = build_model(ck["architecture"]).to(device).eval()
        target.load_state_dict(ck["target"])
    source = source or RustDataSource(cfg["puzzle"], cfg["min_scramble"], cfg["max_scramble"], cfg.get("gpss_bin"))
    batch = source.sample(states, seed=seed)
    y = bellman_targets(target, batch, chunk)
    x = torch.from_numpy(batch.states.astype(np.int64))
    j = torch.cat([model(x[i:i + chunk].to(device)).cpu() for i in range(0, len(x), chunk)]).numpy()
    rows = []
    for d in np.unique(batch.scramble_depth):
        m = batch.scramble_depth == d
        rows.append({"depth": int(d), "n": int(m.sum()), "mean_j": float(j[m].mean()),
                     "mean_y": float(y[m].mean()), "mse": float(((j[m] - y[m]) ** 2).mean())})
    return {"checkpoint": str(checkpoint), "update": ck["state"]["update"], "syncs": ck["state"]["syncs"],
            "states": states, "seed": seed, "mse": float(((j - y) ** 2).mean()), "depths": rows}


def render(profile: dict) -> str:
    out = [f"update {profile['update']}, {profile['syncs']} syncs, held-out MSE {profile['mse']:.4f}", "",
           "| scramble depth | n | mean J | mean y | y - J | MSE |", "|---|---|---|---|---|---|"]
    for r in profile["depths"]:
        out.append(f"| {r['depth']} | {r['n']} | {r['mean_j']:.2f} | {r['mean_y']:.2f} | "
                   f"{r['mean_y'] - r['mean_j']:+.3f} | {r['mse']:.4f} |")
    return "\n".join(out)


def main(checkpoint: Path, states: int, seed: int, out: Optional[Path]) -> str:
    profile = depth_profile(checkpoint, states, seed)
    if out:
        out.write_text(json.dumps(profile, indent=2))
    return render(profile)
