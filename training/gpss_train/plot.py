"""Training-progress charts from ``log.jsonl`` files and depth profiles (``profile.json``).

``plot --log pilot=a/log.jsonl --log check=b/log.jsonl --out progress.png`` draws, for runs
that continue one another (a warm start or a resume), training and held-out loss, the mean and
max Bellman target, and target-network syncs against GPU hours. Rounds are numbered
continuously across the runs in the order given. ``--profile name=profile.json`` (repeatable)
adds a panel with mean ``J`` per scramble depth, the clearest view of how far value
propagation has reached. matplotlib is imported lazily so training never depends on it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

COLOURS = ["tab:gray", "tab:blue", "tab:green", "tab:purple", "tab:red", "tab:brown"]


def load_log(path: Path) -> List[dict]:
    """Records keyed by update; a resumed run's repeated updates keep the latest record."""
    recs = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r["update"]] = r
    return [recs[k] for k in sorted(recs)]


def _named(spec: str) -> Tuple[str, Path]:
    name, sep, path = spec.partition("=")
    return (name, Path(path)) if sep else (Path(spec).parent.name, Path(spec))


def plot(logs: List[str], profiles: List[str], out: Path, title: str = "") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [(name, load_log(p)) for name, p in map(_named, logs)]
    panels = 5 if profiles else 4
    fig, axes = plt.subplots(2, 3, figsize=(18, 9)) if profiles else plt.subplots(2, 2, figsize=(13, 9))
    ax = list(axes.flat)
    colour = {}
    offset = 0
    for i, (name, recs) in enumerate(runs):
        c = colour.setdefault(name, COLOURS[i % len(COLOURS)])
        rounds = [offset + r["update"] for r in recs]
        ax[0].plot(rounds, [r["loss"] for r in recs], color=c, label=f"{name}: training loss")
        if any("fresh_loss" in r for r in recs):
            ax[0].plot(rounds, [r.get("fresh_loss", float("nan")) for r in recs], color=c, ls="--",
                       label=f"{name}: held-out loss")
        ax[1].plot(rounds, [r["target_mean"] for r in recs], color=c, label=name)
        ax[2].plot(rounds, [r["target_max"] for r in recs], color=c, label=name)
        ax[3].plot([r["wall_s"] / 3600 for r in recs], [r["syncs"] for r in recs], color=c, label=name)
        offset = rounds[-1] if rounds else offset
    ax[0].set_ylim(0, 0.3)
    ax[0].set_title("Loss (MSE vs Bellman target)")
    ax[1].set_title("Mean Bellman target")
    ax[2].set_title("Max Bellman target per round")
    for a in ax[:3]:
        a.set_xlabel("training round (continuous across runs)")
    ax[3].set_title("Target-network syncs vs training time")
    ax[3].set_xlabel("hours of training in that run")
    if profiles:
        for i, (name, p) in enumerate(map(_named, profiles)):
            prof = json.loads(p.read_text())
            d = [r["depth"] for r in prof["depths"]]
            c = colour.setdefault(name, COLOURS[(len(runs) + i) % len(COLOURS)])
            ax[4].plot(d, [r["mean_j"] for r in prof["depths"]], marker=".", color=c,
                       label=f"{name} (update {prof['update']}, MSE {prof['mse']:.3f})")
        ax[4].set_title("Mean prediction J by scramble depth")
        ax[4].set_xlabel("scramble depth (random moves)")
        ax[5].axis("off")
    for a in ax[:panels]:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
