"""``local``: run training as a subprocess on this machine."""

from __future__ import annotations

import os
import sys

from .base import Launcher, LaunchPlan, MachineType, Resources
from .bootstrap import train_args
from ..davi import DaviConfig


class LocalLauncher(Launcher):
    name = "local"
    is_cloud = False
    default_machine = "this-machine"

    @property
    def machines(self):  # type: ignore[override]
        cpus = os.cpu_count() or 1
        return {"this-machine": MachineType("this-machine", cpus, 0.0, speed=cpus / 8, measured=cpus == 8)}

    def plan(self, cfg: DaviConfig, res: Resources) -> LaunchPlan:
        cmd = [sys.executable, "-m", "gpss_train", "train", *train_args(cfg, cfg.run_dir)]
        return LaunchPlan(self.name, self.estimate(cfg, res), [cmd], artefacts=cfg.run_dir)
