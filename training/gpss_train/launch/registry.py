"""Backend lookup."""

from __future__ import annotations

from .aws import AwsLauncher
from .base import Launcher, Runner, subprocess_runner
from .gcp import GcpLauncher
from .local import LocalLauncher

BACKENDS = {"local": LocalLauncher, "aws": AwsLauncher, "gcp": GcpLauncher}


def get_launcher(name: str, runner: Runner = subprocess_runner) -> Launcher:
    try:
        return BACKENDS[name](runner)
    except KeyError:
        raise ValueError(f"unknown backend {name!r}; choose one of {', '.join(BACKENDS)}") from None
