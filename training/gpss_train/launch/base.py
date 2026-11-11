"""The launcher interface, cost/time estimation and the confirmation gate."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from ..davi import DaviConfig

#: Runs one command; returns its stdout. Tests replace this with a recorder.
Runner = Callable[[Sequence[str]], str]


def subprocess_runner(cmd: Sequence[str]) -> str:
    return subprocess.run(list(cmd), check=True, stdout=subprocess.PIPE, text=True).stdout


@dataclass(frozen=True)
class MachineType:
    """A machine and its (approximate) DAVI throughput and price.

    ``speed`` is relative to the reference machine on which ``REFERENCE_UPDATE_S`` was
    measured (8 vCPU x86, no GPU: the machine that trained the shipped 2x2x2 weights).
    Prices are on-demand list prices in USD/hour at the time of writing for the default
    region; they change - always check the provider's pricing page. Speeds for machines we
    have not run on are extrapolated (vCPU count for CPU machines, a conservative 10x for
    a single T4/L4-class GPU) and marked ``measured=False``.
    """

    name: str
    vcpus: int
    usd_per_hour: float
    speed: float
    gpu: Optional[str] = None
    measured: bool = False


#: Seconds per DAVI update on the reference machine for the default network
#: (hidden1=512, hidden2=256, 3 blocks), 100k states/update, 250 iterations of 1000.
REFERENCE_UPDATE_S = 16.0
#: Default network size the reference timing applies to (parameters).
REFERENCE_PARAMS = 630_000
#: One-off overhead for a cloud VM: boot, toolchain install, cargo build (hours).
CLOUD_SETUP_HOURS = 0.25


@dataclass
class Resources:
    """What the user asks for."""

    backend: str
    machine: Optional[str] = None
    region: Optional[str] = None
    max_hours: float = 4.0
    budget_usd: Optional[float] = None
    spot: bool = False
    bucket: Optional[str] = None
    repo: str = "https://github.com/diodotos/geometry-puzzle-solver-suite"
    ref: str = "main"
    extra: Dict[str, str] = field(default_factory=dict)


@dataclass
class Estimate:
    machine: MachineType
    train_hours: float
    billed_hours: float
    usd: float
    notes: List[str] = field(default_factory=list)

    def render(self) -> str:
        m = self.machine
        lines = [
            f"machine:   {m.name} ({m.vcpus} vCPU{', ' + m.gpu if m.gpu else ''})"
            f"{'' if m.measured else '  [throughput extrapolated]'}",
            f"training:  ~{self.train_hours:.2f} h",
            f"billed:    ~{self.billed_hours:.2f} h at ${m.usd_per_hour:.3f}/h",
            f"cost:      ~${self.usd:.2f} (compute only; storage/egress extra)",
        ]
        lines += [f"note:      {n}" for n in self.notes]
        return "\n".join(lines)


@dataclass
class LaunchPlan:
    """Everything a launch would do. Printing it is the dry run."""

    backend: str
    estimate: Estimate
    commands: List[List[str]]
    files: Dict[str, str] = field(default_factory=dict)
    artefacts: Optional[str] = None

    def render(self) -> str:
        out = [f"backend:   {self.backend}", self.estimate.render()]
        if self.artefacts:
            out.append(f"artefacts: {self.artefacts}")
        out.append("commands:")
        out += ["  " + shlex.join(c) for c in self.commands]
        for name, body in self.files.items():
            out.append(f"--- {name} ---")
            out.append(body.rstrip())
        return "\n".join(out)


def estimate_hours(cfg: DaviConfig, machine: MachineType, params: int) -> float:
    """Training wall time: updates x per-update time, scaled by work and machine speed.

    Per-update work is proportional to states (target computation over all successors)
    plus minibatch iterations, both times network size. This is a planning number,
    accurate to maybe +-50%; ``max_hours`` is the hard limit.
    """
    work = (cfg.states_per_update / 100_000 * 0.6 + cfg.iters_per_update * cfg.batch_size / 250_000 * 0.4)
    per_update = REFERENCE_UPDATE_S * work * (params / REFERENCE_PARAMS) / machine.speed
    return cfg.updates * per_update / 3600


def default_params(cfg: DaviConfig, length: int = 8, classes: int = 24) -> int:
    h1, h2 = cfg.hidden1, cfg.hidden2
    return length * classes * h1 + h1 + h1 * h2 + h2 + cfg.blocks * 2 * (h2 * h2 + h2) + h2 + 1


class Launcher:
    """Base class. Subclasses define ``machines`` and ``commands``."""

    name = "base"
    machines: Dict[str, MachineType] = {}
    default_machine = ""
    is_cloud = True

    def __init__(self, runner: Runner = subprocess_runner):
        self.runner = runner

    def machine(self, res: Resources) -> MachineType:
        key = res.machine or self.default_machine
        if key not in self.machines:
            raise ValueError(f"{self.name}: unknown machine {key!r}; known: {', '.join(sorted(self.machines))}")
        return self.machines[key]

    def estimate(self, cfg: DaviConfig, res: Resources) -> Estimate:
        m = self.machine(res)
        train_h = estimate_hours(cfg, m, default_params(cfg))
        notes = []
        billed = train_h + (CLOUD_SETUP_HOURS if self.is_cloud else 0.0)
        if self.is_cloud and billed > res.max_hours:
            notes.append(f"estimate exceeds max_hours={res.max_hours}; the VM is killed at the limit "
                         "(training resumes from the last checkpoint if rerun with --resume)")
            billed = res.max_hours
        if res.spot:
            notes.append("spot/preemptible pricing is lower but not modelled; the estimate is on-demand")
        return Estimate(m, train_h, billed, billed * m.usd_per_hour, notes)

    def plan(self, cfg: DaviConfig, res: Resources) -> LaunchPlan:
        raise NotImplementedError

    def launch(self, plan: LaunchPlan) -> List[str]:
        """Execute the plan's commands; returns their outputs."""
        return [self.runner(c) for c in plan.commands]


class LaunchRefused(Exception):
    pass


def confirm_and_launch(launcher: Launcher, cfg: DaviConfig, res: Resources, *, execute: bool, assume_yes: bool,
                       ask: Callable[[str], str] = input, show: Callable[[str], None] = print) -> Optional[List[str]]:
    """The only path to a launch: estimate -> plan -> (dry run | budget check -> confirm -> run)."""
    plan = launcher.plan(cfg, res)
    show(plan.render())
    if not execute:
        show("\ndry run: nothing was launched (pass --execute to launch)")
        return None
    if res.budget_usd is not None and plan.estimate.usd > res.budget_usd:
        raise LaunchRefused(f"estimated ${plan.estimate.usd:.2f} exceeds --budget-usd {res.budget_usd:.2f}")
    if launcher.is_cloud and res.budget_usd is None:
        raise LaunchRefused("cloud launches require --budget-usd")
    if not assume_yes:
        answer = ask(f"\nLaunch on {launcher.name} for up to ~${plan.estimate.usd:.2f}? Type 'yes' to continue: ")
        if answer.strip().lower() != "yes":
            raise LaunchRefused("not confirmed")
    return launcher.launch(plan)
