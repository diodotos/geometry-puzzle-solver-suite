"""``gpss_train launch``: argument parsing for the launchers."""

from __future__ import annotations

import argparse
import sys

from .base import LaunchRefused, Resources, confirm_and_launch
from .registry import BACKENDS, get_launcher


def add_launch_parser(sub) -> None:
    from ..cli import _add_train_args

    p = sub.add_parser("launch", help="train on local / aws / gcp (estimate, plan, confirm)",
                       description="Prints a cost/time estimate and the exact commands. Launches only with "
                                   "--execute and an interactive 'yes' (or --yes); cloud launches also need "
                                   "--budget-usd. See docs/training.md.")
    p.add_argument("backend", choices=sorted(BACKENDS))
    p.add_argument("--machine", help="instance / machine type (default per backend)")
    p.add_argument("--region", help="AWS region or GCP zone")
    p.add_argument("--max-hours", type=float, default=4.0, help="hard wall-clock limit for the VM")
    p.add_argument("--budget-usd", type=float, help="refuse to launch if the estimate exceeds this")
    p.add_argument("--spot", action="store_true", help="spot / preemptible capacity")
    p.add_argument("--bucket", help="s3://... or gs://... prefix for artefacts")
    p.add_argument("--repo", default=Resources.repo)
    p.add_argument("--ref", default="main", help="git ref to train from")
    p.add_argument("--extra", action="append", default=[], metavar="KEY=VALUE",
                   help="backend options: instance_profile, key_name (aws); project, service_account (gcp)")
    p.add_argument("--execute", action="store_true", help="actually launch (default: dry run)")
    p.add_argument("--yes", action="store_true", help="skip the interactive confirmation")
    _add_train_args(p)


def run_launch(args: argparse.Namespace, cfg) -> int:
    extra = dict(kv.split("=", 1) for kv in args.extra)
    res = Resources(args.backend, args.machine, args.region, args.max_hours, args.budget_usd, args.spot,
                    args.bucket, args.repo, args.ref, extra)
    try:
        confirm_and_launch(get_launcher(args.backend), cfg, res, execute=args.execute, assume_yes=args.yes)
    except (LaunchRefused, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0
