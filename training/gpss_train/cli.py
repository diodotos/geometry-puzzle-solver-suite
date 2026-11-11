"""Command line: ``python -m gpss_train <command>`` (also reachable as ``gpss train -- <command>``).

Commands:
    train   run DAVI locally (resumable)
    export  checkpoint -> ONNX + weights manifest
    profile per-scramble-depth predictions, targets and held-out loss of a checkpoint
    report  render `gpss evaluate` JSON reports as Markdown
    launch  run training on local / aws / gcp with a cost estimate and confirmation
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import List, Optional

from .davi import PRESETS, DaviConfig


def _add_train_args(p: argparse.ArgumentParser) -> None:
    defaults = DaviConfig()
    for f in fields(DaviConfig):
        flag = "--" + f.name.replace("_", "-")
        default = getattr(defaults, f.name)
        kind = type(default) if default is not None else str
        p.add_argument(flag, type=kind, default=None, help=f"default: {default}")
    p.add_argument("--resume", action="store_true", help="continue from run_dir/checkpoint.pt")


def config_from_args(args: argparse.Namespace) -> DaviConfig:
    """Defaults < puzzle preset < explicit flags."""
    puzzle = args.puzzle or DaviConfig.puzzle
    values = {**PRESETS.get(puzzle, {}), "puzzle": puzzle}
    for f in fields(DaviConfig):
        v = getattr(args, f.name, None)
        if v is not None:
            values[f.name] = v
    if "run_dir" not in values or getattr(args, "run_dir", None) is None:
        values["run_dir"] = f"runs/{puzzle}"
    return DaviConfig(**values)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="gpss_train", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="run DAVI locally")
    _add_train_args(p_train)

    p_export = sub.add_parser("export", help="export a checkpoint to ONNX + manifest")
    p_export.add_argument("--checkpoint", type=Path, required=True)
    p_export.add_argument("--out", type=Path, required=True, help="output path without extension")
    p_export.add_argument("--name", default=None)
    p_export.add_argument("--evaluation", type=Path, default=None,
                          help="JSON evaluation summary to embed in the manifest")

    p_profile = sub.add_parser("profile", help="per-scramble-depth prediction / target / MSE table")
    p_profile.add_argument("--checkpoint", type=Path, required=True)
    p_profile.add_argument("--states", type=int, default=30_000)
    p_profile.add_argument("--seed", type=int, default=12345)
    p_profile.add_argument("--out", type=Path, default=None, help="also write the profile as JSON")

    p_report = sub.add_parser("report", help="render `gpss evaluate` JSON as Markdown tables")
    p_report.add_argument("what", choices=["heuristic", "depth", "uniform-depth", "settings", "summary"])
    p_report.add_argument("reports", type=Path, nargs="+")

    from .launch.cli import add_launch_parser
    add_launch_parser(sub)

    args = ap.parse_args(argv)
    if args.command == "train":
        from .davi import train
        cfg = config_from_args(args)
        train(cfg, resume=args.resume)
        return 0
    if args.command == "export":
        from .export import export
        evaluation = json.loads(args.evaluation.read_text()) if args.evaluation else None
        print(export(args.checkpoint, args.out, args.name, evaluation))
        return 0
    if args.command == "profile":
        from .profile import main as profile_main
        print(profile_main(args.checkpoint, args.states, args.seed, args.out))
        return 0
    if args.command == "report":
        from .report import main as report_main
        print(report_main(args.reports, args.what))
        return 0
    if args.command == "launch":
        from .launch.cli import run_launch
        return run_launch(args, config_from_args(args))
    ap.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
