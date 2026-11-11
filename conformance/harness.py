#!/usr/bin/env python3
"""Black-box conformance harness for puzzle solvers.

Runs an implementation's CLI once per corpus case, then checks the result with the
independent sticker-level reference model (``reference/cube_model.py``):

* ``expect: solve``  - the process exits 0, the returned move sequence is valid, applying
  ``scramble + solution`` leaves every face a single colour, and (for solvers declared
  optimal) the solution length equals the corpus ``optimal_length``. A ``null``
  ``optimal_length`` (long 3x3x3 scrambles) means only correctness is checked.
* ``expect: reject`` - the process exits with a non-zero status *without crashing*
  (no signal) and returns no solution.

Per case it records wall time, the solver-reported search time (if the implementation
prints one) and the peak resident set size of the child process (via ``os.wait4``).

Known, documented deviations of an implementation live in ``known_deviations.json``;
they are reported as XFAIL (or XPASS if they unexpectedly pass) and do not fail the run.

Examples:
    python3 conformance/harness.py --impl legacy-cpp --solver astardual
    python3 conformance/harness.py --impl gpss --solver idastar --heuristic dual --tags top
    python3 conformance/harness.py --impl legacy-cpp --solver bfs --max-depth 8 --report out.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import selectors
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / "reference"))
from cube_model import CubeModel  # noqa: E402

PUZZLE_SIZES = {"cube2": 2, "cube3": 3}


def puzzle_dims(name: str):
    """``cube2``/``cube3`` or ``AxBxC`` (width x depth x height) -> CubeModel argument."""
    if name in PUZZLE_SIZES:
        return PUZZLE_SIZES[name]
    a, b, c = (int(v) for v in name.split("x"))
    return (a, b, c)


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    wall_s: float
    peak_rss_kb: int
    timed_out: bool = False
    solution: Optional[str] = None
    reported_time_s: Optional[float] = None


# ----------------------------------------------------------------------------------------
# Implementations. Each knows how to build a command line and parse the output.
# ----------------------------------------------------------------------------------------
class Implementation:
    name = ""
    cwd: Path = ROOT
    optimal_solvers: set = set()

    def command(self, case: dict, args) -> List[str]:
        raise NotImplementedError

    def parse(self, result: RunResult) -> None:
        raise NotImplementedError

    def build(self) -> None:
        pass


class LegacyCpp(Implementation):
    """The original C++ CubeSolver in ``legacy/cpp`` (text output)."""

    name = "legacy-cpp"
    cwd = ROOT / "legacy" / "cpp"
    optimal_solvers = {"bfs", "astardual", "astarori", "astarperm", "astarperf"}

    def build(self) -> None:
        subprocess.run(["make", "-s", "CubeSolver"], cwd=self.cwd, check=True)

    def command(self, case, args):
        return ["./CubeSolver", "solve", "--scramble", case["scramble"], "--solver", args.solver]

    def parse(self, result):
        sol = re.search(r"^Solution: ?(.*)$", result.stdout, re.M)
        t = re.search(r"^Time: ([0-9.]+) seconds", result.stdout, re.M)
        result.solution = sol.group(1).strip() if sol else None
        result.reported_time_s = float(t.group(1)) if t else None


class Gpss(Implementation):
    """The Rust ``gpss`` CLI (JSON output)."""

    name = "gpss"
    optimal_solvers = {"bfs", "astar", "idastar"}

    def __init__(self):
        self.binary = Path(os.environ.get("GPSS_BIN", ROOT / "target" / "release" / "gpss"))

    def build(self) -> None:
        if "GPSS_BIN" not in os.environ:
            subprocess.run(["cargo", "build", "--release", "-q", "-p", "gpss-cli"], cwd=ROOT, check=True)

    def command(self, case, args):
        cmd = [str(self.binary), "solve", "--puzzle", case["puzzle"], "--scramble", case["scramble"]]
        cmd += ["--solver", args.solver, "--format", "json"]
        if args.heuristic:
            cmd += ["--heuristic", args.heuristic]
        return cmd

    def parse(self, result):
        if result.returncode != 0:
            return
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            return
        result.solution = data.get("solution")
        result.reported_time_s = data.get("stats", {}).get("elapsed_s")


IMPLEMENTATIONS = {cls.name: cls for cls in (LegacyCpp, Gpss)}


# ----------------------------------------------------------------------------------------
def run_with_rusage(cmd: List[str], cwd: Path, timeout: float) -> RunResult:
    """Spawn the child and reap it with os.wait4 so we get its own peak RSS."""
    start = time.perf_counter()
    out_r, out_w = os.pipe()
    err_r, err_w = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        try:
            os.chdir(cwd)
            os.dup2(out_w, 1)
            os.dup2(err_w, 2)
            for fd in (out_r, out_w, err_r, err_w):
                os.close(fd)
            os.execvp(cmd[0], cmd)
        finally:
            os._exit(127)
    os.close(out_w)
    os.close(err_w)
    chunks = {out_r: [], err_r: []}
    open_fds = [out_r, err_r]
    sel = selectors.DefaultSelector()
    for fd in open_fds:
        sel.register(fd, selectors.EVENT_READ)
    timed_out = False
    deadline = start + timeout
    while open_fds:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            timed_out = True
            os.kill(pid, 9)
            break
        for key, _ in sel.select(timeout=min(remaining, 1.0)):
            data = os.read(key.fd, 65536)
            if data:
                chunks[key.fd].append(data)
            else:
                sel.unregister(key.fd)
                open_fds.remove(key.fd)
    sel.close()
    _, status, usage = os.wait4(pid, 0)
    for fd in (out_r, err_r):
        try:
            os.close(fd)
        except OSError:
            pass
    wall = time.perf_counter() - start
    rc = os.waitstatus_to_exitcode(status)
    decode = lambda fd: b"".join(chunks[fd]).decode(errors="replace")  # noqa: E731
    return RunResult(rc, decode(out_r), decode(err_r), wall, usage.ru_maxrss, timed_out)


# ----------------------------------------------------------------------------------------
@dataclass
class Outcome:
    case: dict
    status: str  # PASS FAIL XFAIL XPASS
    detail: str
    result: RunResult
    solution_length: Optional[int] = None
    extra: dict = field(default_factory=dict)


def check(case: dict, result: RunResult, model: CubeModel, optimal: bool) -> tuple[bool, str, Optional[int]]:
    if result.timed_out:
        return False, "timeout", None
    if case["expect"] == "reject":
        if result.returncode < 0:
            return False, f"crashed with signal {-result.returncode}", None
        if result.returncode == 0:
            return False, "accepted invalid input (exit 0)", None
        if result.solution:
            return False, "returned a solution for invalid input", None
        return True, "rejected", None

    if result.returncode != 0:
        tail = (result.stderr or result.stdout).strip().splitlines()[-1:] or [""]
        return False, f"exit status {result.returncode}: {tail[0][:120]}", None
    if result.solution is None:
        return False, "no solution in output", None
    tokens = result.solution.split()
    try:
        final = model.apply(model.apply(model.solved, case["scramble"]), tokens)
    except ValueError as exc:
        return False, f"solution not parseable: {exc}", len(tokens)
    if not model.is_solved_up_to_rotation(final):
        return False, f"solution does not solve the puzzle: {result.solution!r}", len(tokens)
    if case.get("optimal_length") is None:
        return True, "ok (optimal length unknown)", len(tokens)
    if optimal and len(tokens) != case["optimal_length"]:
        return False, f"length {len(tokens)} != optimal {case['optimal_length']}", len(tokens)
    if len(tokens) < case["optimal_length"]:
        return False, f"length {len(tokens)} < optimal {case['optimal_length']} (corpus bug?)", len(tokens)
    return True, "ok", len(tokens)


def load_cases(path: Path, args) -> List[dict]:
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if args.tags:
        wanted = set(args.tags.split(","))
        cases = [c for c in cases if wanted & set(c["tags"])]
    if args.max_depth is not None:
        cases = [c for c in cases if c["expect"] != "solve" or (c.get("optimal_length") is not None and c["optimal_length"] <= args.max_depth)]
    if args.ids:
        wanted_ids = set(args.ids.split(","))
        cases = [c for c in cases if c["id"] in wanted_ids]
    return cases


def percentile(values: List[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def summarize(outcomes: List[Outcome]) -> dict:
    buckets: dict = {}
    for o in outcomes:
        if o.case["expect"] != "solve" or o.status not in ("PASS", "XPASS"):
            continue
        d = o.case.get("optimal_length")
        buckets.setdefault(d, []).append(o)
    rows = {}
    for d in sorted(buckets, key=lambda k: (k is None, k or 0)):
        items = buckets[d]
        t = [o.result.reported_time_s if o.result.reported_time_s is not None else o.result.wall_s for o in items]
        w = [o.result.wall_s for o in items]
        rows[str(d)] = {
            "n": len(items),
            "solve_s_median": statistics.median(t),
            "solve_s_p95": percentile(t, 0.95),
            "solve_s_max": max(t),
            "wall_s_median": statistics.median(w),
            "peak_rss_kb_max": max(o.result.peak_rss_kb for o in items),
        }
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impl", required=True, choices=sorted(IMPLEMENTATIONS))
    ap.add_argument("--solver", required=True)
    ap.add_argument("--heuristic", default=None, help="heuristic name (implementations that take one)")
    ap.add_argument("--corpus", type=Path, default=HERE / "corpus" / "cube2.jsonl")
    ap.add_argument("--tags", help="comma-separated: run cases having any of these tags")
    ap.add_argument("--ids", help="comma-separated case ids")
    ap.add_argument("--max-depth", type=int, help="skip cases whose optimal length exceeds this")
    ap.add_argument("--timeout", type=float, default=120.0, help="per-case timeout in seconds")
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--report", type=Path, help="write a JSON report here")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    impl = IMPLEMENTATIONS[args.impl]()
    if not args.no_build:
        impl.build()
    deviations = json.loads((HERE / "known_deviations.json").read_text()).get(impl.name, {})
    optimal = args.solver in impl.optimal_solvers
    cases = load_cases(args.corpus, args)
    models = {}

    outcomes: List[Outcome] = []
    for case in cases:
        size = puzzle_dims(case["puzzle"])
        model = models.setdefault(size, CubeModel(size))
        result = run_with_rusage(impl.command(case, args), impl.cwd, args.timeout)
        impl.parse(result)
        ok, detail, length = check(case, result, model, optimal)
        known = deviations.get(case["id"])
        if known:
            status = "XPASS" if ok else "XFAIL"
            detail = f"{detail} (known deviation: {known})"
        else:
            status = "PASS" if ok else "FAIL"
        outcomes.append(Outcome(case, status, detail, result, length))
        if args.verbose or status in ("FAIL", "XPASS"):
            print(f"{status:5} {case['id']:28} {result.wall_s:8.3f}s  {detail}")

    counts = {s: sum(o.status == s for o in outcomes) for s in ("PASS", "FAIL", "XFAIL", "XPASS")}
    table = summarize(outcomes)
    print()
    print(f"{impl.name} / solver={args.solver}" + (f" / heuristic={args.heuristic}" if args.heuristic else ""))
    print("depth   n   median_s     p95_s     max_s   peak_rss_MB")
    for d, row in table.items():
        print(
            f"{d:>5} {row['n']:>3} {row['solve_s_median']:10.4f} {row['solve_s_p95']:9.4f} "
            f"{row['solve_s_max']:9.4f} {row['peak_rss_kb_max'] / 1024:10.1f}"
        )
    print("  ".join(f"{k}={v}" for k, v in counts.items()))

    if args.report:
        report = {
            "implementation": impl.name,
            "solver": args.solver,
            "heuristic": args.heuristic,
            "corpus": str(args.corpus.relative_to(ROOT) if args.corpus.is_relative_to(ROOT) else args.corpus),
            "counts": counts,
            "latency_by_depth": table,
            "cases": [
                {
                    "id": o.case["id"],
                    "status": o.status,
                    "detail": o.detail,
                    "optimal_length": o.case.get("optimal_length"),
                    "solution_length": o.solution_length,
                    "wall_s": round(o.result.wall_s, 6),
                    "reported_s": o.result.reported_time_s,
                    "peak_rss_kb": o.result.peak_rss_kb,
                }
                for o in outcomes
            ],
        }
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
