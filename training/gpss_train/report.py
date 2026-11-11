"""Render ``gpss evaluate`` JSON reports as Markdown tables (for ``docs/benchmarks/``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List


def heuristic_table(report: dict) -> str:
    he = report["heuristic_error"]
    rows = ["| true depth | states | mean J | MAE | RMSE | over-estimates | max over | rounds to exact |",
            "|-----------:|-------:|-------:|----:|-----:|---------------:|---------:|----------------:|"]
    for d in he["by_depth"] + [he["overall"]]:
        depth = "all" if d["depth"] is None else str(d["depth"])
        rows.append(f"| {depth} | {d['count']:,} | {d['mean_estimate']:.2f} | {d['mae']:.3f} | {d['rmse']:.3f} | "
                    f"{100 * d['overestimate_rate']:.1f}% | {d['max_overestimate']:.2f} | "
                    f"{100 * d['rounded_exact_rate']:.1f}% |")
    return "\n".join(rows)


def _solve_row(label: str, d: dict) -> str:
    return (f"| {label} | {d['count']} | {100 * d['solved_rate']:.1f}% | {d['mean_optimal_length']:.2f} | "
            f"{d['mean_length']:.2f} | {100 * d['optimal_rate']:.1f}% | {d['max_extra_moves']} | "
            f"{d['mean_expanded']:.0f} | {d['median_expanded']} | {d['max_expanded']} | {1000 * d['mean_time_s']:.1f} |")


SOLVE_HEADER = ["| {label} | n | solved | mean optimal | mean length | optimal | max extra | mean expanded | median expanded | max expanded | mean ms |",
                "|------:|--:|-------:|-------------:|------------:|--------:|----------:|--------------:|----------------:|-------------:|--------:|"]


def solve_by_depth(report: dict, key: str = "solve_stratified") -> str:
    rep = report[key]["report"]
    rows = [SOLVE_HEADER[0].format(label="depth"), SOLVE_HEADER[1]]
    rows += [_solve_row(str(d["depth"]), d) for d in rep["by_depth"]]
    rows.append(_solve_row("all", rep["overall"]))
    return "\n".join(rows)


def settings_table(reports: List[dict]) -> str:
    """One row per (lambda, batch) setting: uniform-random and stratified overall stats."""
    rows = [SOLVE_HEADER[0].format(label="lambda / N"), SOLVE_HEADER[1]]
    for r in reports:
        key = "solve_uniform" if r["solve_uniform"]["report"] else "solve_stratified"
        rows.append(_solve_row(f"{r['lambda']:g} / {r['batch_size']}", r[key]["report"]["overall"]))
    return "\n".join(rows)


def summary(report: dict) -> dict:
    """Compact numbers to embed in a weights manifest (``export --evaluation``)."""
    he = report["heuristic_error"]["overall"]
    out = {"solver": report["solver"], "lambda": round(report["lambda"], 4), "batch_size": report["batch_size"],
           "heuristic_mae_all_states": round(he["mae"], 4),
           "heuristic_overestimate_rate": round(he["overestimate_rate"], 4)}
    for key in ("solve_uniform", "solve_stratified"):
        rep = report[key]["report"]
        if rep:
            o = rep["overall"]
            out[key] = {"n": o["count"], "solved_rate": o["solved_rate"], "optimal_rate": round(o["optimal_rate"], 4),
                        "mean_length": round(o["mean_length"], 3), "mean_optimal_length": round(o["mean_optimal_length"], 3),
                        "mean_expanded": round(o["mean_expanded"], 1), "median_expanded": o["median_expanded"]}
    out["report"] = "docs/benchmarks/cube2-learned.md"
    return out


def main(paths: List[Path], what: str) -> str:
    reports = [json.loads(p.read_text()) for p in paths]
    if what == "heuristic":
        return heuristic_table(reports[0])
    if what == "depth":
        return solve_by_depth(reports[0])
    if what == "uniform-depth":
        return solve_by_depth(reports[0], "solve_uniform")
    if what == "settings":
        return settings_table(reports)
    if what == "summary":
        return json.dumps(summary(reports[0]), indent=2)
    raise ValueError(what)
