"""
Compact tables for RESULTS.md, built from the cached per-run rows of all
experiments (v1, v2, v3, real-time). Prints Markdown to stdout.

    python -m experiments.results_tables > /tmp/tables.md
"""

from __future__ import annotations

import glob
import json

import numpy as np
import pandas as pd

from data.scene_generator.pick_place_benchmark import FACTORS, SCENE_VERSION

RUNS = "data_results/benchmark_runs"
FACTOR_METHODS = ["Static (no motion)", "POGS", "POGS (release: no per-frame depth)", "POGS + KF",
                  "POGS + KF + check"]
BUDGET_METHODS = ["POGS", "POGS + KF", "POGS + KF + check", "POGS, 10 iters", "POGS + KF, 10 iters",
                  "POGS + KF + check, 10 iters", "POGS, 5 iters", "POGS + KF, 5 iters",
                  "POGS + KF + check, 5 iters"]
SPEED_METHODS = ["POGS", "POGS + KF + check", "POGS, 10 iters", "POGS + KF + check, 10 iters",
                 "POGS, 5 iters", "POGS + KF + check, 5 iters"]
ABLATION_METHODS = ["Static (no motion)", "POGS", "No Depth", "No DINO (RGB)",
                    "POGS (release: no per-frame depth)"]


def load_rows() -> pd.DataFrame:
    rows = []
    for f in glob.glob(f"{RUNS}/*/*/row.json"):
        with open(f) as fh:
            r = json.load(fh)
        if r.get("scene_version") == SCENE_VERSION:
            rows.append(r)
    df = pd.DataFrame(rows)
    return df.drop_duplicates(["case", "method"])


def _cell(g: pd.DataFrame) -> str:
    if g.empty:
        return "-"
    s, lost = g["adds_success"] * 100, g["lost_rate"] * 100
    sd = f" ±{s.std(ddof=1):.0f}" if len(s) > 1 else ""
    return f"{s.mean():.0f}%{sd} / {lost.mean():.0f}%"


def _case_rows(df: pd.DataFrame, cond: str, speed: int, depth: str) -> pd.DataFrame:
    return df[(df["condition"] == cond) & (df["speed"] == speed) & (df["depth"] == depth)
              & ~df["case"].str.contains("_30hz")]


def factor_table(df: pd.DataFrame, methods) -> str:
    out = ["| Factor | Level | " + " | ".join(methods) + " |", "|---|---|" + "---:|" * len(methods)]
    for factor, levels in FACTORS.items():
        for label, cond, speed, depth in levels:
            d = _case_rows(df, cond, speed, depth)
            out.append(f"| {factor} | {label} | " + " | ".join(_cell(d[d["method"] == m]) for m in methods) + " |")
    return "\n".join(out)


def speed_table(df: pd.DataFrame) -> str:
    out = ["| Level | " + " | ".join(SPEED_METHODS) + " |", "|---|" + "---:|" * len(SPEED_METHODS)]
    for label, cond, speed, depth in FACTORS["speed"]:
        d = _case_rows(df, cond, speed, depth)
        out.append(f"| {label} | " + " | ".join(_cell(d[d["method"] == m]) for m in SPEED_METHODS) + " |")
    return "\n".join(out)


def budget_table(df: pd.DataFrame) -> str:
    d = df[~df["case"].str.contains("_30hz")]
    out = ["| Method | Iters/frame | FPS | ADD-S success | Lost (>10 cm) | Median pos (cm) | Rot err (deg) |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for m in BUDGET_METHODS:
        g = d[d["method"] == m]
        if g.empty:
            continue
        iters = 10 if "10 iters" in m else 5 if "5 iters" in m else 25
        out.append(f"| {m} | {iters} | {g['fps'].mean():.1f} | {100 * g['adds_success'].mean():.0f}% | "
                   f"{100 * g['lost_rate'].mean():.0f}% | {g['median_pos_err_cm'].mean():.1f} | "
                   f"{g['mean_rot_err_deg'].mean():.1f} |")
    out.append(f"\n{d[d['method'] == 'POGS']['case'].nunique()} cases per method.")
    return "\n".join(out)


def ablation_table(df: pd.DataFrame) -> str:
    d = _case_rows(df, "base", 1, "stereo")
    out = ["| Method | ADD-S success | Lost | Median pos (cm) | Rot err (deg) |", "|---|---:|---:|---:|---:|"]
    for m in ABLATION_METHODS:
        g = d[d["method"] == m]
        if g.empty:
            continue
        out.append(f"| {m} | {100 * g['adds_success'].mean():.0f}% | {100 * g['lost_rate'].mean():.0f}% | "
                   f"{g['median_pos_err_cm'].mean():.2f} | {g['mean_rot_err_deg'].mean():.1f} |")
    return "\n".join(out)


def main() -> None:
    df = load_rows()
    print("### Per factor: ADD-S success ± seed std / lost-frame rate (25 iterations)\n")
    print(factor_table(df, FACTOR_METHODS))
    print("\n### Compute budget: all offline cases\n")
    print(budget_table(df))
    print("\n### Speed factor at reduced budgets (ADD-S success ± std / lost)\n")
    print(speed_table(df))
    print("\n### Paper ablations (base condition, 3 seeds)\n")
    print(ablation_table(df))


if __name__ == "__main__":
    main()
