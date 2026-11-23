"""
E2b (plan rev. 2 decision table, "features or loss mismatch"): does adding the RGB photometric term
(``TrackingConfig.use_rgb``, already in the tracker; configuration only, CLAUDE.md rule 2) remove the
DINO roll bias on O4? Found by `docs/results/e2_loss_bias_2026-11-23.md`: the DINO term's minimum is ~10° of
bit-axis roll off the truth at the scan pose; RGB's minimum is at the truth.

Method "POGS + RGB" on the same O4 trials as E0 (seeds 0-24, 4 settings). Compared against the E0 rows
(POGS, Static) for the same cases; both rules reported, pre-registered first.

    source tools/cuda_env.sh && .venv/bin/python -m experiments.e2b_rgb [--seeds 25] [--summarise]
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os

import numpy as np

from data.scene_generator.pick_place_benchmark import load_sequence
from data.scene_generator.tool_perturbation import TOOL_DATASET_DIR, o4_cases
from experiments.env_info import write_env_json
from experiments.o3_perturbation import score_trial, setup_tool_context
from experiments.o4_ingrasp import judge
from metrics.calibration import paired_difference_bootstrap
from metrics.stats import wilson_interval
from model.pogs.checkpoint import write_json
from model.pogs.evaluate import track_sequence, variant_config

OUT = "data_results/e2b_rgb"
E0_RUNS = "data_results/o4_ingrasp/runs"
# Config-only variants (CLAUDE.md rule 2). "rgb1" is the original E2b; the depth-weight ladder is direction B,
# step 1 (docs/results/b_bias_diagnostics_2026-11-23.md: depth is unbiased in roll but too flat at 3.7).
VARIANTS = {
    "rgb1": ("POGS + RGB", dict(use_rgb=True), "data_results/e2b_rgb"),
    "depth30": ("POGS (depth x30)", dict(depth_loss_mult=30.0), "data_results/e2c_depth30"),
    "depth100": ("POGS (depth x100)", dict(depth_loss_mult=100.0), "data_results/e2c_depth100"),
    # control for E2c: is a heavy depth weight a fix, or does it just switch DINO off? (POGS Table III ablation:
    # tracking without DINO features gave zero consecutive successful resets)
    "depthonly": ("POGS (depth only, no DINO)", dict(use_dino=False), "data_results/e2c_depthonly"),
}


def run(seeds: int, shard: int = 0, nshards: int = 1, device: str = "cuda:0", variant: str = "rgb1") -> None:
    name, over, OUT = VARIANTS[variant]
    os.makedirs(OUT, exist_ok=True)
    if shard == 0:
        write_env_json(OUT)
    ctx = setup_tool_context(device)
    cfg = ctx["cfg"]
    tcfg = dataclasses.replace(variant_config(cfg.tracking, "POGS"), **over)
    for i, c in enumerate(o4_cases(range(seeds))):
        if i % nshards != shard:
            continue
        rp = os.path.join(OUT, "runs", c.name, "row.json")
        if os.path.exists(rp):
            continue
        seq = load_sequence(TOOL_DATASET_DIR, c.name)
        res = track_sequence(ctx["model"], ctx["groups"], seq, ctx["dino"], cfg, ctx["g2gt"], name=name,
                             tracking_cfg=tcfg, model_points=ctx["mp"], verbose=False)
        sc = score_trial(res, seq, ctx["means_local"], ctx["covs"], ctx["opac"], ctx["tip_local"], ctx["mp"],
                         axis_local=ctx["bit_axis_local"])
        row = {"case": c.name, "angle": c.angle, "seed": c.seed, "method": name, **judge(sc["per_frame"])}
        os.makedirs(os.path.dirname(rp), exist_ok=True)
        write_json(os.path.join(os.path.dirname(rp), "per_frame.json"), sc["per_frame"])
        write_json(rp, row)
        print(f"[e2b:{variant}] {c.name} done", flush=True)


def summarise(variant: str = "rgb1") -> None:
    import pandas as pd
    name, _, OUT = VARIANTS[variant]
    new = [json.load(open(p)) for p in glob.glob(os.path.join(OUT, "runs", "*", "row.json"))]
    if not new:
        print("[e2b] no rows")
        return
    cases = {r["case"] for r in new}
    old = []
    for slug, _base in [("static__no_motion", "Static (no motion)"), ("pogs", "POGS")]:
        for cname in cases:
            p = os.path.join(E0_RUNS, cname, slug, "row.json")
            if os.path.exists(p):
                r = json.load(open(p))
                if r.get("schema", 1) >= 2:
                    old.append(r)
    df = pd.DataFrame(new + old)
    meths = ["Static (no motion)", "POGS", name]
    L = [f"# E2b/E2c: {name} on O4 (same cases as E0)", "", f"{len(cases)} cases. Single splat.", ""]
    for rule, col in [("Pre-registered rule", "success"), ("Paper-faithful rule (roll free)", "success_rollfree")]:
        L += [f"## {rule}", "", "| Setting | " + " | ".join(meths) + " |", "|---|" + "---|" * len(meths)]
        for ang, g in df.groupby("angle"):
            cells = []
            for m in meths:
                s = g[g.method == m][col].astype(bool)
                lo, hi = wilson_interval(int(s.sum()), len(s)) if len(s) else (np.nan, np.nan)
                cells.append(f"{int(s.sum())}/{len(s)} [{lo:.2f}, {hi:.2f}]")
            L.append(f"| {ang:+g}° | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## Roll before the perturbation and paired difference (paper-faithful)", "",
          f"| Setting | abs roll pre, POGS (deg) | abs roll pre, {name} (deg) | variant − POGS success [95% CI] |", "|---|---|---|---|"]
    for ang, g in df.groupby("angle"):
        piv = g.pivot_table(index="case", columns="method", values="success_rollfree", aggfunc="first").dropna()
        b = paired_difference_bootstrap(piv[name].astype(float), piv["POGS"].astype(float), []) \
            if {"POGS", name} <= set(piv.columns) and len(piv) else None
        rp = g[g.method == "POGS"].roll_pre_deg.abs().mean()
        rr = g[g.method == name].roll_pre_deg.abs().mean()
        L.append(f"| {ang:+g}° | {rp:.1f} | {rr:.1f} | " + (f"{b['estimate']:+.2f} [{b['lo']:+.2f}, {b['hi']:+.2f}]" if b else "–") + " |")
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=25)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--variant", default="rgb1", choices=sorted(VARIANTS))
    a = ap.parse_args()
    if a.summarise:
        summarise(a.variant)
    else:
        run(a.seeds, a.shard, a.nshards, variant=a.variant)
        if a.nshards == 1:
            summarise(a.variant)
