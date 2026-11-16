"""
PLAN Phase 3 (O4): in-grasp tool perturbation recovery rate, target 80 %.
Protocol: experiments/protocols/o4.md. 4 settings (+-15, +-30 deg) x 50 trials x methods.

Trial = data/scene_generator/tool_perturbation.generate_grasp_sequence (gripper holds the drill, the
drill turns inside the closed gripper at frame p0). Success (fixed in advance): at frame p0 + N
(N = 10, i.e. 2 s at 5 Hz) tool-tip error < 3 cm AND rotation error < 5 deg, and the drill was never
lost (> 10 cm) after p0. Rate with Wilson 95 % CI; verdict vs 80 % via metrics.stats.target_verdict.
Same trials for every method (Static, POGS, POGS (release), POGS + KF + check).

    source tools/cuda_env.sh && .venv/bin/python -m experiments.o4_ingrasp [--trials 50]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from experiments import pipeline
from experiments.env_info import write_env_json
from experiments.o3_perturbation import LOST_CM, METHODS, score_trial, setup_tool_context
from experiments.pipeline import _slug
from data.scene_generator.pick_place_benchmark import load_sequence
from data.scene_generator.tool_perturbation import (
    TOOL_DATASET_DIR, generate_grasp_sequence, generate_tool_capture, o4_cases,
)
from metrics.pose import SUCCESS_POS_CM, SUCCESS_ROT_DEG
from metrics.stats import mean_std, target_verdict, wilson_interval
from model.pogs.checkpoint import write_json
from model.pogs.evaluate import static_baseline, track_sequence, variant_config

OUT = "data_results/o4_ingrasp"
N_AFTER = 10
TARGET = 0.80


def judge(per_frame: dict) -> dict:
    ph = per_frame["phase"]
    p0 = ph.index("perturbed")
    k = min(p0 + N_AFTER, len(ph) - 1)
    tip, rot, pos = (np.asarray(per_frame[x]) for x in ("tip_err_cm", "rot_err_deg", "pos_err_cm"))
    ok = bool(tip[k] < SUCCESS_POS_CM and rot[k] < SUCCESS_ROT_DEG and np.nanmax(pos[p0:]) <= LOST_CM)
    return {"success": ok, "tip_err_at_N_cm": float(tip[k]), "rot_err_at_N_deg": float(rot[k]),
            "max_pos_err_after_cm": float(np.nanmax(pos[p0:])), "tip_err_pre_cm": float(tip[p0 - 1]),
            "rot_err_pre_deg": float(rot[p0 - 1]), "p0": p0}


def main(trials: int, device: str = "cuda:0") -> pd.DataFrame:
    os.makedirs(OUT, exist_ok=True)
    write_env_json(OUT)
    if not os.path.exists(os.path.join(TOOL_DATASET_DIR, "capture", "images.npy")):
        generate_tool_capture(TOOL_DATASET_DIR, device=device)
    cases = o4_cases(range(trials))
    for c in cases:
        if not os.path.exists(os.path.join(TOOL_DATASET_DIR, c.name, "tracking_metadata.json")):
            generate_grasp_sequence(c, TOOL_DATASET_DIR, device=device, verbose=False)
    ctx = setup_tool_context(device)
    cfg, model, dino, groups, g2gt, mp = (ctx[k] for k in ("cfg", "model", "dino", "groups", "g2gt", "mp"))

    rows = []
    for c in cases:
        seq = load_sequence(TOOL_DATASET_DIR, c.name)
        for name, variant, prior_f in [("Static (no motion)", None, None)] + METHODS:
            run_dir = os.path.join(OUT, "runs", c.name, _slug(name))
            rp = os.path.join(run_dir, "row.json")
            if os.path.exists(rp):
                row = json.load(open(rp))
                if row.get("ckpt_id") == pipeline.CKPT_ID:
                    rows.append(row)
                    continue
            if variant is None:
                res = static_baseline(groups, seq, g2gt, mp)
            else:
                prior = prior_f(1, 1.0 / seq.fps) if prior_f else None
                res = track_sequence(model, groups, seq, dino, cfg, g2gt, name=name,
                                     tracking_cfg=variant_config(cfg.tracking, variant), model_points=mp,
                                     motion_prior=prior, verbose=False)
            sc = score_trial(res, seq, ctx["means_local"], ctx["covs"], ctx["opac"], ctx["tip_local"], mp)
            row = {"case": c.name, "angle": c.angle, "seed": c.seed, "method": name, "ckpt_id": pipeline.CKPT_ID,
                   **judge(sc["per_frame"]), "kl_xy": sc["summary"]["kl_xy"]}
            os.makedirs(run_dir, exist_ok=True)
            write_env_json(run_dir)
            write_json(os.path.join(run_dir, "per_frame.json"), sc["per_frame"])
            write_json(rp, row)
            rows.append(row)
        if c.seed % 10 == 0:
            print(f"[o4] {c.name} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    write_summary(df)
    return df


def write_summary(df: pd.DataFrame) -> None:
    lines = ["| Setting | Method | Recovered (k/n) | Rate | 95% CI (Wilson) | vs 80% | Tip err at N (cm) | Rot err at N (deg) |",
             "|---|---|---|---|---|---|---|---|"]
    for (ang, meth), g in df.groupby(["angle", "method"], sort=False):
        k, n = int(g.success.sum()), len(g)
        lo, hi = wilson_interval(k, n)
        t, ts = mean_std(g.tip_err_at_N_cm)
        r, rs = mean_std(g.rot_err_at_N_deg)
        lines.append(f"| {ang:+g}° | {meth} | {k}/{n} | {k / n:.0%} | [{lo:.2f}, {hi:.2f}] | "
                     f"{target_verdict(k, n, TARGET)} | {t:.2f} ± {ts:.2f} | {r:.1f} ± {rs:.1f} |")
    lines.append("\nSuccess: tip < 3 cm and rot < 5 deg at 10 frames (2 s) after the perturbation, never lost after it. "
                 "Paper (physical robot, reference only): 48/50 at 15 deg, 40/50 at 30 deg. "
                 "Jaws are not collision-checked against the turning drill.")
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    a = ap.parse_args()
    main(a.trials)
