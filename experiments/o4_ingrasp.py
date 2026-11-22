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
from metrics.pose import SUCCESS_POS_CM, SUCCESS_ROT_DEG, tool_success_roll_free
from metrics.stats import mean_std, target_verdict, wilson_interval
from model.pogs.checkpoint import write_json
from model.pogs.evaluate import static_baseline, track_sequence, variant_config

OUT = "data_results/o4_ingrasp"
N_AFTER = 10
TARGET = 0.80
ROW_SCHEMA = 2
"""2: per-frame record stores full SE(3) plus roll/swing about the bit axis, and rows carry the
paper-faithful roll-free rule (plan rev. 2, E0; LEDGER deviation 2026-11-22). Rows of an older schema
are re-run, not reused."""


def judge(per_frame: dict) -> dict:
    ph = per_frame["phase"]
    p0 = ph.index("perturbed")
    k = min(p0 + N_AFTER, len(ph) - 1)
    tip, rot, pos = (np.asarray(per_frame[x]) for x in ("tip_err_cm", "rot_err_deg", "pos_err_cm"))
    never_lost = bool(np.nanmax(pos[p0:]) <= LOST_CM)
    ok = bool(tip[k] < SUCCESS_POS_CM and rot[k] < SUCCESS_ROT_DEG and never_lost)
    out = {"success": ok, "tip_err_at_N_cm": float(tip[k]), "rot_err_at_N_deg": float(rot[k]),
           "max_pos_err_after_cm": float(np.nanmax(pos[p0:])), "tip_err_pre_cm": float(tip[p0 - 1]),
           "rot_err_pre_deg": float(rot[p0 - 1]), "p0": p0}
    if "swing_err_deg" in per_frame:
        roll, swing = np.asarray(per_frame["roll_err_deg"]), np.asarray(per_frame["swing_err_deg"])
        out.update({"success_rollfree": bool(tool_success_roll_free(tip[k], swing[k]) and never_lost),
                    "swing_at_N_deg": float(swing[k]), "roll_at_N_deg": float(roll[k]),
                    "swing_pre_deg": float(swing[p0 - 1]), "roll_pre_deg": float(roll[p0 - 1])})
    return out


def _method_names():
    return ["Static (no motion)"] + [m[0] for m in METHODS]


def aggregate(trials: int) -> pd.DataFrame:
    """Collect every cached row.json (all shards) into summary.csv / summary.md."""
    from experiments.o3_perturbation import CKPT_DIR
    pipeline._set_ckpt_id(CKPT_DIR)
    rows = []
    for c in o4_cases(range(trials)):
        for name in _method_names():
            rp = os.path.join(OUT, "runs", c.name, _slug(name), "row.json")
            if os.path.exists(rp):
                row = json.load(open(rp))
                if row.get("ckpt_id") == pipeline.CKPT_ID:
                    rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    write_summary(df)
    print(f"[o4] aggregated {len(df)} rows ({len(df) / len(_method_names()):.0f} cases of {trials * 4})", flush=True)
    return df


def main(trials: int, device: str = "cuda:0", shard: int = 0, nshards: int = 1) -> pd.DataFrame:
    """Shard k of n handles cases with index % n == k, so n processes can share one GPU."""
    os.makedirs(OUT, exist_ok=True)
    if shard == 0:
        write_env_json(OUT)
    if not os.path.exists(os.path.join(TOOL_DATASET_DIR, "capture", "images.npy")):
        generate_tool_capture(TOOL_DATASET_DIR, device=device)
    cases = [c for i, c in enumerate(o4_cases(range(trials))) if i % nshards == shard]
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
                if row.get("ckpt_id") == pipeline.CKPT_ID and row.get("schema", 1) >= ROW_SCHEMA:
                    rows.append(row)
                    continue
            if variant is None:
                res = static_baseline(groups, seq, g2gt, mp)
            else:
                prior = prior_f(1, 1.0 / seq.fps) if prior_f else None
                res = track_sequence(model, groups, seq, dino, cfg, g2gt, name=name,
                                     tracking_cfg=variant_config(cfg.tracking, variant), model_points=mp,
                                     motion_prior=prior, verbose=False)
            sc = score_trial(res, seq, ctx["means_local"], ctx["covs"], ctx["opac"], ctx["tip_local"], mp,
                             axis_local=ctx["bit_axis_local"])
            row = {"case": c.name, "angle": c.angle, "seed": c.seed, "method": name, "ckpt_id": pipeline.CKPT_ID,
                   "schema": ROW_SCHEMA,
                   **judge(sc["per_frame"]), "kl_xy": sc["summary"]["kl_xy"]}
            os.makedirs(run_dir, exist_ok=True)
            write_env_json(run_dir)
            write_json(os.path.join(run_dir, "per_frame.json"), sc["per_frame"])
            write_json(rp, row)
            rows.append(row)
        if c.seed % 10 == 0:
            print(f"[o4] {c.name} done", flush=True)
    if nshards == 1:
        return aggregate(trials)
    return pd.DataFrame(rows)


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
    if "success_rollfree" in df and df["success_rollfree"].notna().any():
        lines += ["", "**Paper-faithful rule** (deviation logged 2026-11-22, taken from POGS Sec. V-B, not from our "
                  "results): tip < 3 cm and bit-axis direction (swing) < 5 deg at N, never lost; roll about the bit "
                  "axis free. Pre-registered rule above stays the primary result.", "",
                  "| Setting | Method | Recovered (k/n) | 95% CI (Wilson) | vs 80% | Swing at N (deg) | Roll at N (deg) | Roll pre (deg) |",
                  "|---|---|---|---|---|---|---|---|"]
        for (ang, meth), g in df.dropna(subset=["success_rollfree"]).groupby(["angle", "method"], sort=False):
            k, n = int(g.success_rollfree.astype(bool).sum()), len(g)
            lo, hi = wilson_interval(k, n)
            w, ws = mean_std(g.swing_at_N_deg)
            r, rs = mean_std(g.roll_at_N_deg.abs())
            r0, r0s = mean_std(g.roll_pre_deg.abs())
            lines.append(f"| {ang:+g}° | {meth} | {k}/{n} | [{lo:.2f}, {hi:.2f}] | {target_verdict(k, n, TARGET)} | "
                         f"{w:.1f} ± {ws:.1f} | {r:.1f} ± {rs:.1f} | {r0:.1f} ± {r0s:.1f} |")
    lines.append("\nSuccess: tip < 3 cm and rot < 5 deg at 10 frames (2 s) after the perturbation, never lost after it. "
                 "Paper (physical robot, reference only): 48/50 at 15 deg, 40/50 at 30 deg. "
                 "Jaws are not collision-checked against the turning drill.")
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--aggregate", action="store_true", help="only rebuild summary from cached rows")
    a = ap.parse_args()
    if a.aggregate:
        aggregate(a.trials)
    else:
        main(a.trials, shard=a.shard, nshards=a.nshards)
