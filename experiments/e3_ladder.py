"""
Plan rev. 2, E3 (protocol experiments/protocols/e_series.md): does an information-adaptive R beat fixed
and trace R? Methods S, P, KF, TR, FU, OR on O4 trials (MS, the equal-compute multi-start, is not
implemented yet and is listed as missing in the summary rather than silently dropped).

Steps
  calibrate  FU with scale 1 on calibration seeds 40-49: per rest_x frame, rotation NEES of the
             measurement error (optimiser output vs GT, world frame) under R_r. Since R scales with c,
             c = mean NEES(1) / 3. Writes data_results/e3_ladder/calibration.json.
  run        all methods on test seeds 0-39 (refuses unless calibration.json exists AND the git tag
             e-series-frozen exists, so test seeds cannot be run before the protocol is frozen).
  summarise  rules in protocol order (pre-registered, paper-faithful), Static included, paired
             bootstrap FU - TR and FU - KF (single splat until T7: trial level only, labelled).

    source tools/cuda_env.sh && .venv/bin/python -m experiments.e3_ladder calibrate|run|summarise
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess

import numpy as np

from data.scene_generator.pick_place_benchmark import load_sequence
from data.scene_generator.tool_perturbation import TOOL_DATASET_DIR, o4_cases
from experiments.env_info import write_env_json
from experiments.o3_perturbation import score_trial, setup_tool_context
from experiments.o4_ingrasp import judge
from metrics.calibration import paired_difference_bootstrap
from metrics.stats import wilson_interval
from model.pogs.checkpoint import write_json
from model.pogs.evaluate import gt_pose_for_group, static_baseline, track_sequence, variant_config
from model.pogs.geometry import rotmat_to_quat, so3_log
from model.pogs.kalman import KalmanConfig
from model.pogs.kalman_info import InfoKalmanPrior

import torch

OUT = "data_results/e3_ladder"
CAL_SEEDS = range(40, 50)
TEST_SEEDS = range(0, 40)
METHODS = ["S", "P", "KF", "TR", "FU", "OR"]
NAMES = {"S": "Static (no motion)", "P": "POGS", "KF": "POGS + KF (fixed R)", "TR": "POGS + KF (trace R)",
         "FU": "POGS + KF (full R)", "OR": "POGS + KF (oracle R, GT ablation)"}


def _rot_err_world(T_meas: np.ndarray, T_gt: np.ndarray) -> np.ndarray:
    R = torch.as_tensor(T_meas[:3, :3] @ T_gt[:3, :3].T, dtype=torch.float64)
    return so3_log(rotmat_to_quat(R.unsqueeze(0))[0]).numpy()


def _make_prior(mode: str, scale: float, seq, gt_id: int, c0: np.ndarray, use_depth: bool):
    fi = {"i": 0}           # frame counter for the oracle (observe is called once per frame)
    prior = InfoKalmanPrior(1, mode=mode, scale=scale, cfg=KalmanConfig(dt=1.0 / seq.fps), use_depth=use_depth)
    if mode == "oracle":
        def observe(opt, obs, _p=prior):
            T_gt = gt_pose_for_group(seq, fi["i"], gt_id, c0)
            T_m = opt.get_part2world(0).detach().cpu().numpy()
            _p.set_oracle_cov(0, _rot_err_world(T_m, T_gt), T_m[:3, 3] - T_gt[:3, 3])
            fi["i"] += 1
        prior.observe = observe
    return prior


def _run_case(ctx, c, method: str, scale: float):
    cfg, model, dino, groups, g2gt, mp = (ctx[k] for k in ("cfg", "model", "dino", "groups", "g2gt", "mp"))
    seq = load_sequence(TOOL_DATASET_DIR, c.name)
    c0 = groups.init_p2w[0, :3].detach().cpu().numpy()
    tcfg = variant_config(cfg.tracking, "POGS")
    if method == "S":
        res, prior = static_baseline(groups, seq, g2gt, mp), None
    else:
        prior = None if method == "P" else _make_prior(
            {"KF": "fixed", "TR": "trace", "FU": "full", "OR": "oracle"}[method], scale, seq, g2gt[0], c0,
            tcfg.use_depth)
        res = track_sequence(model, groups, seq, dino, cfg, g2gt, name=NAMES[method], tracking_cfg=tcfg,
                             model_points=mp, motion_prior=prior, verbose=False)
    sc = score_trial(res, seq, ctx["means_local"], ctx["covs"], ctx["opac"], ctx["tip_local"], mp,
                     axis_local=ctx["bit_axis_local"])
    return res, sc, prior, seq


def calibrate(device="cuda:0") -> None:
    os.makedirs(OUT, exist_ok=True)
    ctx = setup_tool_context(device)
    nees = []
    for c in o4_cases(CAL_SEEDS):
        res, sc, prior, seq = _run_case(ctx, c, "FU", 1.0)
        ph = sc["per_frame"]["phase"]
        for f, rec in enumerate(prior.log):
            if f >= len(ph) or ph[f] != "rest_x" or 0 not in rec:
                continue
            e = _rot_err_world(res.pred_poses[f, 0], res.gt_poses[f, 0])
            nees.append(float(e @ np.linalg.solve(rec[0]["R_r"], e)))
        print(f"[e3-cal] {c.name}: {len(nees)} frames", flush=True)
    c_hat = float(np.mean(nees)) / 3.0
    write_json(os.path.join(OUT, "calibration.json"), {"scale": c_hat, "n_frames": len(nees),
               "median_nees_scale1": float(np.median(nees)), "seeds": list(CAL_SEEDS),
               "note": "c = mean rotation NEES at scale 1 / 3 over rest_x frames of calibration seeds"})
    print(f"[e3-cal] scale c = {c_hat:.4g} from {len(nees)} frames")


def _frozen() -> bool:
    r = subprocess.run(["git", "tag", "-l", "e-series-frozen"], capture_output=True, text=True)
    return r.stdout.strip() == "e-series-frozen"


def run(device="cuda:0", shard: int = 0, nshards: int = 1) -> None:
    cal = os.path.join(OUT, "calibration.json")
    if not os.path.exists(cal) or not _frozen():
        raise SystemExit("E3 test run refused: needs calibration.json and git tag e-series-frozen (protocol)")
    scale = json.load(open(cal))["scale"]
    if shard == 0:
        write_env_json(OUT)
    ctx = setup_tool_context(device)
    cases = [c for i, c in enumerate(o4_cases(TEST_SEEDS)) if i % nshards == shard]
    for c in cases:
        for m in METHODS:
            rp = os.path.join(OUT, "runs", c.name, m, "row.json")
            if os.path.exists(rp):
                continue
            _, sc, _, _ = _run_case(ctx, c, m, scale)
            row = {"case": c.name, "angle": c.angle, "seed": c.seed, "method": m, **judge(sc["per_frame"])}
            os.makedirs(os.path.dirname(rp), exist_ok=True)
            write_json(os.path.join(os.path.dirname(rp), "per_frame.json"), sc["per_frame"])
            write_json(rp, row)
        print(f"[e3] {c.name} done", flush=True)


def summarise() -> None:
    import pandas as pd
    rows = [json.load(open(p)) for p in glob.glob(os.path.join(OUT, "runs", "*", "*", "row.json"))]
    if not rows:
        print("[e3] no rows")
        return
    df = pd.DataFrame(rows)
    L = ["# E3: information-adaptive R ladder (protocol e_series.md)", "",
         "Single splat until T7 (≥ 3 splats): intervals are over trials only. MS (equal-compute multi-start) "
         "NOT run yet.", ""]
    for rule, col in [("Pre-registered rule (tip < 3 cm, geodesic < 5°)", "success"),
                      ("Paper-faithful rule (tip < 3 cm, bit-axis swing < 5°, roll free)", "success_rollfree")]:
        L += [f"## {rule}", "", "| Setting | " + " | ".join(NAMES[m] for m in METHODS) + " |",
              "|---|" + "---|" * len(METHODS)]
        for ang, g in df.groupby("angle"):
            cells = []
            for m in METHODS:
                s = g[g.method == m][col].astype(bool)
                lo, hi = wilson_interval(int(s.sum()), len(s)) if len(s) else (np.nan, np.nan)
                cells.append(f"{int(s.sum())}/{len(s)} [{lo:.2f}, {hi:.2f}]")
            L.append(f"| {ang:+g}° | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## P2: paired differences (paper-faithful rule), bootstrap over trials", "",
          "| Setting | FU − TR | FU − KF |", "|---|---|---|"]
    for ang, g in df.groupby("angle"):
        piv = g.pivot_table(index="case", columns="method", values="success_rollfree", aggfunc="first").dropna()
        cells = []
        for other in ("TR", "KF"):
            if {"FU", other} <= set(piv.columns) and len(piv):
                b = paired_difference_bootstrap(piv["FU"].astype(float), piv[other].astype(float), [])
                cells.append(f"{b['estimate']:+.2f} [{b['lo']:+.2f}, {b['hi']:+.2f}]")
            else:
                cells.append("–")
        L.append(f"| {ang:+g}° | " + " | ".join(cells) + " |")
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["calibrate", "run", "summarise"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    {"calibrate": calibrate, "summarise": summarise}.get(a.step, lambda: run(shard=a.shard, nshards=a.nshards))()
