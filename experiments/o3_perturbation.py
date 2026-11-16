"""
PLAN Phase 2 (O3): tool perturbation recovery to 30 deg, splats + pose + KL.
Protocol: experiments/protocols/o3.md (must be FROZEN before results are reported).

Pipeline: tool-scene capture + sequences (data/scene_generator/tool_perturbation.py)
-> own checkpoint (model_checkpoints/tool_table) -> grouping -> track the drill only
(operator click, Sec. IV-C) with each method -> per-frame metrics -> summary.

Per frame (drill group, POGS pose convention x_world = R (x_scan - c) + t):
    pos / rot error, ADD-S, tool-tip error (cm, bit tip)
    kl_gt      weighted_kl( GT-posed scan Gaussians || tracked Gaussians )  [= pose error, PLAN §2.1]
Per trial:
    kl_xy      weighted_kl( tracked at end of rest_x || tracked at end of rest_y )   [supervisor's x vs y]
    recovered  final tip error < 3 cm AND rot error < 5 deg AND never lost (> 10 cm)

    source tools/cuda_env.sh && .venv/bin/python -m experiments.o3_perturbation [--seeds 0 1 2 3 4]
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from experiments import pipeline
from experiments.env_info import write_env_json
from experiments.pipeline import _slug, build_groups, lost_rate, train_or_load
from experiments.v1_base_pogs.config import get_v1_config
from experiments.v2_kalman_pogs.config import _kf
from data.scene_generator.pick_place_benchmark import OBJECT_QUERIES, load_sequence
from data.scene_generator.tool_perturbation import (
    DRILL_ID, DRILL_TIP_LOCAL, TOOL_DATASET_DIR, TOOL_OBJECT_NAMES, ToolCase, build_tool_scene,
    generate_tool_capture, generate_tool_sequence, o3_cases,
)
from metrics.gaussian import gaussian_covariances, weighted_kl
from metrics.pose import SUCCESS_POS_CM, SUCCESS_ROT_DEG, adds_success_rate, object_diameter
from metrics.stats import mean_std, wilson_interval
from model.pogs.checkpoint import write_json
from model.pogs.evaluate import object_model_points, static_baseline, track_sequence, variant_config
from model.pogs.geometry import quat_normalize
from model.pogs.grouping import subset_groups

OUT = "data_results/o3_perturbation"
CKPT_DIR = "model_checkpoints/tool_table"
LOST_CM = 10.0
METHODS = [  # (name, variant, motion prior factory)
    ("POGS", "POGS", None),
    ("POGS (release: no per-frame depth)", "POGS (release: no per-frame depth)", None),
    ("POGS + KF + check", "POGS", _kf(True)),
]


def _posed(means_local: torch.Tensor, covs: torch.Tensor, T: np.ndarray):
    Tt = torch.as_tensor(T, dtype=torch.float64, device=means_local.device)
    R, t = Tt[:3, :3], Tt[:3, 3]
    return means_local @ R.T + t, R @ covs @ R.T


def _tip_err_cm(P: np.ndarray, G: np.ndarray, tip_local: np.ndarray) -> float:
    a = P[:3, :3] @ tip_local + P[:3, 3]
    b = G[:3, :3] @ tip_local + G[:3, 3]
    return float(np.linalg.norm(a - b) * 100)


def score_trial(res, seq, means_local, covs, opac, tip_local, mp) -> Dict:
    phases = [fm["phase"] for fm in seq.meta["frames"]]
    P, G = res.pred_poses[:, 0], res.gt_poses[:, 0]
    F = len(phases)
    tip = np.array([_tip_err_cm(P[f], G[f], tip_local) for f in range(F)])
    kl_gt = []
    for f in range(F):
        mg, cg = _posed(means_local, covs, G[f])
        mpred, cpred = _posed(means_local, covs, P[f])
        kl_gt.append(weighted_kl(mg, cg, mpred, cpred, opac)["kl_weighted_mean"])
    fx = max(i for i, p in enumerate(phases) if p == "rest_x")          # end of rest before perturbation
    fy = F - 1
    mx, cx = _posed(means_local, covs, P[fx])
    my, cy = _posed(means_local, covs, P[fy])
    kl_xy = weighted_kl(mx, cx, my, cy, opac, symmetric=True)
    rot = res.rot_err_deg[:, 0]
    pos = res.trans_err_cm[:, 0]
    recovered = bool(tip[-1] < SUCCESS_POS_CM and rot[-1] < SUCCESS_ROT_DEG and np.nanmax(pos) <= LOST_CM)
    return {
        "per_frame": {"phase": phases, "commanded_deg": [fm["commanded_deg"] for fm in seq.meta["frames"]],
                      "pos_err_cm": pos.tolist(), "rot_err_deg": rot.tolist(), "tip_err_cm": tip.tolist(),
                      "adds_m": res.adds[:, 0].tolist(), "kl_gt": kl_gt},
        "summary": {
            "recovered": recovered, "final_tip_err_cm": float(tip[-1]), "final_rot_err_deg": float(rot[-1]),
            "max_pos_err_cm": float(np.nanmax(pos)), "x_tip_err_cm": float(tip[fx]), "x_rot_err_deg": float(rot[fx]),
            "kl_xy": kl_xy["kl_weighted_mean"], "kl_xy_median": kl_xy["kl_median"],
            "kl_gt_final": kl_gt[-1], "kl_gt_max": float(np.nanmax(kl_gt)),
            "lost_rate": lost_rate(pos),
            "adds_success": adds_success_rate(res.adds[:, 0], object_diameter(mp[0])) if mp else float("nan"),
        },
    }


def setup_tool_context(device: str = "cuda:0") -> Dict:
    """Checkpoint + grouping of the tool scene and everything ``score_trial`` needs about the drill group
    (shared by O3 and O4). Needs the tool capture on disk."""
    cfg = get_v1_config(device)
    model, capture, dino, clip = train_or_load(cfg, dataset_dir=TOOL_DATASET_DIR, ckpt_dir=CKPT_DIR,
                                               objects=build_tool_scene())
    grouping = build_groups(model, capture, cfg, ckpt_dir=CKPT_DIR, clip=clip, object_names=TOOL_OBJECT_NAMES,
                            object_queries={**OBJECT_QUERIES, DRILL_ID: "power drill"})
    drill_groups = [g for g, gt in grouping.group_to_gt.items() if gt == DRILL_ID]
    if not drill_groups:
        raise RuntimeError("grouping found no cluster for the drill; see object_clusters.json")
    g_d = drill_groups[0]
    groups = subset_groups(grouping.groups, [g_d])
    g2gt = {0: DRILL_ID}
    mp = {0: object_model_points(grouping.groups, model)[g_d]}

    mask = grouping.groups.group_masks[g_d]
    gp = model.gauss_params
    c0 = groups.init_p2w[0, :3].detach().double()
    means_local = gp["means"].data[mask].double() - c0
    covs = gaussian_covariances(quat_normalize(gp["quats"].data[mask]).double(), torch.exp(gp["scales"].data[mask]).double())
    opac = torch.sigmoid(gp["opacities"].data[mask]).squeeze(-1).double()
    scan_drill = next(o for o in build_tool_scene() if o.obj_id == DRILL_ID)
    tip_world = np.asarray(scan_drill.position) + DRILL_TIP_LOCAL      # drill yaw 0 at scan
    tip_local = tip_world - c0.cpu().numpy()
    return dict(cfg=cfg, model=model, dino=dino, groups=groups, g2gt=g2gt, mp=mp, means_local=means_local,
                covs=covs, opac=opac, tip_local=tip_local)


def main(seeds: List[int], axes: List[str], device: str = "cuda:0") -> pd.DataFrame:
    os.makedirs(OUT, exist_ok=True)
    write_env_json(OUT)
    if not os.path.exists(os.path.join(TOOL_DATASET_DIR, "capture", "images.npy")):
        generate_tool_capture(TOOL_DATASET_DIR, device=device)
    cases = o3_cases(seeds, axes)
    for c in cases:
        if not os.path.exists(os.path.join(TOOL_DATASET_DIR, c.name, "tracking_metadata.json")):
            generate_tool_sequence(c, TOOL_DATASET_DIR, device=device)

    ctx = setup_tool_context(device)
    cfg, model, dino, groups, g2gt, mp = (ctx[k] for k in ("cfg", "model", "dino", "groups", "g2gt", "mp"))
    means_local, covs, opac, tip_local = (ctx[k] for k in ("means_local", "covs", "opac", "tip_local"))

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
            sc = score_trial(res, seq, means_local, covs, opac, tip_local, mp)
            os.makedirs(run_dir, exist_ok=True)
            write_env_json(run_dir)
            write_json(os.path.join(run_dir, "per_frame.json"), sc["per_frame"])
            row = {"case": c.name, "kind": c.kind, "angle": c.angle, "step": c.step if c.kind == "step" else None,
                   "axis": c.axis, "seed": c.seed, "method": name, "ckpt_id": pipeline.CKPT_ID, **sc["summary"]}
            write_json(rp, row)
            rows.append(row)
            print(f"[o3] {c.name} :: {name}: recovered={row['recovered']} tip={row['final_tip_err_cm']:.2f}cm "
                  f"rot={row['final_rot_err_deg']:.1f}deg kl_xy={row['kl_xy']:.3g}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    write_summary(df)
    return df


def write_summary(df: pd.DataFrame) -> None:
    lines = ["| Condition | Method | Recovered (k/n) | 95% CI | Final tip err (cm) | Final rot err (deg) | KL(x‖y) |",
             "|---|---|---|---|---|---|---|"]
    df = df.copy()
    df["condition"] = df.apply(lambda r: f"step {r.angle:g}/{r.step:g} ({r.axis})" if r.kind == "step"
                               else f"jump {r.angle:+g} ({r.axis})", axis=1)
    for (cond, meth), g in df.groupby(["condition", "method"], sort=False):
        k, n = int(g.recovered.sum()), len(g)
        lo, hi = wilson_interval(k, n)
        t, ts = mean_std(g.final_tip_err_cm)
        r, rs = mean_std(g.final_rot_err_deg)
        kl, kls = mean_std(g.kl_xy)
        lines.append(f"| {cond} | {meth} | {k}/{n} | [{lo:.2f}, {hi:.2f}] | {t:.2f} ± {ts:.2f} | "
                     f"{r:.1f} ± {rs:.1f} | {kl:.3g} ± {kls:.2g} |")
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=list(range(5)))
    ap.add_argument("--axes", nargs="*", default=["z"])
    a = ap.parse_args()
    main(a.seeds, a.axes)
