"""
G2: the RSRD tracking objective (Kerr et al., CoRL 2024; kerrj/rsrd `motion_optimizer.py`), evaluated statically
with our splat and renderer, to test whether the background-silhouette mechanism is a property of the objective
family rather than of POGS. Pre-registered in `docs/note/g2_second_tracker_preregistration.md` (Q1-Q3).

    source tools/cuda_env.sh
    .venv/bin/python -m experiments.g2_rsrd sweeps
    .venv/bin/python -m experiments.g2_rsrd summarise
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch

from data.scene_generator.tool_perturbation import TOOL_OBJECT_NAMES, build_tool_scene
from experiments.b_bias import _argmin, _obs, _optimizer, _render
from experiments.env_info import write_env_json
from experiments.g1_generality import AXES, GRID, PRIMARY, _ctx, _gripper_frame0, _kappa, _ring_cam, load_splat
from model.pogs import losses as LL
from model.pogs.checkpoint import write_json
from model.pogs.evaluate import tracking_camera, variant_config
from model.pogs.information import perturbed_deltas

OUT = "data_results/g2_rsrd"
RGB_W = 0.05             # rsrd RigidGroupOptimizerConfig.rgb_loss_weight
MASK_T = 0.8             # rsrd mask_threshold
DEPTH_IGNORE = 0.1       # rsrd depth_ignore_threshold (m)


def _curves(opt, obs, axis6, grid):
    rf = obs.roi_frames[0]
    dev = opt.init_means.device
    real = rf.dino_feats.to(dev)
    rgb_live = rf.rgb.to(dev)
    depth_live = rf.depth.to(dev) if getattr(rf, "depth", None) is not None else None
    base = opt.part_deltas.detach().clone()
    k = opt.config.blur_kernel_size
    out = {n: [] for n in ("D", "D_in", "D_out", "C", "C_in", "C_out", "Z", "L")}
    with torch.no_grad():
        for g in grid:
            pd = perturbed_deltas(base, 0, torch.as_tensor(axis6 * math.radians(g), dtype=torch.float64))
            o = opt._render(rf.camera, pd, 0, True)
            m = o["accumulation"].squeeze(-1) > MASK_T
            rD = torch.where(m.unsqueeze(-1), o["dino"], LL.gaussian_blur2d(o["dino"], k))
            e = (real - rD).norm(dim=-1)
            v = torch.isfinite(e)
            n = v.sum().clamp(min=1)
            D_in, D_out = float(e[m & v].sum() / n), float(e[~m & v].sum() / n)
            c = (o["rgb"] - rgb_live).abs().mean(dim=-1)
            C_in, C_out = float(c[m].sum() / c.numel()), float(c[~m].sum() / c.numel())
            Z = float("nan")
            if depth_live is not None:
                dl = depth_live.reshape(o["depth"].shape)
                pix = (o["depth"] - dl) ** 2
                sel = m.unsqueeze(-1) & torch.isfinite(dl) & (dl > 0) & (pix < DEPTH_IGNORE ** 2)
                Z = float(pix[sel].mean()) if sel.any() else float("nan")
            for name, val in (("D", D_in + D_out), ("D_in", D_in), ("D_out", D_out), ("C", C_in + C_out),
                              ("C_in", C_in), ("C_out", C_out), ("Z", Z)):
                out[name].append(val)
            out["L"].append(D_in + D_out + RGB_W * (C_in + C_out) + (0.0 if not np.isfinite(Z) else Z))
    return out


def sweeps(device="cuda:0"):
    from data.scene_generator.pick_place_benchmark import BenchmarkConfig
    from data.scene_generator.pick_place_benchmark import tracking_camera as bench_cam
    os.makedirs(OUT, exist_ok=True)
    write_env_json(OUT)
    cfg, model, dino, grouping = load_splat(0, device)
    tcfg = variant_config(cfg.tracking, "POGS")
    bcfg = BenchmarkConfig()
    scene = build_tool_scene()
    grip, grip_tr = _gripper_frame0(scene)
    track = bench_cam(bcfg, device)
    name_to_id = {v: k for k, v in TOOL_OBJECT_NAMES.items()}
    rows = []
    for name in PRIMARY:
        ctx = _ctx(cfg, model, dino, grouping, name_to_id[name])
        obj = next(o for o in scene if o.name == name)
        target = np.asarray(obj.position, float) + np.array([0.0, 0.0, 0.05])
        conds = [("track", scene, {}, track, True), ("track, no shadows", scene, {}, track, False)]
        if name == "power_drill":
            conds.append(("track, gripper (frame 0)", scene + [grip], grip_tr, track, True))
        conds += [(f"heldout az{az:03d}", scene, {}, _ring_cam(target, az, bcfg, device), True)
                  for az in range(0, 360, 30)]
        for label, objs, tr, cam, shadows in conds:
            rgb, depth = _render(objs, cam, tr, shadows, device)
            opt = _optimizer(ctx, tcfg)
            try:
                obs = _obs(rgb, depth, tracking_camera(cam, tcfg, device), ctx, tcfg, opt)
            except RuntimeError as e:
                print(f"[g2] {name} {label}: skipped ({e})", flush=True)
                continue
            for ax, a6 in AXES.items():
                cv = _curves(opt, obs, a6, GRID)
                r = {"object": name, "view": label, "axis": ax, "shadows": shadows, "curves": cv,
                     "kappa": {k: _kappa(GRID, cv[k]) for k in cv},
                     "argmin": {k: _argmin(GRID, cv[k]) for k in ("D", "L", "C", "Z")}}
                rows.append(r)
                print(f"[g2] {name:14s} {label:26s} {ax}: argmin L {r['argmin']['L']:+.0f} D {r['argmin']['D']:+.0f} "
                      f"C {r['argmin']['C']:+.0f}", flush=True)
    write_json(os.path.join(OUT, "sweeps.json"), {"grid_deg": GRID.tolist(), "rows": rows})


def summarise():
    d = json.load(open(os.path.join(OUT, "sweeps.json")))
    L = ["# G2: RSRD objective, static sweeps (seed-0 splat)", "",
         "Pre-registered in `docs/note/g2_second_tracker_preregistration.md`. Evaluation only; no trajectories, so "
         "Static (no motion) does not apply. Rules first.", "",
         "| Object | Q1 median b(D) | Q2 median b(L_RSRD) | n |", "|---|---|---|---|"]
    q1 = q2 = 0
    for name in PRIMARY:
        rows = [r for r in d["rows"] if r["object"] == name]
        bD, bL = [], []
        for r in rows:
            k = r["kappa"]
            if k["D"] > 0:
                bD.append(k["D_out"] / k["D"])
            if k["L"] > 0:
                bL.append((k["D_out"] + RGB_W * k["C_out"]) / k["L"])
        mD, mL = float(np.median(bD)), float(np.median(bL))
        q1 += mD > 0.5
        q2 += mL > 0.5
        L.append(f"| {name} | {mD:.2f} | {mL:.2f} | {len(bD)}/{len(bL)} |")
    g = [r for r in d["rows"] if r["object"] == "power_drill" and r["view"] == "track, gripper (frame 0)"
         and r["axis"] == "x"]
    a3 = g[0]["argmin"]["L"] if g else float("nan")
    L += ["", f"Q1: {q1}/4 → **{'PASS' if q1 >= 3 else 'FAIL'}**", f"Q2: {q2}/4 → **{'PASS' if q2 >= 3 else 'FAIL'}**",
          f"Q3: drill roll argmin of L_RSRD with the frame-0 gripper = {a3:+.0f}° → "
          f"**{'PASS' if a3 <= -5 else 'FAIL'}**", ""]
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["sweeps", "summarise"])
    a = ap.parse_args()
    sweeps() if a.step == "sweeps" else summarise()
