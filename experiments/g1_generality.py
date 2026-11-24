"""
G1: does the background-silhouette mechanism of the DINO term generalise across splats and objects? Static loss
sweeps only (no trajectories). Pre-registered in `docs/note/g1_generality_preregistration.md` (P1-P5); evaluation
only: ground truth places the sweep and names the group, the tracker never sees it.

    source tools/cuda_env.sh
    .venv/bin/python -m experiments.g1_generality train --seed 1      # new splat (seed 0 = existing tool_table)
    .venv/bin/python -m experiments.g1_generality sweeps --seed 0
    .venv/bin/python -m experiments.g1_generality dose --seed 0
    .venv/bin/python -m experiments.g1_generality summarise
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Dict, List

import numpy as np
import torch

from data.scene_generator.tool_perturbation import DRILL_ID, TOOL_DATASET_DIR, TOOL_OBJECT_NAMES, build_tool_scene
from data.scene_generator.pick_place_benchmark import OBJECT_QUERIES
from experiments.b_bias import _argmin, _obs, _optimizer, _render, _terms
from experiments.env_info import write_env_json
from experiments.pipeline import build_groups, train_or_load
from experiments.v1_base_pogs.config import get_v1_config
from model.pogs import losses as LL
from model.pogs.checkpoint import write_json
from model.pogs.evaluate import tracking_camera, variant_config
from model.pogs.grouping import subset_groups
from model.pogs.information import perturbed_deltas

OUT = "data_results/g1_generality"
GRID = np.arange(-20.0, 20.0 + 1e-9, 1.0)
DOSE_GRID = np.arange(-20.0, 20.0 + 1e-9, 0.5)
DOSE_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
PRIMARY = ("power_drill", "ceramic_mug", "mustard_bottle", "cereal_box")
EXPLORATORY = ("soup_can", "red_apple")
AXES = {"x": np.r_[1.0, 0, 0, 0, 0, 0], "y": np.r_[0, 1.0, 0, 0, 0, 0], "z": np.r_[0, 0, 1.0, 0, 0, 0]}


def ckpt_dir(seed: int) -> str:
    return "model_checkpoints/tool_table" if seed == 0 else f"model_checkpoints/tool_table_seed{seed}"


def load_splat(seed: int, device: str = "cuda:0"):
    cfg = get_v1_config(device)
    cfg.train.seed = seed
    model, capture, dino, clip = train_or_load(cfg, dataset_dir=TOOL_DATASET_DIR, ckpt_dir=ckpt_dir(seed),
                                               objects=build_tool_scene())
    grouping = build_groups(model, capture, cfg, ckpt_dir=ckpt_dir(seed), clip=clip, object_names=TOOL_OBJECT_NAMES,
                            object_queries={**OBJECT_QUERIES, DRILL_ID: "power drill"})
    return cfg, model, dino, grouping


def _ctx(cfg, model, dino, grouping, gt_id):
    gs = [g for g, gt in grouping.group_to_gt.items() if gt == gt_id]
    if not gs:
        return None
    return dict(cfg=cfg, model=model, dino=dino, groups=subset_groups(grouping.groups, [gs[0]]))


def _gripper_frame0(objects):
    from data.scene_generator.pick_place_benchmark import GRIPPER_ID
    from data.scene_generator.photoreal_scene import make_robot_gripper
    from data.scene_generator.tool_perturbation import (APPROACH_FRAMES, GRASP_APPROACH_H, GRASP_PIVOT_LOCAL,
                                                        _smooth, _translate)
    drill = next(o for o in objects if o.obj_id == DRILL_ID)
    c = np.asarray(drill.position) + GRASP_PIVOT_LOCAL
    g = make_robot_gripper(scale=1.0)
    g.position, g.yaw, g.obj_id, g.movable = (c[0], c[1], drill.position[2] + 0.138), drill.yaw + math.pi / 2, \
        GRIPPER_ID, True
    return g, {GRIPPER_ID: _translate([0, 0, GRASP_APPROACH_H * (1 - _smooth(1 / APPROACH_FRAMES))])}


def _ring_cam(target, az, bcfg, device, radius=0.87, elev=30.0):
    from model.pogs.geometry import PinholeCamera, look_at
    a, e = math.radians(az), math.radians(elev)
    eye = target + radius * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    return PinholeCamera(torch.from_numpy(look_at(eye, target)).float().to(device), bcfg.track_fx, bcfg.track_fx,
                         bcfg.track_width / 2.0, bcfg.track_height / 2.0, bcfg.track_width, bcfg.track_height)


def _curves(opt, obs, axis6, grid, with_depth=True) -> Dict[str, List[float]]:
    """Per grid point: release term L and its pixel-sum split S_in + S_out (mask at the evaluated pose), the
    inside-object mean, the rendered-background term, and the depth term."""
    rf = obs.roi_frames[0]
    real = rf.dino_feats.to(opt.init_means.device)
    base = opt.part_deltas.detach().clone()
    k = opt.config.blur_kernel_size
    out = {n: [] for n in ("L", "S_in", "S_out", "inside_mean", "bg_rendered", "depth", "mask_frac", "split_err")}
    with torch.no_grad():
        for g in grid:
            pd = perturbed_deltas(base, 0, torch.as_tensor(axis6 * math.radians(g), dtype=torch.float64))
            o = opt._render(rf.camera, pd, 0, True)
            e = (real - LL.gaussian_blur2d(o["dino"], k)).norm(dim=-1)
            valid = torch.isfinite(e)
            n = valid.sum().clamp(min=1)
            m = (o["accumulation"].squeeze(-1) >= opt.config.dino_alpha_thresh) & valid
            L = float(e[valid].sum() / n)
            s_in, s_out = float(e[m].sum() / n), float(e[valid & ~m].sum() / n)
            out["L"].append(L)
            out["S_in"].append(s_in)
            out["S_out"].append(s_out)
            out["split_err"].append(abs(L - s_in - s_out))
            out["inside_mean"].append(float(e[m].mean()) if m.any() else float("nan"))
            out["mask_frac"].append(float(m.sum() / n))
            ob = opt._render(rf.camera, pd, None, True, all_gaussians=True)
            out["bg_rendered"].append(float((real - LL.gaussian_blur2d(ob["dino"], k)).norm(dim=-1).nanmean()))
            if with_depth:
                out["depth"].append(_terms(opt, obs, base, axis6 * math.radians(g))["depth"])
    return out


def _kappa(grid, v, half=5.0) -> float:
    grid, v = np.asarray(grid), np.asarray(v, float)
    sel = (np.abs(grid) <= half + 1e-9) & np.isfinite(v)
    return float(np.polyfit(grid[sel], v[sel], 2)[0]) if sel.sum() >= 3 else float("nan")


def _row(label, axis, grid, cv, shadows, extra=None):
    r = {"view": label, "axis": axis, "shadows": shadows, **(extra or {}), "curves": cv}
    r["kappa"] = {k: _kappa(grid, cv[k]) for k in ("L", "S_in", "S_out")}
    r["argmin"] = {k: _argmin(grid, cv[k]) for k in ("L", "inside_mean", "bg_rendered", "depth") if cv[k]}
    return r


def sweeps(seed: int, device: str = "cuda:0", objects_only=None) -> None:
    from data.scene_generator.pick_place_benchmark import BenchmarkConfig
    from data.scene_generator.pick_place_benchmark import tracking_camera as bench_cam
    os.makedirs(OUT, exist_ok=True)
    write_env_json(OUT)
    cfg, model, dino, grouping = load_splat(seed, device)
    tcfg = variant_config(cfg.tracking, "POGS")
    bcfg = BenchmarkConfig()
    scene = build_tool_scene()
    grip, grip_tr = _gripper_frame0(scene)
    track = bench_cam(bcfg, device)
    name_to_id = {v: k for k, v in TOOL_OBJECT_NAMES.items()}
    rows = []
    for name in (objects_only or PRIMARY + EXPLORATORY):
        ctx = _ctx(cfg, model, dino, grouping, name_to_id[name])
        if ctx is None:
            print(f"[g1] seed {seed} {name}: no group, skipped", flush=True)
            continue
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
                print(f"[g1] seed {seed} {name} {label}: skipped ({e})", flush=True)
                continue
            for ax, a6 in AXES.items():
                r = _row(label, ax, GRID, _curves(opt, obs, a6, GRID), shadows, {"object": name, "seed": seed})
                rows.append(r)
                kL = r["kappa"]["L"]
                b = r["kappa"]["S_out"] / kL if kL > 0 else float("nan")
                print(f"[g1] s{seed} {name:14s} {label:26s} {ax}: argmin L {r['argmin']['L']:+.0f} in "
                      f"{r['argmin']['inside_mean']:+.0f} bg {r['argmin']['bg_rendered']:+.0f} depth "
                      f"{r['argmin']['depth']:+.0f}  bg-share {b:.2f}", flush=True)
    write_json(os.path.join(OUT, f"sweeps_seed{seed}.json"), {"grid_deg": GRID.tolist(), "rows": rows})


def dose(seed: int, device: str = "cuda:0") -> None:
    from data.scene_generator.pick_place_benchmark import BenchmarkConfig
    from data.scene_generator.pick_place_benchmark import tracking_camera as bench_cam
    from data.scene_generator.photoreal_scene import render_photoreal_scene
    from data.scene_generator.scene_builder import FlatScene
    os.makedirs(OUT, exist_ok=True)
    cfg, model, dino, grouping = load_splat(seed, device)
    tcfg = variant_config(cfg.tracking, "POGS")
    ctx = _ctx(cfg, model, dino, grouping, DRILL_ID)
    scene = build_tool_scene()
    grip, grip_tr = _gripper_frame0(scene)
    cam = bench_cam(BenchmarkConfig(), device)
    rows = []
    for s in DOSE_LEVELS:
        o = render_photoreal_scene(FlatScene(scene + [grip], device=device).clone_with_transforms(grip_tr), cam,
                                   cast_shadows=True, shadow_strength=s)
        opt = _optimizer(ctx, tcfg)
        obs = _obs(o["rgb"].float(), o["depth"].float(), tracking_camera(cam, tcfg, device), ctx, tcfg, opt)
        r = _row(f"shadow {s:g}", "x", DOSE_GRID, _curves(opt, obs, AXES["x"], DOSE_GRID, with_depth=False), True,
                 {"seed": seed, "shadow_strength": s})
        rows.append(r)
        print(f"[g1-dose] s{seed} strength {s:.2f}: roll argmin L {r['argmin']['L']:+.1f} "
              f"bg {r['argmin']['bg_rendered']:+.1f} inside {r['argmin']['inside_mean']:+.1f}", flush=True)
    write_json(os.path.join(OUT, f"dose_seed{seed}.json"), {"grid_deg": DOSE_GRID.tolist(), "rows": rows})


# --------------------------------------------------------------------------- #
# summary (pre-registered rules first)
# --------------------------------------------------------------------------- #

def _share(r):
    kL = r["kappa"]["L"]
    return r["kappa"]["S_out"] / kL if (kL is not None and kL > 0) else float("nan")


def summarise() -> None:
    from scipy.stats import spearmanr
    L = ["# G1: generality of the background-silhouette mechanism (static sweeps)", "",
         "Pre-registered in `docs/note/g1_generality_preregistration.md`. Evaluation only; no trajectories. "
         "Static (no motion) is not applicable: no tracking is run. Rules first, then exploratory tables.", ""]
    S = {int(p.split("seed")[-1].split(".")[0]): json.load(open(p))
         for p in sorted(glob.glob(os.path.join(OUT, "sweeps_seed*.json")))}
    D = {int(p.split("seed")[-1].split(".")[0]): json.load(open(p))
         for p in sorted(glob.glob(os.path.join(OUT, "dose_seed*.json")))}
    split = max((max(r["curves"]["split_err"]) for s in S.values() for r in s["rows"]), default=float("nan"))
    L += [f"Split check: max |L - S_in - S_out| = {split:.1e}.", ""]

    def med_share(rows):
        v = [_share(r) for r in rows]
        v = [x for x in v if np.isfinite(x)]
        return (float(np.median(v)) if v else float("nan")), len(v)

    # P1
    L += ["## Pre-registered results", "", "### P1: background share of the restoring curvature", "",
          "| Splat | Object | median b | n (views x axes, κ_L > 0) | b > 0.5 |", "|---|---|---|---|---|"]
    p1_obj = 0
    p1_drill_new = []
    for seed, s in S.items():
        for name in PRIMARY + EXPLORATORY:
            rows = [r for r in s["rows"] if r["object"] == name]
            if not rows:
                continue
            m, n = med_share(rows)
            tag = "" if name in PRIMARY else " (exploratory)"
            L.append(f"| seed {seed} | {name}{tag} | {m:.2f} | {n} | {'yes' if m > 0.5 else 'no'} |")
            if seed == 0 and name in PRIMARY and m > 0.5:
                p1_obj += 1
            if seed != 0 and name == "power_drill":
                p1_drill_new.append(m > 0.5)
    p1 = p1_obj >= 3 and len(p1_drill_new) == 2 and all(p1_drill_new)
    L += ["", f"P1: {p1_obj}/4 primary objects on seed 0 with median b > 0.5; drill on new splats: "
          f"{p1_drill_new}. **{'PASS' if p1 else 'FAIL'}**" + (" (incomplete)" if len(p1_drill_new) < 2 else ""), ""]

    # P2
    p2 = []
    L += ["### P2: drill roll bias with the frame-0 gripper, by splat", "", "| Splat | argmin L (deg) |", "|---|---|"]
    for seed, s in S.items():
        r = [r for r in s["rows"] if r["object"] == "power_drill" and r["view"] == "track, gripper (frame 0)"
             and r["axis"] == "x"]
        if r:
            a = r[0]["argmin"]["L"]
            L.append(f"| seed {seed} | {a:+.0f} |")
            if seed != 0:
                p2.append(a <= -5)
    ok2 = len(p2) == 2 and all(p2)
    L += ["", f"P2: **{'PASS' if ok2 else 'FAIL'}**" + (" (incomplete)" if len(p2) < 2 else ""), ""]

    # P3
    L += ["### P3: held-out views, mean |argmin| of release vs rendered background (seed 0)", "",
          "| Object | axis (largest κ_L) | release | rendered background | lower |", "|---|---|---|---|---|"]
    n3 = 0
    if 0 in S:
        for name in PRIMARY:
            ho = [r for r in S[0]["rows"] if r["object"] == name and r["view"].startswith("heldout")]
            if not ho:
                continue
            if name == "power_drill":
                ax = "x"
            else:
                ax = max(AXES, key=lambda a: np.nanmedian([r["kappa"]["L"] for r in ho if r["axis"] == a]))
            sel = [r for r in ho if r["axis"] == ax]
            rel = float(np.mean([abs(r["argmin"]["L"]) for r in sel]))
            bg = float(np.mean([abs(r["argmin"]["bg_rendered"]) for r in sel]))
            n3 += bg < rel
            L.append(f"| {name} | {ax} | {rel:.1f} | {bg:.1f} | {'yes' if bg < rel else 'no'} |")
    L += ["", f"P3: {n3}/4 objects. **{'PASS' if n3 >= 3 else 'FAIL'}**", ""]

    # P4/P5
    L += ["### P4/P5: shadow dose-response (drill, tracking camera, frame-0 gripper; roll argmin, deg)", "",
          "| Splat | " + " | ".join(f"s={s:g}" for s in DOSE_LEVELS) + " | Spearman ρ | move 0→1 | bg move 0→1 |",
          "|---|" + "---|" * (len(DOSE_LEVELS) + 3)]
    n4a = n4b = n5 = 0
    for seed, d in D.items():
        st = [r["shadow_strength"] for r in d["rows"]]
        aL = [r["argmin"]["L"] for r in d["rows"]]
        aB = [r["argmin"]["bg_rendered"] for r in d["rows"]]
        rho = float(spearmanr(st, aL).correlation)
        mv = aL[st.index(1.0)] - aL[st.index(0.0)]
        mvb = aB[st.index(1.0)] - aB[st.index(0.0)]
        n4a += rho <= -0.8
        n4b += abs(mv) >= 4
        n5 += abs(mvb) < 0.5 * abs(mv)
        L.append(f"| seed {seed} | " + " | ".join(f"{a:+.1f}" for a in aL) + f" | {rho:+.2f} | {mv:+.1f} | {mvb:+.1f} |")
    nd = len(D)
    L += ["", f"P4: ρ ≤ −0.8 on {n4a}/{nd}, movement ≥ 4° on {n4b}/{nd}. "
          f"**{'PASS' if n4a >= 2 and n4b >= 2 else 'FAIL'}**" + (" (incomplete)" if nd < 3 else ""),
          f"P5: rendered-background movement < half of release on {n5}/{nd}. "
          f"**{'PASS' if n5 >= 2 else 'FAIL'}**" + (" (incomplete)" if nd < 3 else ""), ""]

    # exploratory: argmin tables
    L += ["## Exploratory: argmins by object and view set (median [min, max], deg)", "",
          "| Splat | Object | axis | views | release L | inside mean | rendered bg | depth |", "|---|---|---|---|---|---|---|---|"]

    def cell(v):
        v = [x for x in v if np.isfinite(x)]
        return f"{np.median(v):+.0f} [{min(v):+.0f}, {max(v):+.0f}]" if v else "n/a"
    for seed, s in S.items():
        for name in PRIMARY + EXPLORATORY:
            for ax in AXES:
                for vs, pred in (("tracking cam", lambda r: r["view"].startswith("track")),
                                 ("held-out ring", lambda r: r["view"].startswith("heldout"))):
                    sel = [r for r in s["rows"] if r["object"] == name and r["axis"] == ax and pred(r)]
                    if sel:
                        L.append(f"| {seed} | {name} | {ax} | {vs} ({len(sel)}) | " + " | ".join(
                            cell([r["argmin"][k] for r in sel]) for k in ("L", "inside_mean", "bg_rendered", "depth"))
                            + " |")
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["train", "sweeps", "dose", "summarise"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--objects", nargs="*", default=None)
    a = ap.parse_args()
    if a.step == "train":
        load_splat(a.seed)
    elif a.step == "sweeps":
        sweeps(a.seed, objects_only=a.objects)
    elif a.step == "dose":
        dose(a.seed)
    else:
        summarise()
