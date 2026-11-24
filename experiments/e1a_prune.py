"""
E1a: pruning the drill group's over-covering Gaussians (pre-registered, `docs/note/e1a_overcoverage_prune_preregistration.md`).
Ground truth (simulator masks) is used for diagnosis only.

    source tools/cuda_env.sh && .venv/bin/python -m experiments.e1a_prune
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from data.scene_generator.pick_place_benchmark import BenchmarkConfig
from data.scene_generator.pick_place_benchmark import tracking_camera as bench_cam
from data.scene_generator.photoreal_scene import render_photoreal_scene
from data.scene_generator.scene_builder import FlatScene
from data.scene_generator.tool_perturbation import DRILL_ID, build_tool_scene
from experiments.b_bias import _argmin, _obs, _optimizer
from experiments.g1_generality import AXES, DOSE_GRID, _ctx, _curves, _gripper_frame0, load_splat
from experiments.off_repro.silhouette_diag import stats as sil_stats
from model.pogs.evaluate import tracking_camera, variant_config
from model.pogs.grouping import build_object_groups

OUT = "data_results/e1a_prune"
CAP = "datasets/synthetic_scenes/tool_table/capture"


def outside_mask(means: np.ndarray) -> np.ndarray:
    cams = json.load(open(os.path.join(CAP, "cameras.json")))
    ids = np.load(os.path.join(CAP, "obj_masks.npy"), mmap_mode="r")
    n_in = np.zeros(len(means), int)
    n_out = np.zeros(len(means), int)
    X = np.c_[means, np.ones(len(means))]
    for v, c in enumerate(cams):
        w2c = np.linalg.inv(np.asarray(c["c2w"], float))          # OpenCV convention
        p = X @ w2c.T
        z = p[:, 2]
        u = c["fx"] * p[:, 0] / np.maximum(z, 1e-9) + c["cx"]
        y = c["fy"] * p[:, 1] / np.maximum(z, 1e-9) + c["cy"]
        ok = (z > 0) & (u >= 0) & (u < c["width"]) & (y >= 0) & (y < c["height"])
        ui, vi = u[ok].astype(int), y[ok].astype(int)
        lab = np.asarray(ids[v])[vi, ui]
        n_in[ok] += 1
        n_out[np.nonzero(ok)[0][lab != DRILL_ID]] += 1
    return (n_in > 0) & (n_out >= 0.5 * n_in)


def regroup(ctx, keep_bool: torch.Tensor):
    model = ctx["model"]
    labels = torch.full((keep_bool.shape[0],), -1, dtype=torch.long, device=keep_bool.device)
    labels[keep_bool] = 0
    g = build_object_groups(model, labels, keep_clusters=[0], names={0: "power_drill"})
    return dict(ctx, groups=g)


def dose_pair(ctx, frames, cam, device):
    tcfg = variant_config(ctx["cfg"].tracking, "POGS")
    out = {}
    for s, (rgb, depth) in frames.items():
        opt = _optimizer(ctx, tcfg)
        obs = _obs(rgb, depth, tracking_camera(cam, tcfg, device), ctx, tcfg, opt)
        cv = _curves(opt, obs, AXES["x"], DOSE_GRID, with_depth=False)
        out[s] = _argmin(DOSE_GRID, cv["L"])
    return out


def main(device="cuda:0"):
    os.makedirs(OUT, exist_ok=True)
    scene = build_tool_scene()
    grip, grip_tr = _gripper_frame0(scene)
    cam = bench_cam(BenchmarkConfig(), device)
    frames, obj1 = {}, None
    for s in (0.0, 1.0):
        o = render_photoreal_scene(FlatScene(scene + [grip], device=device).clone_with_transforms(grip_tr), cam,
                                   cast_shadows=True, shadow_strength=s)
        frames[s] = (o["rgb"].float(), o["depth"].float())
        if s == 1.0:
            obj1 = o["obj_id"]
    rows = []
    for seed in (0, 1, 2):
        cfg, model, dino, grouping = load_splat(seed, device)
        ctx = _ctx(cfg, model, dino, grouping, DRILL_ID)
        m = ctx["groups"].group_masks[0]
        idx = torch.nonzero(m).squeeze(-1)
        means = model.gauss_params["means"].detach()[idx].cpu().numpy()
        outs = torch.from_numpy(outside_mask(means)).to(m.device)
        pruned = m.clone()
        pruned[idx[outs]] = False
        rng = np.random.default_rng(0)
        drop = torch.from_numpy(rng.choice(len(idx), int(outs.sum()), replace=False)).to(m.device)
        control = m.clone()
        control[idx[drop]] = False
        tcfg = variant_config(cfg.tracking, "POGS")
        r = {"seed": seed, "n_group": int(m.sum()), "n_outside": int(outs.sum())}
        for name, c in (("original", ctx), ("pruned", regroup(ctx, pruned)), ("control", regroup(ctx, control))):
            a = dose_pair(c, frames, cam, device)
            st = sil_stats(c, frames[1.0][0], frames[1.0][1], obj1, cam, device, tcfg)
            r[name] = {"argmin_s0": a[0.0], "argmin_s1": a[1.0], "mask_outside_gt_px": st["mask_outside_gt_px"],
                       "halo_outside_gt_px": st["halo_outside_gt_px"], "gt_missed_px": st["gt_missed_px"],
                       "iou": st["iou"]}
            print(f"[e1a] s{seed} {name:8s} argmin s0 {a[0.0]:+.1f} s1 {a[1.0]:+.1f}  outside {st['mask_outside_gt_px']}"
                  f" halo {st['halo_outside_gt_px']} missed {st['gt_missed_px']}", flush=True)
        rows.append(r)
        del model
        torch.cuda.empty_cache()
    pr1 = sum(abs(r["pruned"]["argmin_s1"]) <= 3.0 for r in rows) >= 2
    pr2 = sum(r["control"]["argmin_s1"] <= -5.0 for r in rows) >= 2
    pr3 = sum(abs(r["pruned"]["argmin_s1"] - r["pruned"]["argmin_s0"]) < 2.0 for r in rows) >= 2
    json.dump(rows, open(os.path.join(OUT, "prune.json"), "w"), indent=1)
    L = ["# E1a: pruning over-covering Gaussians from the drill group", "",
         "Pre-registered in `docs/note/e1a_overcoverage_prune_preregistration.md`. Static (no motion) not applicable."
         " Rules first.", "", "| Rule | Result |", "|---|---|",
         f"| PR1 pruned abs(argmin s=1) ≤ 3° on ≥ 2/3 | {'pass' if pr1 else 'FAIL'} |",
         f"| PR2 control argmin s=1 ≤ −5° on ≥ 2/3 | {'pass' if pr2 else 'FAIL'} |",
         f"| PR3 pruned abs(shift s 0→1) < 2° on ≥ 2/3 | {'pass' if pr3 else 'FAIL'} |", "",
         "| Seed | group / outside | variant | argmin s=0 | argmin s=1 | mask outside GT px | halo outside GT px | GT missed px |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for name in ("original", "pruned", "control"):
            v = r[name]
            L.append(f"| {r['seed']} | {r['n_group']} / {r['n_outside']} | {name} | {v['argmin_s0']:+.1f} | "
                     f"{v['argmin_s1']:+.1f} | {v['mask_outside_gt_px']} | {v['halo_outside_gt_px']} | {v['gt_missed_px']} |")
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    for k, ok in (("PR1", pr1), ("PR2", pr2), ("PR3", pr3)):
        print(f"FINDING: E1a-{k} | pruning over-covering drill Gaussians: {k} {'pass' if ok else 'FAIL'} | {OUT}/summary.md")


if __name__ == "__main__":
    main()
