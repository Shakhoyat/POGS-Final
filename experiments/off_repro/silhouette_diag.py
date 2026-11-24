"""8.OFF-S2 (exploratory): rendered drill silhouette of the official vs our splat against the simulator's drill mask,
at the tracking camera (frame-0 gripper, s = 1), at the true pose. Same code path as the sweeps (our renderer).

    source tools/cuda_env.sh && .venv/bin/python -m experiments.off_repro.silhouette_diag
"""
import json
import os

import numpy as np
import torch

from data.scene_generator.pick_place_benchmark import BenchmarkConfig
from data.scene_generator.pick_place_benchmark import tracking_camera as bench_cam
from data.scene_generator.photoreal_scene import render_photoreal_scene
from data.scene_generator.scene_builder import FlatScene
from data.scene_generator.tool_perturbation import DRILL_ID, build_tool_scene
from experiments.b_bias import _obs, _optimizer
from experiments.g1_generality import _ctx, _gripper_frame0, load_splat
from experiments.off_repro.swap_ours_on_official import official_ctx
from model.pogs.evaluate import tracking_camera, variant_config

OUT = "data_results/off_repro"


def stats(ctx, rgb, depth, obj, cam, device, tcfg):
    opt = _optimizer(ctx, tcfg)
    obs = _obs(rgb, depth, tracking_camera(cam, tcfg, device), ctx, tcfg, opt)
    rf = obs.roi_frames[0]
    with torch.no_grad():
        o = opt._render(rf.camera, opt.part_deltas.detach().clone(), 0, True)
    a = o["accumulation"].squeeze(-1).float()
    # GT drill mask in the same ROI crop / resolution
    gt_full = (obj == DRILL_ID).float()[None, None]
    x0, x1, y0, y1 = rf.bbox
    gt = gt_full[..., y0:y1, x0:x1]
    gt = torch.nn.functional.interpolate(gt, size=a.shape, mode="nearest")[0, 0] > 0.5
    m = a >= 0.8
    halo = (a > 0.05) & (a < 0.8)
    inter, union = (m & gt).sum().item(), (m | gt).sum().item()
    g = rf.dino_feats.to(a.device).norm(dim=-1)
    gval = torch.isfinite(g)
    return {"roi_px": int(a.numel()), "mask_px": int(m.sum()), "gt_px": int(gt.sum()), "iou": inter / max(union, 1),
            "mask_outside_gt_px": int((m & ~gt).sum()), "gt_missed_px": int((gt & ~m).sum()),
            "halo_px": int(halo.sum()), "halo_outside_gt_px": int((halo & ~gt).sum()),
            "alpha_sum_outside_gt": float(a[~gt].sum()), "g_mean_roi": float(g[gval].mean()),
            "n_gaussians_group": int(ctx["groups"].group_masks[0].sum())}


def main(device="cuda:0"):
    scene = build_tool_scene()
    grip, grip_tr = _gripper_frame0(scene)
    cam = bench_cam(BenchmarkConfig(), device)
    o = render_photoreal_scene(FlatScene(scene + [grip], device=device).clone_with_transforms(grip_tr), cam,
                               cast_shadows=True, shadow_strength=1.0)
    rgb, depth, obj = o["rgb"].float(), o["depth"].float(), o["obj_id"]
    res = {}
    ctx_o, _ = official_ctx(device)
    tcfg = variant_config(ctx_o["cfg"].tracking, "POGS")
    res["official"] = stats(ctx_o, rgb, depth, obj, cam, device, tcfg)
    del ctx_o
    torch.cuda.empty_cache()
    for seed in (0, 1, 2):
        cfg, model, dino, grouping = load_splat(seed, device)
        res[f"ours_seed{seed}"] = stats(_ctx(cfg, model, dino, grouping, DRILL_ID), rgb, depth, obj, cam, device, tcfg)
        del model
        torch.cuda.empty_cache()
    json.dump(res, open(os.path.join(OUT, "silhouette_diag.json"), "w"), indent=1)
    keys = list(res["official"])
    L = ["# 8.OFF-S2 (exploratory): drill silhouette at the tracking camera (gripper, s = 1, true pose)", "",
         "| Splat | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    L += [f"| {k} | " + " | ".join(f"{v[c]:.3f}" if isinstance(v[c], float) else str(v[c]) for c in keys) + " |"
          for k, v in res.items()]
    open(os.path.join(OUT, "summary_silhouette.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
