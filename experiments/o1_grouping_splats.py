"""
PLAN Phase 1 (O1): does each object group really hold that object's Gaussians?

For every tracked group k (matched to GT object g_k for scoring only) and every
capture view, render with the trained splat:
    full    all Gaussians
    minus   all Gaussians except group k      ("scene without the object")
    only    group k alone                      ("the object")
and compare with simulator ground truth rendered for the same camera:
    amodal mask of g_k (object rendered alone, so occlusion by other objects
    is not counted as leakage), and the scene rendered WITHOUT g_k.

Metrics per object (mean over views where the object is in frame):
    iou / leakage / completeness   'only' alpha vs amodal GT mask
    purity / recall                opacity-weighted, per Gaussian, using
                                   assign_gt_ids (nearest seed point; approximate)
    psnr_full                      full render vs captured image (reference)
    psnr_minus_out                 minus vs GT-without, OUTSIDE the object mask:
                                   removing the group must not damage the rest
    psnr_minus_in                  minus vs GT-without, INSIDE the mask: low by
                                   construction (surfaces under/behind the object
                                   were never observed); reported, not a target
    residual_alpha_in              mean opacity left inside the mask in 'minus'
                                   from Gaussians whose GT id is g_k (missed
                                   object Gaussians show up here)

Ground truth is used only to score. Writes data_results/o1_grouping/{metrics.json,
metrics.csv, figure.jpg} and env.json.

    source tools/cuda_env.sh && .venv/bin/python -m experiments.o1_grouping_splats
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch

from experiments import pipeline
from experiments.env_info import write_env_json
from experiments.pipeline import build_groups, train_or_load
from experiments.v1_base_pogs.config import get_v1_config
from data.scene_generator.pick_place_benchmark import OBJECT_NAMES, build_pick_place_scene
from data.scene_generator.photoreal_scene import render_photoreal_scene
from data.scene_generator.scene_builder import FlatScene
from metrics.gaussian import image_psnr, mask_metrics
from model.pogs.geometry import quat_normalize
from model.pogs.grouping import assign_gt_ids
from model.pogs.raster import rasterize

OUT = "data_results/o1_grouping"


@torch.no_grad()
def render_subset(model, keep: torch.Tensor, camera, background: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Render only the Gaussians in ``keep`` (N bool), as ``POGSModel.get_outputs`` does."""
    gp = model.gauss_params
    out = rasterize(
        gp["means"][keep], quat_normalize(gp["quats"][keep]), torch.exp(gp["scales"][keep]),
        torch.sigmoid(gp["opacities"][keep]).squeeze(-1), model.colors_sh[keep], camera,
        backend=model.backend, sh_degree=model.sh_degree_to_use(), render_depth=False,
        background=background, antialiased=model.config.rasterize_mode == "antialiased",
    )
    return {"rgb": out["render"].clamp(0, 1), "alpha": out["alpha"][..., 0]}


def main(device: str = "cuda:0") -> pd.DataFrame:
    os.makedirs(OUT, exist_ok=True)
    write_env_json(OUT)
    cfg = get_v1_config(device)
    model, capture, dino, clip = train_or_load(cfg)
    grouping = build_groups(model, capture, cfg, clip=clip)
    groups, g2gt = grouping.groups, grouping.group_to_gt
    model.eval()

    gt_ids = assign_gt_ids(model, capture.seed_points, capture.seed_obj_ids)
    opac = torch.sigmoid(model.gauss_params["opacities"].data).squeeze(-1)
    N = opac.numel()
    bg = torch.zeros(3, device=device)  # black: the object-only alpha and holes are unambiguous
    objects = build_pick_place_scene()

    rows: List[Dict] = []
    fig_rows = []
    for k in range(groups.num_groups):
        gk = g2gt[k]
        in_k = groups.group_masks[k].to(device)
        is_gk = gt_ids == gk
        w = opac
        purity = float((w * (in_k & is_gk)).sum() / (w * in_k).sum().clamp_min(1e-12))
        recall = float((w * (in_k & is_gk)).sum() / (w * is_gk).sum().clamp_min(1e-12))
        scene_alone = FlatScene([o for o in objects if o.obj_id == gk], device=device)
        scene_wo = FlatScene([o for o in objects if o.obj_id != gk], device=device)
        missed = (~in_k) & is_gk                     # object Gaussians outside the group
        per_view, best = [], (-1, None)
        for v, cam in enumerate(capture.cameras):
            cam = cam.to(device)
            amodal = render_photoreal_scene(scene_alone, cam, cast_shadows=False)["obj_id"].to(device) == gk
            if amodal.sum() < 50:
                continue
            gt_wo = render_photoreal_scene(scene_wo, cam, cast_shadows=True)["rgb"].to(device)
            full = render_subset(model, torch.ones(N, dtype=torch.bool, device=device), cam, bg)
            minus = render_subset(model, ~in_k, cam, bg)
            only = render_subset(model, in_k, cam, bg)
            resid = render_subset(model, missed, cam, bg)["alpha"] if missed.any() else torch.zeros_like(only["alpha"])
            img = capture.images[v].to(device)
            m = mask_metrics(only["alpha"], amodal)
            per_view.append({
                **m,
                "psnr_full": image_psnr(full["rgb"], img),
                "psnr_minus_out": image_psnr(minus["rgb"], gt_wo, ~amodal),
                "psnr_minus_in": image_psnr(minus["rgb"], gt_wo, amodal),
                "residual_alpha_in": float(resid[amodal].mean()),
            })
            if int(amodal.sum()) > best[0]:
                best = (int(amodal.sum()), [img, full["rgb"], minus["rgb"], only["rgb"]])
        agg = {key: float(np.nanmean([p[key] for p in per_view])) for key in per_view[0]}
        rows.append({"group": k, "object": OBJECT_NAMES.get(gk, str(gk)), "gt_id": gk,
                     "n_gaussians": int(in_k.sum()), "n_views": len(per_view),
                     "purity": purity, "recall": recall, **agg})
        fig_rows.append(np.concatenate([(t.cpu().numpy() * 255).astype(np.uint8) for t in best[1]], axis=1))
        print(f"[o1] {rows[-1]['object']}: " + ", ".join(f"{a}={rows[-1][a]:.3f}" for a in
              ("iou", "leakage", "completeness", "purity", "recall", "psnr_minus_out", "residual_alpha_in")))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "metrics.csv"), index=False)
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump({"ckpt_id": pipeline.CKPT_ID, "background": "black", "columns": "GT | full | minus | only",
                   "rows": rows}, f, indent=2)
    imageio.imwrite(os.path.join(OUT, "figure.jpg"), np.concatenate(fig_rows, axis=0), quality=92)
    return df


if __name__ == "__main__":
    main()
