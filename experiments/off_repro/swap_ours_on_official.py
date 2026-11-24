"""8.OFF-S (exploratory, `docs/note/off_swap_exploratory.md`): OUR tracking code on the OFFICIAL splat.

    source tools/cuda_env.sh && .venv/bin/python -m experiments.off_repro.swap_ours_on_official

Loads ~/pogs_external/off/official_splat_for_ours.pt (export_official_splat.py), builds our POGSModel from the official
Gaussians, per-Gaussian DINO features and DINO MLP, sets our DinoExtractor to the official PCA basis, makes the drill
group from the official keep set, and runs the G1 dose code path (same frames as G1/E4/OR1-2) plus the tracking
frames without the gripper. Also runs the same on OUR seed-0 splat in the same process as a control.
"""
import json
import math
import os

import numpy as np
import torch

from data.scene_generator.pick_place_benchmark import BenchmarkConfig
from data.scene_generator.pick_place_benchmark import tracking_camera as bench_cam
from data.scene_generator.tool_perturbation import DRILL_ID, build_tool_scene
from experiments.b_bias import _argmin, _obs, _optimizer
from experiments.e4_torque import _render_s
from experiments.g1_generality import AXES, DOSE_GRID, DOSE_LEVELS, _ctx, _curves, _gripper_frame0, load_splat
from experiments.pipeline import load_encoders
from experiments.v1_base_pogs.config import get_v1_config
from model.pogs.checkpoint import write_json
from model.pogs.evaluate import tracking_camera, variant_config
from model.pogs.grouping import build_object_groups
from model.pogs.model import POGSModel

OUT = "data_results/off_repro"
SRC = os.path.expanduser("~/pogs_external/off/official_splat_for_ours.pt")


def official_ctx(device="cuda:0"):
    cfg = get_v1_config(device)
    ck = torch.load(SRC, map_location="cpu", weights_only=False)
    gp = ck["gauss_params"]
    model = POGSModel(seed_points=gp["means"], seed_colors=torch.zeros_like(gp["means"]) + 0.5, config=cfg.model,
                      device=device, raster_backend=cfg.runtime.raster_backend)
    model.gauss_params = torch.nn.ParameterDict({k: torch.nn.Parameter(v.float().to(device)) for k, v in gp.items()})
    model.nn.net.load_state_dict({k: v for k, v in ck["nn"].items()})
    model.eval()
    dino, _ = load_encoders(cfg, with_clip=False)
    dino.set_pca(ck["dino_pca"].float())
    labels = torch.full((gp["means"].shape[0],), -1, dtype=torch.long, device=device)
    labels[ck["keep"].long().to(device)] = 0
    groups = build_object_groups(model, labels, keep_clusters=[0], names={0: "power_drill"})
    return dict(cfg=cfg, model=model, dino=dino, groups=groups), ck


def sweep(ctx, device="cuda:0"):
    tcfg = variant_config(ctx["cfg"].tracking, "POGS")
    scene = build_tool_scene()
    grip, grip_tr = _gripper_frame0(scene)
    cam = bench_cam(BenchmarkConfig(), device)
    conds = [(f"gripper s={s:g}", scene + [grip], grip_tr, True, s) for s in DOSE_LEVELS]
    conds += [("track", scene, {}, True, 1.0), ("track, no shadows", scene, {}, False, 1.0)]
    rows = []
    for label, objs, tr, sh, s in conds:
        rgb, depth = _render_s(objs, cam, tr, sh, s, device)
        opt = _optimizer(ctx, tcfg)
        obs = _obs(rgb, depth, tracking_camera(cam, tcfg, device), ctx, tcfg, opt)
        cv = _curves(opt, obs, AXES["x"], DOSE_GRID, with_depth=False)
        rows.append({"view": label, "shadow_strength": s, "argmin_L": _argmin(DOSE_GRID, cv["L"]),
                     "argmin_inside": _argmin(DOSE_GRID, cv["inside_mean"]),
                     "argmin_bg_rendered": _argmin(DOSE_GRID, cv["bg_rendered"]), "L": cv["L"]})
        print(f"  {label:20s} argmin L {rows[-1]['argmin_L']:+.1f}  inside {rows[-1]['argmin_inside']:+.1f}", flush=True)
    return rows


def main(device="cuda:0"):
    os.makedirs(OUT, exist_ok=True)
    ctx_o, ck = official_ctx(device)
    print("our code, OFFICIAL splat:", flush=True)
    rows_o = sweep(ctx_o, device)
    cfg, model, dino, grouping = load_splat(0, device)
    ctx_s = _ctx(cfg, model, dino, grouping, DRILL_ID)
    print("our code, OUR seed-0 splat (control):", flush=True)
    rows_s = sweep(ctx_s, device)
    write_json(os.path.join(OUT, "swap_ours_on_official.json"), {"grid_deg": DOSE_GRID.tolist(),
                                                                "official_splat": rows_o, "our_splat": rows_s,
                                                                "official_ckpt": ck["ckpt"]})
    L = ["# 8.OFF-S (exploratory): our tracking code on the official splat", "",
         "Note `docs/note/off_swap_exploratory.md` (written before running). No pass rule.", "",
         "| Frame | s | official splat, our code: argmin L | inside-mask | our splat, our code: argmin L | "
         "official code (OR1/OR2) |", "|---|---|---|---|---|---|"]
    off = {}
    try:
        for r in json.load(open(os.path.join(OUT, "official_sweeps.json")))["rows"]:
            off[r["view"]] = r["argmin"]["dino"]
    except Exception:
        pass
    for ro, rs in zip(rows_o, rows_s):
        L.append(f"| {ro['view']} | {ro['shadow_strength']:g} | {ro['argmin_L']:+.1f} | {ro['argmin_inside']:+.1f} | "
                 f"{rs['argmin_L']:+.1f} | {off.get(ro['view'], float('nan')):+.1f} |")
    open(os.path.join(OUT, "summary_swap.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    g1o = next(r for r in rows_o if r["view"] == "gripper s=1")["argmin_L"]
    g1s = next(r for r in rows_s if r["view"] == "gripper s=1")["argmin_L"]
    print(f"FINDING: OFF-S | exploratory swap: our code gives {g1o:+.1f} deg on the official splat vs {g1s:+.1f} on our "
          f"splat (gripper, s=1); official code gave {off.get('gripper s=1', float('nan')):+.1f} | {OUT}/summary_swap.md")


if __name__ == "__main__":
    main()
