"""Render the 8.OFF test frames with our simulator (main venv) for the official-code sweeps.

    source tools/cuda_env.sh && .venv/bin/python -m experiments.off_repro.frames

Same 20 frames as E4 / G1: tracking camera + frame-0 gripper at six shadow strengths, tracking camera without the
gripper (shadows on / off), 12 held-out ring views. Also writes the scripted operator inputs: the click point
(centroid of the drill group from our geometric selection, seed-0 splat) and the table height (the release reads it
from `table_bounding_cube.json`, written at capture time).
"""
import json
import os

import numpy as np
import torch

from data.scene_generator.pick_place_benchmark import TABLE_TOP_Z, BenchmarkConfig
from data.scene_generator.pick_place_benchmark import tracking_camera as bench_cam
from data.scene_generator.tool_perturbation import DRILL_ID, build_tool_scene
from experiments.e4_torque import _render_s
from experiments.g1_generality import DOSE_LEVELS, _ctx, _gripper_frame0, _ring_cam, load_splat

OUT = os.path.expanduser("~/pogs_external/off/frames")


def main(device: str = "cuda:0") -> None:
    os.makedirs(OUT, exist_ok=True)
    cfg, model, dino, grouping = load_splat(0, device)
    ctx = _ctx(cfg, model, dino, grouping, DRILL_ID)
    g = ctx["groups"]
    idx = torch.nonzero(g.group_masks[0]).squeeze(-1) if hasattr(g, "group_masks") else None
    click = model.means[idx].mean(0).detach().cpu().numpy().tolist()
    bcfg = BenchmarkConfig()
    scene = build_tool_scene()
    grip, grip_tr = _gripper_frame0(scene)
    track = bench_cam(bcfg, device)
    drill = next(o for o in scene if o.obj_id == DRILL_ID)
    target = np.asarray(drill.position, float) + np.array([0.0, 0.0, 0.05])
    conds = [(f"gripper s={s:g}", scene + [grip], grip_tr, track, True, s, "dose") for s in DOSE_LEVELS]
    conds += [("track", scene, {}, track, True, 1.0, "track"), ("track, no shadows", scene, {}, track, False, 1.0,
                                                                  "track")]
    conds += [(f"heldout az{az:03d}", scene, {}, _ring_cam(target, az, bcfg, device), True, 1.0, "ring")
              for az in range(0, 360, 30)]
    index = []
    for i, (label, objs, tr, cam, sh, s, kind) in enumerate(conds):
        rgb, depth = _render_s(objs, cam, tr, sh, s, device)
        f = f"frame_{i:02d}.npz"
        np.savez_compressed(os.path.join(OUT, f), rgb=(rgb.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8),
                            depth=depth.cpu().numpy().astype(np.float32), c2w_cv=cam.c2w.cpu().numpy(),
                            K=np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]], float),
                            width=cam.width, height=cam.height)
        index.append({"file": f, "view": label, "kind": kind, "shadow_strength": s, "shadows": sh})
    json.dump({"frames": index, "click_xyz": click, "table_z": float(TABLE_TOP_Z),
               "group_centroid_ours": click}, open(os.path.join(OUT, "index.json"), "w"), indent=1)
    print(f"{len(index)} frames -> {OUT}; click {np.round(click, 4).tolist()}")


if __name__ == "__main__":
    main()
