"""Export the `tool_table` capture to the nerfstudio format the OFFICIAL POGS trainer reads (8.OFF).

    .venv/bin/python -m experiments.off_repro.export [--out ~/pogs_external/off/tool_table_ns]

Writes images/*.png, depths/*.npy (float32 metres; official config depth_unit_scale_factor = 1.0),
masks_sim/*.npy (simulator instance ids, used only to fill the Detic cache), sparse_pc.ply (the capture's seed
points, official `load_3D_points=True`) and transforms.json (OpenGL camera convention, poses unchanged:
orientation/center 'none', auto_scale_poses False in the official config). World coordinates are therefore the
simulator's, so object poses carry over unchanged.
"""
import argparse
import json
import os
import shutil

import numpy as np
from PIL import Image

SRC = "datasets/synthetic_scenes/tool_table/capture"
CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])


def main(out: str) -> None:
    out = os.path.expanduser(out)
    for d in ("images", "depths", "masks_sim"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    cams = json.load(open(os.path.join(SRC, "cameras.json")))
    imgs = np.load(os.path.join(SRC, "images.npy"))
    deps = np.load(os.path.join(SRC, "depths.npy"))
    ids = np.load(os.path.join(SRC, "obj_masks.npy"))
    c0 = cams[0]
    assert all(c["fx"] == c0["fx"] and c["width"] == c0["width"] for c in cams), "one intrinsics block assumed"
    frames = []
    for v, c in enumerate(cams):
        name = f"frame_{v:05d}"
        Image.fromarray(imgs[v]).save(os.path.join(out, "images", name + ".png"))
        np.save(os.path.join(out, "depths", name + ".npy"), deps[v].astype(np.float32))
        np.save(os.path.join(out, "masks_sim", name + ".npy"), ids[v])
        c2w_gl = np.asarray(c["c2w"], dtype=np.float64) @ CV_TO_GL
        frames.append({"file_path": f"images/{name}.png", "depth_file_path": f"depths/{name}.npy",
                       "transform_matrix": c2w_gl.tolist()})
    shutil.copy(os.path.join(SRC, "fused_pointcloud.ply"), os.path.join(out, "sparse_pc.ply"))
    tj = {"camera_model": "OPENCV", "fl_x": c0["fx"], "fl_y": c0["fy"], "cx": c0["cx"], "cy": c0["cy"],
          "w": c0["width"], "h": c0["height"], "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
          "ply_file_path": "sparse_pc.ply", "frames": frames,
          "source": SRC, "note": "exported by experiments/off_repro/export.py for 8.OFF"}
    json.dump(tj, open(os.path.join(out, "transforms.json"), "w"), indent=1)
    meta = json.load(open(os.path.join(SRC, "capture_metadata.json")))
    json.dump(meta, open(os.path.join(out, "capture_metadata.json"), "w"), indent=1)
    print(f"exported {len(frames)} views -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/pogs_external/off/tool_table_ns")
    main(ap.parse_args().out)
