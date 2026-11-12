"""Per-axis (camera frame) view of a tracking run. Reads JSON only, no GPU.

Prints, every few frames, the ground-truth and estimated displacement from frame 0
in the tracking camera's frame: right (x), down (y), depth (z), in mm. Tells apart
a sideways slide from a depth runaway, and shows the step the tracker takes per frame.

    python axes_track.py pendulum_v2_30 [--every 5] [--row 00_baseline_detic_dinov2_clip]
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_config as cfg  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("scene")
ap.add_argument("--row", default=cfg.BASELINE_ROW)
ap.add_argument("--every", type=int, default=5)
a = ap.parse_args()

r = json.load(open(os.path.join(cfg.ARTEFACTS, a.row, a.scene, "track", "tracking.json")))
gtp = json.load(open(os.path.join(cfg.scene_path(a.scene), "tracking", "gt_poses.json")))
R = (np.array(gtp["camera_opengl_c2w"]) @ np.diag([1.0, -1.0, -1.0, 1.0]))[:3, :3]
fr = r["frames"]
g0 = np.array(fr[0]["gt_centroid"]); e0 = np.mean(fr[0]["est_centroids"], axis=0)
cam = lambda d: (d @ R) * 1000.0            # world displacement -> camera frame, mm
print(f"{a.scene}: camera-frame displacement from frame 0 (mm)  [x right, y down, z depth]")
print(f"{'frame':>5} | {'gt x':>7} {'gt y':>7} {'gt z':>7} | {'est x':>7} {'est y':>7} {'est z':>7} | {'step mm':>7}")
prev = e0
for f in fr:
    i = f["frame"]
    e = np.mean(f["est_centroids"], axis=0)
    step = np.linalg.norm(e - prev) * 1000.0; prev = e
    if i % a.every and i != 1 and i != len(fr) - 1:
        continue
    g = cam(np.array(f["gt_centroid"]) - g0); d = cam(e - e0)
    print(f"{i:>5} | {g[0]:>7.1f} {g[1]:>7.1f} {g[2]:>7.1f} | {d[0]:>7.1f} {d[1]:>7.1f} {d[2]:>7.1f} | {step:>7.1f}")
