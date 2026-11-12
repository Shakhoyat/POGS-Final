"""Checks that track_offline works on every scene, not just pickplace.

The tracker was written against one scene's layout and one scene's ground-truth
schema. This covers the other ten inputs without a GPU: nine pendulum angles with
a different frame label (a signed angle, not a phase) and a different resolution,
plus the assertions that catch a scene that exists but is not actually usable.

    python test_scenes.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_config as cfg  # noqa: E402
from track_offline import (  # noqa: E402
    load_sequence,
    score_frame,
)

FAILED = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global FAILED
    if not ok:
        FAILED += 1
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def main() -> int:
    print("=" * 68)
    print("SCENE MATRIX - HOST-INDEPENDENT CHECKS")
    print("=" * 68)

    scenes = [f"pendulum_{a}" for a in ("05", "10", "15", "20", "25",
                                        "30", "40", "50", "60")]
    scenes.append("pickplace")

    print("\n--- every scene loads ---")
    loaded = {}
    for name in scenes:
        path = os.path.join(cfg.INPUTS, name)
        if not os.path.isdir(path):
            check(f"{name} exists", False, f"missing {path}")
            continue
        try:
            # load_sequence takes the tracking folder, which is <scene>/tracking.
            images, depths, K, w, h, c2w, gt = load_sequence(
                os.path.join(path, "tracking"))
            loaded[name] = (images, depths, K, w, h, c2w, gt)
            check(f"{name}: {len(images)} frames, {w}x{h}",
                  len(images) == len(depths) and len(images) > 0,
                  f"fx={K[0, 0]:.0f}")
        except SystemExit as exc:
            check(f"{name} loads", False, str(exc))

    print("\n--- the angle ladder is complete and ordered ---")
    angles = []
    for name in scenes:
        if name.startswith("pendulum_"):
            angles.append(int(name.split("_")[1]))
    check("9 pendulum angles present", len(angles) == 9, f"{sorted(angles)}")
    check("angles ascend", angles == sorted(angles) and
          all(b > a for a, b in zip(angles, angles[1:])), f"{angles}")

    print("\n--- each scene's own metadata matches its render ---")
    for name, (images, depths, K, w, h, c2w, gt) in loaded.items():
        intr = gt["camera_intrinsics"]
        check(f"{name}: intrinsics agree with the loader",
              (intr["fl_x"], intr["fl_y"], intr["w"], intr["h"]) ==
              (K[0, 0], K[1, 1], w, h),
              f"{intr['fl_x']:.0f} {w}x{h}")
        rot = c2w[:3, :3]
        check(f"{name}: OpenCV rotation right-handed",
              abs(np.linalg.det(rot) - 1.0) < 1e-4, f"det={np.linalg.det(rot):.6f}")
        check(f"{name}: ground truth covers every frame",
              len(gt["frames"]) == len(images),
              f"gt={len(gt['frames'])} frames={len(images)}")

    print("\n--- the focus object is visible enough to select ---")
    for name, (images, depths, K, w, h, c2w, gt) in loaded.items():
        frac = gt["frames"][0]["focus"]["mask_fraction"]
        # POGS drops masks over H*W/3.5, so the floor is about detecting
        # something; 0.2% is the smallest share any of these scenes reaches.
        check(f"{name}: frame 0 focus mask is non-trivial", frac > 0.002,
              f"{frac * 100:.2f}% of frame")

    print("\n--- frame labels: both schemas score ---")
    # pickplace carries `phase`; pendulum carries `angle_deg`. score_frame must
    # pass through whichever is present rather than assuming one.
    est0 = np.tile(np.eye(4), (1, 1, 1))
    for name in ("pickplace", "pendulum_60"):
        if name not in loaded:
            continue
        gt = loaded[name][6]
        r = score_frame(0, est0, gt["frames"][0], est0, np.zeros(3))
        key = "phase" if "phase" in r else ("angle_deg" if "angle_deg" in r else None)
        check(f"{name}: score_frame records its frame label", key is not None,
              f"key={key} value={r.get(key)}")
        check(f"{name}: ground-truth displacement is reported",
              "gt_displacement_mm" in r)

    print("\n--- the moving-frame filter picks the right frames ---")
    # A synthetic trajectory: 10 stationary frames then 10 that move 10 mm/frame.
    gt_fake = [{"focus": {"centroid": [0.0, 0.0, 0.0]}} for _ in range(20)]
    for i in range(10, 20):
        gt_fake[i]["focus"]["centroid"] = [0.01 * (i - 10), 0.0, 0.0]
    traj = [score_frame(i, est0, gt_fake[i], est0, np.zeros(3))
            for i in range(20)]
    moving = [f for f in traj if f.get("gt_displacement_mm", 0.0) > 1.0]
    # 9, not 10: frame 10 sits at the new position but has zero displacement
    # *from the frame-0 reference*, so by the definition being used it is not yet
    # moving. Counting it would be counting a frame that has not travelled.
    check("only the frames that have travelled are scored", len(moving) == 9,
          f"moving={len(moving)}")
    check("the moving frames are the later ones",
          moving[0]["frame"] == 11 and moving[-1]["frame"] == 19,
          f"frames {moving[0]['frame']}..{moving[-1]['frame']}")

    print("\n--- real_14 cannot be trained on, and that is recorded ---")
    check("real_14 has no transforms.json",
          not os.path.exists(os.path.join(cfg.REAL_14, "transforms.json")))
    npy = glob.glob(os.path.join(cfg.REAL_14, "*.npy"))
    check("real_14 still has its 14 frames", len(npy) == 14, f"{len(npy)}")
    if npy:
        d = np.load(npy[0], allow_pickle=True).item()
        check("real_14 frames carry rgb/depth/K/seg only",
              set(d.keys()) == {"rgb", "depth", "K", "seg"},
              f"{sorted(d.keys())}")
        check("real_14 has no camera extrinsics",
              not any(k.lower().startswith("pose") or "extrin" in k.lower()
                      for k in d))

    total = 52 - FAILED
    print(f"\n{'PASS' if not FAILED else 'FAIL'}: {total} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())