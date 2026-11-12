"""Checks for track_offline's host-independent logic.

Run on any machine, no GPU or POGS environment needed::

    python test_track_offline.py

Everything that needs a trained model or CUDA is deliberately not covered here
-- the point is that when this fails on the 4070 the failure is the model or the
CUDA install, never the bookkeeping around it.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_config as cfg  # noqa: E402
from track_offline import build_clusters, load_sequence, score_frame  # noqa: E402

FAILED = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global FAILED
    if not ok:
        FAILED += 1
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def main() -> int:
    trk = os.path.join(cfg.PICKPLACE, "tracking")
    print("=" * 64)
    print("TRACK OFFLINE - HOST-INDEPENDENT CHECKS")
    print("=" * 64)

    if not os.path.isdir(trk):
        print(f"no tracking/ under {cfg.PICKPLACE}")
        print("generate it first: python generate_pickplace.py")
        return 1

    images, depths, K, width, height, c2w, gt = load_sequence(trk)
    check("120 frames of left/ and depth/", len(images) == len(depths) == 120,
          f"{len(images)}/{len(depths)}")
    check("intrinsics match the generator's", K[0, 0] == 912.0 and K[0, 2] == 640.0,
          f"fx={K[0, 0]} cx={K[0, 2]}")
    check("resolution matches the intrinsics", (width, height) == (1280, 720),
          f"{width}x{height}")

    rot = c2w[:3, :3]
    check("OpenCV rotation is right-handed", np.linalg.det(rot) > 0.999,
          f"det={np.linalg.det(rot):.6f}")
    fwd = rot[:, 2]
    check("camera looks toward +Y", fwd[1] > 0.9, f"fwd={np.round(fwd, 3)}")
    check("camera is on the -Y side, above the table", c2w[1, 3] < -0.4
          and c2w[2, 3] > 0.9, f"at={np.round(c2w[:3, 3], 3)}")

    # The mug base must project inside the frame, or the tracker would be
    # initialising on an object it cannot see.
    gt0 = gt["frames"][0]["focus"]
    p = np.array(gt0["centroid"], dtype=np.float64)
    cam = c2w[:3, 3]
    d = rot.T @ (p - cam)
    u = K[0, 0] * d[0] / d[2] + K[0, 2]
    v = K[1, 1] * d[1] / d[2] + K[1, 2]
    check("mug base projects inside frame 0", 0 < u < width and 0 < v < height,
          f"u={u:.0f} v={v:.0f}")

    mask0 = os.path.join(trk, "obj_id", os.path.basename(images[0]))
    check("ground-truth obj_id mask present", os.path.exists(mask0))
    check("depth for frame 0 present", os.path.exists(depths[0]))

    # score_frame: identical estimate and ground truth must give 0 mm, and a
    # known 10 mm offset must show up as 10 mm. A silent sign or unit error
    # here would make every tracking number in the report wrong.
    est0 = np.tile(np.eye(4), (2, 1, 1))
    est0[:, :3, 3] = np.array([0.01, 0.02, 0.03])   # part centroid, not base
    gt0 = np.array(gt["frames"][0]["focus"]["centroid"])
    gt1 = np.array(gt["frames"][1]["focus"]["centroid"])

    r = score_frame(0, est0, gt["frames"][0], est0, gt0)
    check("frame 0 is exactly zero",
          all(abs(v) < 1e-9 for v in r["centroid_err_mm"]),
          f"{r['centroid_err_mm']}")

    # A perfect tracker: estimate moves by exactly what the ground truth moved.
    est1 = est0.copy()
    est1[:, :3, 3] += (gt1 - gt0)
    r1 = score_frame(1, est1, gt["frames"][1], est0, gt0)
    check("a tracker matching the true motion scores 0",
          abs(r1["centroid_err_mm_mean"]) < 1e-9,
          f"{r1['centroid_err_mm_mean']} mm")

    # 10 mm off, sideways. Differencing distances instead of displacements would
    # report this as near zero, because the offset is nearly perpendicular to the
    # ray from ground truth to estimate.
    est_bad = est1.copy()
    est_bad[:, :3, 3] += np.array([0.010, 0.0, 0.0])
    r2 = score_frame(1, est_bad, gt["frames"][1], est0, gt0)
    check("a 10 mm sideways error reads as 10 mm",
          abs(r2["centroid_err_mm_mean"] - 10.0) < 1e-6,
          f"{r2['centroid_err_mm_mean']} mm")

    est50 = est1.copy()
    est50[:, :3, 3] += np.array([0.0, 0.030, 0.040])   # 50 mm
    r4 = score_frame(1, est50, gt["frames"][1], est0, gt0)
    check("a 50 mm error reads as 50 mm",
          abs(r4["centroid_err_mm_mean"] - 50.0) < 1e-6,
          f"{r4['centroid_err_mm_mean']} mm")

    # Per-part offsets that differ must not cancel into the mean.
    est_split = est1.copy()
    est_split[0, :3, 3] += np.array([0.0, 0.0, 0.020])
    est_split[1, :3, 3] += np.array([0.0, 0.0, -0.020])
    r5 = score_frame(1, est_split, gt["frames"][1], est0, gt0)
    check("two parts splitting 20 mm each do not average to zero",
          abs(r5["centroid_err_mm_mean"] - 20.0) < 1e-6,
          f"per-part {r5['centroid_err_mm']}")

    class FakeModel:
        def __init__(self, n):
            self.gauss_params = {"means": np.zeros((n, 3))}

    m = FakeModel(5)
    c = build_clusters(m, np.array([0, 2]), np.array([0, 1, 0, 2, 1]))
    check("clusters.npy is the 3-tuple optim expects", len(c) == 3,
          f"len={len(c)}")
    check("keep_inds round-trips", list(np.asarray(c[1])) == [0, 2])
    check("cgtf_stack is (n_clusters, 7)", np.asarray(c[2]).shape == (3, 7),
          f"{np.asarray(c[2]).shape}")

    try:
        build_clusters(FakeModel(4), np.array([0]), np.zeros(5))
        check("length mismatch is rejected", False, "no error raised")
    except SystemExit:
        check("length mismatch is rejected", True)

    print(f"\n{'PASS' if not FAILED else 'FAIL'}: {16 - FAILED} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())