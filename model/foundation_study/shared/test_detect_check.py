"""Checks for detect_check that need neither DETIC nor a GPU.

The detector itself cannot be tested here, but everything around it can, and
that is where the silent failures live: a mask compared against the wrong
resolution grid, or a channel order flipped, both produce a confident number that
means nothing.

    python test_detect_check.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_config as cfg  # noqa: E402
from detect_check import (  # noqa: E402
    AREA_CEILING,
    best_overlap_iou,
    capture_frames,
    resize_masks,
    summarise_capture,
)

FAILED = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global FAILED
    if not ok:
        FAILED += 1
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def main() -> int:
    print("=" * 68)
    print("DETECT CHECK - HOST-INDEPENDENT CHECKS")
    print("=" * 68)

    # --- masks -------------------------------------------------------------
    m = np.zeros((2, 60, 80), dtype=bool)
    m[0, 10:30, 20:40] = True
    m[1, 0:5, 0:5] = True
    same = resize_masks(m, 60, 80)
    check("matching resolution is a no-op",
          same.shape == m.shape and bool((same == m).all()))
    up = resize_masks(m, 120, 160)
    check("half-res masks upscale to the GT grid", up.shape == (2, 120, 160),
          f"{up.shape}")
    # Nearest-neighbour must not invent partial pixels the generator never drew.
    check("upscale stays binary", set(np.unique(up).tolist()) <= {0, 1},
          f"values {np.unique(up).tolist()}")
    check("upscaled area roughly doubles per axis",
          abs(up[0].sum() / m[0].sum() - 4.0) < 0.35,
          f"{up[0].sum() / m[0].sum():.2f}x")

    # --- IoU matching ------------------------------------------------------
    seg = np.zeros((60, 80), dtype=np.float32)
    seg[10:30, 20:40] = 1        # label 1 == m[0] exactly
    seg[40:55, 50:70] = 2        # label 2 overlaps m[1] only a little
    ious = best_overlap_iou(m, seg)
    check("exact mask scores IoU 1.0", abs(ious.get(1, 0.0) - 1.0) < 1e-6,
          f"{ious.get(1)}")
    check("second GT label is scored too", 2 in ious, f"{sorted(ious)}")
    check("partial overlap is below 1", ious.get(2, 1.0) < 0.2,
          f"{ious.get(2)}")

    # Matching must be by overlap, not label index. Swapping which mask is which
    # must not change the score -- POGS numbers mask rows in detection order and
    # `seg` numbers instances its own way, so comparing indices would score a
    # correct detector at zero.
    swapped = np.stack([m[1], m[0]])
    ious2 = best_overlap_iou(swapped, seg)
    check("score is independent of mask row order",
          abs(ious.get(1, 0) - ious2.get(1, 0)) < 1e-9,
          f"{ious.get(1)} vs {ious2.get(1)}")

    big = np.ones((1, 60, 80), dtype=bool)
    check("a full-frame mask is filtered by the area ceiling",
          best_overlap_iou(big, seg).get(1, 0.0) == 0.0,
          f"ceiling {AREA_CEILING:.3f} of frame")

    # --- capture loading ---------------------------------------------------
    if not os.path.isdir(cfg.PICKPLACE):
        print(f"\nno capture under {cfg.PICKPLACE}; run generate_pickplace.py")
        return 1
    frames = capture_frames(cfg.PICKPLACE)
    check("36 capture views load", len(frames) == 36, f"{len(frames)}")

    name, rgb = frames[0]
    check("capture reads RGB, 1280x720, uint8",
          rgb.shape == (720, 1280, 3) and rgb.dtype == np.uint8,
          f"{rgb.shape} {rgb.dtype}")

    # The mug is authored red (0.72, 0.16, 0.14) and nothing else in the scene is
    # saturated, so the reddest 0.1% of pixels must be red in channel 0. If this
    # fails the channel order is flipped and DETIC is being shown a blue mug.
    flat = rgb.reshape(-1, 3).astype(np.float32)
    red = flat[:, 0] - 0.5 * (flat[:, 1] + flat[:, 2])
    top = flat[np.argsort(red)[-200:]]
    check("red mug is red in channel 0 (channels not flipped)",
          float(top[:, 0].mean()) > 1.5 * float(top[:, 2].mean()),
          f"R={top[:, 0].mean():.0f} B={top[:, 2].mean():.0f}")

    # --- summary -----------------------------------------------------------
    per_frame = [
        {"n_masks": 3, "n_masks_kept": 3, "largest_area_frac": 0.031},
        {"n_masks": 1, "n_masks_kept": 1, "largest_area_frac": 0.028},
        {"n_masks": 0, "n_masks_kept": 0, "largest_area_frac": None},
    ]
    s = summarise_capture(per_frame)
    check("counts the frames that found something",
          s["frames_with_any_mask"] == 2, f"{s['frames_with_any_mask']}")
    check("counts mug-sized masks", s["frames_with_mug_sized_mask"] == 2,
          f"{s['frames_with_mug_sized_mask']}")
    check("survives a frame with no masks", s["largest_area_frac_max"] == 0.031,
          f"{s['largest_area_frac_max']}")

    empty = summarise_capture([{"n_masks": 0, "n_masks_kept": 0,
                               "largest_area_frac": None}])
    check("all-empty summary is still well formed",
          empty["frames_with_any_mask"] == 0
          and empty["largest_area_frac_median"] is None)

    total = 15 - FAILED
    print(f"\n{'PASS' if not FAILED else 'FAIL'}: {total} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())