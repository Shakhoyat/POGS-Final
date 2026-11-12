"""Score a detector against the authors' real capture.

This is the only measurement in the study that transfers to real imagery. Our
own frames are raytraced, and a segmenter's accuracy on a raytraced cylinder
says nothing about real photographs, so detector accuracy is measured here
instead, on the 14 ZED frames the authors shipped, using their own ``seg``
labels as ground truth.

POGS' DETIC masks and the ``seg`` labels use different conventions (POGS emits
one boolean row per instance plus a background row; ``seg`` is a single label
map), so this compares them as *label partitions*:

* a predicted instance is matched to the GT label it overlaps most
* that overlap, expressed as IoU, is the score
* a GT label that no prediction touches counts as a miss, not as a zero that
  could be averaged away

Use it as::

    python validate_detector.py --row 00_baseline_detic_dinov2_clip
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_config as cfg  # noqa: E402

# POGS drops any mask larger than H*W/3.5. Mirrored here so both sides of the
# comparison apply the same ceiling -- otherwise DETIC would be penalised for
# masks POGS would discard anyway.
AREA_CEILING = 1.0 / 3.5


def load_frames(directory: str) -> list[dict]:
    frames = []
    for path in sorted(glob.glob(os.path.join(directory, "*.npy")),
                       key=lambda p: int(os.path.basename(p)[:-4])):
        obj = np.load(path, allow_pickle=True).item()
        frames.append(obj)
    return frames


def score_frame(pred_masks: np.ndarray, gt_seg: np.ndarray) -> dict:
    """Best-overlap IoU for each GT label.

    Matching is by *overlap*, never by label number. POGS numbers its mask
    rows in detection order while ``seg`` numbers instances its own way, so
    comparing ``pred == lab`` against ``gt == lab`` silently scores 0 the
    moment the two orderings disagree -- a detector that finds both regions
    perfectly would be reported as a total miss.
    """
    h, w = gt_seg.shape
    ceiling_px = h * w * AREA_CEILING

    keep = np.asarray([m for m in pred_masks if m.sum() <= ceiling_px], dtype=bool)
    gt_labels = [int(v) for v in np.unique(gt_seg) if v != 0]

    ious = {}
    for lab in gt_labels:
        g = gt_seg == lab
        best = 0.0
        for m in keep:
            inter = int(np.logical_and(m, g).sum())
            if inter == 0:
                continue
            union = int(np.logical_or(m, g).sum())
            if union:
                best = max(best, inter / union)
        ious[lab] = best
    return ious


def summarise(per_frame: list[dict[int, float]]) -> dict:
    all_ious = [v for d in per_frame for v in d.values()]
    if not all_ious:
        return {"frames": len(per_frame), "iou_mean": 0.0, "iou_best": 0.0,
                "iou_worst": 0.0, "labels_per_frame": 0.0, "coverage": 0.0}
    return {
        "frames": len(per_frame),
        "iou_mean": round(float(np.mean(all_ious)), 4),
        "iou_median": round(float(np.median(all_ious)), 4),
        "iou_best": round(float(np.max(all_ious)), 4),
        "iou_worst": round(float(np.min(all_ious)), 4),
        "labels_per_frame": round(
            float(np.mean([len(d) for d in per_frame])), 2),
        # Fraction of GT labels some prediction actually overlapped at all.
        # Detects a detector that returns nothing at all, which a mean of
        # zeros would look identical to one that returns bad masks.
        "coverage": round(float(np.mean([v > 0 for v in all_ious])), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Score a detector on real frames")
    ap.add_argument("--row", default=cfg.BASELINE_ROW)
    ap.add_argument("--masks", default=None,
                    help="path to a (M, H, W) bool .npy of predicted masks. "
                         "Omit to record that detection was not run.")
    ap.add_argument("--detector", default="detic")
    args = ap.parse_args()

    frames = load_frames(cfg.REAL_14)
    if not frames:
        print(f"no .npy frames in {cfg.REAL_14}", file=sys.stderr)
        return 1

    result = {
        "detector": args.detector,
        "frames_dir": cfg.REAL_14,
        "n_frames": len(frames),
        "area_ceiling": AREA_CEILING,
        "source": "authors' test_data (real ZED capture, via POGS README)",
        "per_frame": [],
    }

    if args.masks:
        masks = np.load(args.masks, allow_pickle=True)
        if masks.ndim == 4 and masks.shape[0] == 1:
            masks = masks[0]
        result["per_frame"] = [
            {str(k): round(v, 4) for k, v in score_frame(masks, f["seg"]).items()}
            for f in frames
        ]
        result["summary"] = summarise(result["per_frame"])
    else:
        result["summary"] = None
        result["status"] = "not run -- DETIC inference pending on the GPU host"

    out_dir = os.path.join(cfg.ARTEFACTS, args.row, "validate")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "detector_real_iou.json")
    with open(path, "w") as fh:
        json.dump(result, fh, indent=2)

    print("=" * 68)
    print("DETECTOR VALIDATION - authors' real capture")
    print("=" * 68)
    print(f"  frames     : {len(frames)}")
    print(f"  resolution : {frames[0]['rgb'].shape[1]}x{frames[0]['rgb'].shape[0]}")
    print(f"  area ceiling: {AREA_CEILING * 100:.1f}% of frame")
    if result["summary"]:
        s = result["summary"]
        print(f"  IoU mean   : {s['iou_mean']:.4f}")
        print(f"  IoU best   : {s['iou_best']:.4f}   IoU worst: {s['iou_worst']:.4f}")
        print(f"  coverage   : {s['coverage'] * 100:.1f}% of GT labels overlapped")
    else:
        print(f"  {result['status']}")
    print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())