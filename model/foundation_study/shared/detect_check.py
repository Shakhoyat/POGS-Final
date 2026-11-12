"""Does DETIC actually see the mug? A one-minute check before a 40-minute run.

POGS trains on DETIC masks: ``FullImageDatamanager`` builds a ``DeticDataloader``
at ``full_images_datamanager.py:174`` and feeds its masks into the CLIP/lerf loss
from ``lerf_step = 2500`` on. If DETIC returns nothing on our capture, POGS
asserts ``"No masks found"`` at line 437 and training dies -- or worse, survives
with an empty mask set and produces a meaningless field.

That failure would surface only after the long run has started. This script
surfaces it in about a minute, and it is the check that decides whether the
experiment is worth running.

Two modes, sharing one detector call per image:

``--capture`` (default)
    Run DETIC over our 36 capture views. No ground truth: the question is only
    "did it find anything", and a plausible mug-sized mask is the answer.

``--real``
    Run DETIC over the authors' 14 real ZED frames and score against their own
    ``seg`` labels. This is the only detector-accuracy number in the study that
    transfers to real imagery, since accuracy measured on our raytraced frames
    would not.

Detector logic is not reimplemented. ``DeticDataloader.predict()`` already
applies the stock config, the LVIS vocabulary, ``SCORE_THRESH_TEST = 0.5`` and the
``H*W/3.5`` area ceiling POGS itself applies, so this cannot drift from what
training does. Two consequences of reusing it:

* ``predict()`` returns masks at **full** resolution. The half-resolution
  ``downscale_factor=2`` path only runs in ``create()``, the batch entry point the
  datamanager uses. Per-image scoring needs full resolution, so ``predict()`` is
  the right call.
* importing ``detic_dataloader`` chdirs into ``dependencies/Detic`` at line 26 and
  restores at line 52, so every path handed to it must be resolved to absolute
  first. This script resolves them anyway, since it is also the thing the user
  types.

Use::

    python detect_check.py                          # our capture, stock vocab
    python detect_check.py --vocab mug cup          # prompted, for diagnosis
    python detect_check.py --real                   # authors' frames, IoU
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_config as cfg  # noqa: E402

# POGS' own filter, detic_dataloader.py:282 and :287. Mirrored so we can report
# how many raw detections were dropped, which is the usual reason a small object
# "is not detected" -- DETIC found it and POGS discarded it for being too big or
# scoring under threshold.
AREA_CEILING = 1.0 / 3.5


def build_detic(vocab: list[str]):
    """A ``DeticDataloader`` configured the way POGS configures it.

    ``image_list=None`` skips the batch inference in ``create()``; we want the
    per-image ``predict()`` path instead. ``custom_vocab`` empty is stock POGS:
    ``full_images_datamanager.py:179`` passes ``[]``, which selects LVIS. Passing
    words here switches DETIC to open-vocabulary, so it stops being the baseline.
    """
    import torch
    from pogs.data.utils.detic_dataloader import DeticDataloader

    cache = os.path.join(cfg.ARTEFACTS, "detect_check", "detic_cache.npy")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    detic = DeticDataloader(
        image_list=None,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        cfg={"sam": False, "custom_vocab": vocab, "downscale_factor": 1},
        cache_path=Path(cache),
    )
    # A pre-existing cache makes load() succeed and skips create(), which is
    # the only place detic_predictor is built -- predict() needs it either way.
    if not hasattr(detic, "detic_predictor"):
        detic.create()
    return detic


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    return np.asarray(x)


def predict(detic, img_rgb: np.ndarray) -> dict:
    """DETIC masks for one RGB uint8 image, via POGS' own ``predict``.

    ``predict`` reverses channels internally for detectron2, so it wants RGB.
    The last row of ``masks_filtered`` is the background row POGS appends; it is
    dropped here so every downstream count is over foreground instances only.
    """
    out = detic.predict(img_rgb)
    masks = _to_numpy(out["masks_filtered"])
    return {
        "masks": masks[:-1].astype(bool),
        "scores": _to_numpy(out["scores"]).tolist(),
        "boxes": np.asarray(out["boxes"]).tolist(),
    }


def capture_frames(scene: str) -> list[tuple[str, np.ndarray]]:
    paths = sorted(glob.glob(os.path.join(scene, "img", "*.png")))
    if not paths:
        raise SystemExit(f"no capture images under {os.path.join(scene, 'img')}")
    # cv2 reads BGR; DETIC is RGB-native, so convert or the red mug reads blue
    # and "mug" becomes a much harder ask than it should be.
    return [(os.path.basename(p), cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))
            for p in paths]


def real_frames(directory: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    out = []
    for p in sorted(glob.glob(os.path.join(directory, "*.npy")),
                    key=lambda q: int(os.path.basename(q)[:-4])):
        d = np.load(p, allow_pickle=True).item()
        out.append((os.path.basename(p), d["rgb"], d["seg"]))
    if not out:
        raise SystemExit(f"no .npy frames in {directory}")
    return out


def resize_masks(masks: np.ndarray, h: int, w: int) -> np.ndarray:
    """Nearest-neighbour to the GT grid.

    Nearest, not bilinear: a bilinear mask upsampled then thresholded invents
    partial pixels at the mug's edge and quietly inflates IoU. Masks here already
    match the GT resolution, so this is a guard against a resolution change in the
    real frames rather than a code path we expect to take.
    """
    if masks.ndim == 4:
        masks = masks[:, 0]
    if masks.shape[1:] == (h, w):
        return masks.astype(bool)
    return np.stack([cv2.resize(m.astype(np.uint8), (w, h),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
                     for m in masks])


def best_overlap_iou(masks: np.ndarray, gt_seg: np.ndarray) -> dict:
    """Best-overlap IoU per GT label. Shared with ``validate_detector``.

    Imported rather than duplicated: the two must agree on matching-by-overlap,
    because a second copy is exactly how a scoring method starts disagreeing with
    the harness that reads its output.
    """
    from validate_detector import score_frame

    return score_frame(masks, gt_seg)


def summarise_capture(per_frame: list[dict]) -> dict:
    """Is there anything plausible? Reported as counts, not as a verdict.

    The mug is 2.3-3.6% of frame here, so "some mask exists" is weak evidence and
    saying so is the point of reporting areas rather than a pass/fail flag.
    """
    total = [f["n_masks"] for f in per_frame]
    kept = [f["n_masks_kept"] for f in per_frame]
    areas = [f["largest_area_frac"] for f in per_frame if f["largest_area_frac"]]
    mug_like = [f["largest_area_frac"] for f in per_frame
                if 0.005 <= (f["largest_area_frac"] or 0.0) <= 0.10]
    return {
        "frames": len(per_frame),
        "frames_with_any_mask": int(sum(1 for n in total if n > 0)),
        "masks_total": int(sum(total)),
        "masks_after_area_filter": int(sum(kept)),
        "dropped_by_area_filter": int(sum(total) - sum(kept)),
        "largest_area_frac_median": (round(float(np.median(areas)), 5)
                                     if areas else None),
        "largest_area_frac_max": (round(float(np.max(areas)), 5)
                                  if areas else None),
        "frames_with_mug_sized_mask": len(mug_like),
        "note": (
            "the mug occupies 2.3-3.6% of frame, so 'frames_with_mug_sized_mask' "
            "is suggestive, not proof. A definitive answer comes from tracking, "
            "where the ground-truth mask selects which gaussians to follow."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Check DETIC finds the mug")
    ap.add_argument("--row", default=cfg.BASELINE_ROW)
    ap.add_argument("--scene", default=None,
                    help="scene name from the matrix, or a path. Defaults to "
                         "the primary scene.")
    ap.add_argument("--real", action="store_true",
                    help="score the authors' real frames instead of our capture")
    ap.add_argument("--vocab", nargs="*", default=None,
                    help="open-vocabulary words; default is stock LVIS, which is "
                         "what POGS itself uses")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N frames, for a quick smoke check")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Absolute, because importing detic_dataloader chdirs away from here.
    # A bare name is looked up in the scene matrix; anything with a separator is
    # taken as a path, which is how run.sh passes it.
    if args.scene is None:
        scene = os.path.abspath(cfg.PICKPLACE)
        scene_name = os.path.basename(os.path.normpath(scene))
    elif os.sep in args.scene or (os.altsep and os.altsep in args.scene):
        scene = os.path.abspath(args.scene)
        scene_name = os.path.basename(os.path.normpath(scene))
    else:
        scene_name = args.scene
        scene = cfg.scene_path(scene_name)

    vocab = args.vocab or []
    if args.vocab:
        print(f"open-vocabulary: {vocab}  (NOT the stock baseline)")

    print("=" * 68)
    print("DETIC DETECTION CHECK"
          + ("  -- authors' real frames" if args.real
             else f"  -- {scene_name}"))
    print("=" * 68)
    print("loading DETIC (first run downloads ~1GB of weights)...")
    detic = build_detic(vocab)

    # Per-scene by default, so a sweep's ten checks do not overwrite each other.
    # --real keeps a single slot because those 14 frames say the same thing for
    # every scene.
    default_slot = "real_14" if args.real else scene_name
    out_dir = args.out or os.path.join(cfg.ARTEFACTS, args.row, default_slot)
    os.makedirs(out_dir, exist_ok=True)

    if args.real:
        frames = real_frames(os.path.abspath(cfg.REAL_14))
        if args.limit:
            frames = frames[:args.limit]
        print(f"\n{len(frames)} real frames from {cfg.REAL_14}")
        per_frame, ious = [], []
        for name, rgb, seg in frames:
            res = predict(detic, rgb)
            masks = resize_masks(res["masks"], seg.shape[0], seg.shape[1])
            scored = best_overlap_iou(masks, seg)
            per_frame.append({str(k): round(v, 4) for k, v in scored.items()})
            flat = [v for v in scored.values()]
            ious += flat
            best = max(flat) if flat else 0.0
            print(f"  {name:>10}  {len(masks):3d} masks   "
                  f"labels {len(scored):2d}   best IoU {best:.4f}")

        result = {
            "detector": "detic",
            "source": "authors' test_data (real ZED capture, via POGS README)",
            "area_ceiling": AREA_CEILING,
            "n_frames": len(frames),
            "vocab": vocab or "lvis (stock POGS)",
            "per_frame": per_frame,
            "summary": {
                "iou_mean": round(float(np.mean(ious)), 4) if ious else 0.0,
                "iou_median": round(float(np.median(ious)), 4) if ious else 0.0,
                "iou_best": round(float(np.max(ious)), 4) if ious else 0.0,
                "iou_worst": round(float(np.min(ious)), 4) if ious else 0.0,
                "labels_per_frame": round(
                    float(np.mean([len(d) for d in per_frame])), 2) if per_frame else 0.0,
                "coverage": round(float(np.mean([v > 0 for v in ious])), 4)
                            if ious else 0.0,
            },
        }
        path = os.path.join(out_dir, "detector_real_iou.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        s = result["summary"]
        print(f"\n  mean IoU   {s['iou_mean']:.4f}")
        print(f"  coverage   {s['coverage'] * 100:.1f}% of GT labels overlapped")
        print(f"  wrote {path}")
        return 0

    frames = capture_frames(scene)
    if args.limit:
        frames = frames[:args.limit]
    print(f"\n{len(frames)} capture views from {scene}")

    per_frame = []
    for name, rgb in frames:
        res = predict(detic, rgb)
        masks = res["masks"]
        h, w = rgb.shape[:2]
        ceiling = h * w * AREA_CEILING
        areas = masks.reshape(len(masks), -1).sum(1) if len(masks) else []
        kept = [a for a in areas if a <= ceiling]
        rec = {
            "frame": name,
            "n_masks": int(len(masks)),
            "n_masks_kept": int(len(kept)),
            "largest_area_frac": (round(float(max(areas)) / (h * w), 5)
                                  if len(areas) else None),
            "scores": [round(float(s), 3) for s in res["scores"][:8]],
        }
        per_frame.append(rec)
        print(f"  {name:>18}  {rec['n_masks']:3d} masks   "
              f"kept {rec['n_masks_kept']:3d}   "
              f"largest {rec['largest_area_frac'] if rec['largest_area_frac'] is not None else 0:.4f}"
              f" of frame")

    summary = summarise_capture(per_frame)
    result = {
        "scene": scene,
        "vocab": vocab or "lvis (stock POGS)",
        "summary": summary,
        "per_frame": per_frame,
    }
    path = os.path.join(out_dir, "capture_detect.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print("\n" + "-" * 68)
    print(f"  frames with any mask      {summary['frames_with_any_mask']}"
          f" / {summary['frames']}")
    print(f"  mug-sized masks           {summary['frames_with_mug_sized_mask']}"
          f" / {summary['frames']}")
    print(f"  dropped by area filter    {summary['dropped_by_area_filter']}"
          f"  (POGS drops masks > H*W/3.5)")
    print(f"  largest area, median      "
          f"{summary['largest_area_frac_median']} of frame")
    print(f"  wrote {path}")

    if summary["frames_with_any_mask"] == 0:
        print("\n  VERDICT: DETIC found nothing on our capture. POGS would "
              "assert 'No masks found' at full_images_datamanager.py:437 "
              "during training. Do not start the long run; try --vocab mug cup "
              "and see whether that is a vocabulary problem or a "
              "detectability one.")
        return 1
    print("\n  DETIC returns masks, so training should get past the mask "
          "assert. Run 'bash run.sh track' after training to confirm the mug "
          "specifically is trackable -- mask counts alone do not prove it is.")
    return 0


if __name__ == "__main__":
    sys.exit(main())