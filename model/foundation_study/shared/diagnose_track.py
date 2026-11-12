"""Where does a tracking error come from? Reads tracking.json, runs no GPU code.

For each scene prints, every few frames, the ground-truth displacement from frame
0, the tracker's estimated displacement, their error, and the cosine between the
two displacement vectors. Read the shape:

* error already large at frame 1-2 and roughly constant -> the tracker's pose
  at frame 0 / its first step is wrong (initialisation or reference frame)
* estimate displacement >> ground truth, wandering direction (cos near 0 or
  negative) -> drift / a random walk; the optimiser is not following the object
* est ~ k * gt with cos near 1 and k != 1 -> a units or scale mismatch (a harness
  bug, not a model result)

    python diagnose_track.py                      # every scene of row 00
    python diagnose_track.py pendulum_05 pendulum_20 --every 3
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_config as cfg  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", help="scene names (default: all)")
    ap.add_argument("--row", default=cfg.BASELINE_ROW)
    ap.add_argument("--every", type=int, default=5)
    args = ap.parse_args()

    base = os.path.join(cfg.ARTEFACTS, args.row)
    paths = sorted(glob.glob(os.path.join(base, "*", "track", "tracking.json")))
    for p in paths:
        scene = p.split(os.sep)[-3]
        if args.scenes and scene not in args.scenes:
            continue
        r = json.load(open(p))
        fr = r["frames"]
        est0 = np.mean(np.array(fr[0]["est_centroids"]), axis=0)
        gt0 = np.array(fr[0]["gt_centroid"])
        s = r["summary"]
        print(f"\n== {scene}   parts {len(fr[0]['est_centroids'])}   "
              f"kept gaussians {r.get('n_gaussians_kept')}/{r.get('n_gaussians_total')}   "
              f"mean err {s['centroid_err_mm_mean']:.1f} mm   lost_at {s.get('lost_at_frame')}")
        print(f"   {'frame':>5} {'gt mm':>8} {'est mm':>8} {'err mm':>8} {'cos':>6}  note")
        for f in fr:
            i = f["frame"]
            if i % args.every and i != 1 and i != len(fr) - 1:
                continue
            g = (np.array(f["gt_centroid"]) - gt0) * 1000.0
            e = (np.mean(np.array(f["est_centroids"]), axis=0) - est0) * 1000.0
            gn, en = np.linalg.norm(g), np.linalg.norm(e)
            cos = float(g @ e / (gn * en)) if gn > 1e-6 and en > 1e-6 else float("nan")
            print(f"   {i:>5} {gn:>8.1f} {en:>8.1f} {np.linalg.norm(e - g):>8.1f} "
                  f"{cos:>6.2f}  {'LOST' if f.get('lost') else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
