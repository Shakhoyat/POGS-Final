"""
v2 end-to-end (reuses the v1 checkpoint, grouping and base-POGS runs):

    python -m experiments.v2_kalman_pogs.run_all [--factors speed rotation ...]
"""

from __future__ import annotations

import argparse
import time

from experiments.pipeline import build_groups, run_suite, train_or_load, rescore_runs
from experiments.v2_kalman_pogs.config import RESULTS_DIR, get_v2_config, v2_methods
from experiments.visuals import make_dashboards
from scene_generator.pick_place_benchmark import FACTORS, benchmark_cases


def main(factors=None, device: str = "cuda:0", videos: bool = True):
    t0 = time.time()
    cfg = get_v2_config(device)
    model, capture, dino, clip = train_or_load(cfg)
    grouping = build_groups(model, capture, cfg, clip=clip)
    rescore_runs(model, grouping)
    cases = benchmark_cases(factors)
    df = run_suite(model, grouping, dino, cfg, cases, v2_methods(), RESULTS_DIR)
    if videos:
        make_dashboards(model, grouping, cfg, RESULTS_DIR, cases, methods=["POGS + KF", "POGS + KF, 10 iters"])
    print(f"\nv2 done in {(time.time() - t0) / 60:.1f} min -> {RESULTS_DIR}/summary.md")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-videos", action="store_true")
    ap.add_argument("--factors", nargs="*", choices=list(FACTORS), default=None)
    a = ap.parse_args()
    main(factors=a.factors, videos=not a.no_videos)
