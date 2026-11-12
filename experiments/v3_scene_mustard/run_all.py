"""
v3 end-to-end (reuses the v1 checkpoint, grouping and cached runs):

    python -m experiments.v3_scene_mustard.run_all
"""

from __future__ import annotations

import time

from experiments.pipeline import build_groups, run_suite, train_or_load, rescore_runs
from experiments.v3_scene_mustard.config import FACTORS_V3, RESULTS_DIR, get_v3_config, v3_methods
from experiments.visuals import make_dashboards
from scene_generator.pick_place_benchmark import benchmark_cases


def main(device: str = "cuda:0", videos: bool = True):
    t0 = time.time()
    cfg = get_v3_config(device)
    model, capture, dino, clip = train_or_load(cfg)
    grouping = build_groups(model, capture, cfg, clip=clip)
    rescore_runs(model, grouping)
    cases = benchmark_cases(FACTORS_V3)
    df = run_suite(model, grouping, dino, cfg, cases, v3_methods(), RESULTS_DIR)
    if videos:
        make_dashboards(model, grouping, cfg, RESULTS_DIR, cases, methods=["POGS"])
    print(f"\nv3 done in {(time.time() - t0) / 60:.1f} min -> {RESULTS_DIR}/summary.md")
    return df


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--no-videos", action="store_true")
    main(videos=not ap.parse_args().no_videos)
