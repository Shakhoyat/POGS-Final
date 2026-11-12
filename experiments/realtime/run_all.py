"""
Real-time protocol end to end (reuses the v1 checkpoint and grouping):

    python -m experiments.realtime.run_all [--seeds 0 1 2] [--speeds 1 2] [--no-videos]

Renders the 30 Hz sequences first if they are missing (GPU, ~8 min each).
Timing is measured: run it while nothing else uses the GPU.
"""

from __future__ import annotations

import argparse
import time

from experiments.pipeline import DATASET_DIR, LATENCY_COMP, build_groups, run_realtime_suite, train_or_load
from experiments.realtime.config import RESULTS_DIR, get_realtime_config, realtime_methods
from experiments.visuals import make_dashboards
from data.scene_generator.pick_place_benchmark import (
    N_SEEDS, REALTIME_CONDITIONS, REALTIME_SPEEDS, generate_benchmark, realtime_cases,
)


def main(seeds=None, speeds=REALTIME_SPEEDS, conditions=REALTIME_CONDITIONS, device: str = "cuda:0",
         videos: bool = True):
    t0 = time.time()
    seeds = list(range(N_SEEDS)) if seeds is None else list(seeds)
    cases = realtime_cases(seeds, conditions, speeds)
    generate_benchmark(DATASET_DIR, sequences=sorted({c.sequence for c in cases}), device=device)
    cfg = get_realtime_config(device)
    model, capture, dino, clip = train_or_load(cfg)
    grouping = build_groups(model, capture, cfg, clip=clip)
    df = run_realtime_suite(model, grouping, dino, cfg, cases, realtime_methods(), RESULTS_DIR)
    if videos:
        make_dashboards(model, grouping, cfg, RESULTS_DIR, cases,
                        methods=["POGS", "POGS + KF + check, 10 iters" + LATENCY_COMP])
    print(f"\nreal-time done in {(time.time() - t0) / 60:.1f} min -> {RESULTS_DIR}/summary.md")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--speeds", nargs="*", type=int, default=list(REALTIME_SPEEDS))
    ap.add_argument("--no-videos", action="store_true")
    a = ap.parse_args()
    main(seeds=a.seeds, speeds=a.speeds, videos=not a.no_videos)
