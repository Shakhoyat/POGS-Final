"""
v1 end-to-end: benchmark data -> train POGS -> group objects -> track -> visuals.

    python -m experiments.v1_base_pogs.run_all [--force-train] [--regen-data]
                                              [--factors rotation speed ...]
"""

from __future__ import annotations

import argparse
import os
import time

from experiments.pipeline import DATASET_DIR, MethodSpec, build_groups, run_suite, train_or_load, rescore_runs
from experiments.v1_base_pogs.config import RESULTS_DIR, V1_ABLATIONS, V1_MAIN, get_v1_config
from experiments.visuals import make_dashboards
from data.scene_generator.pick_place_benchmark import FACTORS, N_SEEDS, Case, benchmark_cases, generate_benchmark


def main(force_train: bool = False, regen_data: bool = False, factors=None, device: str = "cuda:0",
         videos: bool = True):
    t0 = time.time()
    generate_benchmark(DATASET_DIR, device=device, capture=True if regen_data else None, overwrite=regen_data)
    cfg = get_v1_config(device)
    model, capture, dino, clip = train_or_load(cfg, force_train=force_train)
    grouping = build_groups(model, capture, cfg, clip=clip)
    rescore_runs(model, grouping)
    cases = benchmark_cases(factors)
    df = run_suite(model, grouping, dino, cfg, cases, [MethodSpec(n, v) for n, v in V1_MAIN], RESULTS_DIR)
    run_suite(model, grouping, dino, cfg, [Case("base", s) for s in range(N_SEEDS)],
              [MethodSpec(n, v) for n, v in V1_MAIN[:1] + V1_ABLATIONS], os.path.join(RESULTS_DIR, "ablations"))
    if videos:
        make_dashboards(model, grouping, cfg, RESULTS_DIR, cases, methods=[V1_MAIN[0][0]])
    print(f"\nv1 done in {(time.time() - t0) / 60:.1f} min -> {RESULTS_DIR}/summary.md")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-train", action="store_true")
    ap.add_argument("--regen-data", action="store_true")
    ap.add_argument("--no-videos", action="store_true")
    ap.add_argument("--factors", nargs="*", choices=list(FACTORS), default=None)
    a = ap.parse_args()
    main(force_train=a.force_train, regen_data=a.regen_data, factors=a.factors, videos=not a.no_videos)
