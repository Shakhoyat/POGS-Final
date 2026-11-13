"""
PLAN step 0.4: does the re-implementation reproduce on the target stack?

Regenerates the benchmark data (if absent), trains the checkpoint (if absent),
groups, and tracks the base case (3 seeds) with Static + POGS into a FRESH run
directory, then compares with the 2026-11-10 reference (docs/RESULTS_reproduction.md
and the cached rows of the previous checkpoint). Writes data_results/repro_check/report.json.

    source tools/cuda_env.sh && .venv/bin/python -m experiments.reproduce_check
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from experiments import pipeline
from experiments.env_info import write_env_json
from experiments.pipeline import DATASET_DIR, RUNS_DIR, SCENE_CKPT_DIR, MethodSpec, build_groups, run_suite, train_or_load
from experiments.v1_base_pogs.config import get_v1_config
from data.scene_generator.pick_place_benchmark import N_SEEDS, Case, generate_benchmark
from model.pogs.checkpoint import write_json

OUT = "data_results/repro_check"
REFERENCE = {  # docs/RESULTS_reproduction.md (2026-11-10, torch 2.14 / Python 3.14)
    "psnr_mean": 34.1,
    "iou": {"mustard_bottle": 0.98, "red_apple": 0.95, "cereal_box": 0.88, "ceramic_mug": 0.80, "soup_can": 0.67},
}
IOU_TOL = 0.03
PSNR_TOL = 0.5


def main(device: str = "cuda:0") -> dict:
    t0 = time.time()
    write_env_json(OUT)
    generate_benchmark(DATASET_DIR, device=device)
    cfg = get_v1_config(device)
    model, capture, dino, clip = train_or_load(cfg)
    grouping = build_groups(model, capture, cfg, clip=clip)
    summary = json.load(open(os.path.join(SCENE_CKPT_DIR, "train_summary.json")))

    cases = [Case("base", s) for s in range(N_SEEDS)]
    df = run_suite(model, grouping, dino, cfg, cases, [MethodSpec("POGS", "POGS")], OUT,
                   runs_dir=os.path.join(OUT, "runs"), skip_existing=True)

    # Reference rows from the previous checkpoint (same trajectories), for comparison only.
    ref_rows = []
    for c in cases:
        p = os.path.join(RUNS_DIR, c.name, "pogs", "row.json")
        if os.path.exists(p):
            ref_rows.append(json.load(open(p)))
    new = df[df.method == "POGS"]
    iou = {g["name"]: g["iou_vs_gt"] for g in grouping.report["groups"]}
    rep = {
        "ckpt_id": pipeline.CKPT_ID,
        "psnr_mean": summary.get("psnr_mean"), "psnr_ref": REFERENCE["psnr_mean"],
        "psnr_ok": summary.get("psnr_mean") is not None and abs(summary["psnr_mean"] - REFERENCE["psnr_mean"]) <= PSNR_TOL,
        "n_gaussians": summary.get("n_gaussians"),
        "iou": iou, "iou_ref": REFERENCE["iou"],
        "iou_ok": {k: (k in iou and abs(iou[k] - v) <= IOU_TOL) for k, v in REFERENCE["iou"].items()},
        "gt_objects_missing": grouping.report["gt_objects_missing"],
        "base_pogs_new": {
            "adds_success": [float(x) for x in new.adds_success],
            "mean_pos_err_cm": [float(x) for x in new.mean_pos_err_cm],
            "fps": [float(x) for x in new.fps],
        },
        "base_pogs_ref": {
            "adds_success": [r.get("adds_success") for r in ref_rows],
            "mean_pos_err_cm": [r.get("mean_pos_err_cm") for r in ref_rows],
            "fps": [r.get("fps") for r in ref_rows],
        },
        "exit_100pct_adds": bool(len(new) and np.all(new.adds_success.values >= 1.0)),
        "minutes": (time.time() - t0) / 60,
    }
    write_json(os.path.join(OUT, "report.json"), rep)
    print(json.dumps(rep, indent=2))
    return rep


if __name__ == "__main__":
    main()
