"""
Master Benchmark & Ablation Study Suite (v2).

Ablations:
1. POGS + Kalman Filter (Full Proposed: Predictive warm-start + Covariance fusion, 25 iters)
2. Base POGS (Baseline v1: No Kalman, 25 iters)
3. No Depth Ablation (Kalman + DINO only, 25 iters)
4. No DINO Ablation (Kalman + Depth only, 25 iters)
5. Fast Tracking Ablation (Kalman + DINO + Depth, 10 iters)
6. Ultra-Fast Tracking Ablation (Kalman + DINO + Depth, 5 iters)

Evaluates on the certified pick-and-place benchmark across:
- respected (smooth transit, <=20 deg yaw)
- threshold (nominal transit, <=30 deg yaw)
- violated (rapid transit, >30 deg yaw)
"""

from __future__ import annotations

import json
import os
import sys
import time
import numpy as np
import pandas as pd
import torch

from experiments.v1_base_pogs.track import load_pick_place_sequence
from experiments.v1_base_pogs.train import train_pick_place_pogs
from experiments.v1_base_pogs.cluster import cluster_pick_place_objects
from experiments.v2_kalman_pogs.config import get_v2_config
from experiments.v2_kalman_pogs.track import track_v2_sequence


def run_full_ablation_study(
    dataset_dir: str = "datasets/synthetic_scenes/pick_place_floor_table",
    checkpoint_dir: str = "model_checkpoints/v1_base_pogs",
    results_dir: str = "data_results/v2_kalman",
    scenarios: list[str] | None = None,
    methods: list[str] | None = None,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
):
    os.makedirs(results_dir, exist_ok=True)
    scenarios = scenarios or ["respected", "threshold", "violated"]
    methods = methods or [
        "pogs_kalman",
        "pogs_base",
        "ablation_no_depth",
        "ablation_no_dino",
        "ablation_fast_10iters",
        "ablation_ultra_5iters",
    ]

    print("=" * 80)
    print("  RUNNING POGS V2 KALMAN FILTER & ABLATION STUDY SUITE")
    print(f"  Scenarios: {scenarios}")
    print(f"  Methods: {methods}")
    print(f"  Output directory: {results_dir}")
    print("=" * 80)

    # 1. Load shared trained model & singulated target mug
    print("\nLoading Trained POGS Model & Singulated Object...")
    model, capture, dino_fn, clip_fn, _ = train_pick_place_pogs(
        dataset_dir=dataset_dir,
        checkpoint_dir=checkpoint_dir,
        device=device,
    )
    groups = cluster_pick_place_objects(
        model=model,
        capture=capture,
        checkpoint_dir=checkpoint_dir,
        target_object_id=1,
        verbose=False,
    )

    records = []

    for m_idx, method_name in enumerate(methods):
        print(f"\n[{m_idx+1}/{len(methods)}] >>> Evaluating Method/Ablation: {method_name.upper()} <<<")
        pogs_cfg, kalman_cfg = get_v2_config(preset=method_name, device=device)
        method_out_dir = os.path.join(results_dir, method_name)
        os.makedirs(method_out_dir, exist_ok=True)

        for sc in scenarios:
            print(f"\n  --> Scenario: {sc.upper()}")
            seq = load_pick_place_sequence(dataset_dir, sc, device=device)

            res, profs, deltas_traj = track_v2_sequence(
                seq=seq,
                model=model,
                groups=groups,
                dino_fn=dino_fn,
                pogs_cfg=pogs_cfg,
                kalman_cfg=kalman_cfg,
                device=device,
                verbose=False,
            )

            # Save per-scenario trajectory & telemetry
            np.save(os.path.join(method_out_dir, f"part_deltas_traj__{sc}.npy"), deltas_traj)
            with open(os.path.join(method_out_dir, f"telemetry_{sc}.json"), "w") as f:
                json.dump(profs, f, indent=2)

            pos_errs = res.trans_err_cm[:, 0]
            rot_errs = res.rot_err_deg[:, 0]
            mean_pos = float(np.mean(pos_errs))
            std_pos = float(np.std(pos_errs))
            max_pos = float(np.max(pos_errs))
            mean_rot = float(np.mean(rot_errs))
            std_rot = float(np.std(rot_errs))
            add_rate = float(np.mean(pos_errs < 2.0) * 100.0)

            mean_opt_ms = float(np.mean([p["opt_ms"] for p in profs]))
            mean_dino_ms = float(np.mean([p["dino_ms"] for p in profs]))
            mean_fps = float(np.mean([p["fps"] for p in profs]))

            print(
                f"    Results: Pos Err = {mean_pos:5.2f} ± {std_pos:4.2f} cm (Max {max_pos:5.2f} cm) | "
                f"Rot Err = {mean_rot:5.2f}° | ADD(<2cm) = {add_rate:5.1f}% | Opt = {mean_opt_ms:5.1f} ms | FPS = {mean_fps:4.1f}"
            )

            rec = {
                "method": method_name,
                "scenario": sc,
                "mean_pos_err_cm": mean_pos,
                "std_pos_err_cm": std_pos,
                "max_pos_err_cm": max_pos,
                "mean_rot_err_deg": mean_rot,
                "std_rot_err_deg": std_rot,
                "add_metric_pct": add_rate,
                "mean_opt_ms": mean_opt_ms,
                "mean_dino_ms": mean_dino_ms,
                "fps": mean_fps,
                "n_iters": pogs_cfg.tracking.track_niter,
                "use_kalman": kalman_cfg.enabled,
                "use_depth": pogs_cfg.tracking.use_depth,
                "use_dino": pogs_cfg.tracking.use_dino,
            }
            records.append(rec)

    # Save summary DataFrame
    df = pd.DataFrame(records)
    csv_path = os.path.join(results_dir, "ablation_study_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved summary CSV: {csv_path}")

    # Build Master Markdown Table
    md_lines = [
        "# POGS v2: Kalman Filter & Ablation Study Benchmark Table",
        "",
        "| Method / Ablation | Scenario | Mean Pos Error (cm) | Std Pos (cm) | Max Pos (cm) | Mean Rot Error (deg) | ADD (<2cm) (%) | Opt Time (ms) | FPS | Notes |",
        "|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|",
    ]

    for _, row in df.iterrows():
        notes = []
        if row["use_kalman"]:
            notes.append("Kalman EKF")
        if not row["use_depth"]:
            notes.append("No-Depth")
        if not row["use_dino"]:
            notes.append("No-DINO")
        if row["n_iters"] < 25:
            notes.append(f"{int(row['n_iters'])} iters")
        note_str = ", ".join(notes) if notes else "Full Baseline"

        md_lines.append(
            f"| **{row['method']}** | `{row['scenario']}` | "
            f"**{row['mean_pos_err_cm']:.2f}** | {row['std_pos_err_cm']:.2f} | {row['max_pos_err_cm']:.2f} | "
            f"**{row['mean_rot_err_deg']:.2f}°** | **{row['add_metric_pct']:.1f}%** | "
            f"{row['mean_opt_ms']:.1f} | {row['fps']:.1f} | {note_str} |"
        )

    md_content = "\n".join(md_lines) + "\n"
    md_path = os.path.join(results_dir, "ablation_study_summary.md")
    with open(md_path, "w") as f:
        f.write(md_content)

    print("\n" + "=" * 80)
    print("MASTER BENCHMARK COMPARISON TABLE:")
    print("=" * 80)
    print(md_content)
    return df


if __name__ == "__main__":
    run_full_ablation_study()
