"""
Master Pipeline Runner for Scene 2 (Yellow Mustard Bottle Pick-and-Place).

Full Pipeline:
1. Generate Scene 2 dataset (floor, small table, static mug, moving mustard bottle).
2. Train 3DGS representation with Pearson depth & DINOv2 feature distillation.
3. Singulate mustard bottle Gaussians into persistent object representation.
4. Evaluate tracking on Base POGS and Kalman-filtered POGS across 3 scenarios (Respected, Threshold, Violated).
5. Render 4-panel dashboard videos with Quad 2 showing isolated 3D splats frame-by-frame.
6. Generate benchmark metrics comparison table and report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import pandas as pd
import torch

from experiments.v3_scene_mustard.generate_dataset import generate_mustard_dataset
from experiments.v3_scene_mustard.train import train_mustard_pogs, load_mustard_capture
from experiments.v3_scene_mustard.cluster import cluster_mustard_object
from experiments.v3_scene_mustard.config import get_mustard_config
from experiments.v3_scene_mustard.track import load_mustard_sequence, track_mustard_sequence
from experiments.v3_scene_mustard.generate_visuals import generate_mustard_dashboard_video


def run_scene2_pipeline(
    dataset_dir: str = "datasets/synthetic_scenes/scene2_mustard_pick_place",
    checkpoint_dir: str = "model_checkpoints/v3_mustard_pogs",
    results_dir: str = "data_results/v3_mustard_pogs",
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    force_train: bool = False,
):
    print("=" * 80)
    print("  SCENE 2 BENCHMARK PIPELINE: YELLOW MUSTARD BOTTLE PICK-AND-PLACE")
    print(f"  Device: {device} | Force Retrain: {force_train}")
    print(f"  Dataset: {dataset_dir}")
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"  Results: {results_dir}")
    print("=" * 80)

    t_start = time.time()
    os.makedirs(results_dir, exist_ok=True)

    # ----------------------------------------------------------------------- #
    # STAGE 1: DATASET GENERATION
    # ----------------------------------------------------------------------- #
    if not os.path.exists(os.path.join(dataset_dir, "capture", "images.npy")):
        generate_mustard_dataset(output_dir=dataset_dir, device=device)
    else:
        print(f"Scene 2 dataset already verified at {dataset_dir}.")

    # ----------------------------------------------------------------------- #
    # STAGE 2: 3DGS REPRESENTATION TRAINING
    # ----------------------------------------------------------------------- #
    print("\n" + "#" * 80)
    print("STAGE 2: TRAINING 3DGS MODEL WITH DINOv2 & DEPTH DISTILLATION")
    print("#" * 80)
    model, capture, dino_fn, clip_fn, _ = train_mustard_pogs(
        dataset_dir=dataset_dir,
        checkpoint_dir=checkpoint_dir,
        n_steps=400,
        device=device,
        force_retrain=force_train,
    )

    # ----------------------------------------------------------------------- #
    # STAGE 3: OBJECT SINGULATION
    # ----------------------------------------------------------------------- #
    print("\n" + "#" * 80)
    print("STAGE 3: SINGULATING MUSTARD BOTTLE FROM SCENE")
    print("#" * 80)
    groups = cluster_mustard_object(
        model=model,
        capture=capture,
        checkpoint_dir=checkpoint_dir,
        target_object_id=2,
    )

    # ----------------------------------------------------------------------- #
    # STAGE 4: EVALUATION ACROSS BENCHMARK PRESETS AND SCENARIOS
    # ----------------------------------------------------------------------- #
    print("\n" + "#" * 80)
    print("STAGE 4: TRACKING EVALUATION (BASE POGS & KALMAN POGS)")
    print("#" * 80)

    scenarios = ["respected", "threshold", "violated"]
    methods = ["base", "kalman"]

    summary_rows = []

    for method in methods:
        pogs_cfg, kalman_cfg = get_mustard_config(method, device=device)
        method_out_dir = os.path.join(results_dir, method)
        os.makedirs(method_out_dir, exist_ok=True)

        for sc_name in scenarios:
            print(f"\n---> Running [{method.upper()}] on Scenario: {sc_name.upper()}...")
            seq = load_mustard_sequence(dataset_dir, sc_name, device=device)
            res, telemetry, deltas_traj = track_mustard_sequence(
                seq=seq,
                model=model,
                groups=groups,
                dino_fn=dino_fn,
                pogs_cfg=pogs_cfg,
                kalman_cfg=kalman_cfg,
                device=device,
                verbose=True,
            )

            # Save tracking trajectory and telemetry
            np.save(os.path.join(method_out_dir, f"part_deltas_traj__{sc_name}.npy"), deltas_traj)
            with open(os.path.join(method_out_dir, f"telemetry_{sc_name}.json"), "w") as f:
                json.dump(telemetry, f, indent=2)

            t_errs = [t["trans_err_cm"] for t in telemetry]
            r_errs = [t["rot_err_deg"] for t in telemetry]

            summary_rows.append({
                "Method": "POGS + Kalman" if method == "kalman" else "Base POGS",
                "Scenario": sc_name.capitalize(),
                "Mean Pos Error (cm)": round(float(np.mean(t_errs)), 2),
                "Median Pos Error (cm)": round(float(np.median(t_errs)), 2),
                "Max Pos Error (cm)": round(float(np.max(t_errs)), 2),
                "Mean Rot Error (deg)": round(float(np.mean(r_errs)), 2),
                "FPS": round(res.fps() if callable(res.fps) else getattr(res, "fps_val", 5.0), 1),
                "Total Time (s)": round(res.total_time_sec, 2),
            })

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 80)
    print("SCENE 2 TRACKING PERFORMANCE SUMMARY (MUSTARD BOTTLE):")
    print(summary_df.to_string(index=False))
    print("=" * 80)

    summary_df.to_csv(os.path.join(results_dir, "mustard_tracking_summary.csv"), index=False)
    summary_df.to_markdown(os.path.join(results_dir, "mustard_tracking_summary.md"), index=False)

    # ----------------------------------------------------------------------- #
    # STAGE 5: RENDER 4-PANEL DASHBOARD VIDEOS
    # ----------------------------------------------------------------------- #
    print("\n" + "#" * 80)
    print("STAGE 5: GENERATING 4-PANEL DASHBOARD VIDEOS WITH ISOLATED SPLATS")
    print("#" * 80)

    for sc_name in scenarios:
        # Render dashboard for Kalman results
        generate_mustard_dashboard_video(
            dataset_dir=dataset_dir,
            checkpoint_dir=checkpoint_dir,
            results_dir=results_dir,
            scenario_name=sc_name,
            method="kalman",
            device=device,
            out_name=f"dashboard_mustard_kalman_{sc_name}",
        )
        # Render dashboard for Base results
        generate_mustard_dashboard_video(
            dataset_dir=dataset_dir,
            checkpoint_dir=checkpoint_dir,
            results_dir=results_dir,
            scenario_name=sc_name,
            method="base",
            device=device,
            out_name=f"dashboard_mustard_base_{sc_name}",
        )

    t_total = time.time() - t_start
    print(f"\n Scene 2 Full Pipeline Execution Completed in {t_total:.1f}s.")
    return summary_df


if __name__ == "__main__":
    run_scene2_pipeline()
