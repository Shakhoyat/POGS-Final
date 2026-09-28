"""
Master Orchestrator for Base Original POGS Benchmark (v1 - Pure Base POGS).

Stages:
1. Generate photorealistic pick-and-place dataset (floor, small table, 5 real objects, pure Z yaw rotation).
2. Train 3D Gaussian Splatting representation with DINOv2 & Pearson depth distillation.
3. Cluster discrete persistent object representations (singulate ceramic mug from background).
4. Render 3D Gaussian World Splats & Isolated Target Object Splats.
5. Base Original POGS Tracking Evaluation (Sec. IV-D, 25 iters, feature + depth loss).
6. Render multi-panel dashboard videos with frame-by-frame isolated object splats.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import pandas as pd
import torch

from experiments.v1_base_pogs.generate_dataset import generate_pick_place_dataset
from experiments.v1_base_pogs.train import train_pick_place_pogs
from experiments.v1_base_pogs.cluster import cluster_pick_place_objects
from experiments.v1_base_pogs.visualize_splats import generate_splat_visualizations
from experiments.v1_base_pogs.track import run_base_pogs_evaluation
from experiments.v1_base_pogs.generate_visuals import generate_dashboard_video


def run_pipeline(
    dataset_dir: str = "datasets/synthetic_scenes/pick_place_floor_table",
    checkpoint_dir: str = "model_checkpoints/v1_base_pogs",
    results_dir: str = "data_results/v1_base_pogs",
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    force_train: bool = True,
):
    print("=" * 76)
    print("  BASE ORIGINAL POGS BENCHMARK PIPELINE")
    print(f"  Device: {device} | Force Retrain: {force_train}")
    print(f"  Dataset: {dataset_dir}")
    print(f"  Checkpoints: {checkpoint_dir}")
    print(f"  Results: {results_dir}")
    print("=" * 76)

    t_start = time.time()

    # Stage 1: Generate Dataset
    print("\n" + "#" * 76)
    print("STAGE 1: VERIFYING CLEAN PICK-AND-PLACE DATASET")
    print("#" * 76)
    if not os.path.exists(os.path.join(dataset_dir, "capture", "images.npy")):
        generate_pick_place_dataset(output_dir=dataset_dir, device=device)
    else:
        print(f"Dataset already exists at {dataset_dir}.")

    # Stage 2: Train 3DGS Model
    print("\n" + "#" * 76)
    print("STAGE 2: TRAINING BASE POGS 3DGS REPRESENTATION (DINO + DEPTH)")
    print("#" * 76)
    model, capture, dino_fn, clip_fn, _ = train_pick_place_pogs(
        dataset_dir=dataset_dir,
        checkpoint_dir=checkpoint_dir,
        n_steps=400,
        device=device,
        force_retrain=force_train,
    )

    # Stage 3: Singulate Target Object (Ceramic Coffee Mug)
    print("\n" + "#" * 76)
    print("STAGE 3: ISOLATING TARGET OBJECT FROM FLOOR/TABLE")
    print("#" * 76)
    groups = cluster_pick_place_objects(
        model=model,
        capture=capture,
        checkpoint_dir=checkpoint_dir,
        target_object_id=1,
    )

    # Stage 4: Render World Splats & Isolated Mug Splats
    print("\n" + "#" * 76)
    print("STAGE 4: RENDERING 3D GAUSSIAN WORLD SPLATS & ISOLATED OBJECT SPLATS")
    print("#" * 76)
    generate_splat_visualizations(
        output_dir=os.path.join(results_dir, "splats"),
        checkpoint_dir=checkpoint_dir,
        dataset_dir=dataset_dir,
        device=device,
    )

    # Stage 5: Base Original POGS Tracking Evaluation
    print("\n" + "#" * 76)
    print("STAGE 5: BASE ORIGINAL POGS 6-DoF TRACKING (25 ITERS, DINO + DEPTH LOSS)")
    print("#" * 76)
    results, summary_df = run_base_pogs_evaluation(
        dataset_dir=dataset_dir,
        checkpoint_dir=checkpoint_dir,
        output_dir=results_dir,
        device=device,
    )

    # Stage 6: Multi-Panel Dashboard Videos
    print("\n" + "#" * 76)
    print("STAGE 6: GENERATING MULTI-PANEL DASHBOARD VIDEOS")
    print("#" * 76)
    videos_dir = os.path.join(results_dir, "videos")
    for sc in ["respected", "threshold", "violated"]:
        generate_dashboard_video(
            scenario=sc,
            output_dir=videos_dir,
            dataset_dir=dataset_dir,
            results_dir=results_dir,
            checkpoint_dir=checkpoint_dir,
            device=device,
        )

    print("\n" + "=" * 76)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {(time.time() - t_start) / 60:.1f} MINUTES!")
    print("=" * 76)


if __name__ == "__main__":
    run_pipeline(force_train=False)
