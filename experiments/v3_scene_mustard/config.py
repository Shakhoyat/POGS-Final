"""
Configuration for Scene 2 (Yellow Mustard Bottle Pick-and-Place).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from common.config import POGSConfig
from experiments.v2_kalman_pogs.config import KalmanConfig


def get_mustard_config(
    preset: Literal["base", "kalman"] = "kalman",
    device: str = "cuda:0",
) -> tuple[POGSConfig, KalmanConfig]:
    """Returns (POGSConfig, KalmanConfig) tuned for Scene 2 Mustard Bottle tracking."""
    cfg = POGSConfig(device=device)

    cfg.runtime.torch_raster_max_per_tile = 512
    cfg.tracking.use_roi = False
    cfg.tracking.upright_constraint = True
    cfg.tracking.clip_grad = 0.5
    cfg.tracking.pose_lr = 0.012
    cfg.tracking.pose_lr_final = 0.002
    cfg.tracking.depth_loss_mult = 150.0
    cfg.tracking.depth_ignore_threshold = 0.08
    cfg.tracking.track_niter = 25
    cfg.tracking.use_depth = True
    cfg.tracking.use_dino = True
    cfg.runtime.output_dir = f"data_results/v3_mustard_pogs/{preset}"

    k_cfg = KalmanConfig(
        enabled=(preset == "kalman"),
        warm_start_optimizer=True,
        fuse_measurement=True,
        process_noise_pos=1e-4,
        process_noise_vel=1e-3,
        process_noise_yaw=1e-3,
        process_noise_omega=1e-2,
        measure_noise_pos=5e-4,
        measure_noise_yaw=5e-3,
    )

    return cfg, k_cfg
