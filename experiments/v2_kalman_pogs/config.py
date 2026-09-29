"""
Configuration for v2: Kalman-Enhanced POGS & Comprehensive Ablation Study.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import torch

from common.config import POGSConfig


@dataclass
class KalmanConfig:
    enabled: bool = True
    warm_start_optimizer: bool = True
    fuse_measurement: bool = True
    process_noise_pos: float = 1e-4
    process_noise_vel: float = 1e-3
    process_noise_yaw: float = 1e-3
    process_noise_omega: float = 1e-2
    measure_noise_pos: float = 5e-4
    measure_noise_yaw: float = 5e-3


def get_v2_config(
    preset: Literal[
        "pogs_kalman",
        "pogs_base",
        "ablation_no_depth",
        "ablation_no_dino",
        "ablation_fast_10iters",
        "ablation_ultra_5iters",
    ] = "pogs_kalman",
    device: str = "cuda:0",
) -> tuple[POGSConfig, KalmanConfig]:
    """Returns (POGSConfig, KalmanConfig) tuned for the requested benchmark/ablation preset."""
    cfg = POGSConfig(device=device)

    # Base calibrated tracking settings
    cfg.runtime.torch_raster_max_per_tile = 512
    cfg.tracking.use_roi = False
    cfg.tracking.upright_constraint = True
    cfg.tracking.clip_grad = 0.5
    cfg.tracking.pose_lr = 0.012
    cfg.tracking.pose_lr_final = 0.002
    cfg.tracking.depth_loss_mult = 150.0
    cfg.tracking.depth_ignore_threshold = 0.08
    cfg.runtime.output_dir = f"data_results/v2_kalman/{preset}"

    k_cfg = KalmanConfig(enabled=True)

    if preset == "pogs_kalman":
        cfg.tracking.track_niter = 25
        cfg.tracking.use_depth = True
        cfg.tracking.use_dino = True
        k_cfg.enabled = True
    elif preset == "pogs_base":
        cfg.tracking.track_niter = 25
        cfg.tracking.use_depth = True
        cfg.tracking.use_dino = True
        k_cfg.enabled = False
    elif preset == "ablation_no_depth":
        cfg.tracking.track_niter = 25
        cfg.tracking.use_depth = False
        cfg.tracking.use_dino = True
        k_cfg.enabled = True
    elif preset == "ablation_no_dino":
        cfg.tracking.track_niter = 25
        cfg.tracking.use_depth = True
        cfg.tracking.use_dino = False
        k_cfg.enabled = True
    elif preset == "ablation_fast_10iters":
        cfg.tracking.track_niter = 10
        cfg.tracking.use_depth = True
        cfg.tracking.use_dino = True
        cfg.tracking.pose_lr = 0.015
        cfg.tracking.pose_lr_final = 0.004
        k_cfg.enabled = True
    elif preset == "ablation_ultra_5iters":
        cfg.tracking.track_niter = 5
        cfg.tracking.use_depth = True
        cfg.tracking.use_dino = True
        cfg.tracking.pose_lr = 0.020
        cfg.tracking.pose_lr_final = 0.006
        k_cfg.enabled = True
    else:
        raise ValueError(f"Unknown preset: {preset}")

    return cfg, k_cfg
