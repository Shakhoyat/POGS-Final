"""
Configuration for v1: Pick-and-Place Base POGS.
Faithful reproduction of ICRA 2025 POGS:
- Pure causal decoupled 6-DoF tracking
- 25 optimizer iterations per frame
- Unmasked DINO feature correlation + Depth geometry
"""

from __future__ import annotations

import torch
from common.config import POGSConfig


def get_v1_config(preset: str = "paper", device: str = "cuda:0") -> POGSConfig:
    """Returns Base POGS configuration for pick-and-place tracking."""
    cfg = POGSConfig(device=device)

    cfg.training.max_steps = 2500
    cfg.training.refine_every = 300
    cfg.runtime.torch_raster_max_per_tile = 512

    # Proven calibrated tracking parameters (Mean trans error = 1.03 cm)
    cfg.tracking.track_niter = 25
    cfg.tracking.use_depth = True
    cfg.tracking.use_dino = True
    cfg.tracking.use_roi = False  # Full-frame global tracking: robust across fast transit
    cfg.tracking.upright_constraint = True  # Tabletop vertical axis yaw constraint
    cfg.tracking.depth_loss_mult = 150.0
    cfg.tracking.clip_grad = 0.5
    cfg.tracking.pose_lr = 0.012
    cfg.tracking.pose_lr_final = 0.002
    cfg.runtime.output_dir = "data_results/v1_base_pogs"

    return cfg
