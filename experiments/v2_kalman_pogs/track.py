"""
Kalman-Enhanced POGS Tracking Engine (v2).

Fuses:
1. Extended Constant-Velocity Kalman Filter (Temporal Kinematic Prior)
2. Differential 3D Gaussian Splatting Optimizer (DINOv2 + Pearson Depth)
3. Direct Lie Algebra / Euler Yaw State Transition
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from common.config import POGSConfig
from common.geometry import (
    PinholeCamera,
    identity_posevec,
    posevec_to_matrix,
    matrix_to_posevec,
    quat_normalize,
    quat_to_rotmat,
    rotmat_to_quat,
)
from common.tracking import PosedObservation, RigidGroupOptimizer
from common.evaluate import (
    TrackingResult,
    add_metric,
    rotation_error_deg,
    translation_error_cm,
    results_table,
)
from scene_generator.tracking_generator import TrackingFrame, TrackingSequence
from scene_generator.trajectory_planner import AngleConstraintStatus, FrameEvent, TrajectoryScenario
from experiments.v1_base_pogs.track import load_pick_place_sequence, _gt_pose_for_group
from experiments.v1_base_pogs.train import train_pick_place_pogs
from experiments.v1_base_pogs.cluster import cluster_pick_place_objects
from experiments.v2_kalman_pogs.config import get_v2_config, KalmanConfig
from experiments.v2_kalman_pogs.kalman import PoseKalmanFilter6DoF


def track_v2_sequence(
    seq: TrackingSequence,
    model,
    groups,
    dino_fn,
    pogs_cfg: POGSConfig,
    kalman_cfg: KalmanConfig,
    device: str = "cuda:0",
    verbose: bool = True,
) -> Tuple[TrackingResult, List[dict], np.ndarray]:
    """Runs Kalman-enhanced 6-DoF tracking on a pick-and-place sequence."""
    cam = seq.camera.to(device)
    K = groups.num_groups
    F_n = len(seq.frames)

    raster_kwargs = dict(
        max_per_tile=pogs_cfg.runtime.torch_raster_max_per_tile,
        tile_chunk=pogs_cfg.runtime.torch_raster_tile_chunk,
    )
    opt = RigidGroupOptimizer(pogs_cfg.tracking, model, groups, raster_kwargs=raster_kwargs)
    opt.reset_transforms()

    # Initial frame observation
    obs0 = PosedObservation(
        seq.frames[0].rgb.to(device),
        cam,
        dino_fn,
        seq.frames[0].depth.to(device),
        pogs_cfg.tracking.max_roi_resolution,
    )
    opt.is_initialized = True
    opt.set_frame(obs0)
    opt.part_deltas = torch.nn.Parameter(identity_posevec(K, device))
    opt.part_optimizer = torch.optim.Adam([opt.part_deltas], lr=pogs_cfg.tracking.pose_lr)
    opt.prev_part_deltas = opt.part_deltas.data.clone()

    base_centroids = {gi: groups.init_p2w[gi, :3].detach().cpu().numpy() for gi in range(K)}

    # Initialize Kalman Filter
    kf = None
    if kalman_cfg.enabled:
        dt = 1.0 / max(seq.fps, 1.0)
        kf = PoseKalmanFilter6DoF(
            dt=dt,
            process_noise_pos=kalman_cfg.process_noise_pos,
            process_noise_vel=kalman_cfg.process_noise_vel,
            process_noise_yaw=kalman_cfg.process_noise_yaw,
            process_noise_omega=kalman_cfg.process_noise_omega,
            measure_noise_pos=kalman_cfg.measure_noise_pos,
            measure_noise_yaw=kalman_cfg.measure_noise_yaw,
        )
        # Scan start pose: translation=(0, 0, 0), yaw=0.0
        kf.initialize(np.zeros(3, dtype=np.float32), yaw_rad=0.0)

    trans_errs = np.full((F_n, K), np.nan)
    rot_errs = np.full((F_n, K), np.nan)
    preds = np.tile(np.eye(4), (F_n, K, 1, 1))
    gts = np.tile(np.eye(4), (F_n, K, 1, 1))
    part_deltas_traj = np.zeros((F_n, K, 7), dtype=np.float32)
    frame_profiling = []

    for fi, fr in enumerate(seq.frames):
        t_frame_start = time.time()

        # Step 1: Kalman Prediction (Time Update)
        t_kalman_start = time.time()
        pred_pos, pred_yaw = None, None
        if kf is not None and fi > 0:
            pred_pos, pred_yaw = kf.predict()
            if kalman_cfg.warm_start_optimizer:
                with torch.no_grad():
                    # Set part_deltas translation to predicted position
                    opt.part_deltas[:, :3] = torch.from_numpy(pred_pos).float().to(device)
                    # Set quaternion from predicted yaw
                    cy = np.cos(pred_yaw * 0.5)
                    sy = np.sin(pred_yaw * 0.5)
                    # Upright: q = [w, x, y, z] = [cy, 0, 0, sy]
                    q_pred = torch.tensor([cy, 0.0, 0.0, sy], device=device, dtype=torch.float32)
                    opt.part_deltas[:, 3:] = q_pred
                opt.apply_to_model()

        # Step 2: DINO feature extraction
        torch.cuda.synchronize()
        t_dino_start = time.time()
        obs = PosedObservation(
            fr.rgb.to(device),
            cam,
            dino_fn,
            fr.depth.to(device) if pogs_cfg.tracking.use_depth else None,
            pogs_cfg.tracking.max_roi_resolution,
        )
        if pogs_cfg.tracking.use_dino:
            _ = obs.frame.dino_feats
        torch.cuda.synchronize()
        dino_time_ms = (time.time() - t_dino_start) * 1000.0

        opt.set_frame(obs)

        # Step 3: POGS 6-DoF Optimization
        torch.cuda.synchronize()
        t_opt_start = time.time()
        step_info = opt.step(
            niter=pogs_cfg.tracking.track_niter,
            use_depth=pogs_cfg.tracking.use_depth,
            use_rgb=pogs_cfg.tracking.use_rgb,
        )
        torch.cuda.synchronize()
        opt_time_ms = (time.time() - t_opt_start) * 1000.0

        # Step 4: Kalman Measurement Update
        meas_pos = opt.part_deltas[0, :3].detach().cpu().numpy()
        meas_quat = opt.part_deltas[0, 3:].detach().cpu().numpy() # [w, x, y, z]
        meas_yaw = 2.0 * np.arctan2(meas_quat[3], meas_quat[0])

        fused_pos, fused_yaw = meas_pos, meas_yaw
        if kf is not None:
            fused_pos, fused_yaw = kf.update(meas_pos, meas_yaw)
            if kalman_cfg.fuse_measurement:
                with torch.no_grad():
                    opt.part_deltas[0, :3] = torch.from_numpy(fused_pos).float().to(device)
                    cy = np.cos(fused_yaw * 0.5)
                    sy = np.sin(fused_yaw * 0.5)
                    opt.part_deltas[0, 3:] = torch.tensor([cy, 0.0, 0.0, sy], device=device)
                opt.apply_to_model()

        kalman_time_ms = (time.time() - t_kalman_start) * 1000.0 - opt_time_ms - dino_time_ms

        total_frame_ms = (time.time() - t_frame_start) * 1000.0
        fps = 1000.0 / max(total_frame_ms, 1e-3)

        curr_deltas = opt.part_deltas.detach().cpu().numpy()
        part_deltas_traj[fi] = curr_deltas

        # Step 5: Evaluate against Ground Truth
        P = opt.get_all_part2world().detach().cpu().numpy()
        for gi in range(K):
            gt_id = groups.cluster_ids[gi]
            T_gt = _gt_pose_for_group(seq, fi, gt_id, base_centroids[gi])
            preds[fi, gi], gts[fi, gi] = P[gi], T_gt
            trans_errs[fi, gi] = translation_error_cm(P[gi], T_gt)
            rot_errs[fi, gi] = rotation_error_deg(P[gi], T_gt)

        prof = {
            "frame": fi,
            "dino_ms": float(dino_time_ms),
            "opt_ms": float(opt_time_ms),
            "kalman_ms": float(max(kalman_time_ms, 0.0)),
            "total_ms": float(total_frame_ms),
            "fps": float(fps),
            "loss": float(step_info.loss) if not np.isnan(step_info.loss) else 0.0,
            "trans_err_cm": float(trans_errs[fi, 0]),
            "rot_err_deg": float(rot_errs[fi, 0]),
        }
        frame_profiling.append(prof)

        if verbose and (fi % 5 == 0 or fi == F_n - 1):
            k_flag = "[KF] " if kalman_cfg.enabled else ""
            print(
                f"  {k_flag}Frame {fi+1:2d}/{F_n} | "
                f"Trans: {trans_errs[fi, 0]:5.2f}cm | Rot: {rot_errs[fi, 0]:5.2f}° | "
                f"Opt: {opt_time_ms:5.1f}ms | FPS: {fps:4.1f}"
            )

    name = f"POGS_{'Kalman' if kalman_cfg.enabled else 'Base'}_{seq.scenario}"
    result = TrackingResult(
        name=name,
        trans_err_cm=trans_errs,
        rot_err_deg=rot_errs,
        add=np.zeros_like(trans_errs),
        adds=np.zeros_like(trans_errs),
        frame_times=np.array([p["total_ms"] / 1000.0 for p in frame_profiling]),
        iters=np.full(F_n, pogs_cfg.tracking.track_niter),
        losses=np.array([p["loss"] for p in frame_profiling]),
        reset_frames=[],
        hand_frames=[],
        pred_poses=preds,
        gt_poses=gts,
        init_time=0.0,
    )

    opt.reset_transforms()
    return result, frame_profiling, part_deltas_traj
