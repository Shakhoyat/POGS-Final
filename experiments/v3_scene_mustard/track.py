"""
Tracking Engine for Scene 2 (Yellow Mustard Bottle Pick-and-Place).
Supports Base POGS and Kalman-filtered POGS tracking across Respected, Threshold, and Violated scenarios.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
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
from experiments.v2_kalman_pogs.kalman import PoseKalmanFilter6DoF
from experiments.v2_kalman_pogs.config import KalmanConfig
from experiments.v3_scene_mustard.config import get_mustard_config


def load_mustard_sequence(
    dataset_dir: str,
    scenario_name: str,
    device: str = "cuda:0",
) -> TrackingSequence:
    sc_dir = os.path.join(dataset_dir, f"tracking_{scenario_name}")
    meta_path = os.path.join(sc_dir, "tracking_metadata.json")

    with open(meta_path) as f:
        meta = json.load(f)

    cam_info = meta["camera"]
    camera = PinholeCamera(
        c2w=torch.tensor(cam_info["c2w"], device=device, dtype=torch.float32),
        fx=cam_info["fx"], fy=cam_info["fy"], cx=cam_info["cx"], cy=cam_info["cy"],
        width=cam_info["width"], height=cam_info["height"],
    )

    rgb_dir = os.path.join(sc_dir, "rgb")
    depth_dir = os.path.join(sc_dir, "depth_npy")
    frames = []

    for item in meta["frames"]:
        f_idx = item["frame"]
        f_name = f"frame_{f_idx:04d}"

        bgr = cv2.imread(os.path.join(rgb_dir, f"{f_name}.png"))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_t = torch.from_numpy(rgb).float() / 255.0

        d_np = np.load(os.path.join(depth_dir, f"{f_name}.npy"))
        d_t = torch.from_numpy(d_np).float()

        delta_mat = np.array(item["gt_deltas"]["2"], dtype=np.float32)

        status_enum = AngleConstraintStatus(item.get("constraint_status", "RESPECTED"))
        ev = FrameEvent(
            frame_idx=f_idx,
            timestamp_sec=item["timestamp_sec"],
            yaw_deg=item["yaw_deg"],
            pitch_deg=item["pitch_deg"],
            roll_deg=item["roll_deg"],
            translation_cm=item["translation_cm"],
            constraint_status=status_enum,
            hand_occlusion=False,
            reset_event=False,
            event_type="",
            description=item["description"],
        )

        frames.append(
            TrackingFrame(
                frame_idx=f_idx,
                timestamp_sec=item["timestamp_sec"],
                rgb=rgb_t,
                depth=d_t,
                obj_id=torch.zeros((camera.height, camera.width), dtype=torch.long),
                gt_delta={2: delta_mat},
                event=ev,
            )
        )

    return TrackingSequence(
        scenario=TrajectoryScenario(scenario_name),
        frames=frames,
        camera=camera,
        object_ids=[2],
        trajectories={},
        fps=meta.get("fps", 5.0),
    )


def track_mustard_sequence(
    seq: TrackingSequence,
    model,
    groups,
    dino_fn,
    pogs_cfg: POGSConfig,
    kalman_cfg: KalmanConfig,
    device: str = "cuda:0",
    verbose: bool = True,
) -> Tuple[TrackingResult, List[dict], np.ndarray]:
    cam = seq.camera.to(device)
    K = groups.num_groups
    F_n = len(seq.frames)

    raster_kwargs = dict(
        max_per_tile=pogs_cfg.runtime.torch_raster_max_per_tile,
        tile_chunk=pogs_cfg.runtime.torch_raster_tile_chunk,
    )
    opt = RigidGroupOptimizer(pogs_cfg.tracking, model, groups, raster_kwargs=raster_kwargs)
    opt.reset_transforms()

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
        kf.initialize(np.zeros(3, dtype=np.float32), yaw_rad=0.0)

    part_deltas_traj = np.zeros((F_n, K, 7), dtype=np.float32)
    telemetry = []
    trans_errs = np.zeros((F_n, K), dtype=np.float32)
    rot_errs = np.zeros((F_n, K), dtype=np.float32)
    preds = np.zeros((F_n, K, 4, 4), dtype=np.float32)
    gts = np.zeros((F_n, K, 4, 4), dtype=np.float32)

    t0_track = time.time()

    for f_idx, frame in enumerate(seq.frames):
        t_frame_start = time.time()
        obs = PosedObservation(
            frame.rgb.to(device),
            cam,
            dino_fn,
            frame.depth.to(device),
            pogs_cfg.tracking.max_roi_resolution,
        )

        kf_pred_pos, kf_pred_yaw = None, None
        if kalman_cfg.enabled and kf is not None and f_idx > 0:
            kf_pred_pos, kf_pred_yaw = kf.predict()
            if kalman_cfg.warm_start_optimizer:
                with torch.no_grad():
                    opt.part_deltas.data[0, :3] = torch.from_numpy(kf_pred_pos).to(device)
                    half_yaw = kf_pred_yaw * 0.5
                    qw = float(np.cos(half_yaw))
                    qz = float(np.sin(half_yaw))
                    opt.part_deltas.data[0, 3:] = torch.tensor([qw, 0.0, 0.0, qz], device=device)

        opt.set_frame(obs)
        for _ in range(pogs_cfg.tracking.track_niter):
            opt.step()

        if kalman_cfg.enabled and kf is not None and kalman_cfg.fuse_measurement and f_idx > 0:
            meas_trans = opt.part_deltas.data[0, :3].cpu().numpy()
            meas_q = opt.part_deltas.data[0, 3:].cpu().numpy()
            meas_yaw = float(2.0 * np.arctan2(meas_q[3], meas_q[0]))

            fused_trans, fused_yaw = kf.update(meas_trans, meas_yaw)
            with torch.no_grad():
                opt.part_deltas.data[0, :3] = torch.from_numpy(fused_trans).to(device)
                half_yaw = fused_yaw * 0.5
                qw = float(np.cos(half_yaw))
                qz = float(np.sin(half_yaw))
                opt.part_deltas.data[0, 3:] = torch.tensor([qw, 0.0, 0.0, qz], device=device)

        with torch.no_grad():
            curr_deltas = opt.part_deltas.data.clone()
            part_deltas_traj[f_idx] = curr_deltas.cpu().numpy()

        # Telemetry and Ground Truth Error Calculation
        gt_delta_mat = frame.gt_delta[2]
        pred_delta_vec = curr_deltas[0]
        pred_delta_mat = posevec_to_matrix(pred_delta_vec.unsqueeze(0))[0].cpu().numpy()

        c_scan = base_centroids[0]
        R_gt = gt_delta_mat[:3, :3]
        t_gt = gt_delta_mat[:3, 3]
        gt_center = R_gt @ c_scan + t_gt

        R_pred = pred_delta_mat[:3, :3]
        t_pred = pred_delta_mat[:3, 3]
        pred_center = R_pred @ c_scan + t_pred

        trans_err = float(np.linalg.norm(pred_center - gt_center) * 100.0)  # cm
        rot_err = float(rotation_error_deg(R_pred, R_gt))                   # deg

        q_pred = pred_delta_vec[3:].cpu().numpy()
        pred_yaw_deg = float(np.degrees(2.0 * np.arctan2(q_pred[3], q_pred[0])))
        gt_yaw_deg = float(np.degrees(np.arctan2(R_gt[1, 0], R_gt[0, 0])))

        trans_errs[f_idx, 0] = trans_err
        rot_errs[f_idx, 0] = rot_err
        preds[f_idx, 0] = pred_delta_mat
        gts[f_idx, 0] = gt_delta_mat

        frame_duration = time.time() - t_frame_start
        telemetry.append({
            "frame": f_idx,
            "timestamp_sec": frame.timestamp_sec,
            "trans_err_cm": trans_err,
            "rot_err_deg": rot_err,
            "gt_center": gt_center.tolist(),
            "pred_center": pred_center.tolist(),
            "gt_yaw_deg": gt_yaw_deg,
            "pred_yaw_deg": pred_yaw_deg,
            "constraint_status": frame.event.constraint_status.value,
            "frame_time_sec": frame_duration,
        })

        method_tag = "pogs_kalman" if kalman_cfg.enabled else "pogs_base"
        if verbose and (f_idx % 5 == 0 or f_idx == F_n - 1):
            print(f"  [{method_tag}] Frame {f_idx:2d}/{F_n} | Pos Err: {trans_err:4.2f} cm | Rot Err: {rot_err:4.2f}° | {frame_duration * 1000:.0f}ms")

    res = TrackingResult(
        name=f"scene2_mustard_{seq.scenario.value}",
        trans_err_cm=trans_errs,
        rot_err_deg=rot_errs,
        add=np.zeros_like(trans_errs),
        adds=np.zeros_like(trans_errs),
        frame_times=np.array([p["frame_time_sec"] for p in telemetry]),
        iters=np.full(F_n, pogs_cfg.tracking.track_niter),
        losses=np.zeros(F_n),
        reset_frames=[],
        hand_frames=[],
        pred_poses=preds,
        gt_poses=gts,
        init_time=0.0,
    )
    res.total_time_sec = time.time() - t0_track
    res.fps_val = F_n / max(res.total_time_sec, 1e-4)
    return res, telemetry, part_deltas_traj
