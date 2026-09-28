"""
Base Original POGS Tracking Runner for Pick-and-Place (v1 - Pure Base POGS).
Strictly implements the ICRA 2025 paper:
- RigidGroupOptimizer with 25 iterations per frame
- Feature loss (DINOv2) + Depth loss (Pearson)
- Decoupled single-view 6-DoF tracking
- Granular latency profiling per frame (DINO extraction, pose optimization, FPS)
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

from common.config import POGSConfig, TrackingConfig
from common.geometry import PinholeCamera, identity_posevec
from common.grouping import ObjectGroups
from common.model import POGSModel
from common.tracking import DinoFn, PosedObservation, RigidGroupOptimizer
from common.evaluate import (
    TrackingResult,
    _gt_pose_for_group,
    rotation_error_deg,
    translation_error_cm,
    results_table,
)
from scene_generator.tracking_generator import TrackingFrame, TrackingSequence
from scene_generator.trajectory_planner import AngleConstraintStatus, FrameEvent, TrajectoryScenario
from experiments.v1_base_pogs.config import get_v1_config


def load_pick_place_sequence(
    dataset_dir: str,
    scenario: str,
    device: str = "cuda:0",
) -> TrackingSequence:
    sc_dir = os.path.join(dataset_dir, f"tracking_{scenario}")
    with open(os.path.join(sc_dir, "tracking_metadata.json")) as f:
        meta = json.load(f)

    cam_info = meta["camera"]
    camera = PinholeCamera(
        c2w=torch.tensor(cam_info["c2w"], dtype=torch.float32, device=device),
        fx=cam_info["fx"],
        fy=cam_info["fy"],
        cx=cam_info["cx"],
        cy=cam_info["cy"],
        width=cam_info["width"],
        height=cam_info["height"],
    )

    frames = []
    for fr_meta in meta["frames"]:
        f = fr_meta["frame"]
        rgb_path = os.path.join(sc_dir, "rgb", f"frame_{f:04d}.png")
        depth_path = os.path.join(sc_dir, "depth_npy", f"frame_{f:04d}.npy")
        mask_path = os.path.join(sc_dir, "masks", f"frame_{f:04d}.png")

        rgb_bgr = cv2.imread(rgb_path)
        rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        rgb_t = torch.from_numpy(rgb_rgb).float() / 255.0

        depth_np = np.load(depth_path)
        depth_t = torch.from_numpy(depth_np).float()

        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_t = torch.from_numpy((mask_img // 40).astype(np.int64))

        gt_deltas = {int(k): np.array(v, dtype=np.float64) for k, v in fr_meta["gt_deltas"].items()}
        status_enum = AngleConstraintStatus(fr_meta.get("constraint_status", "RESPECTED"))

        ev = FrameEvent(
            frame_idx=f,
            timestamp_sec=fr_meta["timestamp_sec"],
            yaw_deg=fr_meta["yaw_deg"],
            pitch_deg=fr_meta["pitch_deg"],
            roll_deg=fr_meta["roll_deg"],
            translation_cm=fr_meta["translation_cm"],
            constraint_status=status_enum,
            hand_occlusion=False,
            reset_event=False,
            event_type="",
            description=fr_meta["description"],
        )

        frames.append(
            TrackingFrame(
                frame_idx=f,
                timestamp_sec=fr_meta["timestamp_sec"],
                rgb=rgb_t,
                depth=depth_t,
                obj_id=mask_t,
                gt_delta=gt_deltas,
                event=ev,
            )
        )

    return TrackingSequence(
        scenario=TrajectoryScenario(scenario),
        frames=frames,
        camera=camera,
        object_ids=[1],
        trajectories={},
    )


def track_base_pogs_sequence(
    model: POGSModel,
    groups: ObjectGroups,
    seq: TrackingSequence,
    dino_fn: DinoFn,
    cfg: POGSConfig,
    name: str = "Base_POGS",
    verbose: bool = True,
) -> Tuple[TrackingResult, List[Dict[str, float]], np.ndarray]:
    """Runs Base Original POGS tracking (Sec. IV-D) with granular latency profiling."""
    device = cfg.runtime.device
    tcfg = cfg.tracking
    raster_kwargs = (
        dict(max_per_tile=cfg.runtime.torch_raster_max_per_tile,
             tile_chunk=cfg.runtime.torch_raster_tile_chunk)
        if model.backend == "torch" else {}
    )

    # Optimizer instance
    opt = RigidGroupOptimizer(tcfg, model, groups, raster_kwargs=raster_kwargs)
    opt.reset_transforms()
    K = groups.num_groups
    F_n = len(seq.frames)
    cam = seq.camera.to(device)

    # 1. Initialize Frame 0
    t0 = time.time()
    obs0 = PosedObservation(
        seq.frames[0].rgb.to(device),
        cam,
        dino_fn,
        seq.frames[0].depth.to(device),
        tcfg.max_roi_resolution,
    )
    if tcfg.init_niter_coarse > 0 or tcfg.init_niter_fine > 0:
        init_info = opt.initialize_obj_pose(obs0, n_seeds=1, verbose=verbose)
        init_time = time.time() - t0
        if verbose:
            print(f"  [{name}] Init pose: loss={init_info.get('loss', 0.0):.4f} in {init_time:.2f}s")
    else:
        for i in range(K):
            try:
                obs0.add_roi(*opt.calculate_roi(i, cam))
            except Exception:
                obs0.add_roi(0.0, 1.0, 0.0, 1.0)
        opt.is_initialized = True
        opt.set_frame(obs0)
        opt.part_deltas = torch.nn.Parameter(identity_posevec(K, device))
        opt.part_optimizer = torch.optim.Adam([opt.part_deltas], lr=tcfg.pose_lr)
        opt.prev_part_deltas = opt.part_deltas.data.clone()
        if verbose:
            print(f"  [{name}] Init pose: canonical scan identity at (0, 0, 0)")

    base_centroids = {gi: groups.init_p2w[gi, :3].detach().cpu().numpy() for gi in range(K)}

    trans_errs = np.full((F_n, K), np.nan)
    rot_errs = np.full((F_n, K), np.nan)
    preds = np.tile(np.eye(4), (F_n, K, 1, 1))
    gts = np.tile(np.eye(4), (F_n, K, 1, 1))
    part_deltas_traj = np.zeros((F_n, K, 7), dtype=np.float32)

    frame_profiling = []

    for fi, fr in enumerate(seq.frames):
        t_frame_start = time.time()

        # Phase 1: DINO feature extraction
        torch.cuda.synchronize()
        t_dino_start = time.time()
        obs = PosedObservation(
            fr.rgb.to(device),
            cam,
            dino_fn,
            fr.depth.to(device),
            tcfg.max_roi_resolution,
        )
        if tcfg.use_roi:
            for i in range(K):
                try:
                    obs.add_roi(*opt.calculate_roi(i, obs.frame.camera))
                except Exception:
                    obs.add_roi(0.0, 1.0, 0.0, 1.0)
        _ = obs.frame.dino_feats # force DINO inference
        torch.cuda.synchronize()
        dino_time_ms = (time.time() - t_dino_start) * 1000.0

        opt.set_frame(obs)

        # Phase 2: 6-DoF Optimization (25 iterations per paper)
        torch.cuda.synchronize()
        t_opt_start = time.time()
        step_info = opt.step(niter=tcfg.track_niter, use_depth=tcfg.use_depth, use_rgb=tcfg.use_rgb)
        torch.cuda.synchronize()
        opt_time_ms = (time.time() - t_opt_start) * 1000.0

        total_frame_ms = (time.time() - t_frame_start) * 1000.0
        fps = 1000.0 / max(total_frame_ms, 1e-3)

        # Save part_deltas
        curr_deltas = opt.part_deltas.detach().cpu().numpy()
        part_deltas_traj[fi] = curr_deltas

        # Phase 3: Evaluate against Ground Truth
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
            "total_ms": float(total_frame_ms),
            "fps": float(fps),
            "loss": float(step_info.loss),
            "trans_err_cm": float(trans_errs[fi, 0]),
            "rot_err_deg": float(rot_errs[fi, 0]),
        }
        frame_profiling.append(prof)

        if verbose and (fi % 5 == 0 or fi == F_n - 1):
            print(f"  Frame {fi + 1:2d}/{F_n} | Trans Err: {trans_errs[fi, 0]:5.2f}cm | Rot Err: {rot_errs[fi, 0]:5.2f}° | Opt: {opt_time_ms:4.1f}ms | FPS: {fps:4.1f}")

    result = TrackingResult(
        name=name,
        trans_err_cm=trans_errs,
        rot_err_deg=rot_errs,
        add=np.zeros_like(trans_errs),
        adds=np.zeros_like(trans_errs),
        frame_times=np.array([p["total_ms"] / 1000.0 for p in frame_profiling]),
        iters=np.full(F_n, tcfg.track_niter),
        losses=np.array([p["loss"] for p in frame_profiling]),
        reset_frames=[],
        hand_frames=[],
        pred_poses=preds,
        gt_poses=gts,
        init_time=init_time,
    )

    opt.reset_transforms()
    return result, frame_profiling, part_deltas_traj


def run_base_pogs_evaluation(
    dataset_dir: str = "datasets/synthetic_scenes/pick_place_floor_table",
    checkpoint_dir: str = "model_checkpoints/v1_base_pogs",
    output_dir: str = "data_results/v1_base_pogs",
    scenarios: Optional[List[str]] = None,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
):
    os.makedirs(output_dir, exist_ok=True)
    scenarios = scenarios or ["respected", "threshold", "violated"]
    cfg = get_v1_config(preset="paper", device=device)

    from experiments.v1_base_pogs.train import train_pick_place_pogs
    from experiments.v1_base_pogs.cluster import cluster_pick_place_objects

    # 1. Load trained model & capture
    print("Loading Base POGS Model...")
    model, capture, dino_fn, clip_fn, _ = train_pick_place_pogs(
        dataset_dir=dataset_dir,
        checkpoint_dir=checkpoint_dir,
        device=device,
    )

    # 2. Extract clean target mug representation
    groups = cluster_pick_place_objects(
        model=model,
        capture=capture,
        checkpoint_dir=checkpoint_dir,
        target_object_id=1,
    )

    print("\n" + "=" * 76)
    print("RUNNING BASE ORIGINAL POGS TRACKING EVALUATION")
    print(f"Output directory: {output_dir}")
    print("=" * 76)

    all_results = {}
    summary_rows = []

    for sc in scenarios:
        print(f"\n---> Evaluating Scenario: {sc.upper()}")
        seq = load_pick_place_sequence(dataset_dir, sc, device=device)

        res, prof, deltas_traj = track_base_pogs_sequence(
            model=model,
            groups=groups,
            seq=seq,
            dino_fn=dino_fn,
            cfg=cfg,
            name=f"Base_POGS_{sc}",
            verbose=True,
        )
        all_results[sc] = res

        # Save trajectory deltas
        np.save(os.path.join(output_dir, f"part_deltas_traj__{sc}.npy"), deltas_traj)

        # Save granular profiling
        with open(os.path.join(output_dir, f"telemetry_{sc}.json"), "w") as f:
            json.dump(prof, f, indent=2)

        summary_rows.append({
            "scenario": sc,
            "mean_pos_err_cm": res.mean_position_error_cm(),
            "std_pos_err_cm": res.std_position_error_cm(),
            "mean_rot_err_deg": res.mean_rotation_error_deg(),
            "fps": res.fps(),
            "mean_dino_ms": float(np.mean([p["dino_ms"] for p in prof])),
            "mean_opt_ms": float(np.mean([p["opt_ms"] for p in prof])),
            "mean_frame_ms": float(np.mean([p["total_ms"] for p in prof])),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, "base_pogs_benchmark.csv"), index=False)
    print("\n" + "=" * 76)
    print("BASE POGS BENCHMARK SUMMARY:")
    print("=" * 76)
    print(summary_df.to_string(index=False))

    return all_results, summary_df


if __name__ == "__main__":
    run_base_pogs_evaluation()
