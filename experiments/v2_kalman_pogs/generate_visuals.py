"""
Multi-Panel Comparison Dashboard Video Generator (v2 - Kalman POGS).

Compares:
- Base POGS (v1 Baseline) vs Kalman POGS (v2 Proposed)
- Displays predicted isolated 3D splats in Quad 2
- Real-time comparison telemetry curves (Pos Error cm & Rot Error deg)
"""

from __future__ import annotations

import json
import os
import cv2
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common.geometry import PinholeCamera, quat_to_rotmat
from experiments.v1_base_pogs.generate_visuals import get_3d_box_corners, project_points, draw_3d_box
from experiments.v1_base_pogs.train import train_pick_place_pogs
from experiments.v1_base_pogs.cluster import cluster_pick_place_objects
from experiments.v1_base_pogs.visualize_splats import render_isolated_target_splats


def render_comparison_telemetry_plot(
    frames_hist: list[int],
    base_pos_errs: list[float],
    kalman_pos_errs: list[float],
    base_rot_errs: list[float],
    kalman_rot_errs: list[float],
    curr_frame: int,
    total_frames: int,
    width: int = 640,
    height: int = 360,
) -> np.ndarray:
    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    fig.patch.set_facecolor("#0b0f19")

    # Top: Position Error (cm) comparison
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.set_facecolor("#111827")
    if frames_hist:
        ax1.plot(frames_hist, base_pos_errs, color="#f59e0b", lw=2.0, label="v1 Base POGS", ls="--")
        ax1.plot(frames_hist, kalman_pos_errs, color="#00e5ff", lw=2.4, label="v2 Kalman POGS")
        ax1.scatter([curr_frame], [kalman_pos_errs[-1]], color="#ffffff", s=40, zorder=5)
    ax1.set_xlim(0, max(total_frames - 1, 1))
    ax1.set_ylim(0, max(max(base_pos_errs + kalman_pos_errs + [3.0]) * 1.25, 4.0))
    ax1.set_ylabel("Pos Error (cm)", color="#94a3b8", fontsize=9, fontweight="bold")
    ax1.tick_params(colors="#64748b", labelsize=8)
    ax1.grid(True, color="#1e293b", linestyle="--", alpha=0.7)
    ax1.legend(loc="upper left", facecolor="#1e293b", edgecolor="none", labelcolor="#e2e8f0", fontsize=8)

    # Bottom: Rotation Error (deg) comparison
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.set_facecolor("#111827")
    if frames_hist:
        ax2.plot(frames_hist, base_rot_errs, color="#f59e0b", lw=2.0, label="v1 Base POGS", ls="--")
        ax2.plot(frames_hist, kalman_rot_errs, color="#a855f7", lw=2.4, label="v2 Kalman POGS")
        ax2.scatter([curr_frame], [kalman_rot_errs[-1]], color="#ffffff", s=40, zorder=5)
    ax2.set_xlim(0, max(total_frames - 1, 1))
    ax2.set_ylim(0, max(max(base_rot_errs + kalman_rot_errs + [8.0]) * 1.25, 15.0))
    ax2.set_xlabel("Sequence Frame Index", color="#94a3b8", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Rot Error (°)", color="#94a3b8", fontsize=9, fontweight="bold")
    ax2.tick_params(colors="#64748b", labelsize=8)
    ax2.grid(True, color="#1e293b", linestyle="--", alpha=0.7)
    ax2.legend(loc="upper left", facecolor="#1e293b", edgecolor="none", labelcolor="#e2e8f0", fontsize=8)

    plt.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)

    img = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
    return cv2.resize(img, (width, height))


def generate_v2_dashboard_video(
    scenario: str,
    output_dir: str = "data_results/v2_kalman/videos",
    dataset_dir: str = "datasets/synthetic_scenes/pick_place_floor_table",
    results_dir: str = "data_results/v2_kalman",
    checkpoint_dir: str = "model_checkpoints/v1_base_pogs",
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
):
    os.makedirs(output_dir, exist_ok=True)
    sc_dir = os.path.join(dataset_dir, f"tracking_{scenario}")
    with open(os.path.join(sc_dir, "tracking_metadata.json")) as f:
        meta = json.load(f)

    # Load trajectories
    kalman_deltas = np.load(os.path.join(results_dir, "pogs_kalman", f"part_deltas_traj__{scenario}.npy"))
    base_deltas = np.load(os.path.join(results_dir, "pogs_base", f"part_deltas_traj__{scenario}.npy"))
    with open(os.path.join(results_dir, "pogs_kalman", f"telemetry_{scenario}.json")) as f:
        kalman_telem = json.load(f)
    with open(os.path.join(results_dir, "pogs_base", f"telemetry_{scenario}.json")) as f:
        base_telem = json.load(f)

    cam_info = meta["camera"]
    c2w = np.array(cam_info["c2w"])
    w2c = np.linalg.inv(c2w)
    fx, fy, cx, cy = cam_info["fx"], cam_info["fy"], cam_info["cx"], cam_info["cy"]

    model, capture, _, _, _ = train_pick_place_pogs(dataset_dir=dataset_dir, checkpoint_dir=checkpoint_dir, device=device)
    groups = cluster_pick_place_objects(model=model, capture=capture, checkpoint_dir=checkpoint_dir, target_object_id=1, verbose=False)

    c_scan = np.array(meta["frames"][0]["c_scan"])
    extents = np.array(meta["frames"][0]["extents"])
    local_center = np.array(meta["frames"][0]["local_center"])
    mug_base = c_scan - local_center

    N = min(len(meta["frames"]), len(kalman_deltas))
    dashboard_frames = []

    frames_hist = []
    base_pos_errs, kalman_pos_errs = [], []
    base_rot_errs, kalman_rot_errs = [], []
    gt_traj, kalman_traj = [], []

    print(f"\nGenerating Comparison Dashboard Video for Scenario: {scenario.upper()} ({N} frames)...")

    for f in range(N):
        fr_meta = meta["frames"][f]

        # Quad 1: Observed Camera View + Dual Bounding Boxes
        rgb_path = os.path.join(sc_dir, "rgb", f"frame_{f:04d}.png")
        q1 = cv2.imread(rgb_path)

        delta_gt = np.array(fr_meta["gt_deltas"]["1"])
        corners_local = get_3d_box_corners(local_center, extents)
        corners_scan = mug_base + corners_local

        # 1. Ground Truth Box (Green)
        corners_gt = (delta_gt[:3, :3] @ corners_scan.T).T + delta_gt[:3, 3]
        c2d_gt, v_gt = project_points(corners_gt, w2c, fx, fy, cx, cy)
        if v_gt.all():
            draw_3d_box(q1, c2d_gt, (40, 220, 80), thickness=2)

        # 2. Kalman POGS Box (Cyan)
        pd_k = kalman_deltas[f, 0]
        R_k = quat_to_rotmat(torch.from_numpy(pd_k[3:]).unsqueeze(0))[0].numpy()
        corners_k = (R_k @ (corners_scan - c_scan).T).T + c_scan + pd_k[:3]
        c2d_k, v_k = project_points(corners_k, w2c, fx, fy, cx, cy)
        if v_k.all():
            draw_3d_box(q1, c2d_k, (255, 230, 0), thickness=2)

        # 3. Base POGS Box (Orange)
        pd_b = base_deltas[f, 0]
        R_b = quat_to_rotmat(torch.from_numpy(pd_b[3:]).unsqueeze(0))[0].numpy()
        corners_b = (R_b @ (corners_scan - c_scan).T).T + c_scan + pd_b[:3]
        c2d_b, v_b = project_points(corners_b, w2c, fx, fy, cx, cy)
        if v_b.all():
            draw_3d_box(q1, c2d_b, (0, 140, 255), thickness=1)

        # Centroid dot
        c_gt = np.array(fr_meta["gt_pos"])
        c_2d_gt, _ = project_points(c_gt[None], w2c, fx, fy, cx, cy)
        cv2.circle(q1, (int(c_2d_gt[0, 0]), int(c_2d_gt[0, 1])), 4, (0, 0, 255), -1)

        # Overlay text
        cv2.rectangle(q1, (10, 10), (380, 52), (20, 24, 33), -1)
        cv2.putText(q1, "OBSERVED CAMERA VIEW", (18, 28), cv2.FONT_HERSHEY_DUPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(q1, "GT (Green) | v2 Kalman (Cyan) | v1 Base (Orange)", (18, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (148, 163, 184), 1, cv2.LINE_AA)
        q1 = cv2.resize(cv2.cvtColor(q1, cv2.COLOR_BGR2RGB), (640, 360))

        # Quad 2: POGS Isolated 3D Splat
        track_cam = PinholeCamera(
            c2w=torch.from_numpy(c2w).float().to(device),
            fx=fx, fy=fy, cx=cx, cy=cy,
            width=int(cam_info["width"]), height=int(cam_info["height"]),
        )
        q2_raw = render_isolated_target_splats(
            model=model,
            groups=groups,
            part_delta_7vec=torch.from_numpy(pd_k).to(device),
            camera=track_cam,
        )
        q2 = cv2.resize(q2_raw, (640, 360))
        cv2.rectangle(q2, (10, 10), (380, 52), (15, 23, 42), -1)
        cv2.putText(q2, "V2 KALMAN POGS: ISOLATED 3D SPLAT", (18, 28), cv2.FONT_HERSHEY_DUPLEX, 0.46, (0, 229, 255), 1, cv2.LINE_AA)
        cv2.putText(q2, "Fused EKF + 3DGS Pose (Frame-by-Frame)", (18, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (148, 163, 184), 1, cv2.LINE_AA)

        # Quad 3: 3D Trajectory
        c_k = c_scan + pd_k[:3]
        gt_traj.append(c_gt)
        kalman_traj.append(c_k)
        from experiments.v1_base_pogs.generate_visuals import render_3d_trajectory_plot
        q3 = render_3d_trajectory_plot(gt_traj, kalman_traj, f, width=640, height=360)

        # Quad 4: Real-time Comparative Telemetry Curves
        frames_hist.append(f)
        base_pos_errs.append(base_telem[f]["trans_err_cm"])
        kalman_pos_errs.append(kalman_telem[f]["trans_err_cm"])
        base_rot_errs.append(base_telem[f]["rot_err_deg"])
        kalman_rot_errs.append(kalman_telem[f]["rot_err_deg"])
        q4 = render_comparison_telemetry_plot(
            frames_hist, base_pos_errs, kalman_pos_errs, base_rot_errs, kalman_rot_errs,
            f, N, width=640, height=360
        )

        composite = np.vstack([np.hstack([q1, q2]), np.hstack([q3, q4])])
        dashboard_frames.append(composite)

    mp4_out = os.path.join(output_dir, f"v2_comparison_{scenario}.mp4")
    gif_out = os.path.join(output_dir, f"v2_comparison_{scenario}.gif")
    imageio.mimsave(mp4_out, dashboard_frames, fps=5.0)
    imageio.mimsave(gif_out, dashboard_frames[::2], fps=3.0)
    sample_path = os.path.join(output_dir, f"sample_v2_comparison_{scenario}.png")
    cv2.imwrite(sample_path, cv2.cvtColor(dashboard_frames[len(dashboard_frames) // 2], cv2.COLOR_RGB2BGR))
    print(f"Saved Comparison Video: {mp4_out}")
    return mp4_out


def generate_all_v2_visuals():
    for sc in ["respected", "threshold", "violated"]:
        generate_v2_dashboard_video(sc)


if __name__ == "__main__":
    generate_all_v2_visuals()
