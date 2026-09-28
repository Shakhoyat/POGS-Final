"""
Multi-Panel Dashboard Video & Publication Visuals Generator (v1 - Base POGS Certified).

Dashboard Layout (4 Quads):
- Quad 1 (Top-Left): Observed RGB Camera View + 3D Bounding Box Overlay (GT Green, POGS Tracked Cyan)
- Quad 2 (Top-Right): POGS Output -- Isolated Object 3D Gaussian Splats Frame-by-Frame
- Quad 3 (Bottom-Left): 3D Spatial Trajectory & Waypoint Tracking
- Quad 4 (Bottom-Right): Dynamic Error Telemetry & Execution Profiling (DINO ms, Opt ms, FPS)
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
import pandas as pd
import torch

from common.geometry import PinholeCamera, look_at, quat_normalize, se3_compose, se3_inverse
from common.model import POGSModel, sh_to_rgb
from common.raster import rasterize
from experiments.v1_base_pogs.train import train_pick_place_pogs
from experiments.v1_base_pogs.cluster import cluster_pick_place_objects


def get_3d_box_corners(center: np.ndarray, extents: np.ndarray) -> np.ndarray:
    dx, dy, dz = extents / 2.0
    corners = np.array([
        [-dx, -dy, -dz],
        [ dx, -dy, -dz],
        [ dx,  dy, -dz],
        [-dx,  dy, -dz],
        [-dx, -dy,  dz],
        [ dx, -dy,  dz],
        [ dx,  dy,  dz],
        [-dx,  dy,  dz],
    ])
    return corners + center


def project_points(pts: np.ndarray, w2c: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> tuple[np.ndarray, np.ndarray]:
    pts_h = np.concatenate([pts, np.ones((len(pts), 1))], axis=-1)
    pts_c = (w2c @ pts_h.T).T
    z = pts_c[:, 2]
    valid = z > 1e-2
    u = fx * (pts_c[:, 0] / np.maximum(z, 1e-3)) + cx
    v = fy * (pts_c[:, 1] / np.maximum(z, 1e-3)) + cy
    return np.stack([u, v], axis=-1), valid


def draw_3d_box(img: np.ndarray, corners_2d: np.ndarray, color: tuple[int, int, int], thickness: int = 2):
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in edges:
        p1 = tuple(np.round(corners_2d[i]).astype(int))
        p2 = tuple(np.round(corners_2d[j]).astype(int))
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


def render_dynamic_telemetry_plot(
    frames_history: list[int],
    trans_errs: list[float],
    rot_errs: list[float],
    curr_frame: int,
    total_frames: int,
    latest_prof: dict,
    width: int = 640,
    height: int = 360,
) -> np.ndarray:
    """Renders real-time telemetry curves and execution stats."""
    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    fig.patch.set_facecolor("#0b0f19")

    # Top plot: Translation error (cm)
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.set_facecolor("#111827")
    if frames_history:
        ax1.plot(frames_history, trans_errs, color="#00e5ff", lw=2.2, label="Position Error (cm)")
        ax1.scatter([curr_frame], [trans_errs[-1]], color="#ffffff", s=45, zorder=5)
    ax1.set_xlim(0, max(total_frames - 1, 1))
    ax1.set_ylim(0, max(max(trans_errs + [4.0]) * 1.25, 5.0))
    ax1.set_ylabel("Pos Error (cm)", color="#94a3b8", fontsize=9, fontweight="bold")
    ax1.tick_params(colors="#64748b", labelsize=8)
    ax1.grid(True, color="#1e293b", linestyle="--", alpha=0.7)
    ax1.legend(loc="upper left", facecolor="#1e293b", edgecolor="none", labelcolor="#e2e8f0", fontsize=8)

    # Bottom plot: Rotation error (deg)
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.set_facecolor("#111827")
    if frames_history:
        ax2.plot(frames_history, rot_errs, color="#a855f7", lw=2.2, label="Rotation Error (°)")
        ax2.scatter([curr_frame], [rot_errs[-1]], color="#ffffff", s=45, zorder=5)
    ax2.set_xlim(0, max(total_frames - 1, 1))
    ax2.set_ylim(0, max(max(rot_errs + [8.0]) * 1.25, 15.0))
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
    img = cv2.resize(img, (width, height))

    # Add HUD stats overlay banner at bottom
    fps_val = latest_prof.get("fps", 0.0)
    dino_ms = latest_prof.get("dino_ms", 0.0)
    opt_ms = latest_prof.get("opt_ms", 0.0)
    hud_txt = f"DINO: {dino_ms:4.1f}ms | Opt (25 iters): {opt_ms:4.1f}ms | FPS: {fps_val:4.1f}"
    cv2.putText(img, hud_txt, (20, height - 12), cv2.FONT_HERSHEY_DUPLEX, 0.42, (203, 213, 225), 1, cv2.LINE_AA)

    return img


def render_3d_trajectory_plot(
    gt_traj: list[np.ndarray],
    pred_traj: list[np.ndarray],
    curr_frame: int,
    width: int = 640,
    height: int = 360,
) -> np.ndarray:
    """Renders the 3D trajectory plot showing floor, table, and motion path."""
    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    fig.patch.set_facecolor("#0b0f19")
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.set_facecolor("#0b0f19")

    # Floor plane outline
    ax.plot([-0.25, 0.25, 0.25, -0.25, -0.25], [-0.25, -0.25, 0.25, 0.25, -0.25], [0, 0, 0, 0, 0], color="#334155", lw=1.2, ls=":")
    # Small table outline (0.18, 0.12, 0.06)
    tx, ty, tz = 0.18, 0.12, 0.06
    ax.plot([tx-0.1, tx+0.1, tx+0.1, tx-0.1, tx-0.1], [ty-0.08, ty-0.08, ty+0.08, ty+0.08, ty-0.08], [tz, tz, tz, tz, tz], color="#b45309", lw=1.8, label="Staging Table (z=6cm)")

    if len(gt_traj) > 0:
        g = np.array(gt_traj)
        p = np.array(pred_traj)
        ax.plot(g[:, 0], g[:, 1], g[:, 2], color="#22c55e", lw=2.5, label="Ground Truth Trajectory")
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], color="#00e5ff", s=35, label="Base POGS Tracked")
        ax.scatter([g[-1, 0]], [g[-1, 1]], [g[-1, 2]], color="#ef4444", s=60, marker="o", label="Target Mug Pose")

    ax.set_xlim(-0.25, 0.30)
    ax.set_ylim(-0.25, 0.30)
    ax.set_zlim(0.0, 0.20)
    ax.set_xlabel("X (m)", color="#94a3b8", fontsize=8)
    ax.set_ylabel("Y (m)", color="#94a3b8", fontsize=8)
    ax.set_zlabel("Z (m)", color="#94a3b8", fontsize=8)
    ax.tick_params(colors="#64748b", labelsize=7)
    ax.view_init(elev=28, azim=-55)
    ax.legend(loc="upper left", facecolor="#1e293b", edgecolor="none", labelcolor="#e2e8f0", fontsize=7.5)

    plt.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)

    img = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
    return cv2.resize(img, (width, height))


def generate_dashboard_video(
    scenario: str,
    output_dir: str = "data_results/v1_base_pogs/videos",
    dataset_dir: str = "datasets/synthetic_scenes/pick_place_floor_table",
    results_dir: str = "data_results/v1_base_pogs",
    checkpoint_dir: str = "model_checkpoints/v1_base_pogs",
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
):
    os.makedirs(output_dir, exist_ok=True)
    sc_dir = os.path.join(dataset_dir, f"tracking_{scenario}")
    with open(os.path.join(sc_dir, "tracking_metadata.json")) as f:
        meta = json.load(f)

    telemetry_path = os.path.join(results_dir, f"telemetry_{scenario}.json")
    deltas_path = os.path.join(results_dir, f"part_deltas_traj__{scenario}.npy")
    with open(telemetry_path) as f:
        telemetry = json.load(f)
    deltas_np = np.load(deltas_path)

    cam_info = meta["camera"]
    c2w = np.array(cam_info["c2w"])
    w2c = np.linalg.inv(c2w)
    fx, fy, cx, cy = cam_info["fx"], cam_info["fy"], cam_info["cx"], cam_info["cy"]

    # Load trained model and mug cluster for Quad 2 rendering
    model, capture, _, _, _ = train_pick_place_pogs(dataset_dir=dataset_dir, checkpoint_dir=checkpoint_dir, device=device)
    groups = cluster_pick_place_objects(model=model, capture=capture, checkpoint_dir=checkpoint_dir, target_object_id=1)

    c_scan = np.array(meta["frames"][0]["c_scan"])
    extents = np.array(meta["frames"][0]["extents"])
    local_center = np.array(meta["frames"][0]["local_center"])
    mug_base = c_scan - local_center

    N = min(len(meta["frames"]), len(deltas_np))
    dashboard_frames = []

    frames_hist, trans_errs, rot_errs = [], [], []
    gt_traj, pred_traj = [], []

    print(f"\nGenerating 4-Panel Dashboard Video for Scenario: {scenario.upper()} ({N} frames)...")

    for f in range(N):
        fr_meta = meta["frames"][f]
        prof = telemetry[f]

        # ------------------------------------------------------------------- #
        # Quad 1: Observed Camera View with Aligned 3D Bounding Boxes
        # ------------------------------------------------------------------- #
        rgb_path = os.path.join(sc_dir, "rgb", f"frame_{f:04d}.png")
        q1 = cv2.imread(rgb_path)

        delta_gt = np.array(fr_meta["gt_deltas"]["1"])
        corners_local = get_3d_box_corners(local_center, extents)
        corners_scan = mug_base + corners_local

        # 1. Ground Truth 3D Box (Green)
        corners_world_gt = (delta_gt[:3, :3] @ corners_scan.T).T + delta_gt[:3, 3]
        c2d_gt, v_gt = project_points(corners_world_gt, w2c, fx, fy, cx, cy)
        if v_gt.all():
            draw_3d_box(q1, c2d_gt, (40, 220, 80), thickness=2)

        # 2. Base POGS Tracked 3D Box (Cyan)
        pd = deltas_np[f, 0] # (7,) [t | quat_wxyz]
        from common.geometry import quat_to_rotmat
        R_pred = quat_to_rotmat(torch.from_numpy(pd[3:]).unsqueeze(0))[0].numpy()
        t_pred = pd[:3]
        corners_world_p = (R_pred @ (corners_scan - c_scan).T).T + c_scan + t_pred
        c2d_p, v_p = project_points(corners_world_p, w2c, fx, fy, cx, cy)
        if v_p.all():
            draw_3d_box(q1, c2d_p, (255, 230, 0), thickness=2)

        # Red centroid dot
        c_gt = np.array(fr_meta["gt_pos"])
        c_2d_gt, _ = project_points(c_gt[None], w2c, fx, fy, cx, cy)
        cv2.circle(q1, (int(c_2d_gt[0, 0]), int(c_2d_gt[0, 1])), 4, (0, 0, 255), -1)

        # Overlay text
        cv2.rectangle(q1, (10, 10), (320, 48), (20, 24, 33), -1)
        cv2.putText(q1, "OBSERVED CAMERA VIEW", (18, 30), cv2.FONT_HERSHEY_DUPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(q1, "GT (Green) | POGS Tracked (Cyan)", (18, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (148, 163, 184), 1, cv2.LINE_AA)
        q1 = cv2.resize(cv2.cvtColor(q1, cv2.COLOR_BGR2RGB), (640, 360))

        # ------------------------------------------------------------------- #
        # Quad 2: POGS Output -- Isolated Object 3D Gaussian Splats
        # ------------------------------------------------------------------- #
        from experiments.v1_base_pogs.visualize_splats import render_isolated_target_splats
        track_cam = PinholeCamera(
            c2w=torch.from_numpy(c2w).float().to(device),
            fx=fx, fy=fy, cx=cx, cy=cy,
            width=int(cam_info["width"]), height=int(cam_info["height"]),
        )
        q2_raw = render_isolated_target_splats(
            model=model,
            groups=groups,
            part_delta_7vec=torch.from_numpy(pd).to(device),
            camera=track_cam,
        )
        q2 = cv2.resize(q2_raw, (640, 360))
        cv2.rectangle(q2, (10, 10), (360, 48), (15, 23, 42), -1)
        cv2.putText(q2, "POGS OUTPUT: ISOLATED 3D SPLAT", (18, 30), cv2.FONT_HERSHEY_DUPLEX, 0.48, (0, 229, 255), 1, cv2.LINE_AA)
        cv2.putText(q2, "Tracked Ceramic Mug Splat (Frame-by-Frame)", (18, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (148, 163, 184), 1, cv2.LINE_AA)

        # ------------------------------------------------------------------- #
        # Quad 3: 3D Spatial Trajectory
        # ------------------------------------------------------------------- #
        c_pred = c_scan + t_pred
        gt_traj.append(c_gt)
        pred_traj.append(c_pred)
        q3 = render_3d_trajectory_plot(gt_traj, pred_traj, f, width=640, height=360)

        # ------------------------------------------------------------------- #
        # Quad 4: Real-Time Telemetry & Profiling Plot
        # ------------------------------------------------------------------- #
        frames_hist.append(f)
        trans_errs.append(prof["trans_err_cm"])
        rot_errs.append(prof["rot_err_deg"])
        q4 = render_dynamic_telemetry_plot(frames_hist, trans_errs, rot_errs, f, N, prof, width=640, height=360)

        # Assemble 2x2 Composite Dashboard (1280 x 720)
        top_row = np.hstack([q1, q2])
        bottom_row = np.hstack([q3, q4])
        composite = np.vstack([top_row, bottom_row])
        dashboard_frames.append(composite)

        if f % 5 == 0 or f == N - 1:
            print(f"  Rendered Dashboard Frame {f + 1:2d}/{N}")

    # Save MP4 and GIF
    mp4_out = os.path.join(output_dir, f"dashboard_{scenario}.mp4")
    gif_out = os.path.join(output_dir, f"dashboard_{scenario}.gif")
    imageio.mimsave(mp4_out, dashboard_frames, fps=5.0)
    imageio.mimsave(gif_out, dashboard_frames[::2], fps=3.0)
    print(f"Saved Dashboard Video: {mp4_out}")

    # Save sample frame as artifact image
    sample_path = os.path.join(output_dir, f"sample_dashboard_{scenario}.png")
    cv2.imwrite(sample_path, cv2.cvtColor(dashboard_frames[len(dashboard_frames) // 2], cv2.COLOR_RGB2BGR))
    return mp4_out


def generate_all_visuals():
    scenarios = ["respected", "threshold", "violated"]
    for sc in scenarios:
        generate_dashboard_video(sc)


if __name__ == "__main__":
    generate_all_visuals()
