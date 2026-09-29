"""
Multi-Panel Dashboard Video Generator for Scene 2 (Mustard Bottle Pick-and-Place).

Dashboard Layout (4 Quads):
- Quad 1 (Top-Left): Observed RGB Camera View + 3D Bounding Box Overlay (GT Green, POGS Tracked Cyan)
- Quad 2 (Top-Right): POGS Output -- Isolated Mustard Bottle 3D Gaussian Splats Frame-by-Frame
- Quad 3 (Bottom-Left): 3D Spatial Trajectory & Waypoint Tracking
- Quad 4 (Bottom-Right): Dynamic Error Telemetry & Execution Profiling (Pos Error cm, Rot Error deg, FPS)
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

    fps_val = latest_prof.get("fps", 0.0)
    dur_ms = latest_prof.get("frame_time_sec", 0.0) * 1000.0

    hud_bar = np.zeros((40, width, 3), dtype=np.uint8)
    hud_bar[:] = (15, 23, 42)
    txt = f"LATENCY: {dur_ms:4.0f} ms | SPEED: {fps_val:4.1f} FPS | CONSTRAINT: {latest_prof.get('status', 'RESPECTED')}"
    cv2.putText(hud_bar, txt, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (226, 232, 240), 1, cv2.LINE_AA)
    img = np.vstack([img[:-40], hud_bar])
    return img


def render_3d_trajectory_plot(
    gt_traj: list[np.ndarray],
    pred_traj: list[np.ndarray],
    curr_frame: int,
    total_frames: int,
    width: int = 640,
    height: int = 360,
) -> np.ndarray:
    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    fig.patch.set_facecolor("#0b0f19")
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.set_facecolor("#111827")

    gt_arr = np.array(gt_traj) * 100.0
    pred_arr = np.array(pred_traj) * 100.0

    ax.plot(gt_arr[:, 0], gt_arr[:, 1], gt_arr[:, 2], color="#22c55e", lw=2.2, linestyle="--", label="Ground Truth Path")
    ax.plot(pred_arr[:, 0], pred_arr[:, 1], pred_arr[:, 2], color="#00e5ff", lw=2.5, label="POGS Tracked Path")

    ax.scatter([gt_arr[curr_frame, 0]], [gt_arr[curr_frame, 1]], [gt_arr[curr_frame, 2]], color="#22c55e", s=65, edgecolor="#ffffff")
    ax.scatter([pred_arr[curr_frame, 0]], [pred_arr[curr_frame, 1]], [pred_arr[curr_frame, 2]], color="#00e5ff", s=75, edgecolor="#ffffff")

    ax.set_title("3D Workspace Trajectory (Floor to Table)", color="#e2e8f0", fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("X (cm)", color="#94a3b8", fontsize=8)
    ax.set_ylabel("Y (cm)", color="#94a3b8", fontsize=8)
    ax.set_zlabel("Z (cm)", color="#94a3b8", fontsize=8)
    ax.tick_params(colors="#64748b", labelsize=7)

    ax.w_xaxis.set_pane_color((0.07, 0.09, 0.15, 1.0))
    ax.w_yaxis.set_pane_color((0.07, 0.09, 0.15, 1.0))
    ax.w_zaxis.set_pane_color((0.07, 0.09, 0.15, 1.0))
    ax.legend(loc="upper right", facecolor="#1e293b", edgecolor="none", labelcolor="#e2e8f0", fontsize=8)

    plt.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)

    img = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
    return cv2.resize(img, (width, height))


def render_isolated_mustard_splats(
    model: POGSModel,
    groups,
    part_delta_7vec: torch.Tensor,
    camera: PinholeCamera,
    group_idx: int = 0,
    background: tuple[float, float, float] = (0.07, 0.08, 0.11),
) -> np.ndarray:
    device = model.gauss_params["means"].device
    cam = camera.to(device)

    mustard_mask = groups.group_masks[group_idx]
    means = model.gauss_params["means"].data[mustard_mask]
    quats = model.gauss_params["quats"].data[mustard_mask]
    scales = torch.exp(model.gauss_params["scales"].data[mustard_mask])
    opacities = torch.sigmoid(model.gauss_params["opacities"].data[mustard_mask]).squeeze(-1)
    colors = model.colors_sh[mustard_mask]

    p2w = groups.init_p2w[group_idx]
    g2w = torch.cat([means, quat_normalize(quats)], dim=-1)
    g2p = se3_compose(se3_inverse(p2w.unsqueeze(0)), g2w)
    new_g2w = se3_compose(se3_compose(p2w.unsqueeze(0), part_delta_7vec.to(device).unsqueeze(0)), g2p)

    means_t = new_g2w[:, :3]
    quats_t = new_g2w[:, 3:]

    bg_t = torch.tensor(background, dtype=torch.float32, device=device)

    with torch.no_grad():
        out = rasterize(
            means=means_t,
            quats=quats_t,
            scales=scales,
            opacities=opacities,
            colors=colors,
            camera=cam,
            backend=model.backend,
            sh_degree=model.sh_degree_to_use(),
            render_depth=False,
            background=bg_t,
            tile_size=8,
            antialiased=True,
        )
    rgb = torch.clamp(out["render"], 0.0, 1.0).cpu().numpy()
    return (rgb * 255).astype(np.uint8)


def generate_mustard_dashboard_video(
    dataset_dir: str,
    checkpoint_dir: str,
    results_dir: str,
    scenario_name: str = "respected",
    method: str = "kalman",
    device: str = "cuda:0",
    out_name: Optional[str] = None,
) -> str:
    from experiments.v3_scene_mustard.train import train_mustard_pogs
    from experiments.v3_scene_mustard.cluster import cluster_mustard_object

    model, capture, _, _, _ = train_mustard_pogs(
        dataset_dir=dataset_dir,
        checkpoint_dir=checkpoint_dir,
        device=device,
        force_retrain=False,
    )
    groups = cluster_mustard_object(model, capture, checkpoint_dir=checkpoint_dir, target_object_id=2)

    sc_dir = os.path.join(dataset_dir, f"tracking_{scenario_name}")
    meta_path = os.path.join(sc_dir, "tracking_metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)

    # Check method subdirectory or root results_dir
    candidate_dirs = [os.path.join(results_dir, method), results_dir]
    data_dir = None
    for cd in candidate_dirs:
        if os.path.exists(os.path.join(cd, f"part_deltas_traj__{scenario_name}.npy")):
            data_dir = cd
            break
    if data_dir is None:
        raise FileNotFoundError(f"Could not find part_deltas_traj__{scenario_name}.npy in {candidate_dirs}")

    traj_path = os.path.join(data_dir, f"part_deltas_traj__{scenario_name}.npy")
    part_deltas = np.load(traj_path)

    telem_path = os.path.join(data_dir, f"telemetry_{scenario_name}.json")
    with open(telem_path) as f:
        telem = json.load(f)

    cam_info = meta["camera"]
    camera = PinholeCamera(
        c2w=torch.tensor(cam_info["c2w"], device=device, dtype=torch.float32),
        fx=cam_info["fx"], fy=cam_info["fy"], cx=cam_info["cx"], cy=cam_info["cy"],
        width=cam_info["width"], height=cam_info["height"],
    )

    w2c = camera.w2c.cpu().numpy()
    rgb_dir = os.path.join(sc_dir, "rgb")
    n_frames = len(meta["frames"])

    gt_centers = [np.array(t["gt_center"]) for t in telem]
    pred_centers = [np.array(t["pred_center"]) for t in telem]

    q_w, q_h = 640, 360
    dashboard_frames = []

    print(f"\n[Scene 2 Visuals] Rendering 4-Panel Dashboard for '{scenario_name}' ({n_frames} frames)...")

    for f_idx in range(n_frames):
        bgr = cv2.imread(os.path.join(rgb_dir, f"frame_{f_idx:04d}.png"))
        rgb_raw = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_q1 = cv2.resize(rgb_raw, (q_w, q_h)).copy()

        # Quad 1: Bounding boxes
        gt_corners = get_3d_box_corners(gt_centers[f_idx], extents=np.array([0.072, 0.048, 0.155]))
        gt_2d, v_gt = project_points(gt_corners, w2c, camera.fx * (q_w / camera.width), camera.fy * (q_h / camera.height), camera.cx * (q_w / camera.width), camera.cy * (q_h / camera.height))
        if v_gt.all():
            draw_3d_box(rgb_q1, gt_2d, color=(34, 197, 94), thickness=2)

        pred_corners = get_3d_box_corners(pred_centers[f_idx], extents=np.array([0.072, 0.048, 0.155]))
        pred_2d, v_pr = project_points(pred_corners, w2c, camera.fx * (q_w / camera.width), camera.fy * (q_h / camera.height), camera.cx * (q_w / camera.width), camera.cy * (q_h / camera.height))
        if v_pr.all():
            draw_3d_box(rgb_q1, pred_2d, color=(0, 229, 255), thickness=2)

        cv2.putText(rgb_q1, "QUAD 1: CAMERA OBS + 3D BOX (GT:GREEN, POGS:CYAN)", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

        # Quad 2: POGS PREDICTED ISOLATED MUSTARD 3D SPLATS
        delta_f = torch.from_numpy(part_deltas[f_idx, 0]).to(device)
        isolated_rgb = render_isolated_mustard_splats(model, groups, delta_f, camera=camera, group_idx=0)
        isolated_q2 = cv2.resize(isolated_rgb, (q_w, q_h))
        cv2.putText(isolated_q2, "QUAD 2: POGS PREDICTED ISOLATED 3D SPLAT", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 229, 255), 1, cv2.LINE_AA)
        cv2.putText(isolated_q2, f"Pose Delta: [{delta_f[0]:.2f}, {delta_f[1]:.2f}, {delta_f[2]:.2f}] m", (14, q_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1, cv2.LINE_AA)

        # Quad 3: 3D Trajectory Map
        q3_img = render_3d_trajectory_plot(gt_centers, pred_centers, f_idx, n_frames, width=q_w, height=q_h)
        cv2.putText(q3_img, "QUAD 3: 3D WORKSPACE TRAJECTORY", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

        # Quad 4: Real-time Telemetry Plot & HUD
        hist_f = list(range(f_idx + 1))
        hist_t = [telem[i]["trans_err_cm"] for i in range(f_idx + 1)]
        hist_r = [telem[i]["rot_err_deg"] for i in range(f_idx + 1)]
        prof_info = {
            "fps": 1.0 / max(telem[f_idx]["frame_time_sec"], 1e-4),
            "frame_time_sec": telem[f_idx]["frame_time_sec"],
            "status": telem[f_idx]["constraint_status"],
        }
        q4_img = render_dynamic_telemetry_plot(hist_f, hist_t, hist_r, f_idx, n_frames, prof_info, width=q_w, height=q_h)
        cv2.putText(q4_img, "QUAD 4: TELEMETRY & CONSTRAINT MONITOR", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

        # 4-Panel Canvas assembly (1280x720)
        top_row = np.hstack([rgb_q1, isolated_q2])
        bot_row = np.hstack([q3_img, q4_img])
        quad_canvas = np.vstack([top_row, bot_row])

        # Header banner (60px)
        header = np.zeros((60, 1280, 3), dtype=np.uint8)
        header[:] = (11, 15, 25)
        title_txt = f"SCENE 2 (MUSTARD BOTTLE PICK-AND-PLACE) | POGS 6-DoF TRACKING DASHBOARD [{scenario_name.upper()}]"
        cv2.putText(header, title_txt, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
        status_color = (34, 197, 94) if telem[f_idx]["constraint_status"] == "respected" else (245, 158, 11) if telem[f_idx]["constraint_status"] == "near_threshold" else (239, 68, 68)
        cv2.putText(header, f"STATUS: {telem[f_idx]['constraint_status'].upper()}", (980, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.60, status_color, 2, cv2.LINE_AA)

        full_frame = np.vstack([header, quad_canvas])
        dashboard_frames.append(full_frame)

    os.makedirs(os.path.join(results_dir, "videos"), exist_ok=True)
    base_name = out_name or f"dashboard_mustard_{scenario_name}"
    mp4_out = os.path.join(results_dir, "videos", f"{base_name}.mp4")
    gif_out = os.path.join(results_dir, "videos", f"{base_name}.gif")
    png_sample = os.path.join(results_dir, "videos", f"sample_{base_name}.png")

    imageio.mimsave(mp4_out, dashboard_frames, fps=5.0)
    imageio.mimsave(gif_out, dashboard_frames[::2], fps=3.0)
    cv2.imwrite(png_sample, cv2.cvtColor(dashboard_frames[len(dashboard_frames) // 2], cv2.COLOR_RGB2BGR))

    print(f"  [Visuals] Saved video: {mp4_out} and sample: {png_sample}")
    return mp4_out
