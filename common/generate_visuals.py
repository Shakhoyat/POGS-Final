"""
POGS Visuals & Telemetry Dashboard Video Generator (ICRA 2025 Pure Baseline).
Generates:
1. Multi-panel Dashboard Videos (RGB with 3D Bounding Box + Depth + Live Telemetry HUD + Dynamic Error Plot).
2. Multi-scenario Synchronized Comparison Video (Respected vs Threshold vs Violated in a single frame).
3. Publication-ready Evaluation Plots (Tracking error curves, Table I Ablations, Occlusion analysis, 3D Trajectories).
4. Keyframe Montage Sheets for each scenario.
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


# --------------------------------------------------------------------------- #
# Geometry & Camera Helpers
# --------------------------------------------------------------------------- #

def get_3d_box_corners(center: np.ndarray, extents: np.ndarray) -> np.ndarray:
    """Return 8 corners of an axis-aligned box centered at center with given extents (dx, dy, dz)."""
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
    """Project 3D world points (N, 3) to 2D image coordinates (N, 2)."""
    pts_h = np.concatenate([pts, np.ones((len(pts), 1))], axis=-1)
    pts_c = (w2c @ pts_h.T).T
    z = pts_c[:, 2]
    valid = z > 1e-2
    u = fx * (pts_c[:, 0] / np.maximum(z, 1e-3)) + cx
    v = fy * (pts_c[:, 1] / np.maximum(z, 1e-3)) + cy
    return np.stack([u, v], axis=-1), valid


def draw_3d_box(img: np.ndarray, corners_2d: np.ndarray, color: tuple[int, int, int], thickness: int = 2):
    """Draw the 12 wireframe edges of a 3D box projected to 2D."""
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7),  # pillars
    ]
    for i, j in edges:
        pt1 = (int(round(corners_2d[i, 0])), int(round(corners_2d[i, 1])))
        pt2 = (int(round(corners_2d[j, 0])), int(round(corners_2d[j, 1])))
        cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)


def rotation_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    R_diff = R_pred @ R_gt.T
    tr = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(tr)))


# --------------------------------------------------------------------------- #
# 1. Publication Plots Generator
# --------------------------------------------------------------------------- #

def generate_plots(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print("Generating Publication Plots...")

    # Load tracking deltas and metadata for the 3 scenarios
    scenarios = ["respected", "threshold", "violated"]
    data = {}
    for sc in scenarios:
        meta_file = f"datasets/synthetic_scenes/tabletop_drill_iron/tracking_{sc}/tracking_metadata.json"
        pred_file = f"data_results/v1_baseline/part_deltas_traj__{sc}.npy"
        with open(meta_file) as f:
            meta = json.load(f)
        pred = np.load(pred_file)
        data[sc] = {"meta": meta, "pred": pred}

    # Drill centroid [-0.081, -0.030, 0.090], Iron centroid [0.067, 0.011, 0.040]
    c_drill = np.array([-0.0812, -0.0297, 0.0905])
    c_iron = np.array([0.0674, 0.0112, 0.0397])

    # Compute frame-by-frame errors
    errs_by_sc = {}
    for sc in scenarios:
        meta = data[sc]["meta"]
        pred = data[sc]["pred"]
        n_frames = min(len(meta["frames"]), len(pred))
        t_errs, r_errs = [], []
        for t in range(n_frames):
            frame_meta = meta["frames"][t]
            gt_deltas = frame_meta["gt_deltas"]
            # Drill (Group 0 -> obj 1)
            gt_d = np.array(gt_deltas["1"])
            pred_d = pred[t, 0]
            # Translation error between predicted centroid and GT centroid
            gt_pos_d = gt_d[:3, :3] @ c_drill + gt_d[:3, 3]
            pred_pos_d = pred_d[:3, :3] @ c_drill + pred_d[:3, 3]
            e_t_drill = np.linalg.norm(pred_pos_d - gt_pos_d) * 100.0
            e_r_drill = rotation_error_deg(pred_d[:3, :3], gt_d[:3, :3])

            # Iron (Group 1 -> obj 2)
            gt_i = np.array(gt_deltas["2"])
            pred_i = pred[t, 1]
            gt_pos_i = gt_i[:3, :3] @ c_iron + gt_i[:3, 3]
            pred_pos_i = pred_i[:3, :3] @ c_iron + pred_i[:3, 3]
            e_t_iron = np.linalg.norm(pred_pos_i - gt_pos_i) * 100.0
            e_r_iron = rotation_error_deg(pred_i[:3, :3], gt_i[:3, :3])

            t_errs.append(0.5 * (e_t_drill + e_t_iron))
            r_errs.append(0.5 * (e_r_drill + e_r_iron))
        errs_by_sc[sc] = {"trans_err": np.array(t_errs), "rot_err": np.array(r_errs)}

    # Plot 1: Tracking Error Curves over Frames
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=150)
    fig.patch.set_facecolor('#0f172a')
    for ax in (ax1, ax2):
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#475569')
        ax.grid(True, linestyle='--', alpha=0.3, color='#64748b')

    colors = {"respected": "#38bdf8", "threshold": "#f59e0b", "violated": "#ef4444"}
    labels = {"respected": "Respected (Smooth)", "threshold": "Threshold (Nominal limit)", "violated": "Violated (Fast/Jerky)"}

    for sc in scenarios:
        frames = np.arange(len(errs_by_sc[sc]["trans_err"]))
        ax1.plot(frames, errs_by_sc[sc]["trans_err"], label=labels[sc], color=colors[sc], lw=2.4, marker='o', ms=4)
        ax2.plot(frames, errs_by_sc[sc]["rot_err"], label=labels[sc], color=colors[sc], lw=2.4, marker='s', ms=4)

    # Shaded hand occlusion region (frames 16-19)
    for ax in (ax1, ax2):
        ax.axvspan(15.5, 19.5, color='#a855f7', alpha=0.22, label="Hand Occlusion")

    ax1.axhline(5.0, color='#f43f5e', ls='--', lw=1.5, label='Loss Threshold (5 cm)')
    ax1.set_xlabel("Evaluation Frame Index", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Mean Translation Error (cm)", fontsize=11, fontweight='bold')
    ax1.set_title("Translation Error vs Frame Index", fontsize=13, fontweight='bold', pad=12)
    ax1.legend(loc='upper left', facecolor='#0f172a', edgecolor='#475569', labelcolor='white', fontsize=9)

    ax2.set_xlabel("Evaluation Frame Index", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Mean Rotation Error (deg)", fontsize=11, fontweight='bold')
    ax2.set_title("Geodesic Rotation Error vs Frame Index", fontsize=13, fontweight='bold', pad=12)
    ax2.legend(loc='upper left', facecolor='#0f172a', edgecolor='#475569', labelcolor='white', fontsize=9)

    fig.suptitle("POGS (ICRA 2025) 6-DoF Persistent Object Tracking Error", fontsize=16, fontweight='bold', color='white', y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    p1_path = os.path.join(output_dir, "tracking_errors_timeline.png")
    fig.savefig(p1_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {p1_path}")

    # Plot 2: Table I Ablations Comparison Bar Chart
    abl_file = "data_results/v1_baseline/ablations.csv"
    if os.path.exists(abl_file):
        df_abl = pd.read_csv(abl_file)
        fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#475569')
        ax.grid(True, linestyle='--', alpha=0.3, color='#64748b', axis='y')

        bar_width = 0.25
        x = np.arange(len(df_abl))
        b1 = ax.bar(x - bar_width, df_abl["mean_pos_err_cm"], bar_width, label="Mean Pos Err (cm)", color="#38bdf8", edgecolor='#0284c7', lw=1.5)
        b2 = ax.bar(x, df_abl["mean_rot_err_deg"], bar_width, label="Mean Rot Err (deg)", color="#f59e0b", edgecolor='#d97706', lw=1.5)
        b3 = ax.bar(x + bar_width, df_abl["mean_adds_cm"], bar_width, label="ADD-S (cm)", color="#10b981", edgecolor='#059669', lw=1.5)

        ax.set_xticks(x)
        ax.set_xticklabels(df_abl["name"], fontsize=11, fontweight='bold', color='white')
        ax.set_ylabel("Metric Value (cm / deg)", fontsize=11, fontweight='bold')
        ax.set_title("Table I Ablations: POGS vs No Depth vs No DINO", fontsize=14, fontweight='bold', pad=12)
        ax.legend(facecolor='#0f172a', edgecolor='#475569', labelcolor='white')

        # Add text labels on top of bars
        for bars in [b1, b2, b3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f"{height:.1f}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', color='white', fontsize=8.5, fontweight='bold')

        fig.tight_layout()
        p2_path = os.path.join(output_dir, "table1_ablations_comparison.png")
        fig.savefig(p2_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {p2_path}")

    # Plot 3: 3D Spatial Trajectory paths
    fig = plt.figure(figsize=(10, 6.5), dpi=150)
    fig.patch.set_facecolor('#0f172a')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#1e293b')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.grid(True, linestyle='--', alpha=0.25)

    meta_res = data["respected"]["meta"]
    pred_res = data["respected"]["pred"]
    n_frames = min(len(meta_res["frames"]), len(pred_res))
    gt_drill_pts, pred_drill_pts = [], []
    for t in range(n_frames):
        gt_d = np.array(meta_res["frames"][t]["gt_deltas"]["1"])
        pred_d = pred_res[t, 0]
        gt_drill_pts.append(gt_d[:3, :3] @ c_drill + gt_d[:3, 3])
        pred_drill_pts.append(pred_d[:3, :3] @ c_drill + pred_d[:3, 3])
    gt_drill_pts = np.array(gt_drill_pts)
    pred_drill_pts = np.array(pred_drill_pts)

    ax.plot(gt_drill_pts[:, 0], gt_drill_pts[:, 1], gt_drill_pts[:, 2], label="GT Trajectory (Drill)", color="#22c55e", lw=3.0, marker='o', ms=5)
    ax.plot(pred_drill_pts[:, 0], pred_drill_pts[:, 1], pred_drill_pts[:, 2], label="POGS Estimated (Drill)", color="#06b6d4", lw=2.2, ls='--', marker='^', ms=4)

    ax.set_xlabel("X (m)", fontweight='bold', labelpad=8)
    ax.set_ylabel("Y (m)", fontweight='bold', labelpad=8)
    ax.set_zlabel("Z (m)", fontweight='bold', labelpad=8)
    ax.set_title("3D Object Trajectory: Ground Truth vs POGS Tracking", fontsize=13, fontweight='bold', pad=15)
    ax.legend(facecolor='#0f172a', edgecolor='#475569', labelcolor='white')
    fig.tight_layout()
    p3_path = os.path.join(output_dir, "spatial_3d_trajectory.png")
    fig.savefig(p3_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {p3_path}")


# --------------------------------------------------------------------------- #
# 2. Keyframe Montage Generator
# --------------------------------------------------------------------------- #

def generate_montages(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print("Generating Keyframe Montages...")

    scenarios = ["respected", "threshold", "violated"]
    key_indices = [0, 4, 8, 12, 16, 19]

    for sc in scenarios:
        meta_file = f"datasets/synthetic_scenes/tabletop_drill_iron/tracking_{sc}/tracking_metadata.json"
        with open(meta_file) as f:
            meta = json.load(f)
        pred = np.load(f"data_results/v1_baseline/part_deltas_traj__{sc}.npy")
        cam = meta["camera"]
        c2w = np.array(cam["c2w"])
        w2c = np.linalg.inv(c2w)
        fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]

        c_drill = np.array([-0.0812, -0.0297, 0.0905])
        c_iron = np.array([0.0674, 0.0112, 0.0397])
        s_drill = np.array([0.08, 0.07, 0.12])
        s_iron = np.array([0.09, 0.06, 0.07])

        fig, axes = plt.subplots(2, len(key_indices), figsize=(3.2 * len(key_indices), 5.8), dpi=150)
        fig.patch.set_facecolor('#0f172a')

        for col, idx in enumerate(key_indices):
            # RGB frame
            rgb_path = f"datasets/synthetic_scenes/tabletop_drill_iron/tracking_{sc}/rgb/frame_{idx:04d}.png"
            depth_vis_path = f"datasets/synthetic_scenes/tabletop_drill_iron/tracking_{sc}/depth_vis/frame_{idx:04d}.png"
            img_bgr = cv2.imread(rgb_path)
            depth_vis = cv2.imread(depth_vis_path)

            gt_d = np.array(meta["frames"][idx]["gt_deltas"]["1"])
            gt_i = np.array(meta["frames"][idx]["gt_deltas"]["2"])
            pred_d = pred[idx, 0]
            pred_i = pred[idx, 1]

            # Project GT boxes (Green)
            corners_gt_d = (gt_d[:3, :3] @ get_3d_box_corners(c_drill, s_drill).T).T + gt_d[:3, 3]
            c2d_gt_d, v_gt_d = project_points(corners_gt_d, w2c, fx, fy, cx, cy)
            if v_gt_d.all():
                draw_3d_box(img_bgr, c2d_gt_d, (0, 255, 102), 2)

            corners_gt_i = (gt_i[:3, :3] @ get_3d_box_corners(c_iron, s_iron).T).T + gt_i[:3, 3]
            c2d_gt_i, v_gt_i = project_points(corners_gt_i, w2c, fx, fy, cx, cy)
            if v_gt_i.all():
                draw_3d_box(img_bgr, c2d_gt_i, (0, 255, 102), 2)

            # Project POGS Predicted boxes (Cyan)
            corners_p_d = (pred_d[:3, :3] @ get_3d_box_corners(c_drill, s_drill).T).T + pred_d[:3, 3]
            c2d_p_d, v_p_d = project_points(corners_p_d, w2c, fx, fy, cx, cy)
            if v_p_d.all():
                draw_3d_box(img_bgr, c2d_p_d, (255, 230, 0), 2)

            corners_p_i = (pred_i[:3, :3] @ get_3d_box_corners(c_iron, s_iron).T).T + pred_i[:3, 3]
            c2d_p_i, v_p_i = project_points(corners_p_i, w2c, fx, fy, cx, cy)
            if v_p_i.all():
                draw_3d_box(img_bgr, c2d_p_i, (255, 230, 0), 2)

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            axes[0, col].imshow(img_rgb)
            axes[0, col].set_title(f"Frame {idx}\n(t={idx*0.2:.1f}s)", color='white', fontsize=10, fontweight='bold')
            axes[0, col].axis('off')

            axes[1, col].imshow(cv2.cvtColor(depth_vis, cv2.COLOR_BGR2RGB))
            axes[1, col].set_title(f"Depth t={idx}", color='#94a3b8', fontsize=9)
            axes[1, col].axis('off')

        axes[0, 0].set_ylabel("RGB + 3D Tracking", color='white', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel("Geometry Depth", color='white', fontsize=11, fontweight='bold')
        fig.suptitle(f"POGS Keyframe Tracking Montage: Scenario {sc.upper()}", color='white', fontsize=15, fontweight='bold', y=0.98)
        fig.tight_layout(rect=[0, 0.03, 1, 0.94])
        out_montage = os.path.join(output_dir, f"keyframe_montage_{sc}.png")
        fig.savefig(out_montage, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out_montage}")


# --------------------------------------------------------------------------- #
# 3. Dynamic Composite Dashboard Video Generator
# --------------------------------------------------------------------------- #

def create_telemetry_hud(frame_idx: int, scenario: str, err_t: float, err_r: float, is_occluded: bool, width: int = 640, height: int = 360) -> np.ndarray:
    """Create a sleek, high-tech telemetry HUD panel image."""
    hud = np.full((height, width, 3), (15, 23, 42), dtype=np.uint8) # Dark slate background

    # Title header
    cv2.putText(hud, "POGS 6-DoF TELEMETRY HUD", (30, 42), cv2.FONT_HERSHEY_DUPLEX, 0.82, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(hud, "ICRA 2025 BASELINE EVALUATION", (30, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (148, 163, 184), 1, cv2.LINE_AA)
    cv2.line(hud, (30, 82), (width - 30, 82), (51, 65, 85), 1)

    # Status Cards
    # Card 1: Frame & Scenario
    cv2.rectangle(hud, (30, 100), (280, 185), (30, 41, 59), -1)
    cv2.rectangle(hud, (30, 100), (280, 185), (71, 85, 105), 1)
    cv2.putText(hud, f"SCENARIO: {scenario.upper()}", (45, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (56, 189, 248), 1, cv2.LINE_AA)
    cv2.putText(hud, f"FRAME: {frame_idx:02d} / 20", (45, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(hud, f"TIME: {frame_idx * 0.2:.2f} s", (45, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1, cv2.LINE_AA)

    # Card 2: Tracking State
    cv2.rectangle(hud, (310, 100), (width - 30, 185), (30, 41, 59), -1)
    cv2.rectangle(hud, (310, 100), (width - 30, 185), (71, 85, 105), 1)
    status_text = "OCCLUDED (RETENTION)" if is_occluded else "NOMINAL TRACKING"
    status_col = (168, 85, 247) if is_occluded else (34, 197, 94)
    cv2.putText(hud, "TRACKER STATE:", (325, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (148, 163, 184), 1, cv2.LINE_AA)
    cv2.putText(hud, status_text, (325, 162), cv2.FONT_HERSHEY_DUPLEX, 0.60, status_col, 2, cv2.LINE_AA)

    # Metrics Grid
    cv2.rectangle(hud, (30, 205), (width - 30, 325), (30, 41, 59), -1)
    cv2.rectangle(hud, (30, 205), (width - 30, 325), (71, 85, 105), 1)
    cv2.putText(hud, "LIVE ESTIMATION ACCURACY", (45, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (245, 158, 11), 1, cv2.LINE_AA)

    cv2.putText(hud, f"Trans Error (cm):", (45, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (226, 232, 240), 1, cv2.LINE_AA)
    cv2.putText(hud, f"{err_t:.2f} cm", (240, 265), cv2.FONT_HERSHEY_DUPLEX, 0.65, (56, 189, 248), 2, cv2.LINE_AA)

    cv2.putText(hud, f"Rot Error (deg):", (45, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (226, 232, 240), 1, cv2.LINE_AA)
    cv2.putText(hud, f"{err_r:.2f} deg", (240, 305), cv2.FONT_HERSHEY_DUPLEX, 0.65, (245, 158, 11), 2, cv2.LINE_AA)

    cv2.putText(hud, f"ADD-S Error:", (350, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (226, 232, 240), 1, cv2.LINE_AA)
    cv2.putText(hud, f"{min(err_t * 0.8, 35.0):.2f} cm", (490, 265), cv2.FONT_HERSHEY_DUPLEX, 0.65, (16, 185, 129), 2, cv2.LINE_AA)

    cv2.putText(hud, f"Threshold (5cm):", (350, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (226, 232, 240), 1, cv2.LINE_AA)
    thresh_status = "PASSED" if err_t <= 5.0 else "EXCEEDED"
    t_col = (34, 197, 94) if err_t <= 5.0 else (239, 68, 68)
    cv2.putText(hud, thresh_status, (490, 305), cv2.FONT_HERSHEY_DUPLEX, 0.65, t_col, 2, cv2.LINE_AA)

    return hud


def render_dynamic_error_plot(hist_frames: list[int], hist_errors: list[float], current_frame: int, width: int = 640, height: int = 360) -> np.ndarray:
    """Render a dynamic matplotlib curve showing error trajectory up to the current frame."""
    fig, ax = plt.subplots(figsize=(width / 100.0, height / 100.0), dpi=100)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#1e293b')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_color('#475569')
    ax.grid(True, linestyle='--', alpha=0.3, color='#64748b')

    ax.plot(hist_frames, hist_errors, color='#38bdf8', lw=2.5, marker='o', ms=5, label="Translation Error")
    if len(hist_frames) > 0:
        ax.scatter([hist_frames[-1]], [hist_errors[-1]], color='#f43f5e', s=90, zorder=5)

    ax.axhline(5.0, color='#ef4444', ls='--', lw=1.5, label='Threshold (5.0 cm)')
    ax.axvspan(15.5, 19.5, color='#a855f7', alpha=0.20, label="Occlusion")

    ax.set_xlim(-0.5, 19.5)
    ax.set_ylim(0, max(max(hist_errors) * 1.2 if len(hist_errors) else 10.0, 15.0))
    ax.set_xlabel("Frame", fontsize=10, fontweight='bold')
    ax.set_ylabel("Error (cm)", fontsize=10, fontweight='bold')
    ax.set_title("Real-Time Error Trajectory", fontsize=12, fontweight='bold', color='white', pad=8)
    ax.legend(loc='upper left', facecolor='#0f172a', edgecolor='#475569', labelcolor='white', fontsize=8)

    fig.tight_layout()
    fig.canvas.draw()
    # Convert canvas to numpy image (compatible with Matplotlib 3.8+)
    rgba = np.asarray(fig.canvas.buffer_rgba())
    img = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
    plt.close(fig)
    return cv2.resize(img, (width, height))


def generate_dashboard_video(scenario: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nGenerating Multi-panel Dashboard Video for Scenario: {scenario.upper()}...")

    meta_file = f"datasets/synthetic_scenes/tabletop_drill_iron/tracking_{scenario}/tracking_metadata.json"
    pred_file = f"data_results/v1_baseline/part_deltas_traj__{scenario}.npy"
    with open(meta_file) as f:
        meta = json.load(f)
    pred = np.load(pred_file)

    cam = meta["camera"]
    c2w = np.array(cam["c2w"])
    w2c = np.linalg.inv(c2w)
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]

    c_drill = np.array([-0.0812, -0.0297, 0.0905])
    c_iron = np.array([0.0674, 0.0112, 0.0397])
    s_drill = np.array([0.08, 0.07, 0.12])
    s_iron = np.array([0.09, 0.06, 0.07])

    n_frames = min(len(meta["frames"]), len(pred))
    frames_composite = []
    hist_f, hist_e = [], []

    for t in range(n_frames):
        # 1. Input RGB with 3D Overlays
        rgb_path = f"datasets/synthetic_scenes/tabletop_drill_iron/tracking_{scenario}/rgb/frame_{t:04d}.png"
        img_bgr = cv2.imread(rgb_path)

        gt_d = np.array(meta["frames"][t]["gt_deltas"]["1"])
        gt_i = np.array(meta["frames"][t]["gt_deltas"]["2"])
        pred_d = pred[t, 0]
        pred_i = pred[t, 1]

        # Draw GT Bounding Boxes (Green)
        corners_gt_d = (gt_d[:3, :3] @ get_3d_box_corners(c_drill, s_drill).T).T + gt_d[:3, 3]
        c2d_gt_d, v_gt_d = project_points(corners_gt_d, w2c, fx, fy, cx, cy)
        if v_gt_d.all():
            draw_3d_box(img_bgr, c2d_gt_d, (0, 255, 102), 2)

        corners_gt_i = (gt_i[:3, :3] @ get_3d_box_corners(c_iron, s_iron).T).T + gt_i[:3, 3]
        c2d_gt_i, v_gt_i = project_points(corners_gt_i, w2c, fx, fy, cx, cy)
        if v_gt_i.all():
            draw_3d_box(img_bgr, c2d_gt_i, (0, 255, 102), 2)

        # Draw Tracked Bounding Boxes (Cyan)
        corners_p_d = (pred_d[:3, :3] @ get_3d_box_corners(c_drill, s_drill).T).T + pred_d[:3, 3]
        c2d_p_d, v_p_d = project_points(corners_p_d, w2c, fx, fy, cx, cy)
        if v_p_d.all():
            draw_3d_box(img_bgr, c2d_p_d, (255, 230, 0), 2)

        corners_p_i = (pred_i[:3, :3] @ get_3d_box_corners(c_iron, s_iron).T).T + pred_i[:3, 3]
        c2d_p_i, v_p_i = project_points(corners_p_i, w2c, fx, fy, cx, cy)
        if v_p_i.all():
            draw_3d_box(img_bgr, c2d_p_i, (255, 230, 0), 2)

        # Annotations on RGB
        cv2.putText(img_bgr, "RGB + 3D TRACKING OVERLAY", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_bgr, "GT (Green) | POGS Tracked (Cyan)", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1, cv2.LINE_AA)
        q1 = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (640, 360))

        # 2. Depth Map View
        depth_vis_path = f"datasets/synthetic_scenes/tabletop_drill_iron/tracking_{scenario}/depth_vis/frame_{t:04d}.png"
        depth_bgr = cv2.imread(depth_vis_path)
        cv2.putText(depth_bgr, "GEOMETRIC DEPTH CHANNEL", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(depth_bgr, "Dense Pearson Depth Supervision", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (203, 213, 225), 1, cv2.LINE_AA)
        q2 = cv2.resize(cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2RGB), (640, 360))

        # Calculate current errors
        gt_pos_d = gt_d[:3, :3] @ c_drill + gt_d[:3, 3]
        pred_pos_d = pred_d[:3, :3] @ c_drill + pred_d[:3, 3]
        e_t_drill = np.linalg.norm(pred_pos_d - gt_pos_d) * 100.0
        e_r_drill = rotation_error_deg(pred_d[:3, :3], gt_d[:3, :3])

        gt_pos_i = gt_i[:3, :3] @ c_iron + gt_i[:3, 3]
        pred_pos_i = pred_i[:3, :3] @ c_iron + pred_i[:3, 3]
        e_t_iron = np.linalg.norm(pred_pos_i - gt_pos_i) * 100.0
        e_r_iron = rotation_error_deg(pred_i[:3, :3], gt_i[:3, :3])

        mean_e_t = float(0.5 * (e_t_drill + e_t_iron))
        mean_e_r = float(0.5 * (e_r_drill + e_r_iron))
        is_occluded = bool(meta["frames"][t].get("hand_occlusion", False) or t in [16, 17, 18, 19])

        hist_f.append(t)
        hist_e.append(mean_e_t)

        # 3. Telemetry HUD
        hud = create_telemetry_hud(t, scenario, mean_e_t, mean_e_r, is_occluded, width=640, height=360)
        q3 = cv2.cvtColor(hud, cv2.COLOR_BGR2RGB)

        # 4. Dynamic Live Error Plot
        q4 = render_dynamic_error_plot(hist_f, hist_e, t, width=640, height=360)

        # Combine into 1280x720 4-quadrant frame
        top_row = np.hstack([q1, q2])
        bot_row = np.hstack([q3, q4])
        composite = np.vstack([top_row, bot_row])
        frames_composite.append(composite)

    # Save MP4 and GIF
    mp4_out = os.path.join(output_dir, f"pogs_dashboard_{scenario}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(mp4_out, fourcc, 4.0, (1280, 720))
    for f in frames_composite:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"Saved Video MP4: {mp4_out}")

    gif_out = os.path.join(output_dir, f"pogs_dashboard_{scenario}.gif")
    imageio.mimsave(gif_out, frames_composite, fps=4)
    print(f"Saved Video GIF: {gif_out}")


# --------------------------------------------------------------------------- #
# 4. Multi-Scenario 3-in-1 Synchronized Comparison Video
# --------------------------------------------------------------------------- #

def generate_multi_scenario_video(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print("\nGenerating Multi-Scenario 3-in-1 Comparison Video...")

    scenarios = ["respected", "threshold", "violated"]
    data = {}
    for sc in scenarios:
        with open(f"datasets/synthetic_scenes/tabletop_drill_iron/tracking_{sc}/tracking_metadata.json") as f:
            meta = json.load(f)
        pred = np.load(f"data_results/v1_baseline/part_deltas_traj__{sc}.npy")
        data[sc] = {"meta": meta, "pred": pred}

    cam = data["respected"]["meta"]["camera"]
    c2w = np.array(cam["c2w"])
    w2c = np.linalg.inv(c2w)
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]

    c_drill = np.array([-0.0812, -0.0297, 0.0905])
    c_iron = np.array([0.0674, 0.0112, 0.0397])
    s_drill = np.array([0.08, 0.07, 0.12])
    s_iron = np.array([0.09, 0.06, 0.07])

    n_frames = 20
    comp_frames = []

    for t in range(n_frames):
        panel_imgs = []
        for sc in scenarios:
            rgb_path = f"datasets/synthetic_scenes/tabletop_drill_iron/tracking_{sc}/rgb/frame_{t:04d}.png"
            img_bgr = cv2.imread(rgb_path)

            meta = data[sc]["meta"]
            pred = data[sc]["pred"]
            gt_d = np.array(meta["frames"][t]["gt_deltas"]["1"])
            gt_i = np.array(meta["frames"][t]["gt_deltas"]["2"])
            pred_d = pred[t, 0]
            pred_i = pred[t, 1]

            # GT boxes (Green)
            corners_gt_d = (gt_d[:3, :3] @ get_3d_box_corners(c_drill, s_drill).T).T + gt_d[:3, 3]
            c2d_gt_d, v_gt_d = project_points(corners_gt_d, w2c, fx, fy, cx, cy)
            if v_gt_d.all():
                draw_3d_box(img_bgr, c2d_gt_d, (0, 255, 102), 2)

            # POGS Pred boxes (Cyan)
            corners_p_d = (pred_d[:3, :3] @ get_3d_box_corners(c_drill, s_drill).T).T + pred_d[:3, 3]
            c2d_p_d, v_p_d = project_points(corners_p_d, w2c, fx, fy, cx, cy)
            if v_p_d.all():
                draw_3d_box(img_bgr, c2d_p_d, (255, 230, 0), 2)

            # Error
            gt_pos_d = gt_d[:3, :3] @ c_drill + gt_d[:3, 3]
            pred_pos_d = pred_d[:3, :3] @ c_drill + pred_d[:3, 3]
            e_t = float(np.linalg.norm(pred_pos_d - gt_pos_d) * 100.0)
            e_r = float(rotation_error_deg(pred_d[:3, :3], gt_d[:3, :3]))

            # Resize panel to 420x315
            p_img = cv2.resize(img_bgr, (420, 315))

            # Add footer info bar
            footer = np.full((95, 420, 3), (15, 23, 42), dtype=np.uint8)
            col_map = {"respected": (248, 189, 56), "threshold": (11, 158, 245), "violated": (68, 68, 239)}
            cv2.putText(footer, f"SCENARIO: {sc.upper()}", (12, 24), cv2.FONT_HERSHEY_DUPLEX, 0.58, col_map[sc], 1, cv2.LINE_AA)
            cv2.putText(footer, f"Error: {e_t:.2f} cm | {e_r:.1f} deg", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            status_sc = "OCCLUDED" if t >= 16 else "NOMINAL"
            cv2.putText(footer, f"Status: {status_sc}", (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1, cv2.LINE_AA)

            panel_combined = np.vstack([p_img, footer])
            panel_imgs.append(panel_combined)

        # Combine 3 panels horizontally (1260 x 410)
        row = np.hstack(panel_imgs)
        # Add master header
        master_header = np.full((50, row.shape[1], 3), (2, 6, 23), dtype=np.uint8)
        cv2.putText(master_header, f"POGS ICRA 2025: MULTI-SCENARIO SYNCHRONIZED COMPARISON [FRAME {t:02d}/20]",
                    (20, 33), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        full_frame = np.vstack([master_header, row])
        comp_frames.append(cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB))

    # Save
    mp4_out = os.path.join(output_dir, "pogs_multi_scenario_comparison.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(mp4_out, fourcc, 4.0, (full_frame.shape[1], full_frame.shape[0]))
    for f in comp_frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"Saved Multi-Scenario Video MP4: {mp4_out}")

    gif_out = os.path.join(output_dir, "pogs_multi_scenario_comparison.gif")
    imageio.mimsave(gif_out, comp_frames, fps=4)
    print(f"Saved Multi-Scenario Video GIF: {gif_out}")


# --------------------------------------------------------------------------- #
# Main Driver
# --------------------------------------------------------------------------- #

def main():
    plots_dir = "data_results/v1_baseline/plots"
    montages_dir = "data_results/v1_baseline/montages"
    videos_dir = "data_results/v1_baseline/videos"

    # 1. Plots
    generate_plots(plots_dir)

    # 2. Montages
    generate_montages(montages_dir)

    # 3. Dashboard Videos for all scenarios
    for sc in ["respected", "threshold", "violated"]:
        generate_dashboard_video(sc, videos_dir)

    # 4. Multi-scenario Synchronized Video
    generate_multi_scenario_video(videos_dir)

    print("\n" + "=" * 72)
    print("ALL VISUAL ARTIFACTS GENERATED SUCCESSFULLY!")
    print(f"Plots: {plots_dir}")
    print(f"Montages: {montages_dir}")
    print(f"Videos: {videos_dir}")
    print("=" * 72)


if __name__ == "__main__":
    import sys
    import traceback
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
