"""
Pick-and-Place Photorealistic Dataset Generator (v1 - Base POGS Certified).

Scene Layout:
1. Large floor plane.
2. Small height table (staging table) on the floor at (0.18, 0.12, 0.0), top height z=0.060m.
3. 5 real-world objects resting on the floor (target: red ceramic coffee mug; distractors: mustard bottle, soup can, apple, cereal box).
4. Pure pick-and-place motion:
   - Smooth vertical lift from floor to clearance height.
   - Smooth horizontal transit to the small table with pure Z-axis yaw rotation (handle turns across).
   - Smooth descent and placement onto the small table surface.
5. Exact SE(3) pivot transform and ground truth bounding boxes.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from common.geometry import PinholeCamera, look_at
from scene_generator.capture_generator import dbscan_filter, hemisphere_trajectory, save_pointcloud_ply
from scene_generator.photoreal_scene import render_photoreal_scene
from scene_generator.pick_place_scene import build_pick_place_scene
from scene_generator.primitives import euler_to_R, se3, smoothstep
from scene_generator.scene_builder import FlatScene, deproject
from scene_generator.trajectory_planner import AngleConstraintStatus, TrajectoryScenario
from scene_generator.video_annotator import VideoAnnotator


def plan_pick_and_place_trajectory(
    scenario: TrajectoryScenario,
    n_frames: int = 20,
    start_pos: Tuple[float, float, float] = (-0.08, -0.06, 0.0),
    place_pos: Tuple[float, float, float] = (0.18, 0.12, 0.060),
    clearance_z: float = 0.15,
) -> List[Dict[str, any]]:
    """Generates smooth pick-and-place trajectories from floor to small table with pure Z-axis yaw rotation."""
    frames_info = []
    sx, sy, sz = start_pos
    px, py, pz = place_pos

    local_center = np.array([0.012, 0.0, 0.048])
    c_scan = np.array([sx, sy, sz]) + local_center

    for f in range(n_frames):
        u = f / max(n_frames - 1, 1)

        # 3-Phase Motion Profile:
        # Phase 1 [0.00 - 0.25]: Vertical lift from floor to clearance height
        # Phase 2 [0.25 - 0.75]: Horizontal transit across to table + pure Z-axis rotation (handle turns across)
        # Phase 3 [0.75 - 1.00]: Smooth descent onto table surface (z=0.060m)

        mug_pitch = 0.0  # Strictly 0 (upright)
        mug_roll = 0.0   # Strictly 0 (upright)

        if u <= 0.25:
            # Phase 1: Vertical Lift
            w = u / 0.25
            s = smoothstep(w)
            z_lift = sz + (clearance_z - sz) * s
            mug_pos = np.array([sx, sy, z_lift])
            mug_yaw = 0.0
            status = AngleConstraintStatus.RESPECTED
            desc = "Smooth Vertical Lift from Floor"

        elif u <= 0.75:
            # Phase 2: Horizontal Transit + Pure Z-Axis Yaw Rotation
            w = (u - 0.25) / 0.50
            s = smoothstep(w)
            tx = sx + (px - sx) * s
            ty = sy + (py - sy) * s
            tz = clearance_z + 0.015 * math.sin(math.pi * s)
            mug_pos = np.array([tx, ty, tz])

            # Rotation strictly around real world's Z-axis (upward/yaw)
            rot_envelope = math.sin(math.pi * s)
            if scenario == TrajectoryScenario.RESPECTED:
                yaw_deg = 18.0 * rot_envelope
                status = AngleConstraintStatus.RESPECTED
                desc = "Transit & Gentle Yaw Rotation (<=20°)"
            elif scenario == TrajectoryScenario.THRESHOLD:
                yaw_deg = 29.5 * rot_envelope
                status = AngleConstraintStatus.NEAR_THRESHOLD
                desc = "Transit & Near-Threshold Yaw (~29°)"
            else:  # VIOLATED
                yaw_deg = 90.0 * rot_envelope
                status = AngleConstraintStatus.VIOLATED
                desc = "Transit & Violated Yaw Rotation (>60°)"

            mug_yaw = math.radians(yaw_deg)

        else:
            # Phase 3: Descend onto Table Surface (z=0.060m)
            w = (u - 0.75) / 0.25
            s = smoothstep(w)
            tz = clearance_z + (pz - clearance_z) * s
            mug_pos = np.array([px, py, tz])
            mug_yaw = 0.0
            status = AngleConstraintStatus.RESPECTED
            desc = "Place on Small Staging Table Top"

        yaw_deg = float(math.degrees(mug_yaw))
        trans_cm = float(np.linalg.norm(mug_pos - np.array([sx, sy, sz])) * 100.0)

        # Delta transform using the exact SE(3) pivot formula:
        # Rotates primitives about the object's scan centroid c_scan, then translates
        R_delta = euler_to_R(mug_roll, mug_pitch, mug_yaw)
        t_trans = mug_pos - np.array([sx, sy, sz])
        delta_mat = se3(R_delta, c_scan - R_delta @ c_scan + t_trans)
        gt_pos = c_scan + t_trans

        frames_info.append({
            "frame": f,
            "timestamp_sec": f / 5.0,
            "yaw_deg": yaw_deg,
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
            "translation_cm": trans_cm,
            "constraint_status": status.value,
            "hand_occlusion": False,
            "is_lifted": u > 0.05,
            "description": desc,
            "mug_delta": delta_mat,
            "gt_pos": gt_pos,
            "c_scan": c_scan,
        })

    return frames_info


def generate_pick_place_dataset(
    output_dir: str = "datasets/synthetic_scenes/pick_place_floor_table",
    n_views: int = 35,
    n_frames: int = 20,
    width: int = 400,
    height: int = 300,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)
    capture_dir = os.path.join(output_dir, "capture")
    os.makedirs(capture_dir, exist_ok=True)

    print("=" * 76)
    print("GENERATING PHOTOREALISTIC PICK-AND-PLACE DATASET (v1 - Base POGS)")
    print(f"Output directory: {output_dir}")
    print(f"Views: {n_views} | Frames/Scenario: {n_frames} | Device: {device}")
    print("=" * 76)

    # 1. Base Scene with Floor, Small Table, and 5 Real Objects
    base_objects = build_pick_place_scene()
    flat_capture_scene = FlatScene(base_objects, device=device)

    # ----------------------------------------------------------------------- #
    # 2. Render Multi-View Hemisphere Captures
    # ----------------------------------------------------------------------- #
    print(f"\n--- 1. Rendering {n_views} Hemisphere Capture Views with Cast Shadows ---")
    t0 = time.time()
    cameras = hemisphere_trajectory(
        n_views=n_views,
        radius=0.68,
        elevation_deg=(30.0, 75.0),
        center=(0.02, 0.02, 0.08),
        width=width,
        height=height,
        fov_deg=62.0,
        device=device,
    )

    all_rgb, all_depth, all_mask = [], [], []
    cloud_pts, cloud_cols, cloud_ids = [], [], []

    for idx, cam in enumerate(cameras):
        out = render_photoreal_scene(flat_capture_scene, cam, cast_shadows=True)
        rgb = out["rgb"]
        depth = out["depth"]
        obj_id = out["obj_id"]

        # Realistic sensor depth noise
        noise = torch.randn_like(depth) * 0.003 * (depth / 0.7).clamp(min=0.2)
        noisy_depth = torch.where(depth > 0, depth + noise, depth)

        all_rgb.append(rgb.cpu())
        all_depth.append(noisy_depth.cpu())
        all_mask.append(obj_id.cpu())

        # Multi-rate subsampling: sample points across the workspace
        stride = 3
        deproj_pts = deproject(noisy_depth, cam).cpu().numpy()[::stride, ::stride].reshape(-1, 3)
        rgb_raw = rgb.cpu().numpy()[::stride, ::stride].reshape(-1, 3)
        id_raw = obj_id.cpu().numpy()[::stride, ::stride].reshape(-1)
        d_raw = noisy_depth.cpu().numpy()[::stride, ::stride].reshape(-1)

        v = (d_raw > 0.15) & (d_raw < 1.35)
        if v.any():
            cloud_pts.append(deproj_pts[v])
            cloud_cols.append(rgb_raw[v])
            cloud_ids.append(id_raw[v])

        if verbose and (idx % 10 == 0 or idx == n_views - 1):
            print(f"  [Capture] Rendered view {idx + 1:2d}/{n_views}")

    # Save capture dataset files
    images_np = (torch.stack(all_rgb, dim=0).numpy() * 255).astype(np.uint8)
    depths_np = torch.stack(all_depth, dim=0).numpy().astype(np.float32)
    masks_np = torch.stack(all_mask, dim=0).numpy().astype(np.int64)

    np.save(os.path.join(capture_dir, "images.npy"), images_np)
    np.save(os.path.join(capture_dir, "depths.npy"), depths_np)
    np.save(os.path.join(capture_dir, "obj_masks.npy"), masks_np)

    all_pts = np.concatenate(cloud_pts, axis=0)
    all_cols = np.concatenate(cloud_cols, axis=0)
    all_ids = np.concatenate(cloud_ids, axis=0)

    filtered_pts, filtered_cols, kept_mask = dbscan_filter(all_pts, all_cols, eps=0.018, min_samples=5, return_mask=True)
    filtered_ids = all_ids[kept_mask]

    # Balanced object allocation:
    # Target mug (id=1): up to 1200 points
    # Table (id=9): up to 1500 points
    # Floor & others: up to 3000 points
    rng = np.random.default_rng(0)
    final_indices = []

    for obj_val, max_count in [(1, 1200), (9, 1500)]:
        idx = np.where(filtered_ids == obj_val)[0]
        if len(idx) > max_count:
            idx = rng.choice(idx, max_count, replace=False)
        final_indices.append(idx)

    other_idx = np.where((filtered_ids != 1) & (filtered_ids != 9))[0]
    if len(other_idx) > 3000:
        other_idx = rng.choice(other_idx, 3000, replace=False)
    final_indices.append(other_idx)

    keep_idx = np.sort(np.concatenate(final_indices))
    filtered_pts, filtered_cols, filtered_ids = filtered_pts[keep_idx], filtered_cols[keep_idx], filtered_ids[keep_idx]

    ply_path = os.path.join(capture_dir, "fused_pointcloud.ply")
    save_pointcloud_ply(ply_path, filtered_pts, filtered_cols)
    np.save(os.path.join(capture_dir, "seed_obj_ids.npy"), filtered_ids)

    # Save camera metadata
    cam_meta = [
        {
            "view_idx": i,
            "c2w": c.c2w.cpu().tolist(),
            "fx": float(c.fx), "fy": float(c.fy),
            "cx": float(c.cx), "cy": float(c.cy),
            "width": c.width, "height": c.height,
        }
        for i, c in enumerate(cameras)
    ]
    with open(os.path.join(capture_dir, "cameras.json"), "w") as f:
        json.dump(cam_meta, f, indent=2)

    print(f"Capture completed in {time.time() - t0:.1f}s. Seed points: {len(filtered_pts)}")

    # ----------------------------------------------------------------------- #
    # 3. Generate Dynamic Pick-and-Place Scenarios (Respected, Threshold, Violated)
    # ----------------------------------------------------------------------- #
    print("\n--- 2. Generating Dynamic Pick-and-Place Sequences ---")

    # Front-quarter tracking camera (ZED 2 viewpoint)
    eye = np.array([0.22, -0.68, 0.52])
    target = np.array([0.00, 0.02, 0.08])
    up = np.array([0.0, 0.0, 1.0])
    c2w_track = torch.from_numpy(look_at(eye, target, up)).float()
    track_cam = PinholeCamera(c2w=c2w_track, fx=460.0, fy=460.0, cx=width / 2.0, cy=height / 2.0, width=width, height=height)

    scenarios = [
        TrajectoryScenario.RESPECTED,
        TrajectoryScenario.THRESHOLD,
        TrajectoryScenario.VIOLATED,
    ]

    target_obj_id = 1  # ceramic_mug

    for sc in scenarios:
        sc_name = sc.value
        print(f"\n---> Generating Scenario: {sc_name.upper()}")
        sc_dir = os.path.join(output_dir, f"tracking_{sc_name}")
        rgb_dir = os.path.join(sc_dir, "rgb")
        depth_npy_dir = os.path.join(sc_dir, "depth_npy")
        depth_vis_dir = os.path.join(sc_dir, "depth_vis")
        masks_dir = os.path.join(sc_dir, "masks")
        for d in (rgb_dir, depth_npy_dir, depth_vis_dir, masks_dir):
            os.makedirs(d, exist_ok=True)

        plan = plan_pick_and_place_trajectory(sc, n_frames=n_frames)

        frames_rgb, frames_annotated = [], []
        frames_meta = []

        for f, p in enumerate(plan):
            # Move target object (ceramic_mug) by delta across floor to table with pure Z yaw
            deltas = {target_obj_id: p["mug_delta"]}
            scene_f = FlatScene(base_objects, device=device).clone_with_transforms(deltas)

            out = render_photoreal_scene(scene_f, track_cam, cast_shadows=True)
            rgb_f = (out["rgb"].cpu().numpy() * 255).astype(np.uint8)
            d_f = out["depth"].cpu().numpy()
            m_f = out["obj_id"].cpu().numpy()

            # Save frame files
            f_name = f"frame_{f:04d}.png"
            cv2.imwrite(os.path.join(rgb_dir, f_name), cv2.cvtColor(rgb_f, cv2.COLOR_RGB2BGR))
            np.save(os.path.join(depth_npy_dir, f"frame_{f:04d}.npy"), d_f)
            cv2.imwrite(os.path.join(masks_dir, f_name), (m_f * 40).clip(0, 255).astype(np.uint8))

            # Depth visualization
            d_vis = cv2.applyColorMap((np.clip((d_f - 0.2) / 0.9, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            cv2.imwrite(os.path.join(depth_vis_dir, f_name), d_vis)

            frames_rgb.append(rgb_f)
            frames_meta.append({
                "frame": f,
                "timestamp_sec": p["timestamp_sec"],
                "yaw_deg": p["yaw_deg"],
                "pitch_deg": p["pitch_deg"],
                "roll_deg": p["roll_deg"],
                "translation_cm": p["translation_cm"],
                "constraint_status": p["constraint_status"],
                "hand_occlusion": False,
                "is_lifted": p["is_lifted"],
                "description": p["description"],
                "gt_deltas": {str(target_obj_id): p["mug_delta"].tolist()},
                "gt_pos": p["gt_pos"].tolist(),
                "c_scan": p["c_scan"].tolist(),
                "extents": [0.110, 0.080, 0.096],
                "local_center": [0.012, 0.0, 0.048],
            })

            # Annotated video frame
            ann_bgr = VideoAnnotator.render_hud_frame(
                rgb_hwc=rgb_f.astype(np.float32) / 255.0,
                depth_hw=d_f,
                scenario_name=sc_name,
                frame_idx=f,
                total_frames=n_frames,
                time_sec=p["timestamp_sec"],
                yaw_deg=p["yaw_deg"],
                trans_cm=p["translation_cm"],
                constraint_status=AngleConstraintStatus(p["constraint_status"]),
                hand_occlusion=False,
                reset_event=False,
                description=p["description"],
            )
            frames_annotated.append(ann_bgr)

            if verbose and (f % 10 == 0 or f == n_frames - 1):
                print(f"    Frame {f + 1:2d}/{n_frames} | yaw={p['yaw_deg']:+5.1f}° | trans={p['translation_cm']:4.1f}cm | {p['description']}")

        # Save metadata and video
        meta_dict = {
            "scenario": sc_name,
            "fps": 5.0,
            "n_frames": n_frames,
            "target_object": "ceramic_mug",
            "target_object_id": target_obj_id,
            "camera": {
                "c2w": track_cam.c2w.cpu().tolist(),
                "fx": float(track_cam.fx), "fy": float(track_cam.fy),
                "cx": float(track_cam.cx), "cy": float(track_cam.cy),
                "width": track_cam.width, "height": track_cam.height,
            },
            "frames": frames_meta,
        }
        with open(os.path.join(sc_dir, "tracking_metadata.json"), "w") as f:
            json.dump(meta_dict, f, indent=2)

        mp4_path = os.path.join(sc_dir, "tracking_annotated.mp4")
        gif_path = os.path.join(sc_dir, "tracking_annotated.gif")
        if frames_annotated:
            imageio.mimsave(mp4_path, [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_annotated], fps=5.0)
            imageio.mimsave(gif_path, [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_annotated[::2]], fps=3.0)

    print(f"\nAll datasets generated successfully in {time.time() - t0:.1f}s.")
    return output_dir


if __name__ == "__main__":
    generate_pick_place_dataset()
