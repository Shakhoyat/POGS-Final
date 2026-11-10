"""
Phase 1 Multi-View Hemisphere Capture Generator.
Produces multi-view RGB-D images, camera extrinsics, instance masks, and a
DBSCAN-filtered fused 3D point cloud for initializing POGS Gaussians.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np
import torch

from .scene_builder import FlatScene, render_scene, deproject
from .objects import SceneObject
from common.geometry import PinholeCamera, look_at


@dataclass
class CaptureData:
    scene_objects: List[SceneObject]
    images: torch.Tensor          # (V, H, W, 3) in [0, 1]
    depths: torch.Tensor          # (V, H, W) in meters
    obj_masks: torch.Tensor       # (V, H, W) integer object IDs
    cameras: List[PinholeCamera]  # length V
    seed_points: torch.Tensor     # (N, 3) fused 3D point cloud
    seed_colors: torch.Tensor     # (N, 3) in [0, 1]
    metadata: Dict


def hemisphere_trajectory(
    n_views: int = 35,
    radius: float = 0.55,
    elevation_deg: Tuple[float, float] = (25.0, 75.0),
    center: Tuple[float, float, float] = (0.0, 0.0, 0.045),
    width: int = 400,
    height: int = 300,
    fov_deg: float = 60.0,
    device: str = "cpu",
) -> List[PinholeCamera]:
    """Generates 35 viewpoints arranged along a hemisphere above the tabletop."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0  # golden ratio
    el_min, el_max = math.radians(elevation_deg[0]), math.radians(elevation_deg[1])
    center_pt = np.asarray(center, dtype=np.float64)
    cams = []

    for i in range(n_views):
        u = (i + 0.5) / n_views
        el = el_min + (el_max - el_min) * u
        az = 2.0 * math.pi * (i / phi)

        eye = center_pt + radius * np.array([
            math.cos(el) * math.cos(az),
            math.cos(el) * math.sin(az),
            math.sin(el),
        ])
        c2w = look_at(eye, center_pt, up=np.array([0.0, 0.0, 1.0]))
        cams.append(
            PinholeCamera.from_fov(
                torch.as_tensor(c2w, dtype=torch.float32, device=device),
                fov_deg, width, height
            )
        )
    return cams


def dbscan_filter(
    points: np.ndarray, colors: np.ndarray, eps: float = 0.015, min_samples: int = 8,
    return_mask: bool = False, min_cluster_points: int = 30,
):
    """Sec. IV-A: 'We run DBSCAN on the fused pointcloud to filter noise and
    floater points.' Drops DBSCAN noise and tiny floater clusters; every real
    surface (floor, table, every object) is kept."""
    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    valid = labels >= 0
    if not valid.any():
        if return_mask:
            return points, colors, np.ones(len(points), dtype=bool)
        return points, colors
    u_labels, counts = np.unique(labels[valid], return_counts=True)
    kept_labels = u_labels[counts >= min_cluster_points]
    keep = np.isin(labels, kept_labels)
    if return_mask:
        return points[keep], colors[keep], keep
    return points[keep], colors[keep]


def save_pointcloud_ply(filepath: str, points: np.ndarray, colors: np.ndarray):
    """Write point cloud to ASCII PLY file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pts = points.astype(np.float32)
    cols = (np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8)
    N = len(pts)

    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {N}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(filepath, "w", encoding="ascii") as f:
        f.write(header)
        for i in range(N):
            f.write(f"{pts[i, 0]:.6f} {pts[i, 1]:.6f} {pts[i, 2]:.6f} {cols[i, 0]} {cols[i, 1]} {cols[i, 2]}\n")


class CaptureGenerator:
    """Orchestrates Phase 1 scene capture and point cloud synthesis."""

    def __init__(self, device: str = "cpu"):
        self.device = device

    @torch.no_grad()
    def generate(
        self,
        scene_objects: List[SceneObject],
        n_views: int = 35,
        width: int = 400,
        height: int = 300,
        fov_deg: float = 60.0,
        depth_noise_std: float = 0.003,
        supersample: int = 1,
        verbose: bool = True,
    ) -> CaptureData:
        scene = FlatScene(scene_objects, device=self.device)
        cameras = hemisphere_trajectory(
            n_views=n_views,
            radius=0.52,
            elevation_deg=(25.0, 75.0),
            center=(0.0, 0.0, 0.045),
            width=width,
            height=height,
            fov_deg=fov_deg,
            device=self.device,
        )

        all_rgb, all_depth, all_mask = [], [], []
        cloud_pts, cloud_cols = [], []

        for idx, cam in enumerate(cameras):
            out = render_scene(scene, cam, supersample=supersample)
            rgb = out["rgb"]
            depth = out["depth"]
            obj_id = out["obj_id"]

            # Add sensor depth noise (increasing with depth, matching ZED Mini specs)
            noise = torch.randn_like(depth) * depth_noise_std * (depth / 0.6).clamp(min=0.2)
            noisy_depth = torch.where(depth > 0, depth + noise, depth)

            all_rgb.append(rgb.cpu())
            all_depth.append(noisy_depth.cpu())
            all_mask.append(obj_id.cpu())

            # Subsample for fused point cloud initialization
            stride = 6
            world_pts = deproject(noisy_depth, cam).cpu().numpy()[::stride, ::stride].reshape(-1, 3)
            rgb_pts = rgb.cpu().numpy()[::stride, ::stride].reshape(-1, 3)
            d_flat = noisy_depth.cpu().numpy()[::stride, ::stride].reshape(-1)
            valid = (d_flat > 0.15) & (d_flat < 1.1)
            cloud_pts.append(world_pts[valid])
            cloud_cols.append(rgb_pts[valid])

            if verbose and (idx % 10 == 0 or idx == n_views - 1):
                print(f"  [Capture] Rendered view {idx + 1}/{n_views}")

        images_t = torch.stack(all_rgb, dim=0)
        depths_t = torch.stack(all_depth, dim=0)
        masks_t = torch.stack(all_mask, dim=0)

        all_pts = np.concatenate(cloud_pts, axis=0)
        all_cols = np.concatenate(cloud_cols, axis=0)

        # DBSCAN filtering
        filtered_pts, filtered_cols = dbscan_filter(all_pts, all_cols, eps=0.012, min_samples=6)
        if len(filtered_pts) > 6000:
            sub_idx = np.random.default_rng(0).choice(len(filtered_pts), 6000, replace=False)
            filtered_pts = filtered_pts[sub_idx]
            filtered_cols = filtered_cols[sub_idx]

        metadata = {
            "n_views": n_views,
            "width": width,
            "height": height,
            "fov_deg": fov_deg,
            "depth_noise_std": depth_noise_std,
            "num_seed_points": len(filtered_pts),
            "object_ids": [obj.obj_id for obj in scene_objects if obj.movable],
            "object_names": [obj.name for obj in scene_objects if obj.movable],
        }

        return CaptureData(
            scene_objects=scene_objects,
            images=images_t,
            depths=depths_t,
            obj_masks=masks_t,
            cameras=cameras,
            seed_points=torch.from_numpy(filtered_pts).float(),
            seed_colors=torch.from_numpy(filtered_cols).float(),
            metadata=metadata,
        )

    def export(self, capture_data: CaptureData, output_dir: str):
        """Saves all capture artifacts to disk for user inspection."""
        rgb_dir = os.path.join(output_dir, "rgb")
        depth_npy_dir = os.path.join(output_dir, "depth_npy")
        depth_vis_dir = os.path.join(output_dir, "depth_vis")
        masks_dir = os.path.join(output_dir, "masks")

        for d in [rgb_dir, depth_npy_dir, depth_vis_dir, masks_dir]:
            os.makedirs(d, exist_ok=True)

        poses_data = []
        V = capture_data.images.shape[0]

        for i in range(V):
            prefix = f"view_{i:03d}"

            # RGB Image
            rgb_np = (capture_data.images[i].numpy() * 255.0).astype(np.uint8)
            cv2.imwrite(os.path.join(rgb_dir, f"{prefix}.png"), cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))

            # Depth map (raw float32 meters)
            depth_np = capture_data.depths[i].numpy().astype(np.float32)
            np.save(os.path.join(depth_npy_dir, f"{prefix}.npy"), depth_np)

            # Colormapped Depth map (Turbo colormap for visualization)
            d_norm = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colored = cv2.applyColorMap(d_norm, cv2.COLORMAP_TURBO)
            cv2.imwrite(os.path.join(depth_vis_dir, f"{prefix}.png"), depth_colored)

            # Instance Mask
            mask_np = (capture_data.obj_masks[i].numpy() * 60).clip(0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(masks_dir, f"{prefix}.png"), mask_np)

            cam = capture_data.cameras[i]
            poses_data.append({
                "view_idx": i,
                "c2w": cam.c2w.cpu().numpy().tolist(),
                "fx": cam.fx,
                "fy": cam.fy,
                "cx": cam.cx,
                "cy": cam.cy,
                "width": cam.width,
                "height": cam.height,
            })

        # Save poses
        with open(os.path.join(output_dir, "poses.json"), "w", encoding="utf-8") as f:
            json.dump(poses_data, f, indent=2)

        # Save fused point cloud PLY
        save_pointcloud_ply(
            os.path.join(output_dir, "fused_pointcloud.ply"),
            capture_data.seed_points.numpy(),
            capture_data.seed_colors.numpy(),
        )

        # Save metadata
        with open(os.path.join(output_dir, "capture_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(capture_data.metadata, f, indent=2)

        print(f"  [Capture] Exported {V} views and PLY point cloud to: {output_dir}")
