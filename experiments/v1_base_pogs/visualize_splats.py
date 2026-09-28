"""
Render Initial 3D Gaussian World Splats and Isolated Target Object Splats (v1 - Base POGS).
Generates high-fidelity renders of:
1. The full 3D Gaussian scene at capture time.
2. The isolated 3D Gaussian splat of the target ceramic mug with espresso liquid and handle.
"""

from __future__ import annotations

import json
import os
import cv2
import numpy as np
import torch

from common.geometry import PinholeCamera, look_at, quat_normalize, se3_compose, se3_inverse
from common.model import POGSModel
from common.raster import rasterize
from experiments.v1_base_pogs.train import train_pick_place_pogs
from experiments.v1_base_pogs.cluster import cluster_pick_place_objects


def render_full_scene_splats(
    model: POGSModel,
    c2w: torch.Tensor,
    width: int = 640,
    height: int = 480,
    fov_deg: float = 58.0,
) -> np.ndarray:
    """Renders the entire trained 3D Gaussian scene from camera c2w."""
    device = model.gauss_params["means"].device
    cam = PinholeCamera.from_fov(c2w.to(device), fov_deg, width, height)
    with torch.no_grad():
        out = model.get_outputs(cam, rgb_only=True)
    rgb = out["rgb"].cpu().numpy()
    return (rgb * 255).astype(np.uint8)


def render_isolated_target_splats(
    model: POGSModel,
    groups,
    part_delta_7vec: torch.Tensor,
    c2w: Optional[torch.Tensor] = None,
    width: int = 640,
    height: int = 480,
    fov_deg: float = 45.0,
    camera: Optional[PinholeCamera] = None,
    group_idx: int = 0,
    background: tuple[float, float, float] = (0.07, 0.08, 0.11), # dark studio backdrop
) -> np.ndarray:
    """Renders ONLY the isolated 3D Gaussians of the target object transformed by part_delta."""
    device = model.gauss_params["means"].device
    if camera is None:
        cam = PinholeCamera.from_fov(c2w.to(device), fov_deg, width, height)
    else:
        cam = camera.to(device)

    mug_mask = groups.group_masks[group_idx]
    means = model.gauss_params["means"].data[mug_mask]
    quats = model.gauss_params["quats"].data[mug_mask]
    scales = torch.exp(model.gauss_params["scales"].data[mug_mask])
    opacities = torch.sigmoid(model.gauss_params["opacities"].data[mug_mask]).squeeze(-1)
    colors = model.colors_sh[mug_mask]

    # Transform geometry relative to object canonical frame
    p2w = groups.init_p2w[group_idx] # (7,)
    g2w = torch.cat([means, quat_normalize(quats)], dim=-1)
    g2p = se3_compose(se3_inverse(p2w.unsqueeze(0)), g2w)
    new_g2w = se3_compose(se3_compose(p2w.unsqueeze(0), part_delta_7vec.to(device).unsqueeze(0)), g2p)

    means_t = new_g2w[:, :3]
    quats_t = new_g2w[:, 3:]

    if isinstance(background, (tuple, list)):
        bg_t = torch.tensor(background, dtype=torch.float32, device=device)
    else:
        bg_t = background.to(device)

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


def generate_splat_visualizations(
    output_dir: str = "data_results/v1_base_pogs/splat_renders",
    checkpoint_dir: str = "model_checkpoints/v1_base_pogs",
    dataset_dir: str = "datasets/synthetic_scenes/pick_place_floor_table",
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 76)
    print("RENDERING 3D GAUSSIAN WORLD SPLATS & ISOLATED OBJECT SPLATS")
    print("=" * 76)

    model, capture, _, _, _ = train_pick_place_pogs(
        dataset_dir=dataset_dir,
        checkpoint_dir=checkpoint_dir,
        device=device,
        force_retrain=False,
    )

    groups = cluster_pick_place_objects(
        model=model,
        capture=capture,
        checkpoint_dir=checkpoint_dir,
        target_object_id=1,
    )

    # 1. Full 3D World Splats from 3 canonical viewpoints
    views = [
        ("world_front_quarter", np.array([0.28, -0.65, 0.48]), np.array([0.02, 0.04, 0.06])),
        ("world_overhead", np.array([0.02, -0.15, 0.75]), np.array([0.02, 0.02, 0.05])),
        ("world_side_view", np.array([-0.60, -0.05, 0.35]), np.array([0.05, 0.04, 0.06])),
    ]

    world_renders = []
    for name, eye, target in views:
        c2w = torch.from_numpy(look_at(eye, target, np.array([0.0, 0.0, 1.0]))).float()
        img_rgb = render_full_scene_splats(model, c2w, width=640, height=480)
        out_path = os.path.join(output_dir, f"{name}.png")
        cv2.imwrite(out_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        world_renders.append(img_rgb)
        print(f"  Saved Full Scene World Splat: {out_path}")

    # Combine into a single 3-view panorama
    pano = np.hstack(world_renders)
    cv2.imwrite(os.path.join(output_dir, "world_splats_panorama.png"), cv2.cvtColor(pano, cv2.COLOR_RGB2BGR))

    # 2. Isolated Object Splats of Ceramic Coffee Mug at Initial Scan Pose
    c_scan = groups.init_p2w[0, :3].cpu().numpy()
    mug_cam_eye = c_scan + np.array([0.18, -0.22, 0.16])
    c2w_mug = torch.from_numpy(look_at(mug_cam_eye, c_scan, np.array([0.0, 0.0, 1.0]))).float()

    identity_delta = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], device=device) # [t, quat_wxyz]
    mug_splat = render_isolated_target_splats(
        model=model,
        groups=groups,
        part_delta_7vec=identity_delta,
        c2w=c2w_mug,
        width=640,
        height=480,
        fov_deg=35.0,
    )
    cv2.imwrite(os.path.join(output_dir, "isolated_ceramic_mug_splat.png"), cv2.cvtColor(mug_splat, cv2.COLOR_RGB2BGR))
    print(f"  Saved Isolated Target Object Splat: {os.path.join(output_dir, 'isolated_ceramic_mug_splat.png')}")

    print("\nAll splat visualizations rendered successfully!")


if __name__ == "__main__":
    generate_splat_visualizations()
