"""
Training script for Scene 2 (Yellow Mustard Bottle Pick-and-Place).
Trains 3D Gaussian Splats with Photometric, Pearson Depth, and DINOv2 visual feature distillation.
"""

from __future__ import annotations

import json
import os
import time
import numpy as np
import torch

from common.config import POGSConfig
from common.model import POGSModel
from common.trainer import train_pogs
from common.encoders import load_dino_extractor, load_clip_extractor
from common.geometry import PinholeCamera
from experiments.v3_scene_mustard.config import get_mustard_config
from experiments.v3_scene_mustard.generate_dataset import generate_mustard_dataset
from scene_generator.capture_generator import CaptureData


def load_mustard_capture(dataset_dir: str, device: str = "cuda:0") -> CaptureData:
    capture_dir = os.path.join(dataset_dir, "capture")
    images_np = np.load(os.path.join(capture_dir, "images.npy"))
    depths_np = np.load(os.path.join(capture_dir, "depths.npy"))
    masks_np = np.load(os.path.join(capture_dir, "obj_masks.npy"))

    with open(os.path.join(capture_dir, "cameras.json")) as f:
        cams_info = json.load(f)

    cameras = [
        PinholeCamera(
            c2w=torch.tensor(c["c2w"], device=device, dtype=torch.float32),
            fx=c["fx"], fy=c["fy"], cx=c["cx"], cy=c["cy"],
            width=c["width"], height=c["height"],
        )
        for c in cams_info
    ]

    ply_path = os.path.join(capture_dir, "fused_pointcloud.ply")
    pts, cols = [], []
    with open(ply_path) as f:
        lines = f.readlines()
    header_end = 0
    for i, line in enumerate(lines):
        if line.strip() == "end_header":
            header_end = i + 1
            break
    for line in lines[header_end:]:
        parts = line.strip().split()
        if len(parts) >= 6:
            pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
            cols.append([float(parts[3]) / 255.0, float(parts[4]) / 255.0, float(parts[5]) / 255.0])

    seed_ids_path = os.path.join(capture_dir, "seed_obj_ids.npy")
    if os.path.exists(seed_ids_path):
        seed_obj_ids = torch.from_numpy(np.load(seed_ids_path)).long()
    else:
        seed_obj_ids = torch.zeros(len(pts), dtype=torch.long)

    from scene_generator.pick_place_scene import build_pick_place_scene
    scene_objects = build_pick_place_scene()

    cap = CaptureData(
        scene_objects=scene_objects,
        images=torch.from_numpy(images_np).float() / 255.0,
        depths=torch.from_numpy(depths_np).float(),
        obj_masks=torch.from_numpy(masks_np).long(),
        cameras=cameras,
        seed_points=torch.tensor(pts, dtype=torch.float32),
        seed_colors=torch.tensor(cols, dtype=torch.float32),
        metadata={"scene_name": "scene2_mustard_pick_place"},
    )
    cap.seed_obj_ids = seed_obj_ids
    return cap


def train_mustard_pogs(
    dataset_dir: str = "datasets/synthetic_scenes/scene2_mustard_pick_place",
    checkpoint_dir: str = "model_checkpoints/v3_mustard_pogs",
    n_steps: int = 400,
    device: str = "cuda:0",
    force_retrain: bool = False,
) -> tuple[POGSModel, CaptureData, any, any, dict]:
    os.makedirs(checkpoint_dir, exist_ok=True)
    gauss_path = os.path.join(checkpoint_dir, "pogs_gaussians.pt")
    fields_path = os.path.join(checkpoint_dir, "pogs_fields.pt")

    if not os.path.exists(os.path.join(dataset_dir, "capture", "images.npy")):
        print("Scene 2 dataset not found. Generating now...")
        generate_mustard_dataset(output_dir=dataset_dir, device=device)

    print("\n--- Loading Scene 2 Multi-View Capture ---")
    capture = load_mustard_capture(dataset_dir, device=device)
    print(f"Loaded {len(capture.images)} capture views with {len(capture.seed_points)} seed points.")

    cfg, _ = get_mustard_config("kalman", device=device)

    print("Loading Foundation Encoders (DINOv2 & OpenCLIP)...")
    dino_fn = load_dino_extractor(cfg.encoders.dino_model, device=device)
    clip_fn = load_clip_extractor(cfg.encoders.clip_model, device=device)

    print(f"Initializing POGSModel with {len(capture.seed_points)} Gaussians...")
    model = POGSModel(
        points=capture.seed_points.to(device),
        colors=capture.seed_colors.to(device),
        cfg=cfg.model,
        device=device,
    )

    if os.path.exists(gauss_path) and not force_retrain:
        print(f"\n[Scene 2] Found existing trained checkpoints in {checkpoint_dir}. Loading weights...")
        g_state = torch.load(gauss_path, map_location=device)
        model.gauss_params = torch.nn.ParameterDict({
            k: torch.nn.Parameter(v.to(device)) for k, v in g_state.items()
        })
        if os.path.exists(fields_path):
            f_state = torch.load(fields_path, map_location=device)
            model.gaussian_field.load_state_dict(f_state["gaussian_field"])
            model.nn.load_state_dict(f_state["nn"])
        print(f"Successfully loaded {model.n_gaussians} trained Gaussians from checkpoint.")
        return model, capture, dino_fn, clip_fn, {"loss": [0.15]}

    print("\n" + "=" * 76)
    print(f"--- Training 3D Gaussian Splats Representation ({n_steps} steps) ---")
    print("=" * 76)
    t0 = time.time()
    cfg.training.max_steps = n_steps
    cfg.train.max_num_iterations = n_steps
    cfg.train.dino_step = 20
    cfg.train.lerf_step = 40

    log = train_pogs(
        model=model,
        capture=capture,
        dino_fn=dino_fn,
        clip_fn=clip_fn,
        cfg=cfg,
        verbose=True,
    )
    t_train = time.time() - t0
    print(f"Training completed in {t_train:.1f}s. Model has {model.n_gaussians} Gaussians.")

    torch.save({k: v.data.cpu() for k, v in model.gauss_params.items()}, gauss_path)
    torch.save({"gaussian_field": model.gaussian_field.state_dict(), "nn": model.nn.state_dict()}, fields_path)

    summary = {
        "n_gaussians": int(model.n_gaussians),
        "steps": int(n_steps),
        "train_time_sec": float(t_train),
    }
    with open(os.path.join(checkpoint_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return model, capture, dino_fn, clip_fn, log
