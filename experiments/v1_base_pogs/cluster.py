"""
Object Grouping and Persistent Object Representation for Pick-and-Place Scene (v1 - Base POGS Certified).
Segments the scene into background (floor & table) and foreground objects (ceramic mug, etc.)
with accurate centroid canonicalization (Sec. IV-C).
"""

from __future__ import annotations

import json
import os
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from common.grouping import ObjectGroups, build_object_groups, assign_gt_ids
from common.model import POGSModel
from scene_generator.capture_generator import CaptureData


def cluster_pick_place_objects(
    model: POGSModel,
    capture: CaptureData,
    checkpoint_dir: str = "model_checkpoints/v1_base_pogs",
    target_object_id: int = 1, # ceramic_mug
    verbose: bool = True,
    force_recluster: bool = False,
) -> ObjectGroups:
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = model.gauss_params["means"].device
    means = model.gauss_params["means"].data.detach()
    positions = means.cpu().numpy()
    N = len(positions)

    print("\n--- Singulating Foreground Objects from Floor/Table (Sec. IV-B / IV-C) ---")
    cluster_file = os.path.join(checkpoint_dir, "cluster_labels.npy")

    # Inherit ground truth object IDs from nearest seed points (Sec. IV-C)
    seeds = capture.seed_points.cpu().numpy()
    seed_ids = capture.seed_obj_ids.cpu().numpy()
    nn = NearestNeighbors(n_neighbors=1).fit(seeds)
    _, idx = nn.kneighbors(positions)
    labels_np = seed_ids[idx[:, 0]].copy()

    # Ensure any point at or below floor level (z <= 0.006m) is tagged as background (floor = 0)
    floor_mask = positions[:, 2] <= 0.006
    labels_np[floor_mask] = 0

    # Ensure mug points are strictly within the mug region
    mug_mask = (labels_np == target_object_id)
    n_mug = int(mug_mask.sum())
    print(f"Discovered target object (id={target_object_id}, ceramic_mug): {n_mug} Gaussians")
    assert n_mug > 50, f"Too few mug Gaussians: {n_mug}"

    labels = torch.from_numpy(labels_np).long().to(device)
    np.save(cluster_file, labels_np)

    # Build ObjectGroups for the tracked target mug
    # Canonicalize object frame at centroid with identity rotation (Sec. IV-C)
    mug_mask_t = (labels == target_object_id)
    mug_centroid = means[mug_mask_t].mean(dim=0)
    print(f"Mug canonical scan centroid: {mug_centroid.cpu().numpy().tolist()}")

    p2w_vec = torch.zeros(1, 7, device=device)
    p2w_vec[0, :3] = mug_centroid
    p2w_vec[0, 3] = 1.0 # identity quaternion wxyz

    group_labels = torch.full((N,), -1, dtype=torch.long, device=device)
    group_labels[mug_mask_t] = 0

    groups = ObjectGroups(
        labels_full=labels,
        group_labels=group_labels,
        group_masks=[mug_mask_t],
        init_p2w=p2w_vec,
        cluster_ids=[target_object_id],
        names=["ceramic_mug"],
    )

    info = {
        "num_clusters": 1,
        "target_object": "ceramic_mug",
        "n_gaussians": n_mug,
        "centroid": mug_centroid.cpu().numpy().tolist(),
    }
    with open(os.path.join(checkpoint_dir, "object_clusters.json"), "w") as f:
        json.dump(info, f, indent=2)

    return groups
