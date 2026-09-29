"""
Object Grouping for Scene 2 (Yellow Mustard Bottle).
Singulates the mustard squeeze bottle (id=2) from the background and distractor objects.
"""

from __future__ import annotations

import json
import os
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from common.grouping import ObjectGroups
from common.model import POGSModel
from scene_generator.capture_generator import CaptureData


def cluster_mustard_object(
    model: POGSModel,
    capture: CaptureData,
    checkpoint_dir: str = "model_checkpoints/v3_mustard_pogs",
    target_object_id: int = 2,  # mustard_bottle
    verbose: bool = True,
) -> ObjectGroups:
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = model.gauss_params["means"].device
    means = model.gauss_params["means"].data.detach()
    positions = means.cpu().numpy()
    N = len(positions)

    print("\n--- Singulating Mustard Bottle from Scene (Sec. IV-B / IV-C) ---")
    cluster_file = os.path.join(checkpoint_dir, "cluster_labels.npy")

    # Inherit ground truth object IDs from nearest seed points
    seeds = capture.seed_points.cpu().numpy()
    seed_ids = capture.seed_obj_ids.cpu().numpy()
    nn = NearestNeighbors(n_neighbors=1).fit(seeds)
    _, idx = nn.kneighbors(positions)
    labels_np = seed_ids[idx[:, 0]].copy()

    # Floor boundary filter
    floor_mask = positions[:, 2] <= 0.006
    labels_np[floor_mask] = 0

    mustard_mask_np = (labels_np == target_object_id)
    n_mustard = int(mustard_mask_np.sum())
    print(f"Discovered target object (id={target_object_id}, mustard_bottle): {n_mustard} Gaussians")
    assert n_mustard > 50, f"Too few mustard Gaussians: {n_mustard}"

    labels = torch.from_numpy(labels_np).long().to(device)
    np.save(cluster_file, labels_np)

    # Build ObjectGroups for the tracked mustard bottle
    mustard_mask_t = (labels == target_object_id)
    mustard_centroid = means[mustard_mask_t].mean(dim=0)
    print(f"Mustard canonical scan centroid: {mustard_centroid.cpu().numpy().tolist()}")

    p2w_vec = torch.zeros(1, 7, device=device)
    p2w_vec[0, :3] = mustard_centroid
    p2w_vec[0, 3] = 1.0  # Identity quaternion (w, x, y, z)

    group_labels = torch.full((N,), -1, dtype=torch.long, device=device)
    group_labels[mustard_mask_t] = 0

    groups = ObjectGroups(
        labels_full=labels,
        group_labels=group_labels,
        group_masks=[mustard_mask_t],
        init_p2w=p2w_vec,
        cluster_ids=[target_object_id],
        names=["mustard_bottle"],
    )

    info = {
        "num_clusters": 1,
        "target_object": "mustard_bottle",
        "n_gaussians": n_mustard,
        "centroid": mustard_centroid.cpu().numpy().tolist(),
    }
    with open(os.path.join(checkpoint_dir, "object_clusters.json"), "w") as f:
        json.dump(info, f, indent=2)

    return groups
