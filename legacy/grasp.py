"""
Grasp sampling on the persistent object geometry (Sec. IV-E).

    "The Gaussian means representing that object cluster are passed as a point
     cloud to Contact-GraspNet, which generates potential grasp candidates
     along with their respective scores, and the highest scored grasp is
     executed. By using the Gaussian means, the object grasp is based on the
     full 3D object geometry embedded in POGS, which can be beneficial compared
     to methods that solely deproject depth and are partially occluded as seen
     in Figure 3."

Contact-GraspNet needs TensorFlow 2.5 / CUDA 11.2 / Python 3.8 in a separate
conda environment (see the official README), which cannot exist inside a Kaggle
session.  This module substitutes a geometric **antipodal sampler** with the
same interface and the same output contract as the official wrapper:

    pred_grasps_world.npy   (K, 4, 4) SE(3) gripper poses
    grasp_point_world.npy   (4, 4)    the highest-scoring grasp

The point the paper is actually making does not depend on which sampler runs:
what matters is that the *input point cloud* is POGS's completed geometry
rather than a single depth deprojection.  ``compare_grasp_sources`` measures
exactly that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Local geometry
# --------------------------------------------------------------------------- #


def estimate_normals(points: np.ndarray, k: int = 24, viewpoint: Optional[np.ndarray] = None) -> np.ndarray:
    """Local-PCA surface normals, optionally oriented away from a viewpoint."""
    from sklearn.neighbors import NearestNeighbors

    k = min(k, max(len(points) - 1, 1))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(points)
    _, idx = nn.kneighbors(points)
    nbrs = points[idx]                       # (N, k+1, 3)
    c = nbrs - nbrs.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", c, c) / max(k, 1)
    _, vecs = np.linalg.eigh(cov)
    n = vecs[:, :, 0]                        # smallest eigenvector
    if viewpoint is not None:
        d = viewpoint[None] - points
        flip = (n * d).sum(-1) < 0
        n[flip] *= -1
    return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9)


@dataclass
class GraspSet:
    poses: np.ndarray       # (K, 4, 4)
    scores: np.ndarray      # (K,)
    widths: np.ndarray      # (K,)
    contacts: np.ndarray    # (K, 2, 3)

    def best(self) -> np.ndarray:
        return self.poses[int(np.argmax(self.scores))] if len(self.poses) else np.eye(4)

    def top(self, n: int) -> "GraspSet":
        order = np.argsort(-self.scores)[:n]
        return GraspSet(self.poses[order], self.scores[order], self.widths[order], self.contacts[order])

    def __len__(self) -> int:
        return len(self.poses)


# --------------------------------------------------------------------------- #
# Antipodal sampling
# --------------------------------------------------------------------------- #


def sample_antipodal_grasps(
    points: np.ndarray,
    gripper_min_width: float = 0.012,
    gripper_max_width: float = 0.085,
    friction_cos: float = 0.80,
    n_candidates: int = 20000,
    approach_up: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    table_z: float = 0.0,
    min_clearance: float = 0.010,
    seed: int = 0,
    normals: Optional[np.ndarray] = None,
) -> GraspSet:
    """Parallel-jaw grasps from an (unordered, possibly occluded-completed)
    point cloud.

    A pair (p_a, p_b) is antipodal when the line joining them lies inside both
    contacts' friction cones:  |n_a . u| >= mu-cos  and  |n_b . u| >= mu-cos,
    with u the unit vector from a to b, and the normals opposing.
    """
    rng = np.random.default_rng(seed)
    P = np.asarray(points, dtype=np.float64)
    if len(P) < 32:
        return GraspSet(np.zeros((0, 4, 4)), np.zeros(0), np.zeros(0), np.zeros((0, 2, 3)))

    N = normals if normals is not None else estimate_normals(P)
    centroid = P.mean(axis=0)

    ia = rng.integers(0, len(P), n_candidates)
    ib = rng.integers(0, len(P), n_candidates)
    a, b = P[ia], P[ib]
    na, nb = N[ia], N[ib]

    d = b - a
    w = np.linalg.norm(d, axis=-1)
    ok = (w > gripper_min_width) & (w < gripper_max_width)
    if not ok.any():
        return GraspSet(np.zeros((0, 4, 4)), np.zeros(0), np.zeros(0), np.zeros((0, 2, 3)))
    u = d / np.maximum(w, 1e-9)[:, None]

    align_a = np.abs((na * u).sum(-1))
    align_b = np.abs((nb * u).sum(-1))
    opposing = (na * nb).sum(-1) < 0.0
    ok &= (align_a > friction_cos) & (align_b > friction_cos) & opposing
    if not ok.any():
        return GraspSet(np.zeros((0, 4, 4)), np.zeros(0), np.zeros(0), np.zeros((0, 2, 3)))

    a, b, u, w = a[ok], b[ok], u[ok], w[ok]
    align = 0.5 * (align_a[ok] + align_b[ok])
    center = 0.5 * (a + b)

    # gripper frame: y = closing axis, z = approach, x = y x z
    up = np.asarray(approach_up, dtype=np.float64)
    y = u
    z = np.tile(-up, (len(u), 1))                       # approach downward by default
    z = z - (z * y).sum(-1, keepdims=True) * y          # orthogonalise against the closing axis
    bad = np.linalg.norm(z, axis=-1) < 1e-6
    if bad.any():
        alt = np.array([1.0, 0.0, 0.0])
        z[bad] = alt - (alt * y[bad]).sum(-1, keepdims=True) * y[bad]
    z /= np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-9)
    x = np.cross(y, z)
    x /= np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)

    poses = np.tile(np.eye(4), (len(u), 1, 1))
    poses[:, :3, 0] = x
    poses[:, :3, 1] = y
    poses[:, :3, 2] = z
    poses[:, :3, 3] = center

    # score: friction-cone quality, closeness to the centroid (stability),
    # a preference for narrower grasps, and table clearance
    dist_c = np.linalg.norm(center - centroid, axis=-1)
    score = (
        1.6 * align
        - 2.0 * dist_c
        - 0.8 * (w / gripper_max_width)
        + 0.6 * np.clip((center[:, 2] - table_z) / 0.05, 0, 1)
    )
    score[center[:, 2] - table_z < min_clearance] -= 2.0
    score = 1.0 / (1.0 + np.exp(-score))                # squash to (0, 1)

    order = np.argsort(-score)
    return GraspSet(
        poses[order], score[order], w[order],
        np.stack([a, b], axis=1)[order],
    )


def nms_grasps(gs: GraspSet, radius: float = 0.015, max_keep: int = 200) -> GraspSet:
    """Greedy spatial non-max suppression so the candidate set is diverse."""
    keep: List[int] = []
    used = np.zeros(len(gs), dtype=bool)
    centers = gs.poses[:, :3, 3]
    for i in range(len(gs)):
        if used[i]:
            continue
        keep.append(i)
        used |= np.linalg.norm(centers - centers[i], axis=-1) < radius
        if len(keep) >= max_keep:
            break
    k = np.array(keep, dtype=int)
    return GraspSet(gs.poses[k], gs.scores[k], gs.widths[k], gs.contacts[k])


# --------------------------------------------------------------------------- #
# The comparison the paper's Fig. 3 makes
# --------------------------------------------------------------------------- #


def compare_grasp_sources(
    pogs_points: np.ndarray,
    visible_points: np.ndarray,
    **kwargs,
) -> Dict[str, object]:
    """Grasps from POGS's completed geometry vs. from a single depth
    deprojection of the same object.

    ``coverage`` reports the fraction of the object's surface each source
    carries, and ``occluded_grasp_frac`` the fraction of POGS grasps whose
    contacts fall on surface the visible cloud never saw -- i.e. grasps that
    only the persistent representation can propose.
    """
    from sklearn.neighbors import NearestNeighbors

    g_pogs = nms_grasps(sample_antipodal_grasps(pogs_points, **kwargs))
    g_vis = nms_grasps(sample_antipodal_grasps(visible_points, **kwargs)) if len(visible_points) >= 32 else GraspSet(
        np.zeros((0, 4, 4)), np.zeros(0), np.zeros(0), np.zeros((0, 2, 3))
    )

    occluded_frac = float("nan")
    if len(g_pogs) and len(visible_points) >= 8:
        nn = NearestNeighbors(n_neighbors=1).fit(visible_points)
        d, _ = nn.kneighbors(g_pogs.contacts.reshape(-1, 3))
        d = d.reshape(-1, 2).max(axis=1)
        occluded_frac = float((d > 0.015).mean())

    return {
        "pogs": g_pogs,
        "visible_only": g_vis,
        "n_pogs": len(g_pogs),
        "n_visible_only": len(g_vis),
        "pogs_points": len(pogs_points),
        "visible_points": len(visible_points),
        "occluded_grasp_frac": occluded_frac,
        "best_score_pogs": float(g_pogs.scores.max()) if len(g_pogs) else 0.0,
        "best_score_visible": float(g_vis.scores.max()) if len(g_vis) else 0.0,
    }


def save_grasp_artifacts(gs: GraspSet, out_dir: str, prefix: str = "") -> Dict[str, str]:
    """Writes the same files the official pipeline hands to the UR5."""
    import os

    os.makedirs(out_dir, exist_ok=True)
    p1 = os.path.join(out_dir, f"{prefix}pred_grasps_world.npy")
    p2 = os.path.join(out_dir, f"{prefix}grasp_point_world.npy")
    np.save(p1, gs.poses.astype(np.float32))
    np.save(p2, gs.best().astype(np.float32))
    return {"pred_grasps_world": p1, "grasp_point_world": p2}


def gripper_wireframe(T: np.ndarray, width: float = 0.06, depth: float = 0.055) -> np.ndarray:
    """A simple parallel-jaw wireframe in world coords, for plotting."""
    hw = width / 2
    pts_local = np.array(
        [
            [0, -hw, -depth], [0, -hw, 0], [0, hw, 0], [0, hw, -depth],
            [0, 0, -depth], [0, 0, -depth - 0.03],
        ]
    )
    return pts_local @ T[:3, :3].T + T[:3, 3]