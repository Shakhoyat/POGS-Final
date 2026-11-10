"""
Object grouping and the Persistent Object Representation (Sec. IV-B / IV-C).

    "Before the tracking phase begins, the system must identify and segment
     individual objects within the scene. As proposed in GARField, this is
     accomplished by clustering the group features using HDBSCAN. The result is
     a mapping from each 3D gaussian to a mask and label."

    "For each gaussian cluster, the system uses the centroid of gaussians
     within that cluster as the object frame, canonicalized such that the
     initial pose of each object is rotated to align with the world frame."

Port of ``POGSModel.cluster`` / ``_cluster_scene`` / ``localize_query_cb``
(``pogs/pogs.py``) and ``Optimizer._setup_crops_and_groups``
(``pogs/tracking/optim.py``).

``cuml.cluster.hdbscan`` is swapped for ``sklearn.cluster.HDBSCAN`` (same
algorithm, available in the stock Kaggle image) and Open3D's
``voxel_down_sample_and_trace`` for an equivalent torch voxel hash.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import GroupingConfig
from .encoders import OpenCLIPNetwork
from .model import POGSModel


# --------------------------------------------------------------------------- #
# Voxel downsampling (replaces open3d)
# --------------------------------------------------------------------------- #


def voxel_downsample_trace(points: np.ndarray, voxel: float) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (representative indices, downsampled positions).

    Equivalent to ``pc_o3d.voxel_down_sample_and_trace`` followed by taking the
    first traced index of each voxel, which is what the official code does.
    """
    if voxel <= 0:
        idx = np.arange(len(points))
        return idx, points
    keys = np.floor(points / voxel).astype(np.int64)
    # unique on structured rows
    _, first = np.unique(keys, axis=0, return_index=True)
    first = np.sort(first)
    return first, points[first]


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #


def _hdbscan(features: np.ndarray, cfg: GroupingConfig) -> np.ndarray:
    try:
        from sklearn.cluster import HDBSCAN

        return HDBSCAN(
            cluster_selection_epsilon=cfg.cluster_eps,
            min_samples=cfg.min_samples,
            min_cluster_size=cfg.min_cluster_size,
            allow_single_cluster=cfg.allow_single_cluster,
        ).fit_predict(features)
    except ImportError:
        try:
            import hdbscan as _h

            return _h.HDBSCAN(
                cluster_selection_epsilon=cfg.cluster_eps,
                min_samples=cfg.min_samples,
                min_cluster_size=cfg.min_cluster_size,
                allow_single_cluster=cfg.allow_single_cluster,
            ).fit_predict(features)
        except ImportError:
            from sklearn.cluster import DBSCAN

            print("  [cluster] HDBSCAN unavailable -> DBSCAN fallback")
            return DBSCAN(eps=max(cfg.cluster_eps, 0.05), min_samples=cfg.min_samples).fit_predict(features)


@torch.no_grad()
def cluster_scene(
    model: POGSModel, cfg: GroupingConfig, verbose: bool = True
) -> torch.Tensor:
    """HDBSCAN over the distilled grouping features (``POGSModel.cluster``).
    Returns per-Gaussian labels (noise re-assigned to the nearest cluster).

    The release runs cuML's GPU HDBSCAN on (practically) every Gaussian. On CPU
    (sklearn) that is intractable for 10^5 points in 128-D, so when there are
    more than ``cfg.max_cluster_points`` we cluster a random subsample, scale
    ``min_cluster_size`` / ``min_samples`` by the same ratio, and give every
    other Gaussian the label of its spatially nearest clustered neighbour --
    the same assignment the release uses for voxel-dropped points.
    """
    from sklearn.neighbors import NearestNeighbors

    t0 = time.time()
    hash_enc = model.smoothed_hash(cfg.knn_k)
    group_feats = model.gaussian_field.instance_from_hash(hash_enc).float().cpu().numpy()
    positions = model.gauss_params["means"].data.detach().cpu().numpy()

    id_vec, _ = voxel_downsample_trace(positions, cfg.voxel_size)
    ratio = 1.0
    if len(id_vec) > cfg.max_cluster_points:
        rng = np.random.default_rng(0)
        ratio = cfg.max_cluster_points / len(id_vec)
        id_vec = np.sort(rng.choice(id_vec, cfg.max_cluster_points, replace=False))
    run_cfg = GroupingConfig(**{**cfg.__dict__})
    run_cfg.min_cluster_size = max(10, int(round(cfg.min_cluster_size * ratio)))
    run_cfg.min_samples = max(5, int(round(cfg.min_samples * ratio)))
    if verbose:
        print(f"  clustering {len(id_vec)} / {len(positions)} gaussians "
              f"(min_cluster_size={run_cfg.min_cluster_size}, min_samples={run_cfg.min_samples}) ...",
              end=" ", flush=True)
    labels_ds = _hdbscan(group_feats[id_vec], run_cfg)

    labels = -np.ones(positions.shape[0], dtype=np.int64)
    labels[id_vec] = labels_ds
    rest = np.ones(positions.shape[0], dtype=bool)
    rest[id_vec] = False
    if rest.any():
        nn_model = NearestNeighbors(n_neighbors=1).fit(positions[id_vec])
        _, ind = nn_model.kneighbors(positions[rest])
        labels[rest] = labels_ds[ind[:, 0]]

    # relabel HDBSCAN noise to the spatially nearest clustered gaussian
    noise = labels == -1
    if noise.sum() and (labels >= 0).sum():
        valid = labels >= 0
        nn_model = NearestNeighbors(n_neighbors=1).fit(positions[valid])
        _, ind = nn_model.kneighbors(positions[noise])
        labels[noise] = labels[valid][ind[:, 0]]

    n = int(labels.max()) + 1
    if verbose:
        print(f"done in {time.time() - t0:.1f}s -> {n} clusters "
              f"({np.bincount(labels[labels >= 0], minlength=max(n, 1)).tolist()})")
    return torch.from_numpy(labels).long().to(model.gauss_params["means"].device)


# --------------------------------------------------------------------------- #
# Persistent object representation
# --------------------------------------------------------------------------- #


@dataclass
class ObjectGroups:
    """The tracked objects: one rigid group per selected cluster."""

    labels_full: Optional[torch.Tensor] = None        # (N,) cluster id per gaussian, -1 = untracked
    group_labels: Optional[torch.Tensor] = None       # (N,) index into group list, or -1
    group_masks: List[torch.Tensor] = field(default_factory=list)  # per group: (N,) bool
    init_p2w: Optional[torch.Tensor] = None           # (K, 7) part->world at scan time (identity rot)
    cluster_ids: Optional[List[int]] = None
    names: Optional[List[str]] = None

    def __post_init__(self):
        K = len(self.group_masks)
        if self.cluster_ids is None:
            self.cluster_ids = list(range(K))
        if self.names is None:
            self.names = [f"object_{i}" for i in range(K)]
        if self.init_p2w is None:
            dev = self.group_masks[0].device if K > 0 else "cpu"
            self.init_p2w = torch.zeros(K, 7, device=dev)
            if K > 0:
                self.init_p2w[:, 3] = 1.0  # identity quat
        if self.labels_full is None and K > 0:
            N = self.group_masks[0].shape[0]
            dev = self.group_masks[0].device
            self.labels_full = torch.full((N,), -1, dtype=torch.long, device=dev)
            self.group_labels = torch.full((N,), -1, dtype=torch.long, device=dev)
            for gi, m in enumerate(self.group_masks):
                self.labels_full[m] = self.cluster_ids[gi]
                self.group_labels[m] = gi

    def __len__(self) -> int:
        return self.num_groups

    @property
    def num_groups(self) -> int:
        return len(self.group_masks)

    def centroids(self) -> torch.Tensor:
        return self.init_p2w[:, :3]


def build_object_groups(
    model: POGSModel,
    labels: torch.Tensor,
    keep_clusters: Optional[Sequence[int]] = None,
    min_gaussians: int = 50,
    names: Optional[Dict[int, str]] = None,
) -> ObjectGroups:
    """Selected clusters -> tracked rigid groups (in ``keep_clusters`` order)."""
    """Selects which clusters become tracked objects and canonicalises each
    object frame at its centroid with identity rotation (Sec. IV-C)."""
    means = model.gauss_params["means"].data
    device = means.device
    uniq = [int(c) for c in torch.unique(labels).tolist() if c >= 0]
    if keep_clusters is not None:
        uniq = [int(c) for c in keep_clusters if int(c) in set(uniq)]
    uniq = [c for c in uniq if int((labels == c).sum()) >= min_gaussians]

    masks, p2w = [], []
    group_labels = torch.full((means.shape[0],), -1, dtype=torch.long, device=device)
    for gi, c in enumerate(uniq):
        m = labels == c
        masks.append(m)
        group_labels[m] = gi
        cen = means[m].mean(dim=0)
        vec = torch.zeros(7, device=device)
        vec[:3] = cen
        vec[3] = 1.0                       # identity rotation -> world-aligned
        p2w.append(vec)

    return ObjectGroups(
        labels_full=labels,
        group_labels=group_labels,
        group_masks=masks,
        init_p2w=torch.stack(p2w) if p2w else torch.zeros(0, 7, device=device),
        cluster_ids=uniq,
        names=[(names or {}).get(c, f"cluster_{c}") for c in uniq],
    )


def subset_groups(groups: ObjectGroups, keep: Sequence[int]) -> ObjectGroups:
    """The tracked objects restricted to groups ``keep`` (in that order), e.g.
    the object(s) the operator clicked on (Sec. IV-C)."""
    masks = [groups.group_masks[g] for g in keep]
    N = masks[0].shape[0]
    group_labels = torch.full((N,), -1, dtype=torch.long, device=masks[0].device)
    labels_full = torch.full_like(group_labels, -1)
    for gi, g in enumerate(keep):
        group_labels[masks[gi]] = gi
        labels_full[masks[gi]] = groups.cluster_ids[g]
    return ObjectGroups(
        labels_full=labels_full, group_labels=group_labels, group_masks=masks,
        init_p2w=groups.init_p2w[list(keep)].clone(),
        cluster_ids=[groups.cluster_ids[g] for g in keep],
        names=[groups.names[g] for g in keep],
    )


# --------------------------------------------------------------------------- #
# Language querying (Sec. IV-B, "Language Features")
# --------------------------------------------------------------------------- #


@torch.no_grad()
def find_best_scale(
    model: POGSModel,
    clip: OpenCLIPNetwork,
    query: str,
    scales: Optional[torch.Tensor] = None,
) -> Tuple[float, torch.Tensor]:
    """``POGSModel.get_max_across`` in point space: sweep the LERF scale
    conditioning and keep the one that maximises positive relevancy."""
    device = model.gauss_params["means"].device
    if scales is None:
        scales = torch.linspace(0.0, 0.5, 30, device=device)
    clip.set_positives([query])
    h = model.smoothed_hash(3)
    best_scale, best_score, best_rel = float(scales[0]), -1.0, None
    for s in scales:
        feats = model.gaussian_field.clip_from_hash(
            h, s * torch.ones(h.shape[0], 1, device=device)
        ).float()
        rel = clip.get_relevancy(feats / (feats.norm(dim=-1, keepdim=True) + 1e-6), 0)[..., 0]
        score = float(rel.max())
        if score > best_score:
            best_score, best_scale, best_rel = score, float(s), rel
    model.best_scale = best_scale
    return best_scale, best_rel


@torch.no_grad()
def localize_query(
    model: POGSModel,
    clip: OpenCLIPNetwork,
    groups: ObjectGroups,
    query: str,
    scale: Optional[float] = None,
) -> Tuple[int, np.ndarray, torch.Tensor]:
    """Returns (group index, mean relevancy per group, per-gaussian relevancy).

    Mirrors ``localize_query_cb``: relevancy is averaged inside each cluster and
    the arg-max cluster wins.
    """
    if scale is None:
        scale, rel = find_best_scale(model, clip, query)
    else:
        clip.set_positives([query])
        h = model.smoothed_hash(3)
        feats = model.gaussian_field.clip_from_hash(
            h, scale * torch.ones(h.shape[0], 1, device=h.device)
        ).float()
        rel = clip.get_relevancy(feats / (feats.norm(dim=-1, keepdim=True) + 1e-6), 0)[..., 0]

    per_group = np.array(
        [float(rel[m].mean()) if bool(m.any()) else -1.0 for m in groups.group_masks]
    )
    return int(per_group.argmax()), per_group, rel


# --------------------------------------------------------------------------- #
# Diagnostics: how well did clustering recover the true objects?
# --------------------------------------------------------------------------- #


def cluster_purity(labels: torch.Tensor, gt_ids: torch.Tensor) -> Dict[str, float]:
    """EVALUATION ONLY: cluster-vs-ground-truth agreement for the sim scenes."""
    l = labels.cpu().numpy()
    g = gt_ids.cpu().numpy()
    valid = l >= 0
    if valid.sum() == 0:
        return {"purity": 0.0, "n_clusters": 0}
    total = 0
    for c in np.unique(l[valid]):
        m = (l == c) & valid
        counts = np.bincount(g[m] + 1)
        total += counts.max()
    return {
        "purity": float(total / valid.sum()),
        "n_clusters": int(l[valid].max() + 1),
        "n_gt_objects": int(len(np.unique(g[g >= 0]))),
    }


def match_clusters_to_gt(
    groups: ObjectGroups, gt_ids: torch.Tensor
) -> Dict[int, int]:
    """EVALUATION ONLY: group index -> majority ground-truth object id."""
    g = gt_ids.cpu().numpy()
    out = {}
    for gi, m in enumerate(groups.group_masks):
        sel = g[m.cpu().numpy()]
        sel = sel[sel >= 0]
        out[gi] = int(np.bincount(sel).argmax()) if len(sel) else -1
    return out


def group_gt_iou(groups: ObjectGroups, gt_ids: torch.Tensor, group_to_gt: Dict[int, int]) -> Dict[int, float]:
    """EVALUATION ONLY: IoU between each tracked group and its matched GT object
    (over Gaussians)."""
    out = {}
    for gi, m in enumerate(groups.group_masks):
        g = gt_ids == group_to_gt.get(gi, -999)
        inter = int((m & g).sum())
        union = int((m | g).sum())
        out[gi] = inter / max(union, 1)
    return out


# --------------------------------------------------------------------------- #
# Choosing which clusters become tracked objects
# --------------------------------------------------------------------------- #


@torch.no_grad()
def assign_gt_ids(model: POGSModel, seed_points: torch.Tensor, seed_obj_ids: torch.Tensor) -> torch.Tensor:
    """EVALUATION ONLY: ground-truth object id per current Gaussian, inherited
    from the nearest seed point (densification changes the Gaussian count)."""
    from sklearn.neighbors import NearestNeighbors

    means = model.gauss_params["means"].data.detach().cpu().numpy()
    nn_ = NearestNeighbors(n_neighbors=1).fit(seed_points.cpu().numpy())
    _, idx = nn_.kneighbors(means)
    return seed_obj_ids.cpu()[idx[:, 0]].to(model.gauss_params["means"].device)


@torch.no_grad()
def select_object_clusters(
    model: POGSModel,
    labels: torch.Tensor,
    cfg: GroupingConfig,
    clip: Optional[OpenCLIPNetwork] = None,
    queries: Optional[Sequence[str]] = None,
    min_gaussians: int = 50,
    verbose: bool = True,
) -> Tuple[List[int], torch.Tensor]:
    """Stands in for the release's interactive "click -> crop to click -> add
    crop to group list" step. Returns ``(cluster ids, filtered labels)``.

    Like the release, Gaussians below the support plane (``table_z + 0.015`` in
    ``pogs_pipeline.py``; here ``cfg.support_z`` above the floor) are removed
    from every cluster first.

    ``geometry``  keep clusters that are not the support surface: small XY
                  footprint and a modest share of the scene.
    ``language``  one cluster per natural-language query via CLIP relevancy
                  (Sec. IV-E) -- needs real OpenCLIP weights.
    """
    means = model.gauss_params["means"].data
    labels = labels.clone()
    labels[means[:, 2] < cfg.support_z] = -1
    total = int((labels >= 0).sum())
    ids = [int(c) for c in torch.unique(labels).tolist() if c >= 0]

    if cfg.select_method == "language":
        assert clip is not None and queries, "language selection needs CLIP + queries"
        groups_all = build_object_groups(model, labels, min_gaussians=min_gaussians)
        keep: List[int] = []
        for q in queries:
            gi, per_group, _ = localize_query(model, clip, groups_all, q)
            cid = groups_all.cluster_ids[gi]
            if verbose:
                print(f"  query '{q}' -> cluster {cid} (relevancy {per_group[gi]:.3f})")
            if cid not in keep:
                keep.append(cid)
        return keep, labels

    # The negative-mask term of the contrastive loss pulls every background
    # Gaussian's grouping feature towards the all-ones unit vector (Sec. IV-B:
    # "including a negative mask ... reduces group feature noise for the scene
    # background"), so a cluster whose features sit there IS background.
    feats = model.gaussian_field.instance_from_hash(model.smoothed_hash(cfg.knn_k)).float()
    ones = torch.ones(feats.shape[1], device=feats.device) / feats.shape[1] ** 0.5
    bg_sim = feats @ ones

    keep = []
    min_n = max(min_gaussians, int(cfg.min_object_share * total))
    for c in ids:
        m = labels == c
        n = int(m.sum())
        p = means[m]
        footprint = float((p[:, 0].max() - p[:, 0].min()) * (p[:, 1].max() - p[:, 1].min()))
        share = n / max(total, 1)
        sim = float(bg_sim[m].mean())
        if n < min_n:
            verdict = "too small"
        elif sim > cfg.background_sim_thresh:
            verdict = "background (negative-mask feature)"
        elif footprint > cfg.max_footprint_m2 or share > cfg.max_share:
            verdict = "support surface"
        else:
            verdict = "object"
        if verbose:
            print(f"  cluster {c}: {n} gaussians, footprint {footprint:.3f} m^2, share {share:.2f}, "
                  f"bg-sim {sim:+.2f} -> {verdict}")
        if verdict == "object":
            keep.append(c)
    return keep, labels
