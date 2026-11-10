"""
Offline tracking driver and metrics -- the ONE tracking loop shared by every
experiment (v1 base POGS, v2 Kalman, v3 other scene).

Protocol (mirrors ``track_main_online_demo.py``):
    1. ``initialize_obj_pose`` on frame 0 (100 DINO-only full-frame steps, then
       70 steps with depth + ROI);
    2. for every frame (including frame 0): build the observation, compute each
       object's ROI from the current pose, run ``track_niter`` optimiser steps.

An optional *motion prior* (v2's Kalman filter) can warm-start the optimiser
from a predicted pose and/or replace the output with a fused estimate. With
``motion_prior=None`` the loop is exactly base POGS.

Metrics, per frame and tracked object, against simulator ground truth:
    trans_err_cm   centroid position error
    rot_err_deg    geodesic rotation error
    yaw_err_deg    error about the world vertical axis
    add / adds     ADD / ADD-S over the object's scan-time Gaussians (metres)
    success        the paper's tool-servoing criterion: < 3 cm AND < 5 deg
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence

import numpy as np
import torch

from .config import POGSConfig, TrackingConfig
from .geometry import geodesic_rotation_error, identity_posevec
from .grouping import ObjectGroups
from .model import POGSModel
from .tracking import DinoFn, PosedObservation, RigidGroupOptimizer

SUCCESS_POS_CM = 3.0
SUCCESS_ROT_DEG = 5.0
"""Sec. V-B: 'locate the drill tip within 3cm and 5 deg of the target'."""


# --------------------------------------------------------------------------- #
# Pose metrics
# --------------------------------------------------------------------------- #


def translation_error_cm(T_pred: np.ndarray, T_gt: np.ndarray) -> float:
    return float(np.linalg.norm(T_pred[:3, 3] - T_gt[:3, 3]) * 100.0)


def rotation_error_deg(T_pred: np.ndarray, T_gt: np.ndarray) -> float:
    return geodesic_rotation_error(T_pred[:3, :3], T_gt[:3, :3])


def yaw_deg(R: np.ndarray) -> float:
    return math.degrees(math.atan2(R[1, 0], R[0, 0]))


def yaw_error_deg(T_pred: np.ndarray, T_gt: np.ndarray) -> float:
    d = yaw_deg(T_pred[:3, :3]) - yaw_deg(T_gt[:3, :3])
    return abs((d + 180.0) % 360.0 - 180.0)


def add_metric(points: np.ndarray, T_pred: np.ndarray, T_gt: np.ndarray) -> float:
    """Average Distance of model points (Hinterstoisser et al.), metres.
    ``points`` are object-local (relative to the scan-time centroid)."""
    a = points @ T_pred[:3, :3].T + T_pred[:3, 3]
    b = points @ T_gt[:3, :3].T + T_gt[:3, 3]
    return float(np.linalg.norm(a - b, axis=1).mean())


def adds_metric(points: np.ndarray, T_pred: np.ndarray, T_gt: np.ndarray, max_pts: int = 1500, chunk: int = 256) -> float:
    """ADD-S: symmetry-aware nearest-neighbour variant (chunked, float64)."""
    if len(points) > max_pts:
        idx = np.random.default_rng(0).choice(len(points), max_pts, replace=False)
        points = points[idx]
    a = (points @ T_pred[:3, :3].T + T_pred[:3, 3]).astype(np.float64)
    b = (points @ T_gt[:3, :3].T + T_gt[:3, 3]).astype(np.float64)
    b2 = (b * b).sum(axis=1)
    total = 0.0
    for i in range(0, len(a), chunk):
        block = a[i : i + chunk]
        d2 = (block * block).sum(axis=1)[:, None] - 2.0 * (block @ b.T) + b2[None, :]
        total += float(np.sqrt(np.maximum(d2.min(axis=1), 0.0)).sum())
    return total / max(len(a), 1)


def object_diameter(points: np.ndarray) -> float:
    """Robust object diameter (m): diagonal of the 1-99 percentile box of the
    model points, so a few floater Gaussians do not inflate it."""
    lo, hi = np.percentile(points, 1, axis=0), np.percentile(points, 99, axis=0)
    return float(np.linalg.norm(hi - lo))


ADDS_SUCCESS_FRAC = 0.10
"""Standard ADD(-S) criterion: ADD-S below 10% of the object diameter
(Hinterstoisser et al. 2012; PoseCNN). Symmetry-aware, unlike 3 cm / 5 deg."""


def adds_success_rate(adds_m: np.ndarray, diameter_m: float) -> float:
    v = adds_m[~np.isnan(adds_m)]
    return float((v < ADDS_SUCCESS_FRAC * diameter_m).mean()) if v.size else float("nan")


def gt_pose_for_group(seq, frame_idx: int, gt_obj_id: int, base_centroid: np.ndarray) -> np.ndarray:
    """GT pose in POGS's convention: [R_delta | centroid after the motion].

    ``gt_delta`` is the world-frame rigid motion applied to the object's
    geometry (x' = R x + t), so the scan-time centroid ``c`` moves to
    ``R c + t``; POGS predicts ``c + delta_t`` for the same point."""
    T = seq.frames[frame_idx].gt_delta.get(gt_obj_id, np.eye(4))
    out = np.eye(4)
    out[:3, :3] = T[:3, :3]
    out[:3, 3] = T[:3, :3] @ base_centroid + T[:3, 3]
    return out


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #


@dataclass
class TrackingResult:
    name: str
    trans_err_cm: np.ndarray          # (F, K)
    rot_err_deg: np.ndarray           # (F, K)
    yaw_err_deg: np.ndarray           # (F, K)
    add: np.ndarray                   # (F, K) metres
    adds: np.ndarray                  # (F, K) metres
    frame_times: np.ndarray           # (F,) seconds
    dino_ms: np.ndarray               # (F,)
    opt_ms: np.ndarray                # (F,)
    losses: np.ndarray                # (F,)
    pred_poses: np.ndarray            # (F, K, 4, 4)
    gt_poses: np.ndarray              # (F, K, 4, 4)
    part_deltas: np.ndarray           # (F, K, 7)
    occluded: np.ndarray              # (F,) bool -- GT occluder in view
    group_names: List[str] = field(default_factory=list)
    init_time: float = 0.0
    diagnostics: Dict[str, float] = field(default_factory=dict)

    def _sel(self, arr: np.ndarray, group: Optional[int]) -> np.ndarray:
        return arr if group is None else arr[:, group : group + 1]

    def success(self, group: Optional[int] = None) -> np.ndarray:
        t, r = self._sel(self.trans_err_cm, group), self._sel(self.rot_err_deg, group)
        return (t < SUCCESS_POS_CM) & (r < SUCCESS_ROT_DEG)

    def summary(self, group: Optional[int] = None) -> Dict[str, float]:
        t = self._sel(self.trans_err_cm, group)
        r = self._sel(self.rot_err_deg, group)
        y = self._sel(self.yaw_err_deg, group)
        a = self._sel(self.add, group)
        s = self._sel(self.adds, group)
        occ = self.occluded
        out = {
            "name": self.name,
            "mean_pos_err_cm": float(np.nanmean(t)),
            "median_pos_err_cm": float(np.nanmedian(t)),
            "max_pos_err_cm": float(np.nanmax(t)),
            "mean_rot_err_deg": float(np.nanmean(r)),
            "max_rot_err_deg": float(np.nanmax(r)),
            "mean_yaw_err_deg": float(np.nanmean(y)),
            "final_pos_err_cm": float(np.nanmean(t[-1])),
            "final_rot_err_deg": float(np.nanmean(r[-1])),
            "mean_add_cm": float(np.nanmean(a) * 100),
            "mean_adds_cm": float(np.nanmean(s) * 100),
            "success_rate": float(np.nanmean(self.success(group))),
            "pos_err_occluded_cm": float(np.nanmean(t[occ])) if occ.any() else float("nan"),
            "pos_err_clear_cm": float(np.nanmean(t[~occ])) if (~occ).any() else float("nan"),
            "fps": float(1.0 / max(np.nanmean(self.frame_times), 1e-9)) if self.frame_times.any() else float("nan"),
            "mean_dino_ms": float(np.nanmean(self.dino_ms)),
            "mean_opt_ms": float(np.nanmean(self.opt_ms)),
            "init_time_s": self.init_time,
        }
        out.update({k: v for k, v in self.diagnostics.items()})
        return out

    def telemetry(self) -> List[Dict[str, object]]:
        F, K = self.trans_err_cm.shape
        rows = []
        for f in range(F):
            rows.append({
                "frame": f,
                "occluded": bool(self.occluded[f]),
                "frame_ms": float(self.frame_times[f] * 1000),
                "dino_ms": float(self.dino_ms[f]),
                "opt_ms": float(self.opt_ms[f]),
                "loss": float(self.losses[f]),
                "objects": [
                    {
                        "group": g,
                        "name": self.group_names[g] if g < len(self.group_names) else f"group_{g}",
                        "trans_err_cm": float(self.trans_err_cm[f, g]),
                        "rot_err_deg": float(self.rot_err_deg[f, g]),
                        "yaw_err_deg": float(self.yaw_err_deg[f, g]),
                        "add_cm": float(self.add[f, g] * 100),
                        "pred_pose": self.pred_poses[f, g].tolist(),
                        "gt_pose": self.gt_poses[f, g].tolist(),
                    }
                    for g in range(K)
                ],
            })
        return rows


# --------------------------------------------------------------------------- #
# Motion prior hook (v2)
# --------------------------------------------------------------------------- #


class MotionPrior(Protocol):
    warm_start: bool
    fuse: bool
    check_loss: bool        # optional; see common.kalman.KalmanMotionPrior

    def initialize(self, posevecs: torch.Tensor) -> None: ...
    def predict(self, device=None) -> torch.Tensor: ...
    def update(self, posevecs: torch.Tensor, device=None) -> torch.Tensor: ...


# --------------------------------------------------------------------------- #
# The tracking loop
# --------------------------------------------------------------------------- #


def _score_frame(seq, fi, P, groups, group_to_gt, base_centroids, model_points, out):
    for gi in range(groups.num_groups):
        gt_id = group_to_gt.get(gi, -1)
        if gt_id < 0:
            continue
        T_gt = gt_pose_for_group(seq, fi, gt_id, base_centroids[gi])
        out["preds"][fi, gi], out["gts"][fi, gi] = P[gi], T_gt
        out["trans"][fi, gi] = translation_error_cm(P[gi], T_gt)
        out["rot"][fi, gi] = rotation_error_deg(P[gi], T_gt)
        out["yaw"][fi, gi] = yaw_error_deg(P[gi], T_gt)
        pts = model_points.get(gi) if model_points else None
        if pts is not None:
            out["add"][fi, gi] = add_metric(pts, P[gi], T_gt)
            out["adds"][fi, gi] = adds_metric(pts, P[gi], T_gt)


def _alloc(F_n: int, K: int) -> Dict[str, np.ndarray]:
    return {
        "trans": np.full((F_n, K), np.nan), "rot": np.full((F_n, K), np.nan),
        "yaw": np.full((F_n, K), np.nan), "add": np.full((F_n, K), np.nan),
        "adds": np.full((F_n, K), np.nan),
        "preds": np.tile(np.eye(4), (F_n, K, 1, 1)), "gts": np.tile(np.eye(4), (F_n, K, 1, 1)),
    }


def object_model_points(groups: ObjectGroups, model: POGSModel) -> Dict[int, np.ndarray]:
    """Scan-time Gaussian means of every tracked group, relative to its centroid
    (the 'model points' for ADD / ADD-S)."""
    means = model.gauss_params["means"].data
    out = {}
    for gi, m in enumerate(groups.group_masks):
        c = groups.init_p2w[gi, :3]
        out[gi] = (means[m] - c).detach().cpu().numpy()
    return out


def tracking_camera(seq_camera, tcfg: TrackingConfig, device: str):
    """``Optimizer.MATCH_RESOLUTION``: rescale so the short side is 500 px
    (only ever downscale)."""
    cam = seq_camera.to(device)
    scale = tcfg.match_resolution / min(cam.width, cam.height)
    return cam.rescale(scale) if scale < 1.0 else cam


def track_sequence(
    model: POGSModel,
    groups: ObjectGroups,
    seq,
    dino_fn: DinoFn,
    cfg: POGSConfig,
    group_to_gt: Dict[int, int],
    name: str = "POGS",
    tracking_cfg: Optional[TrackingConfig] = None,
    model_points: Optional[Dict[int, np.ndarray]] = None,
    motion_prior: Optional[MotionPrior] = None,
    max_frames: Optional[int] = None,
    verbose: bool = True,
) -> TrackingResult:
    """Runs POGS tracking over an RGB-D sequence and scores it against GT."""
    tcfg = tracking_cfg or cfg.tracking
    device = cfg.runtime.device
    raster_kwargs = (
        dict(max_per_tile=cfg.runtime.torch_raster_max_per_tile, tile_chunk=cfg.runtime.torch_raster_tile_chunk)
        if model.backend == "torch" else {}
    )
    snapshot = model.state_snapshot()
    opt = RigidGroupOptimizer(tcfg, model, groups, raster_kwargs=raster_kwargs)
    cam = tracking_camera(seq.camera, tcfg, device)

    frames = seq.frames if max_frames is None else seq.frames[:max_frames]
    F_n, K = len(frames), groups.num_groups
    base_centroids = {gi: groups.init_p2w[gi, :3].detach().cpu().numpy() for gi in range(K)}
    acc = _alloc(F_n, K)
    times, dino_ms, opt_ms = np.zeros(F_n), np.zeros(F_n), np.zeros(F_n)
    n_pred_kept = 0
    losses = np.full(F_n, np.nan)
    deltas = np.zeros((F_n, K, 7), dtype=np.float32)

    def make_obs(fr):
        return PosedObservation(fr.rgb.to(device), cam, dino_fn, fr.depth.to(device), tcfg.max_roi_resolution)

    # ---- initialisation on frame 0 -------------------------------------- #
    t0 = time.time()
    init_info = opt.initialize_obj_pose(make_obs(frames[0]), n_seeds=tcfg.init_n_seeds, verbose=False)
    init_time = time.time() - t0
    if verbose:
        print(f"  [{name}] init: coarse={init_info['coarse_loss']:.4f} fine={init_info['fine_loss']:.4f} "
              f"in {init_time:.1f}s")
    if motion_prior is not None:
        motion_prior.initialize(opt.part_deltas.data)

    for fi, fr in enumerate(frames):
        t_frame = time.time()

        # ---- motion prior: warm start ------------------------------------ #
        pred = None
        check = motion_prior is not None and getattr(motion_prior, "check_loss", False)
        if motion_prior is not None and fi > 0:
            pred = motion_prior.predict(device).float()
            if motion_prior.warm_start and not check:
                opt.set_part_deltas(pred)

        # ---- observation + ROIs from the current pose; DINO extraction ---- #
        torch.cuda.synchronize()
        t_d = time.time()
        obs = make_obs(fr)
        opt.set_observation(obs)
        if tcfg.use_dino:
            if tcfg.use_roi:
                for rf in obs.roi_frames:
                    _ = rf.dino_feats
            else:
                _ = obs.frame.dino_feats
        torch.cuda.synchronize()
        dino_ms[fi] = (time.time() - t_d) * 1000.0

        # ---- pose optimisation ------------------------------------------- #
        t_o = time.time()
        if check and pred is not None and motion_prior.warm_start:
            n_pred_kept += _choose_start(opt, obs, pred, tcfg)
        info = opt.step(niter=tcfg.track_niter, use_depth=tcfg.use_depth, use_rgb=tcfg.use_rgb)
        torch.cuda.synchronize()
        opt_ms[fi] = (time.time() - t_o) * 1000.0
        losses[fi] = info.loss

        # ---- motion prior: measurement update ----------------------------- #
        if motion_prior is not None:
            fused = motion_prior.update(opt.part_deltas.data, device)
            if motion_prior.fuse:
                opt.set_part_deltas(fused.float())
                opt.apply_to_model()

        times[fi] = time.time() - t_frame
        deltas[fi] = opt.part_deltas.data.detach().cpu().numpy()
        P = opt.get_all_part2world().detach().cpu().numpy()
        _score_frame(seq, fi, P, groups, group_to_gt, base_centroids, model_points, acc)

        if verbose and (fi % 10 == 0 or fi == F_n - 1):
            print(f"    [{name}] frame {fi:3d}/{F_n}  loss={info.loss:.4f}  "
                  f"pos={np.nanmean(acc['trans'][fi]):.2f} cm  rot={np.nanmean(acc['rot'][fi]):.1f} deg  "
                  f"{times[fi] * 1000:.0f} ms")

    model.load_snapshot(snapshot)
    diag = opt.depth_diagnostics()
    if motion_prior is not None and hasattr(motion_prior, "mean_nis"):
        diag["kalman_mean_nis"] = motion_prior.mean_nis()
    if motion_prior is not None and getattr(motion_prior, "check_loss", False):
        diag["prediction_kept_rate"] = n_pred_kept / max((F_n - 1) * K, 1)
    if verbose and tcfg.use_depth and diag["depth_drop_rate"] > 0.2:
        print(f"  [{name}] WARNING: depth term dropped on {diag['depth_drop_rate']:.0%} of evaluations")

    return TrackingResult(
        name=name, trans_err_cm=acc["trans"], rot_err_deg=acc["rot"], yaw_err_deg=acc["yaw"],
        add=acc["add"], adds=acc["adds"], frame_times=times, dino_ms=dino_ms, opt_ms=opt_ms,
        losses=losses, pred_poses=acc["preds"], gt_poses=acc["gts"], part_deltas=deltas,
        occluded=np.array([bool(getattr(f, "occluded", False)) for f in frames]),
        group_names=list(groups.names or []), init_time=init_time, diagnostics=diag,
    )


@torch.no_grad()
def _choose_start(opt, obs, pred: torch.Tensor, tcfg: TrackingConfig) -> int:
    """Greedy per object: adopt the predicted pose if it lowers the tracking
    loss versus the current estimate. Returns how many objects took it."""
    def loss_at(d):
        l, _ = opt.get_optim_loss(obs, d, use_dino=tcfg.use_dino, use_depth=tcfg.use_depth,
                                  use_rgb=tcfg.use_rgb, use_roi=tcfg.use_roi)
        return float("inf") if l is None else float(l)
    cur = opt.part_deltas.data.clone()
    best = loss_at(cur)
    kept = 0
    for k in range(cur.shape[0]):
        cand = cur.clone()
        cand[k] = pred[k]
        lc = loss_at(cand)
        if lc < best:
            cur, best, kept = cand, lc, kept + 1
    opt.set_part_deltas(cur)
    return kept


def static_baseline(
    groups: ObjectGroups,
    seq,
    group_to_gt: Dict[int, int],
    model_points: Optional[Dict[int, np.ndarray]] = None,
    name: str = "Static (no motion)",
) -> TrackingResult:
    """Trivial reference: every object is predicted to stay at its scan pose.
    Any tracker must beat this to be doing anything at all."""
    F_n, K = len(seq.frames), groups.num_groups
    base_centroids = {gi: groups.init_p2w[gi, :3].detach().cpu().numpy() for gi in range(K)}
    acc = _alloc(F_n, K)
    for fi in range(F_n):
        P = np.tile(np.eye(4), (K, 1, 1))
        for gi in range(K):
            P[gi, :3, 3] = base_centroids[gi]
        _score_frame(seq, fi, P, groups, group_to_gt, base_centroids, model_points, acc)
    z = np.zeros(F_n)
    return TrackingResult(
        name=name, trans_err_cm=acc["trans"], rot_err_deg=acc["rot"], yaw_err_deg=acc["yaw"],
        add=acc["add"], adds=acc["adds"], frame_times=z, dino_ms=z * np.nan, opt_ms=z * np.nan,
        losses=z * np.nan, pred_poses=acc["preds"], gt_poses=acc["gts"],
        part_deltas=np.tile(identity_posevec(K).numpy(), (F_n, 1, 1)),
        occluded=np.array([bool(getattr(f, "occluded", False)) for f in seq.frames]),
        group_names=list(groups.names or []),
    )


# --------------------------------------------------------------------------- #
# Variants (paper ablations)
# --------------------------------------------------------------------------- #


def variant_config(base: TrackingConfig, variant: str) -> TrackingConfig:
    """Tracking settings for the named variant.

    POGS                        paper Sec. IV-D: DINO + depth every frame
    POGS (release: no per-frame depth)
                                the released ``step()``: DINO only per frame,
                                depth only in the init fine pass
    No Depth                    paper ablation: depth off everywhere
    No DINO                     paper ablation: 'optimized with RGB
                                substituting for DINO features'
    N iters                     POGS with N optimiser steps per frame
    """
    tc = copy.deepcopy(base)
    if variant == "POGS":
        pass
    elif variant == "POGS (release: no per-frame depth)":
        tc.use_depth, tc.init_use_depth = False, True
    elif variant == "No Depth":
        tc.use_depth, tc.init_use_depth = False, False
    elif variant == "No DINO":
        tc.use_dino, tc.use_rgb = False, True
    elif variant.endswith(" iters"):
        tc.track_niter = int(variant.split()[0])
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return tc


def results_table(results: Dict[str, TrackingResult], group: Optional[int] = None):
    import pandas as pd

    rows = [r.summary(group) for r in results.values()]
    return pd.DataFrame(rows).set_index("name")
