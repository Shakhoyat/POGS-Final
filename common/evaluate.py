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
from typing import Callable, Dict, List, Optional, Protocol

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
    def predict(self, device=None, dt: Optional[float] = None) -> torch.Tensor: ...
    def update(self, posevecs: torch.Tensor, device=None) -> torch.Tensor: ...


# --------------------------------------------------------------------------- #
# The tracking loop
# --------------------------------------------------------------------------- #


def _score_frame(seq, fi, P, groups, group_to_gt, base_centroids, model_points, out, row: Optional[int] = None):
    """Scores poses ``P`` (K, 4, 4) against GT of frame ``fi`` into ``out[row]``
    (``row`` defaults to ``fi``)."""
    r = fi if row is None else row
    for gi in range(groups.num_groups):
        gt_id = group_to_gt.get(gi, -1)
        if gt_id < 0:
            continue
        T_gt = gt_pose_for_group(seq, fi, gt_id, base_centroids[gi])
        out["preds"][r, gi], out["gts"][r, gi] = P[gi], T_gt
        out["trans"][r, gi] = translation_error_cm(P[gi], T_gt)
        out["rot"][r, gi] = rotation_error_deg(P[gi], T_gt)
        out["yaw"][r, gi] = yaw_error_deg(P[gi], T_gt)
        pts = model_points.get(gi) if model_points else None
        if pts is not None:
            out["add"][r, gi] = add_metric(pts, P[gi], T_gt)
            out["adds"][r, gi] = adds_metric(pts, P[gi], T_gt)


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


def _sync() -> None:
    """Wait for queued GPU work so wall-clock timings are honest (no-op on CPU)."""
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


class _FrameTracker:
    """POGS (+ optional motion prior) on one sequence; ``step`` tracks one
    frame. Shared by the offline loop and the real-time protocol, so both run
    exactly the same per-frame computation."""

    def __init__(self, model: POGSModel, groups: ObjectGroups, seq, dino_fn: DinoFn, cfg: POGSConfig,
                 tcfg: TrackingConfig, motion_prior: Optional["MotionPrior"]):
        self.model, self.cfg, self.tcfg, self.dino_fn, self.prior = model, cfg, tcfg, dino_fn, motion_prior
        self.device = cfg.runtime.device
        raster_kwargs = (
            dict(max_per_tile=cfg.runtime.torch_raster_max_per_tile, tile_chunk=cfg.runtime.torch_raster_tile_chunk)
            if model.backend == "torch" else {}
        )
        self.snapshot = model.state_snapshot()
        self.opt = RigidGroupOptimizer(tcfg, model, groups, raster_kwargs=raster_kwargs)
        self.cam = tracking_camera(seq.camera, tcfg, self.device)
        self.n_pred_kept = 0

    def make_obs(self, fr):
        return PosedObservation(fr.rgb.to(self.device), self.cam, self.dino_fn, fr.depth.to(self.device),
                                self.tcfg.max_roi_resolution)

    def initialize(self, fr) -> Dict[str, float]:
        t0 = time.time()
        info = self.opt.initialize_obj_pose(self.make_obs(fr), n_seeds=self.tcfg.init_n_seeds, verbose=False)
        info = dict(info, init_time=time.time() - t0)
        if self.prior is not None:
            self.prior.initialize(self.opt.part_deltas.data)
        return info

    def step(self, fr, first: bool, dt: Optional[float] = None) -> Dict[str, float]:
        """One tracked frame. ``dt``: seconds since the previously tracked
        frame (motion-prior prediction; None = the prior's nominal period).
        The observation is built from ``fr`` before the clock starts, so disk
        reads are not counted as tracking time."""
        tcfg, prior, opt, device = self.tcfg, self.prior, self.opt, self.device
        rgb, depth = fr.rgb.to(device), fr.depth.to(device)
        _sync()
        t_frame = time.time()

        # ---- motion prior: warm start ------------------------------------ #
        pred = None
        check = prior is not None and getattr(prior, "check_loss", False)
        if prior is not None and not first:
            pred = (prior.predict(device) if dt is None else prior.predict(device, dt=dt)).float()
            if prior.warm_start and not check:
                opt.set_part_deltas(pred)

        # ---- observation + ROIs from the current pose; DINO extraction ---- #
        _sync()
        t_d = time.time()
        obs = PosedObservation(rgb, self.cam, self.dino_fn, depth, tcfg.max_roi_resolution)
        opt.set_observation(obs)
        if tcfg.use_dino:
            if tcfg.use_roi:
                for rf in obs.roi_frames:
                    _ = rf.dino_feats
            else:
                _ = obs.frame.dino_feats
        _sync()
        dino_ms = (time.time() - t_d) * 1000.0

        # ---- pose optimisation ------------------------------------------- #
        t_o = time.time()
        if check and pred is not None and prior.warm_start:
            self.n_pred_kept += _choose_start(opt, obs, pred, tcfg)
        info = opt.step(niter=tcfg.track_niter, use_depth=tcfg.use_depth, use_rgb=tcfg.use_rgb)
        _sync()
        opt_ms = (time.time() - t_o) * 1000.0

        # ---- motion prior: measurement update ----------------------------- #
        if prior is not None:
            fused = prior.update(opt.part_deltas.data, device)
            if prior.fuse:
                opt.set_part_deltas(fused.float())
                opt.apply_to_model()
        _sync()
        return {"time_s": time.time() - t_frame, "dino_ms": dino_ms, "opt_ms": opt_ms, "loss": info.loss}

    def deltas(self) -> torch.Tensor:
        return self.opt.part_deltas.data.detach().clone()

    def poses(self, deltas: Optional[torch.Tensor] = None) -> np.ndarray:
        return self.opt.get_all_part2world(deltas).detach().cpu().numpy()

    def finish(self, n_steps: int, verbose: bool, name: str) -> Dict[str, float]:
        self.model.load_snapshot(self.snapshot)
        diag = self.opt.depth_diagnostics()
        if self.prior is not None and hasattr(self.prior, "mean_nis"):
            diag["kalman_mean_nis"] = self.prior.mean_nis()
        if self.prior is not None and getattr(self.prior, "check_loss", False):
            diag["prediction_kept_rate"] = self.n_pred_kept / max((n_steps - 1) * self.opt.K, 1)
        if verbose and self.tcfg.use_depth and diag["depth_drop_rate"] > 0.2:
            print(f"  [{name}] WARNING: depth term dropped on {diag['depth_drop_rate']:.0%} of evaluations")
        return diag


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
    """Runs POGS tracking over every frame of an RGB-D sequence (offline: the
    tracker never falls behind the camera) and scores it against GT."""
    tcfg = tracking_cfg or cfg.tracking
    trk = _FrameTracker(model, groups, seq, dino_fn, cfg, tcfg, motion_prior)
    frames = seq.frames if max_frames is None else seq.frames[:max_frames]
    F_n, K = len(frames), groups.num_groups
    base_centroids = {gi: groups.init_p2w[gi, :3].detach().cpu().numpy() for gi in range(K)}
    acc = _alloc(F_n, K)
    times, dino_ms, opt_ms = np.zeros(F_n), np.zeros(F_n), np.zeros(F_n)
    losses = np.full(F_n, np.nan)
    deltas = np.zeros((F_n, K, 7), dtype=np.float32)

    init_info = trk.initialize(frames[0])
    if verbose:
        print(f"  [{name}] init: coarse={init_info['coarse_loss']:.4f} fine={init_info['fine_loss']:.4f} "
              f"in {init_info['init_time']:.1f}s")

    for fi, fr in enumerate(frames):
        st = trk.step(fr, first=fi == 0)
        times[fi], dino_ms[fi], opt_ms[fi], losses[fi] = st["time_s"], st["dino_ms"], st["opt_ms"], st["loss"]
        deltas[fi] = trk.deltas().cpu().numpy()
        _score_frame(seq, fi, trk.poses(), groups, group_to_gt, base_centroids, model_points, acc)
        if verbose and (fi % 10 == 0 or fi == F_n - 1):
            print(f"    [{name}] frame {fi:3d}/{F_n}  loss={st['loss']:.4f}  "
                  f"pos={np.nanmean(acc['trans'][fi]):.2f} cm  rot={np.nanmean(acc['rot'][fi]):.1f} deg  "
                  f"{times[fi] * 1000:.0f} ms")

    diag = trk.finish(F_n, verbose, name)
    return TrackingResult(
        name=name, trans_err_cm=acc["trans"], rot_err_deg=acc["rot"], yaw_err_deg=acc["yaw"],
        add=acc["add"], adds=acc["adds"], frame_times=times, dino_ms=dino_ms, opt_ms=opt_ms,
        losses=losses, pred_poses=acc["preds"], gt_poses=acc["gts"], part_deltas=deltas,
        occluded=np.array([bool(getattr(f, "occluded", False)) for f in frames]),
        group_names=list(groups.names or []), init_time=init_info["init_time"], diagnostics=diag,
    )


# --------------------------------------------------------------------------- #
# Real-time protocol
# --------------------------------------------------------------------------- #


def pick_next_frame(timestamps: np.ndarray, t_free: float, last: int) -> Optional[int]:
    """The frame a tracker that becomes free at ``t_free`` works on next: the
    newest frame that has arrived (frames in between are skipped), or, if none
    newer than ``last`` has arrived yet, the next one (it waits for it).
    None when the stream is over."""
    n = len(timestamps)
    if last + 1 >= n:
        return None
    arrived = np.nonzero(timestamps[last + 1:] <= t_free + 1e-9)[0]
    return int(last + 1 + arrived[-1]) if arrived.size else last + 1


def estimate_sources(timestamps: np.ndarray, done_times: np.ndarray) -> np.ndarray:
    """For every camera frame, the index of the newest tracked result already
    finished when that frame arrives (-1: none yet, only the initial pose)."""
    return np.searchsorted(np.asarray(done_times), np.asarray(timestamps) + 1e-9, side="right") - 1


@dataclass
class RealtimeResult:
    """``now``: at every camera frame, the pose the tracker has published so
    far (its last finished estimate, held). ``now_extrap``: the same estimate
    moved forward to the frame's time by the Kalman filter's velocity (motion
    prior methods only). ``processed``: accuracy at the frames actually tracked,
    with ``frame_times`` = measured per-frame latency."""
    name: str
    now: TrackingResult
    processed: TrackingResult
    now_extrap: Optional[TrackingResult]
    timestamps: np.ndarray            # (F,) camera frame times (s)
    proc_frames: np.ndarray           # (P,) camera frame index of every tracked frame
    proc_start: np.ndarray            # (P,) s
    proc_done: np.ndarray             # (P,) s
    cam_src: np.ndarray               # (F,) index into proc_* (-1: initial pose)

    def schedule(self) -> Dict[str, object]:
        return {"timestamps": self.timestamps.tolist(), "proc_frames": self.proc_frames.tolist(),
                "proc_start": self.proc_start.tolist(), "proc_done": self.proc_done.tolist(),
                "cam_src": self.cam_src.tolist()}


def realtime_stats(timestamps: np.ndarray, proc_frames: np.ndarray, proc_start: np.ndarray,
                   proc_done: np.ndarray, cam_src: np.ndarray) -> Dict[str, float]:
    lat = proc_done - proc_start
    ok = cam_src >= 0
    age = np.full(len(timestamps), np.nan)
    age[ok] = timestamps[ok] - timestamps[proc_frames[cam_src[ok]]]
    skipped = np.diff(proc_frames) - 1 if len(proc_frames) > 1 else np.zeros(1)
    cam_hz = 1.0 / float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else float("nan")
    return {
        "camera_hz": cam_hz,
        "latency_ms": float(np.mean(lat) * 1000),
        "tracker_hz": float(1.0 / np.mean(lat)),
        "frames_tracked_frac": len(proc_frames) / len(timestamps),
        "frames_skipped_mean": float(np.mean(skipped)),
        "estimate_age_ms": float(np.nanmean(age) * 1000),
        "estimate_age_max_ms": float(np.nanmax(age) * 1000),
    }


def track_realtime(
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
    latency_fn: Optional[Callable[[Dict[str, float]], float]] = None,
    verbose: bool = True,
) -> RealtimeResult:
    """Real-time protocol on a high-rate camera stream (``seq`` timestamps).

    Simulated clock: the tracker starts on frame 0 at t = 0 (initialisation is
    done beforehand, while the object is still at rest); each tracked frame
    takes its MEASURED compute time (``latency_fn(step_info)`` overrides it,
    e.g. for what-if runs); when it finishes it takes the newest frame that has
    arrived, skipping the rest (``pick_next_frame``). The motion prior predicts
    over the true gap between tracked frames. Scored at every camera frame on
    what the tracker has published by then (``estimate_sources``)."""
    tcfg = tracking_cfg or cfg.tracking
    trk = _FrameTracker(model, groups, seq, dino_fn, cfg, tcfg, motion_prior)
    frames = seq.frames
    ts = np.array([f.timestamp_sec for f in frames], dtype=np.float64)
    ts = ts - ts[0]
    F_n, K = len(frames), groups.num_groups
    base_centroids = {gi: groups.init_p2w[gi, :3].detach().cpu().numpy() for gi in range(K)}
    has_kf = motion_prior is not None and hasattr(motion_prior, "filters")

    init_info = trk.initialize(frames[0])
    init_deltas = trk.deltas()
    if verbose:
        print(f"  [{name}] init in {init_info['init_time']:.1f}s; camera {1.0 / np.median(np.diff(ts)):.0f} Hz, "
              f"{F_n} frames over {ts[-1]:.1f}s")

    proc, starts, dones, steps, p_deltas, p_filters = [], [], [], [], [], []
    t_free, last = 0.0, -1
    while True:
        j = pick_next_frame(ts, t_free, last)
        if j is None or (last >= 0 and t_free > ts[-1]):
            break
        start = max(t_free, ts[j])
        st = trk.step(frames[j], first=last < 0, dt=None if last < 0 else float(ts[j] - ts[last]))
        lat = float(latency_fn(st)) if latency_fn is not None else st["time_s"]
        proc.append(j), starts.append(start), dones.append(start + lat), steps.append(st)
        p_deltas.append(trk.deltas())
        p_filters.append(copy.deepcopy(motion_prior.filters) if has_kf else None)
        t_free, last = start + lat, j
        if verbose and len(proc) % 5 == 1:
            print(f"    [{name}] t={start:5.2f}s  frame {j:3d}/{F_n}  {lat * 1000:.0f} ms  loss={st['loss']:.4f}")

    proc_a, start_a, done_a = np.array(proc), np.array(starts), np.array(dones)
    src = estimate_sources(ts, done_a)

    # ---- score: frames actually tracked ------------------------------------ #
    P_n = len(proc)
    acc_p = _alloc(P_n, K)
    for r, (j, d) in enumerate(zip(proc, p_deltas)):
        _score_frame(seq, j, trk.poses(d), groups, group_to_gt, base_centroids, model_points, acc_p, row=r)

    # ---- score: every camera frame, held and extrapolated estimates ------- #
    acc_h, acc_x = _alloc(F_n, K), (_alloc(F_n, K) if has_kf else None)
    d_h = np.zeros((F_n, K, 7), dtype=np.float32)
    d_x = np.zeros((F_n, K, 7), dtype=np.float32) if has_kf else None
    for c in range(F_n):
        k = src[c]
        d = init_deltas if k < 0 else p_deltas[k]
        d_h[c] = d.cpu().numpy()
        _score_frame(seq, c, trk.poses(d), groups, group_to_gt, base_centroids, model_points, acc_h)
        if has_kf:
            if k >= 0:
                age = float(ts[c] - ts[proc[k]])
                dx = torch.from_numpy(np.stack([f.extrapolate(p, age) for f, p in
                                                zip(p_filters[k], d.cpu().double().numpy())])).to(d.device)
            else:
                dx = d
            d_x[c] = dx.cpu().numpy()
            _score_frame(seq, c, trk.poses(dx.float()), groups, group_to_gt, base_centroids, model_points, acc_x)

    diag = trk.finish(P_n, verbose, name)
    diag.update(realtime_stats(ts, proc_a, start_a, done_a, src))
    occ = np.array([bool(getattr(f, "occluded", False)) for f in frames])
    names = list(groups.names or [])
    zeros = np.zeros(F_n)

    def timeline(acc, deltas):
        return TrackingResult(
            name=name, trans_err_cm=acc["trans"], rot_err_deg=acc["rot"], yaw_err_deg=acc["yaw"],
            add=acc["add"], adds=acc["adds"], frame_times=zeros, dino_ms=zeros * np.nan, opt_ms=zeros * np.nan,
            losses=zeros * np.nan, pred_poses=acc["preds"], gt_poses=acc["gts"], part_deltas=deltas,
            occluded=occ, group_names=names, init_time=init_info["init_time"], diagnostics=dict(diag),
        )

    processed = TrackingResult(
        name=name, trans_err_cm=acc_p["trans"], rot_err_deg=acc_p["rot"], yaw_err_deg=acc_p["yaw"],
        add=acc_p["add"], adds=acc_p["adds"], frame_times=done_a - start_a,
        dino_ms=np.array([s["dino_ms"] for s in steps]), opt_ms=np.array([s["opt_ms"] for s in steps]),
        losses=np.array([s["loss"] for s in steps]), pred_poses=acc_p["preds"], gt_poses=acc_p["gts"],
        part_deltas=np.stack([d.cpu().numpy() for d in p_deltas]), occluded=occ[proc_a],
        group_names=names, init_time=init_info["init_time"], diagnostics=dict(diag),
    )
    if verbose:
        print(f"  [{name}] tracked {P_n}/{F_n} frames, {diag['latency_ms']:.0f} ms/frame "
              f"({diag['tracker_hz']:.1f} Hz), mean estimate age {diag['estimate_age_ms']:.0f} ms")
    return RealtimeResult(
        name=name, now=timeline(acc_h, d_h), processed=processed,
        now_extrap=timeline(acc_x, d_x) if has_kf else None,
        timestamps=ts, proc_frames=proc_a, proc_start=start_a, proc_done=done_a, cam_src=src,
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
