"""
Offline tracking driver, metrics, and the paper's benchmark protocols.

Metrics follow Table I of the paper:
    * mean / std translation error (cm)
    * max and mean *consecutive successful object resets* before tracking is lost
plus the standard 6D-pose measures (ADD, ADD-S, rotation error) that the
simulator's ground truth makes available.

Ablations mirror the paper's two columns:
    "No Depth"  -> depth loss off during tracking
    "No DINO"   -> RGB loss substituted for the DINO feature loss
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import POGSConfig, TrackingConfig
from .geometry import PinholeCamera, geodesic_rotation_error
from .grouping import ObjectGroups
from .model import POGSModel
from .simscene import TrackingSequence, sample_object_model_points
from .tracking import DinoFn, DinoPrefetcher, PosedObservation, RigidGroupOptimizer


# --------------------------------------------------------------------------- #
# Pose metrics
# --------------------------------------------------------------------------- #


def translation_error_cm(T_pred: np.ndarray, T_gt: np.ndarray) -> float:
    return float(np.linalg.norm(T_pred[:3, 3] - T_gt[:3, 3]) * 100.0)


def rotation_error_deg(T_pred: np.ndarray, T_gt: np.ndarray) -> float:
    return geodesic_rotation_error(T_pred[:3, :3], T_gt[:3, :3])


def add_metric(points: np.ndarray, T_pred: np.ndarray, T_gt: np.ndarray) -> float:
    """Average Distance of Model Points (Hinterstoisser et al.), in metres."""
    a = points @ T_pred[:3, :3].T + T_pred[:3, 3]
    b = points @ T_gt[:3, :3].T + T_gt[:3, 3]
    return float(np.linalg.norm(a - b, axis=1).mean())


def adds_metric(
    points: np.ndarray, T_pred: np.ndarray, T_gt: np.ndarray,
    max_pts: int = 1500, chunk: int = 256,
) -> float:
    """ADD-S: symmetry-aware nearest-neighbour variant.

    Computed in row chunks rather than as one ``(N, N, 3)`` difference tensor --
    at the default ``max_pts`` that array alone is 51 MB of float64, which is
    enough to fail on a memory-constrained host.
    """
    if len(points) > max_pts:
        idx = np.random.default_rng(0).choice(len(points), max_pts, replace=False)
        points = points[idx]
    # float64: the |a|^2 - 2a.b + |b|^2 expansion cancels badly in float32, which
    # shows up as a non-zero ADD-S for an exact pose match
    a = (points @ T_pred[:3, :3].T + T_pred[:3, 3]).astype(np.float64)
    b = (points @ T_gt[:3, :3].T + T_gt[:3, 3]).astype(np.float64)
    b2 = (b * b).sum(axis=1)
    total = 0.0
    for i in range(0, len(a), chunk):
        block = a[i : i + chunk]
        # |a - b|^2 = |a|^2 - 2 a.b + |b|^2, so no (n, m, 3) intermediate
        d2 = (block * block).sum(axis=1)[:, None] - 2.0 * (block @ b.T) + b2[None, :]
        total += float(np.sqrt(np.maximum(d2.min(axis=1), 0.0)).sum())
    return total / max(len(a), 1)


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #


@dataclass
class TrackingResult:
    name: str
    trans_err_cm: np.ndarray          # (F, K)
    rot_err_deg: np.ndarray           # (F, K)
    add: np.ndarray                   # (F, K)
    adds: np.ndarray                  # (F, K)
    frame_times: np.ndarray           # (F,)
    iters: np.ndarray                 # (F,)
    losses: np.ndarray                # (F,)
    reset_frames: List[int]
    hand_frames: List[int]
    pred_poses: np.ndarray            # (F, K, 4, 4)
    gt_poses: np.ndarray              # (F, K, 4, 4)
    renders: List[np.ndarray] = field(default_factory=list)
    init_time: float = 0.0
    depth_diag: Dict[str, float] = field(default_factory=dict)
    """From ``RigidGroupOptimizer.depth_diagnostics()`` -- how much of the depth
    term actually reached the loss. A ``depth_drop_rate`` near 1.0 invalidates
    any comparison against the "No Depth" ablation."""

    # -- headline numbers -------------------------------------------------- #

    def mean_position_error_cm(self) -> float:
        return float(np.nanmean(self.trans_err_cm))

    def std_position_error_cm(self) -> float:
        return float(np.nanstd(self.trans_err_cm))

    def mean_rotation_error_deg(self) -> float:
        return float(np.nanmean(self.rot_err_deg))

    def fps(self) -> float:
        return float(1.0 / max(np.nanmean(self.frame_times), 1e-9))

    def tracking_lost(self, thresh_cm: float = 5.0) -> np.ndarray:
        """Per-frame boolean: has any object's error exceeded the threshold?
        The paper defines tracking loss as the point where "repeated grasping
        failures occur due to irrecoverable errors in object state estimation"."""
        return (self.trans_err_cm > thresh_cm).any(axis=1)

    def consecutive_resets(self, thresh_cm: float = 5.0) -> Tuple[int, float]:
        """(max, mean) consecutive successful object resets before tracking is
        lost -- the "Max/Mean Consecutive OR" rows of Table I."""
        lost = self.tracking_lost(thresh_cm)
        runs, cur = [], 0
        for f in self.reset_frames:
            if f < len(lost) and not lost[max(f - 1, 0) : f + 2].any():
                cur += 1
            else:
                runs.append(cur)
                cur = 0
        runs.append(cur)
        return (max(runs) if runs else 0, float(np.mean(runs)) if runs else 0.0)

    def summary(self, thresh_cm: float = 5.0) -> Dict[str, float]:
        mx, mean = self.consecutive_resets(thresh_cm)
        occl = np.zeros(len(self.frame_times), dtype=bool)
        occl[self.hand_frames] = True
        return {
            "name": self.name,
            "mean_pos_err_cm": self.mean_position_error_cm(),
            "std_pos_err_cm": self.std_position_error_cm(),
            "mean_rot_err_deg": self.mean_rotation_error_deg(),
            "mean_add_cm": float(np.nanmean(self.add) * 100),
            "mean_adds_cm": float(np.nanmean(self.adds) * 100),
            "max_consecutive_or": mx,
            "mean_consecutive_or": mean,
            "frac_lost": float(self.tracking_lost(thresh_cm).mean()),
            "pos_err_occluded_cm": float(np.nanmean(self.trans_err_cm[occl])) if occl.any() else float("nan"),
            "pos_err_clear_cm": float(np.nanmean(self.trans_err_cm[~occl])) if (~occl).any() else float("nan"),
            "fps": self.fps(),
            "mean_iters": float(np.mean(self.iters)),
            "init_time_s": self.init_time,
            "depth_drop_rate": self.depth_diag.get("depth_drop_rate", float("nan")),
        }


TrackingMetrics = TrackingResult


# --------------------------------------------------------------------------- #
# The tracking loop
# --------------------------------------------------------------------------- #


def _gt_pose_for_group(
    seq: TrackingSequence, frame_idx: int, gt_obj_id: int, base_centroid: np.ndarray
) -> np.ndarray:
    """GT object pose expressed the same way POGS expresses it: identity
    rotation at the scan-time centroid, then the world-frame delta."""
    T = seq.frames[frame_idx].gt_delta.get(gt_obj_id, np.eye(4))
    out = np.eye(4)
    out[:3, :3] = T[:3, :3]
    # the object rotates about its own centroid, so the induced centroid motion is
    # R * c + t, expressed relative to the scan centroid
    out[:3, 3] = T[:3, :3] @ base_centroid + T[:3, 3]
    return out


def track_sequence(
    model: POGSModel,
    groups: ObjectGroups,
    seq: TrackingSequence,
    dino_fn: DinoFn,
    cfg: POGSConfig,
    group_to_gt: Optional[Dict[int, int]] = None,
    name: str = "POGS",
    tracking_cfg: Optional[TrackingConfig] = None,
    model_points: Optional[Dict[int, np.ndarray]] = None,
    save_renders: bool = False,
    render_every: int = 1,
    prefetch_dino: bool = False,
    verbose: bool = True,
    max_frames: Optional[int] = None,
    **kwargs,
) -> TrackingResult:
    """Runs POGS tracking over an RGB-D sequence and scores it against GT."""
    if group_to_gt is None:
        group_to_gt = kwargs.get("group2gt", {})
    tcfg = tracking_cfg or cfg.tracking
    device = cfg.runtime.device
    raster_kwargs = (
        dict(max_per_tile=cfg.runtime.torch_raster_max_per_tile,
             tile_chunk=cfg.runtime.torch_raster_tile_chunk)
        if model.backend == "torch" else {}
    )

    snapshot = model.state_snapshot()
    opt = RigidGroupOptimizer(tcfg, model, groups, raster_kwargs=raster_kwargs)

    base_cam = seq.camera.to(device)
    scale = tcfg.match_resolution / min(base_cam.width, base_cam.height)
    cam = base_cam.rescale(min(scale, 1.0)) if scale < 1.0 else base_cam

    frames = seq.frames if max_frames is None else seq.frames[:max_frames]
    F_n, K = len(frames), groups.num_groups

    def make_obs(fr):
        return PosedObservation(
            fr.rgb.to(device), cam, dino_fn, fr.depth.to(device), tcfg.max_roi_resolution
        )

    def build_obs_with_rois(fr):
        """Observation with every object's ROI already attached, computed from
        the optimiser's *current* pose estimate."""
        obs = make_obs(fr)
        if tcfg.use_roi:
            for i in range(K):
                try:
                    obs.add_roi(*opt.calculate_roi(i, obs.frame.camera))
                except RuntimeError:
                    obs.add_roi(0.0, 1.0, 0.0, 1.0)
        return obs

    # ---- initialisation -------------------------------------------------- #
    t0 = time.time()
    obs0 = make_obs(frames[0])
    init_info = opt.initialize_obj_pose(obs0, n_seeds=tcfg.init_n_seeds, verbose=verbose)
    init_time = time.time() - t0
    if verbose:
        print(f"  [{name}] init: coarse={init_info['coarse_loss']:.4f} "
              f"fine={init_info['fine_loss']:.4f} in {init_time:.1f}s")

    prefetcher = DinoPrefetcher(enabled=prefetch_dino)

    base_centroids = {gi: groups.init_p2w[gi, :3].detach().cpu().numpy() for gi in range(K)}
    mp = model_points or {}

    trans = np.full((F_n, K), np.nan)
    rot = np.full((F_n, K), np.nan)
    add = np.full((F_n, K), np.nan)
    adds = np.full((F_n, K), np.nan)
    times = np.zeros(F_n)
    iters = np.zeros(F_n, dtype=int)
    losses = np.full(F_n, np.nan)
    preds = np.tile(np.eye(4), (F_n, K, 1, 1))
    gts = np.tile(np.eye(4), (F_n, K, 1, 1))
    renders: List[np.ndarray] = []

    for fi, fr in enumerate(frames):
        t_frame = time.time()

        if fi == 0:
            obs = opt.frame                        # built by initialize_obj_pose
        else:
            cached = prefetcher.take(fi)
            obs = cached if cached is not None else build_obs_with_rois(fr)
            opt.set_frame(obs)

        n_it = tcfg.track_niter
        info = opt.step(niter=n_it, use_depth=tcfg.use_depth, use_rgb=tcfg.use_rgb)

        times[fi] = time.time() - t_frame
        iters[fi] = n_it
        losses[fi] = info.loss

        # ---- prepare frame t+1 ROI if prefetching ----------------------- #
        if fi + 1 < F_n and prefetch_dino:
            prefetcher.submit(fi + 1, build_obs_with_rois(frames[fi + 1]))

        # ---- score ------------------------------------------------------- #
        P = opt.get_all_part2world().detach().cpu().numpy()
        for gi in range(K):
            gt_id = group_to_gt.get(gi, -1)
            if gt_id < 0:
                continue
            T_gt = _gt_pose_for_group(seq, fi, gt_id, base_centroids[gi])
            preds[fi, gi], gts[fi, gi] = P[gi], T_gt
            trans[fi, gi] = translation_error_cm(P[gi], T_gt)
            rot[fi, gi] = rotation_error_deg(P[gi], T_gt)
            pts = mp.get(gt_id, mp.get(gi))
            if pts is not None:
                if isinstance(pts, torch.Tensor):
                    pts = pts.detach().cpu().numpy()
                bc = base_centroids[gi]
                if isinstance(bc, torch.Tensor):
                    bc = bc.detach().cpu().numpy()
                local = pts - bc
                add[fi, gi] = add_metric(local, P[gi], T_gt)
                adds[fi, gi] = adds_metric(local, P[gi], T_gt)

        if save_renders and fi % render_every == 0:
            with torch.no_grad():
                r = opt.render_full(cam)["rgb"].detach().cpu().numpy()
            renders.append((np.clip(r, 0, 1) * 255).astype(np.uint8))

        if verbose and (fi % 10 == 0 or fi == F_n - 1):
            print(f"    [{name}] frame {fi:3d}/{F_n}  loss={info.loss:.4f}  "
                  f"pos_err={np.nanmean(trans[fi]):.2f} cm  rot={np.nanmean(rot[fi]):.1f} deg  "
                  f"iters={n_it}  {times[fi]*1000:.0f} ms")

    model.load_snapshot(snapshot)

    diag = opt.depth_diagnostics()
    if verbose and tcfg.use_depth and diag["depth_drop_rate"] > 0.5:
        print(f"  [{name}] WARNING: the depth term was dropped on "
              f"{diag['depth_drop_rate']:.0%} of evaluations -- the rendered "
              f"accumulation never reached {tcfg.accumulation_thresh}. Train the "
              f"splat longer or lower TrackingConfig.accumulation_thresh; until "
              f"then the 'No Depth' ablation is not a real comparison.")

    return TrackingResult(
        name=name, trans_err_cm=trans, rot_err_deg=rot, add=add, adds=adds,
        frame_times=times, iters=iters, losses=losses,
        reset_frames=[i for i, f in enumerate(frames) if getattr(f, "reset_event", False) or getattr(getattr(f, "event", None), "reset_event", False)],
        hand_frames=[i for i, f in enumerate(frames) if getattr(f, "hand_visible", False) or getattr(getattr(f, "event", None), "hand_occlusion", False)],
        pred_poses=preds, gt_poses=gts,
        renders=renders, init_time=init_time, depth_diag=diag,
    )


# --------------------------------------------------------------------------- #
# Ablations (Table I columns)
# --------------------------------------------------------------------------- #


def run_ablations(
    model: POGSModel,
    groups: ObjectGroups,
    seq: TrackingSequence,
    dino_fn: DinoFn,
    cfg: POGSConfig,
    group_to_gt: Optional[Dict[int, int]] = None,
    model_points: Optional[Dict[int, np.ndarray]] = None,
    variants: Sequence[str] = ("POGS", "No Depth", "No DINO"),
    max_frames: Optional[int] = None,
    verbose: bool = True,
    **kwargs,
) -> Dict[str, TrackingResult]:
    if group_to_gt is None:
        group_to_gt = kwargs.get("group2gt", {})
    import copy

    out: Dict[str, TrackingResult] = {}
    for v in variants:
        tc = copy.deepcopy(cfg.tracking)
        if v == "No Depth":
            tc.use_depth = False
        elif v == "No DINO":
            tc.use_dino = False
            tc.use_rgb = True
        if verbose:
            print(f"\n--- ablation: {v} (depth={tc.use_depth}, dino={tc.use_dino}, rgb={tc.use_rgb}) ---")
        out[v] = track_sequence(
            model, groups, seq, dino_fn, cfg, group_to_gt, name=v,
            tracking_cfg=tc, model_points=model_points,
            max_frames=max_frames, verbose=verbose,
        )
    return out


def results_table(results: Dict[str, TrackingResult], thresh_cm: float = 5.0):
    """Table-I-shaped comparison across variants."""
    rows = [r.summary(thresh_cm) for r in results.values()]
    try:
        import pandas as pd

        df = pd.DataFrame(rows).set_index("name")
        return df.round(3)
    except ImportError:
        return rows


# --------------------------------------------------------------------------- #
# Offline vs online protocols
# --------------------------------------------------------------------------- #


def online_vs_offline(
    model: POGSModel,
    groups: ObjectGroups,
    seq: TrackingSequence,
    dino_fn: DinoFn,
    cfg: POGSConfig,
    group_to_gt: Dict[int, int],
    model_points: Optional[Dict[int, np.ndarray]] = None,
    offline_smoothing: bool = True,
    max_frames: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, TrackingResult]:
    """POGS as shipped is *causal*: frame t only ever sees frames <= t.

    ``offline`` here means running the same causal tracker and then applying an
    RTS-style fixed-interval smoother over the recorded trajectory, which is
    only legal when the whole sequence is available.  This isolates exactly how
    much accuracy the online constraint costs.
    """
    out = {}
    out["online (causal)"] = track_sequence(
        model, groups, seq, dino_fn, cfg, group_to_gt, name="online (causal)",
        model_points=model_points,
        max_frames=max_frames, verbose=verbose,
    )
    if offline_smoothing:
        sm = smooth_trajectory(out["online (causal)"])
        out["offline (smoothed)"] = sm
    return out


def smooth_trajectory(result: TrackingResult, window_size: int = 5) -> TrackingResult:
    """Non-causal zero-phase smoothing of the recorded pose trajectory.

    Translation is filtered forward-backward; rotation is averaged in the
    tangent space around each frame's estimate. Metrics are then recomputed.
    """
    import copy
    from .geometry import quat_to_rotmat, rotmat_to_quat, so3_exp, so3_log

    F_n, K = result.pred_poses.shape[:2]
    win = 5
    half = win // 2
    poses = result.pred_poses.copy()
    for gi in range(K):
        T = poses[:, gi]
        pos = T[:, :3, 3]
        pos_s = np.stack(
            [np.convolve(pos[:, d], np.ones(win) / win, mode="same") for d in range(3)], axis=-1
        )
        # fix the convolution edges
        pos_s[:half] = pos[:half]
        pos_s[-half:] = pos[-half:]
        R = torch.from_numpy(T[:, :3, :3]).float()
        q = rotmat_to_quat(R)
        q_s = q.clone()
        for f in range(half, F_n - half):
            ref = q[f]
            tang = torch.stack(
                [so3_log(_qmul(_qconj(ref), q[f + k])) for k in range(-half, half + 1)]
            ).mean(0)
            q_s[f] = _qmul(ref, so3_exp(tang))
        Rs = quat_to_rotmat(q_s).numpy()
        poses[:, gi, :3, :3] = Rs
        poses[:, gi, :3, 3] = pos_s

    trans = np.full((F_n, K), np.nan)
    rot = np.full((F_n, K), np.nan)
    for f in range(F_n):
        for gi in range(K):
            if np.isnan(result.trans_err_cm[f, gi]):
                continue
            trans[f, gi] = translation_error_cm(poses[f, gi], result.gt_poses[f, gi])
            rot[f, gi] = rotation_error_deg(poses[f, gi], result.gt_poses[f, gi])

    out = copy.copy(result)
    out.name = "offline (smoothed)"
    out.trans_err_cm = trans
    out.rot_err_deg = rot
    out.pred_poses = poses
    out.renders = []
    return out


def _qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    from .geometry import quat_multiply

    return quat_multiply(a.unsqueeze(0), b.unsqueeze(0)).squeeze(0)


def _qconj(a: torch.Tensor) -> torch.Tensor:
    from .geometry import quat_conjugate

    return quat_conjugate(a.unsqueeze(0)).squeeze(0)


# --------------------------------------------------------------------------- #
# Speed study
# --------------------------------------------------------------------------- #


def speed_study(
    model: POGSModel,
    groups: ObjectGroups,
    seq: TrackingSequence,
    dino_fn: DinoFn,
    cfg: POGSConfig,
    group_to_gt: Optional[Dict[int, int]] = None,
    iter_counts: Sequence[int] = (25, 15, 8),
    model_points: Optional[Dict[int, np.ndarray]] = None,
    max_frames: Optional[int] = None,
    verbose: bool = True,
    **kwargs,
) -> Dict[str, TrackingResult]:
    if group_to_gt is None:
        group_to_gt = kwargs.get("group2gt", {})
    """Accuracy/latency trade-off across optimizer iteration counts."""
    import copy

    out: Dict[str, TrackingResult] = {}
    for n in iter_counts:
        tc = copy.deepcopy(cfg.tracking)
        tc.track_niter = n
        out[f"{n} iters"] = track_sequence(
            model, groups, seq, dino_fn, cfg, group_to_gt, name=f"{n} iters",
            tracking_cfg=tc, model_points=model_points,
            max_frames=max_frames, verbose=verbose,
        )
    return out


# --------------------------------------------------------------------------- #
# Sequential Object Reset (Sec. V-A) -- the rest of Table I
# --------------------------------------------------------------------------- #


def _delta_from_pose(T: np.ndarray, scan_centroid: np.ndarray) -> np.ndarray:
    """POGS reports a pose as ``[R | transformed centroid]``. Recover the
    world-frame rigid delta that maps scan-time geometry onto it."""
    out = np.eye(4)
    out[:3, :3] = T[:3, :3]
    out[:3, 3] = T[:3, 3] - T[:3, :3] @ scan_centroid
    return out


def _placement_point(points: np.ndarray) -> np.ndarray:
    """Where the robot would set an object down on this one: the centre of its
    top face."""
    return np.array([points[:, 0].mean(), points[:, 1].mean(), points[:, 2].max()])


@dataclass
class ResetAttempt:
    frame: int
    pick_success: bool
    place_success: bool
    contact_err_cm: float
    """Distance from the planned gripper contacts to the object's true surface.
    Above ``contact_tol_cm`` the jaws close on empty air -- a failed pick."""
    place_err_cm: float
    """'the translation error between the intended and actual placement
    positions' (Sec. V-A)."""
    pick_pose_err_cm: float
    n_grasps: int


@dataclass
class ResetProtocolResult:
    """One trial of the Sec. V-A sequential-object-reset benchmark."""

    name: str
    tier: int
    pick_name: str
    place_name: str
    attempts: List[ResetAttempt]
    tracking: TrackingResult
    contact_tol_cm: float = 1.0
    place_tol_cm: float = 5.0

    @property
    def n(self) -> int:
        return len(self.attempts)

    def successful_pick_rate(self) -> Tuple[int, int]:
        return sum(a.pick_success for a in self.attempts), self.n

    def successful_place_rate(self) -> Tuple[int, int]:
        """The paper only counts a place attempt after a successful pick."""
        picked = [a for a in self.attempts if a.pick_success]
        return sum(a.place_success for a in picked), len(picked)

    def consecutive_resets(self) -> Tuple[int, float]:
        """'Consecutive OR refers to a single trial with repeated Object Resets
        without losing tracking' -- a reset counts only if both the pick and the
        place succeeded."""
        runs, cur = [], 0
        for a in self.attempts:
            if a.pick_success and a.place_success:
                cur += 1
            else:
                runs.append(cur)
                cur = 0
        runs.append(cur)
        return (max(runs) if runs else 0), (float(np.mean(runs)) if runs else 0.0)

    def summary(self) -> Dict[str, object]:
        mx, mean = self.consecutive_resets()
        pick_ok, pick_n = self.successful_pick_rate()
        place_ok, place_n = self.successful_place_rate()
        errs = np.array([a.place_err_cm for a in self.attempts], dtype=float)
        return {
            "name": self.name,
            "tier": self.tier,
            "task": f"{self.pick_name} to {self.place_name}",
            "max_consecutive_or": mx,
            "mean_consecutive_or": round(mean, 2),
            "successful_pick_rate": f"{pick_ok}/{pick_n}",
            "successful_place_rate": f"{place_ok}/{place_n}",
            "mean_position_error_cm": round(float(np.nanmean(errs)), 2) if len(errs) else float("nan"),
            "std_position_error_cm": round(float(np.nanstd(errs)), 2) if len(errs) else float("nan"),
        }


def evaluate_pick(
    pogs_points_at_estimate: np.ndarray,
    true_surface_points: np.ndarray,
    contact_tol_cm: float = 1.0,
    grasp_kwargs: Optional[Dict[str, object]] = None,
) -> Tuple[bool, float, int]:
    """Would the grasp POGS plans actually land on the object?

    The paper's failure mode is explicit: resets continue 'until POGS loses
    tracking of the objects, defined as when repeated grasping failures occur
    due to irrecoverable errors in object state estimation'.  So a pick is
    scored by planning on POGS's *belief* about the object -- the Gaussian means
    at the estimated pose, which is exactly what Sec. IV-E hands to
    Contact-GraspNet -- and then checking the planned contacts against the
    object's *true* surface.  A pose error larger than the gripper pad closes
    the jaws on empty air.
    """
    from sklearn.neighbors import NearestNeighbors

    from .grasp import nms_grasps, sample_antipodal_grasps

    kw = dict(n_candidates=8000, seed=0)
    kw.update(grasp_kwargs or {})
    gs = nms_grasps(sample_antipodal_grasps(pogs_points_at_estimate, **kw))
    if len(gs) == 0 or len(true_surface_points) < 4:
        return False, float("inf"), len(gs)

    jaws = gs.contacts[0]          # (2, 3) the two jaw contacts of the best grasp
    nn = NearestNeighbors(n_neighbors=1).fit(true_surface_points)
    d, _ = nn.kneighbors(jaws)
    err_cm = float(d.max() * 100.0)
    return err_cm <= contact_tol_cm, err_cm, len(gs)


def object_reset_protocol(
    model: POGSModel,
    groups: ObjectGroups,
    seq: TrackingSequence,
    dino_fn: DinoFn,
    cfg: POGSConfig,
    group_to_gt: Dict[int, int],
    pick_group: int,
    place_group: int,
    model_points: Dict[int, np.ndarray],
    name: str = "POGS",
    tier: int = 1,
    pick_name: str = "object",
    place_name: str = "target",
    tracking_cfg: Optional[TrackingConfig] = None,
    contact_tol_cm: float = 1.0,
    place_tol_cm: float = 5.0,
    grasp_kwargs: Optional[Dict[str, object]] = None,
    result: Optional[TrackingResult] = None,
    max_frames: Optional[int] = None,
    verbose: bool = True,
) -> ResetProtocolResult:
    """Run the Sec. V-A protocol and score every row of Table I.

    At each scripted reset the robot (a) plans a grasp on the pick object using
    POGS's estimate of where it is, and (b) sets it down on the place object,
    again using POGS's estimate.  Both actions are then scored against ground
    truth -- the quantity the real experiment measured with calipers.

    Pass ``result`` to score an already-computed :func:`track_sequence` run
    instead of re-tracking.
    """
    res = result if result is not None else track_sequence(
        model, groups, seq, dino_fn, cfg, group_to_gt, name=name,
        tracking_cfg=tracking_cfg,
        model_points=model_points, max_frames=max_frames, verbose=verbose,
    )

    means = model.gauss_params["means"].data.detach().cpu().numpy()
    pick_pts_scan = means[groups.group_masks[pick_group].cpu().numpy()]
    place_pts_scan = means[groups.group_masks[place_group].cpu().numpy()]
    c_pick = groups.init_p2w[pick_group, :3].detach().cpu().numpy()
    c_place = groups.init_p2w[place_group, :3].detach().cpu().numpy()
    place_point_scan = _placement_point(place_pts_scan)

    gt_pick = group_to_gt.get(pick_group, -1)
    mp_pick = model_points.get(gt_pick)
    if mp_pick is None or len(mp_pick) == 0:
        raise ValueError(
            f"object_reset_protocol needs ground-truth surface points for the pick "
            f"object (gt id {gt_pick}); pass them via model_points."
        )

    attempts: List[ResetAttempt] = []
    for f in res.reset_frames:
        if f >= res.pred_poses.shape[0]:
            continue
        d_pick_est = _delta_from_pose(res.pred_poses[f, pick_group], c_pick)
        d_pick_gt = _delta_from_pose(res.gt_poses[f, pick_group], c_pick)
        d_place_est = _delta_from_pose(res.pred_poses[f, place_group], c_place)
        d_place_gt = _delta_from_pose(res.gt_poses[f, place_group], c_place)

        pts_est = pick_pts_scan @ d_pick_est[:3, :3].T + d_pick_est[:3, 3]
        true_surf = mp_pick @ d_pick_gt[:3, :3].T + d_pick_gt[:3, 3]
        ok_pick, contact_err, n_g = evaluate_pick(
            pts_est, true_surf, contact_tol_cm, grasp_kwargs
        )

        p_est = d_place_est[:3, :3] @ place_point_scan + d_place_est[:3, 3]
        p_gt = d_place_gt[:3, :3] @ place_point_scan + d_place_gt[:3, 3]
        place_err = float(np.linalg.norm(p_est - p_gt) * 100.0)

        attempts.append(
            ResetAttempt(
                frame=f,
                pick_success=bool(ok_pick),
                place_success=bool(ok_pick and place_err <= place_tol_cm),
                contact_err_cm=contact_err,
                place_err_cm=place_err,
                pick_pose_err_cm=float(res.trans_err_cm[f, pick_group]),
                n_grasps=n_g,
            )
        )
        if verbose:
            print(f"    reset @frame {f:3d}: pick {'OK ' if ok_pick else 'FAIL'} "
                  f"(contact {contact_err:.2f} cm, {n_g} grasps)  "
                  f"place err {place_err:.2f} cm")

    return ResetProtocolResult(
        name=name, tier=tier, pick_name=pick_name, place_name=place_name,
        attempts=attempts, tracking=res,
        contact_tol_cm=contact_tol_cm, place_tol_cm=place_tol_cm,
    )


def table_one(results: Sequence[ResetProtocolResult]):
    """The Table I layout: one column per (task, tier, variant)."""
    rows = [
        "Max Consecutive OR", "Mean Consecutive OR", "Successful Pick Rate",
        "Successful Place Rate", "Mean Position Error (cm)", "Std Position Error (cm)",
    ]
    cols: Dict[str, List[object]] = {}
    for r in results:
        s = r.summary()
        key = f"{s['task']} | Tier {r.tier} | {r.name}"
        cols[key] = [
            s["max_consecutive_or"], s["mean_consecutive_or"],
            s["successful_pick_rate"], s["successful_place_rate"],
            s["mean_position_error_cm"], s["std_position_error_cm"],
        ]
    try:
        import pandas as pd

        return pd.DataFrame(cols, index=rows)
    except ImportError:
        return {"index": rows, **cols}