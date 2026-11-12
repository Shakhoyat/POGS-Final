"""
Persistent Object Tracking -- Sec. IV-D, "Tracking with POGS".

    "Inspired by RSRD, the core of the tracking algorithm is the computation of
     the loss between the distilled DINO features of the rendered Gaussian Splat
     and the observed images. ... POGS also includes a depth loss term that
     compares the rendered depth maps with depth maps extracted from the real
     observations. The total loss is a weighted sum of the feature loss and
     depth loss, guiding the optimization to adjust per-object pose parameters
     until convergence. Each Gaussian cluster pose parameter is optimized
     independently."

Port of ``pogs/tracking/rigid_group_optimizer.py`` and
``pogs/tracking/observation.py``.  Two substitutions:

* the Warp kernel + ``wp.Tape`` are replaced by
  ``geometry.apply_group_deltas``, which is directly autograd-differentiable,
  so no manual tape is needed;
* kornia's Gaussian blur / morphological erosion are replaced by the
  equivalents in :mod:`pogs_kaggle.losses`.

On seeding: ``Optimizer.init_obj_pose`` calls ``initialize_obj_pose(n_seeds=7)``
but the released ``initialize_obj_pose`` ignores ``n_seeds`` and only ever tries
``z_rot = 0``.  We default to that same single seed (``TrackingConfig.
init_n_seeds = 1``) because it is *correct*, not merely compatible: Sec. IV-C
canonicalises every object frame so that its scan-time pose is identity, and the
tracking sequence starts from the scan configuration.

``n_seeds > 1`` is available and runs a **greedy per-object** yaw sweep -- one
object's yaw at a time, the others held fixed.  It must never be a single yaw
shared by all K objects scored on the aggregate loss, which lets one object's
gain pay for every other object's corruption; on a shallow DINO landscape that
reliably picked a ~150-200 deg yaw and left every object initialised that far
wrong.  See :meth:`RigidGroupOptimizer.initialize_obj_pose`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from . import losses as L
from .config import TrackingConfig
from .geometry import (
    PinholeCamera, apply_group_deltas, identity_posevec, posevec_to_matrix,
    quat_normalize, so3_exp,
)
from .grouping import ObjectGroups
from .model import POGSModel
from .raster import rasterize

DinoFn = Callable[[torch.Tensor], torch.Tensor]
"""(1, 3, H, W) in [0,1] -> (h, w, D) PCA-reduced DINO features."""


# --------------------------------------------------------------------------- #
# Observation containers
# --------------------------------------------------------------------------- #


class _Lazy:
    """``observation.Future`` -- defer an expensive computation until used."""

    def __init__(self, fn):
        self.fn, self._done, self._val = fn, False, None

    def get(self):
        if not self._done:
            self._val = self.fn()
            self._done = True
        return self._val


class Frame:
    """One RGB-D observation rendered/compared at a single camera."""

    def __init__(
        self,
        rgb: torch.Tensor,               # (H, W, 3) in [0, 1]
        camera: PinholeCamera,
        dino_fn: DinoFn,
        depth: Optional[torch.Tensor] = None,   # (H, W) metres
        bbox: Optional[Tuple[int, int, int, int]] = None,
        mask: Optional[torch.Tensor] = None,
        dino_cache: Optional[torch.Tensor] = None,
    ):
        self.camera = camera
        self.bbox = bbox
        self._raw_rgb = rgb
        self.rgb = _resize_hwc(rgb, camera.height, camera.width)
        self._mask = mask
        self._depth_raw = depth

        self._depth = _Lazy(
            lambda: None if depth is None
            else _resize_hwc(depth.unsqueeze(-1), camera.height, camera.width)
        )
        if dino_cache is not None:
            self._dino = _Lazy(lambda: _resize_hwc(dino_cache, camera.height, camera.width))
        else:
            self._dino = _Lazy(
                lambda: _resize_hwc(
                    dino_fn(rgb.permute(2, 0, 1).unsqueeze(0)), camera.height, camera.width
                )
            )

    @property
    def depth(self):
        return self._depth.get()

    @property
    def dino_feats(self):
        return self._dino.get()

    @property
    def mask(self):
        if self._mask is None:
            return None
        return _resize_hwc(self._mask.unsqueeze(-1).float(), self.camera.height, self.camera.width)


def _resize_hwc(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    if x.dim() == 4 and x.shape[0] == 1:
        x = x.squeeze(0)
    if x.shape[0] == h and x.shape[1] == w:
        return x
    mode = "bilinear"
    return F.interpolate(
        x.permute(2, 0, 1).unsqueeze(0).float(), size=(h, w), mode=mode, align_corners=False
    )[0].permute(1, 2, 0)


class PosedObservation:
    """Full frame plus one region-of-interest crop per tracked object."""

    def __init__(
        self,
        rgb: torch.Tensor,
        camera: PinholeCamera,
        dino_fn: DinoFn,
        depth: Optional[torch.Tensor] = None,
        max_roi_resolution: int = 490,
    ):
        # ROI crops index _raw_rgb with pixel coordinates taken from ``camera``,
        # so the two must agree. The official code asserts this; we resize instead,
        # which lets callers pass full-resolution frames with a downscaled camera.
        if rgb.shape[0] != camera.height or rgb.shape[1] != camera.width:
            rgb = _resize_hwc(rgb, camera.height, camera.width)
            if depth is not None:
                depth = _resize_hwc(depth.unsqueeze(-1), camera.height, camera.width).squeeze(-1)
        self._raw_rgb = rgb
        self._raw_depth = depth
        self._camera = camera
        self._dino_fn = dino_fn
        self.max_roi_resolution = max_roi_resolution
        self.frame = Frame(rgb, camera, dino_fn, depth)
        self.roi_frames: List[Frame] = []

    def add_roi(self, xmin: float, xmax: float, ymin: float, ymax: float) -> None:
        self.roi_frames.append(self._make_roi(xmin, xmax, ymin, ymax))

    def update_roi(self, idx: int, xmin: float, xmax: float, ymin: float, ymax: float) -> None:
        self.roi_frames[idx] = self._make_roi(xmin, xmax, ymin, ymax)

    def _make_roi(self, xmin, xmax, ymin, ymax) -> Frame:
        W, H = self._camera.width, self._camera.height
        x0, x1 = int(xmin * (W - 1)), int(xmax * (W - 1))
        y0, y1 = int(ymin * (H - 1)), int(ymax * (H - 1))
        # snap the crop to a multiple of the DINO patch size, as in the official code
        x1 = x0 + max(14, ((x1 - x0) // 14) * 14)
        y1 = y0 + max(14, ((y1 - y0) // 14) * 14)
        x1, y1 = min(x1, W), min(y1, H)

        rgb = self._raw_rgb[y0:y1, x0:x1].clone()
        cam = self._camera.crop(x0, x1, y0, y1)
        if max(cam.width, cam.height) > self.max_roi_resolution:
            cam = cam.rescale(self.max_roi_resolution / max(cam.width, cam.height))
        depth = None if self._raw_depth is None else self._raw_depth[y0:y1, x0:x1].clone()
        return Frame(rgb, cam, self._dino_fn, depth, bbox=(x0, x1, y0, y1))


# --------------------------------------------------------------------------- #
# The optimiser
# --------------------------------------------------------------------------- #


@dataclass
class TrackStepInfo:
    loss: float
    dino_loss: float
    depth_loss: float
    n_iters: int
    wall_time: float
    part_deltas: torch.Tensor


class RigidGroupOptimizer:
    def __init__(
        self,
        config: TrackingConfig,
        model: POGSModel,
        groups: ObjectGroups,
        raster_kwargs: Optional[dict] = None,
    ):
        self.config = config
        self.model = model
        self.groups = groups
        self.device = model.gauss_params["means"].device
        self.K = groups.num_groups
        self.raster_kwargs = raster_kwargs or {}

        self.init_means = model.gauss_params["means"].data.detach().clone()
        self.init_quats = quat_normalize(model.gauss_params["quats"].data.detach().clone())
        self.init_scales = torch.exp(model.gauss_params["scales"].data.detach())
        self.init_opacities = torch.sigmoid(model.gauss_params["opacities"].data.detach()).squeeze(-1)
        self.init_colors = model.colors_sh.data.detach().clone()
        self.init_dino = model.gauss_params["dino_feats"].data.detach().clone()
        self.init_p2w = groups.init_p2w.detach().clone()
        self.group_labels = groups.group_labels.clone()
        # gaussians that belong to no tracked group stay fixed in group 0's frame
        self._label_safe = torch.where(self.group_labels >= 0, self.group_labels,
                                       torch.zeros_like(self.group_labels))
        self._tracked = self.group_labels >= 0
        # Gaussian index lists, so a render only transforms what it draws.
        self._idx_tracked = torch.nonzero(self._tracked).squeeze(-1)
        self._idx_obj = [torch.nonzero(groups.group_masks[k]).squeeze(-1) for k in range(self.K)]

        self.part_deltas = torch.nn.Parameter(identity_posevec(self.K, self.device))
        self.part_optimizer = torch.optim.Adam([self.part_deltas], lr=config.pose_lr)
        self.prev_part_deltas = self.part_deltas.data.clone()
        self.is_initialized = False
        self.frame: Optional[PosedObservation] = None

        # diagnostics: how often the Sec. IV-D depth term had to be rescued or
        # was dropped entirely (see TrackingConfig.depth_mask_relative_fallback)
        self.n_depth_fallback = 0
        self.n_depth_dropped = 0
        self.n_depth_used = 0
        self.n_roi_fallback = 0
        self.n_roi_lost = 0

    # ------------------------------------------------------------------ #
    # geometry
    # ------------------------------------------------------------------ #

    def transformed_geometry(self, part_deltas: torch.Tensor):
        """Apply the per-group SE(3) deltas to every Gaussian (Warp kernel)."""
        means, quats = apply_group_deltas(
            self.init_p2w, part_deltas, self._label_safe, self.init_means, self.init_quats
        )
        # untracked gaussians (background) are left where the scan put them
        means = torch.where(self._tracked.unsqueeze(-1), means, self.init_means)
        quats = torch.where(self._tracked.unsqueeze(-1), quats, self.init_quats)
        return means, quats

    def get_part2world(self, i: int, part_deltas: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``get_part2world_transform``: rotate about the world frame, then
        translate in the world frame, starting from the canonicalised
        (identity-rotation, centroid-translation) scan pose."""
        pd = self.part_deltas if part_deltas is None else part_deltas
        from .geometry import quat_to_rotmat

        T = torch.eye(4, device=self.device)
        T[:3, 3] = self.init_p2w[i, :3]
        R = quat_to_rotmat(pd[i, 3:].unsqueeze(0))[0]
        out = T.clone()
        out[:3, :3] = R @ out[:3, :3]
        out[:3, 3] = out[:3, 3] + pd[i, :3]
        return out

    def get_all_part2world(self, part_deltas: Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.stack([self.get_part2world(i, part_deltas) for i in range(self.K)])

    @torch.no_grad()
    def apply_to_model(self, part_deltas: Optional[torch.Tensor] = None) -> None:
        """Write the current pose estimate back into the model (for rendering /
        exporting the persistent representation)."""
        pd = self.part_deltas.data if part_deltas is None else part_deltas
        means, quats = self.transformed_geometry(pd)
        self.model.gauss_params["means"].data = means
        self.model.gauss_params["quats"].data = quats

    @torch.no_grad()
    def reset_transforms(self) -> None:
        self.model.gauss_params["means"].data = self.init_means.clone()
        self.model.gauss_params["quats"].data = self.init_quats.clone()

    # ------------------------------------------------------------------ #
    # rendering with a *proposed* pose (differentiable in part_deltas)
    # ------------------------------------------------------------------ #

    def _render(
        self,
        camera: PinholeCamera,
        part_deltas: torch.Tensor,
        obj_id: Optional[int],
        want_dino: bool,
    ) -> Dict[str, torch.Tensor]:
        """``model.get_outputs(camera, tracking=True, obj_id=...)`` with the
        proposed pose applied differentiably."""
        if obj_id is not None:
            sel = self._idx_obj[obj_id]
        elif not self.config.render_untracked:
            # the release keeps only the selected objects' Gaussians (keep_inds)
            sel = self._idx_tracked
        else:
            sel = None
        if sel is not None:
            means, quats = apply_group_deltas(self.init_p2w, part_deltas, self._label_safe[sel],
                                              self.init_means[sel], self.init_quats[sel])
            scales, opac = self.init_scales[sel], self.init_opacities[sel]
            colors, dino = self.init_colors[sel], self.init_dino[sel]
        else:
            means, quats = self.transformed_geometry(part_deltas)
            scales, opac = self.init_scales, self.init_opacities
            colors, dino = self.init_colors, self.init_dino

        aa = self.model.config.rasterize_mode == "antialiased"
        out = rasterize(
            means, quats, scales, opac, colors, camera,
            backend=self.model.backend, sh_degree=self.model.sh_degree_to_use(),
            render_depth=True, background=self.model.background_color,
            tile_size=8, antialiased=aa, **self.raster_kwargs,
        )
        alpha = out["alpha"]
        depth = torch.where(alpha > 0, out["depth"], out["depth"].detach().max())
        res = {"rgb": torch.clamp(out["render"], 0, 1), "depth": depth, "accumulation": alpha}
        if want_dino:
            # release: quats / scales / opacities detached in the DINO render
            f = rasterize(
                means, quats.detach(), scales, opac, dino, camera,
                backend=self.model.backend, sh_degree=None, render_depth=False,
                background=None, tile_size=8, antialiased=aa, **self.raster_kwargs,
            )
            a = torch.clamp(f["alpha"], min=1e-6)
            feat = torch.where(f["alpha"] > 0, f["render"] / a.detach(), torch.zeros_like(f["render"]))
            feat = self.model.nn(feat.reshape(-1, feat.shape[-1])).reshape(*feat.shape[:-1], -1)
            # pogs.py (eval mode): dino_feats[dino_alpha < 0.8] = 0
            feat = torch.where(f["alpha"] >= self.config.dino_alpha_thresh, feat, torch.zeros_like(feat))
            res["dino"] = feat
        return res

    # ------------------------------------------------------------------ #
    # ROI
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def render_mask(self, camera: PinholeCamera, obj_id: int) -> torch.Tensor:
        """Release: ``accumulation > 0.9``, raising 'Object left ROI' when empty.
        We fall back to half the peak accumulation before giving up (counted in
        ``n_roi_fallback``) instead of crashing the run."""
        out = self._render(camera, self.part_deltas.data, obj_id, want_dino=False)
        acc = out["accumulation"].squeeze(-1)
        m = acc > self.config.render_mask_thresh
        if not bool(m.any()):
            self.n_roi_fallback += 1
            m = acc > (0.5 * float(acc.max()))
        if not bool(m.any()) or float(acc.max()) < 1e-3:
            raise RuntimeError(f"object {obj_id} left the ROI")
        return m

    @torch.no_grad()
    def calculate_roi(self, obj_id: int, camera: PinholeCamera) -> Tuple[float, float, float, float]:
        """Port of ``RigidGroupOptimizer.calculate_roi``."""
        mask = self.render_mask(camera, obj_id)
        ys, xs = torch.where(mask)
        xs = xs.float() / mask.shape[1]
        ys = ys.float() / mask.shape[0]
        c = self.config
        infl_x = max(c.roi_inflate_proportion * float(xs.max() - xs.min()), c.roi_inflate / camera.width)
        infl_y = max(c.roi_inflate_proportion * float(ys.max() - ys.min()), c.roi_inflate / camera.height)
        return (
            max(0.0, float(xs.min()) - infl_x), min(1.0, float(xs.max()) + infl_x),
            max(0.0, float(ys.min()) - infl_y), min(1.0, float(ys.max()) + infl_y),
        )

    def set_frame(self, frame: PosedObservation) -> None:
        self.frame = frame

    def set_observation(self, frame: PosedObservation) -> None:
        """Attach a new observation and refresh every object's ROI from the
        *current* pose estimate."""
        assert self.is_initialized, "initialize_obj_pose() must run on the first frame"
        if self.config.use_roi:
            for i in range(self.K):
                try:
                    roi = self.calculate_roi(i, frame.frame.camera)
                except RuntimeError:
                    self.n_roi_lost += 1
                    roi = (0.0, 1.0, 0.0, 1.0)
                frame.add_roi(*roi)
        self.frame = frame

    # ------------------------------------------------------------------ #
    # loss
    # ------------------------------------------------------------------ #

    def get_optim_loss(
        self,
        obs: PosedObservation,
        part_deltas: torch.Tensor,
        use_dino: bool = True,
        use_depth: bool = True,
        use_rgb: bool = False,
        use_mask: bool = False,
        use_roi: bool = True,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        cfg = self.config

        def _object_mask(accumulation: torch.Tensor) -> torch.Tensor:
            """``accumulation > 0.85`` (official), with the documented relative
            fallback when the absolute threshold selects nothing."""
            a = accumulation.squeeze(-1)
            m = a > cfg.accumulation_thresh
            if bool(m.any()) or not cfg.depth_mask_relative_fallback:
                return m
            amax = float(a.detach().max())
            if amax <= 1e-3:
                return m
            self.n_depth_fallback += 1
            return a > cfg.accumulation_thresh * amax

        parts: Dict[str, List[torch.Tensor]] = {
            k: [] for k in ("real_rgb", "real_dino", "real_depth", "rendered_rgb",
                            "rendered_dino", "rendered_depth", "valids", "accumulation", "real_mask")
        }

        if not use_roi or not obs.roi_frames:
            frame = obs.frame
            out = self._render(frame.camera, part_deltas, None, use_dino)
            if not bool((out["accumulation"] > 0).any()):
                return None, {}
            parts["real_rgb"].append(frame.rgb)
            parts["rendered_rgb"].append(out["rgb"])
            if use_dino:
                parts["real_dino"].append(frame.dino_feats.to(self.device))
                parts["rendered_dino"].append(L.gaussian_blur2d(out["dino"], cfg.blur_kernel_size))
            if use_depth and frame.depth is not None:
                d = frame.depth
                v = _object_mask(out["accumulation"]) & (d.squeeze(-1) > 0)
                parts["valids"].append(v.unsqueeze(-1))
                parts["real_depth"].append(d)
                parts["rendered_depth"].append(out["depth"])
            parts["accumulation"].append(out["accumulation"])
        else:
            for i in reversed(range(self.K)):
                rf = obs.roi_frames[i]
                out = self._render(rf.camera, part_deltas, i, use_dino)
                out_mask = _object_mask(out["accumulation"])
                real_depth = rf.depth
                valids = out_mask
                if real_depth is not None:
                    valids = valids & (~real_depth.squeeze(-1).isnan()) & (real_depth.squeeze(-1) > 0)

                if use_depth and real_depth is not None:
                    valid_d = L.erode(valids, cfg.erode_depth_kernel)
                    dv = real_depth.squeeze(-1)[valid_d]
                    if dv.numel() > 0:
                        med = torch.median(dv)
                        valid_d = valid_d & ~(real_depth.squeeze(-1) > med * cfg.depth_outlier_ratio)
                    masked_real = real_depth * valid_d.unsqueeze(-1)
                    nonzero = (masked_real != 0)
                    masked_rend = out["depth"] * valid_d.unsqueeze(-1) * nonzero
                    v = L.erode((valid_d.unsqueeze(-1) & nonzero).squeeze(-1), cfg.erode_valid_kernel)
                    parts["valids"].append(v.unsqueeze(-1))
                    parts["real_depth"].append(masked_real * v.unsqueeze(-1))
                    parts["rendered_depth"].append(masked_rend * v.unsqueeze(-1))
                else:
                    parts["valids"].append(L.erode(valids, cfg.erode_valid_kernel).unsqueeze(-1))

                parts["real_rgb"].append(rf.rgb)
                parts["rendered_rgb"].append(out["rgb"])
                if use_dino:
                    parts["real_dino"].append(rf.dino_feats.to(self.device))
                    parts["rendered_dino"].append(L.gaussian_blur2d(out["dino"], cfg.blur_kernel_size))
                parts["accumulation"].append(out["accumulation"])
                if use_mask and rf.mask is not None:
                    parts["real_mask"].append(rf.mask)

        flat = {k: torch.cat([t.reshape(-1, t.shape[-1]) for t in v]) for k, v in parts.items() if v}

        stats: Dict[str, float] = {}
        loss = None
        if use_dino and "real_dino" in flat:
            d_loss = L.tracking_feature_loss(flat["real_dino"], flat["rendered_dino"])
            loss = d_loss if loss is None else loss + d_loss
            stats["dino"] = float(d_loss.detach())
        if use_rgb and "real_rgb" in flat:
            rgbl = (flat["real_rgb"] - flat["rendered_rgb"]).abs().mean()
            loss = rgbl if loss is None else loss + rgbl
            stats["rgb"] = float(rgbl.detach())

        if use_depth and "real_depth" in flat:
            dl = L.tracking_depth_loss(
                flat["real_depth"], flat["rendered_depth"],
                flat["valids"].squeeze(-1).bool(), cfg.depth_ignore_threshold,
            )
            if dl is not None:
                depth_term = cfg.depth_loss_mult * dl
                loss = depth_term if loss is None else loss + depth_term
                stats["depth"] = float(dl.detach())
                self.n_depth_used += 1
            else:
                self.n_depth_dropped += 1
        elif use_depth:
            self.n_depth_dropped += 1
        if use_mask and "real_mask" in flat:
            bce = F.binary_cross_entropy(flat["accumulation"].clamp(1e-6, 1 - 1e-6), flat["real_mask"])
            loss = bce if loss is None else loss + cfg.mask_loss_mult * bce
            stats["mask"] = float(bce.detach())
        return loss, stats

    # ------------------------------------------------------------------ #
    # initialisation (100 DINO-only steps + 70 with depth & ROI)
    # ------------------------------------------------------------------ #

    def initialize_obj_pose(
        self, obs: PosedObservation, n_seeds: int = 7, verbose: bool = False
    ) -> Dict[str, float]:
        assert not self.is_initialized, "initialize once"
        cfg = self.config
        self.frame = obs
        t0 = time.time()

        def try_opt(start: torch.Tensor, niter: int, use_depth: bool, use_roi: bool):
            pose = torch.nn.Parameter(start.detach().clone())
            opt = torch.optim.Adam([pose], lr=cfg.init_lr)
            last = torch.tensor(float("inf"), device=self.device)
            for _ in range(niter):
                opt.zero_grad()
                with torch.no_grad():
                    pose[:, 3:] = quat_normalize(pose[:, 3:])
                loss, _ = self.get_optim_loss(
                    self.frame, pose, use_dino=cfg.use_dino, use_depth=use_depth,
                    use_rgb=cfg.use_rgb, use_mask=False, use_roi=use_roi,
                )
                if loss is None:
                    break
                loss.backward()
                opt.step()
                last = loss.detach()
            return float(last), pose.data.detach()

        # --- coarse pass: DINO only, full frame ---------------------------- #
        # The official ``initialize_obj_pose`` seeds from a SINGLE pose, identity
        # (``z_rot = 0.0``); its ``n_seeds`` argument is accepted and never used.
        # That is the right seed: Sec. IV-C canonicalises every object frame so
        # that its scan-time pose *is* identity, and the tracking sequence starts
        # from the scan configuration.
        #
        # A multi-seed yaw sweep is only sound if each object gets its own yaw.
        # Sweeping a single yaw applied to all K objects at once and scoring it
        # with the aggregate loss -- which is what this used to do -- lets one
        # object's improvement pay for every other object's corruption, and the
        # winning seed is then baked into all K poses.  On a shallow DINO
        # landscape that reliably selected a ~150-200 deg yaw and left every
        # object initialised ~150 deg wrong, which no amount of per-frame
        # tracking can walk back.
        best_loss, best_poses = float("inf"), identity_posevec(self.K, self.device)
        n_seeds = max(int(n_seeds), 1)
        if n_seeds == 1:
            best_loss, best_poses = try_opt(
                identity_posevec(self.K, self.device), cfg.init_niter_coarse,
                use_depth=False, use_roi=False,
            )
        else:
            # Greedy per-object sweep: one object's yaw at a time, the rest held
            # at their current best, so a seed is only ever credited for the
            # object it was applied to.
            cur = identity_posevec(self.K, self.device)
            for i in range(self.K):
                obj_best_loss, obj_best_quat = float("inf"), cur[i, 3:].clone()
                for s in range(n_seeds):
                    z_rot = 2 * math.pi * s / n_seeds
                    trial = cur.clone()
                    trial[i, 3:] = so3_exp(
                        torch.tensor([[0.0, 0.0, z_rot]], device=self.device)
                    )[0]
                    l, _ = try_opt(trial, cfg.init_niter_coarse, use_depth=False, use_roi=False)
                    if verbose:
                        print(f"    obj {i} seed {s} (yaw {math.degrees(z_rot):5.0f} deg): loss {l:.4f}")
                    if l < obj_best_loss:
                        obj_best_loss, obj_best_quat = l, trial[i, 3:].clone()
                cur[i, 3:] = obj_best_quat
            best_loss, best_poses = try_opt(
                cur, cfg.init_niter_coarse, use_depth=False, use_roi=False
            )

        self.is_initialized = True
        self.part_deltas = torch.nn.Parameter(best_poses)

        # --- fine pass: + depth, + ROI ------------------------------------ #
        fine_obs = PosedObservation(
            obs._raw_rgb, obs._camera, obs._dino_fn, obs._raw_depth, cfg.max_roi_resolution
        )
        if cfg.use_roi:
            for i in range(self.K):
                try:
                    fine_obs.add_roi(*self.calculate_roi(i, fine_obs.frame.camera))
                except RuntimeError:
                    self.n_roi_lost += 1
                    fine_obs.add_roi(0.0, 1.0, 0.0, 1.0)
        self.frame = fine_obs
        init_depth = cfg.use_depth if cfg.init_use_depth is None else cfg.init_use_depth
        fine_loss, best_poses = try_opt(
            best_poses, cfg.init_niter_fine, use_depth=init_depth, use_roi=cfg.use_roi
        )

        self.part_deltas = torch.nn.Parameter(best_poses)
        self.part_optimizer = torch.optim.Adam([self.part_deltas], lr=cfg.pose_lr)
        self.prev_part_deltas = best_poses.clone()
        self.apply_to_model()
        return {"coarse_loss": best_loss, "fine_loss": fine_loss, "time": time.time() - t0}

    # ------------------------------------------------------------------ #
    # per-frame update
    # ------------------------------------------------------------------ #

    def step(self, niter: Optional[int] = None, use_depth: Optional[bool] = None,
             use_rgb: Optional[bool] = None) -> TrackStepInfo:
        cfg = self.config
        niter = cfg.track_niter if niter is None else niter
        use_depth = cfg.use_depth if use_depth is None else use_depth
        use_rgb = cfg.use_rgb if use_rgb is None else use_rgb
        t0 = time.time()

        self.prev_part_deltas = self.part_deltas.data.detach().clone()
        stats: Dict[str, float] = {}
        loss_val = float("nan")
        for i in range(niter):
            # nerfstudio ExponentialDecayScheduler(lr_final=pose_lr_final, max_steps=niter)
            lr = cfg.pose_lr * (cfg.pose_lr_final / cfg.pose_lr) ** (i / max(niter, 1))
            for g in self.part_optimizer.param_groups:
                g["lr"] = lr
            with torch.no_grad():
                if cfg.upright_constraint:
                    self.part_deltas[:, 4:6] = 0.0
                self.part_deltas[:, 3:] = quat_normalize(self.part_deltas[:, 3:])
            self.part_optimizer.zero_grad()
            loss, stats = self.get_optim_loss(
                self.frame, self.part_deltas, use_dino=cfg.use_dino, use_depth=use_depth,
                use_rgb=use_rgb, use_mask=cfg.use_mask_loss, use_roi=cfg.use_roi,
            )
            if loss is None:
                break
            loss.backward()
            if cfg.upright_constraint and self.part_deltas.grad is not None:
                self.part_deltas.grad[:, 4:6] = 0.0
            if cfg.clip_grad:
                torch.nn.utils.clip_grad_norm_([self.part_deltas], cfg.clip_grad)
            self.part_optimizer.step()
            loss_val = float(loss.detach())
        for g in self.part_optimizer.param_groups:
            g["lr"] = cfg.pose_lr
        with torch.no_grad():
            if cfg.upright_constraint:
                self.part_deltas[:, 4:6] = 0.0
            self.part_deltas[:, 3:] = quat_normalize(self.part_deltas[:, 3:])
        self.apply_to_model()
        return TrackStepInfo(
            loss=loss_val,
            dino_loss=stats.get("dino", float("nan")),
            depth_loss=stats.get("depth", float("nan")),
            n_iters=niter,
            wall_time=time.time() - t0,
            part_deltas=self.part_deltas.data.detach().clone(),
        )

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def depth_diagnostics(self) -> Dict[str, float]:
        """How much of the Sec. IV-D depth term actually reached the loss.

        A high ``depth_drop_rate`` means the object mask
        (``accumulation > accumulation_thresh``) was empty, so the depth term
        contributed nothing -- and any "No Depth" ablation run against it is
        comparing a configuration to itself.
        """
        total = self.n_depth_used + self.n_depth_dropped
        return {
            "depth_evals": float(total),
            "depth_used": float(self.n_depth_used),
            "depth_dropped": float(self.n_depth_dropped),
            "depth_drop_rate": self.n_depth_dropped / max(total, 1),
            "accum_fallbacks": float(self.n_depth_fallback),
            "roi_mask_fallbacks": float(self.n_roi_fallback),
            "roi_lost": float(self.n_roi_lost),
        }

    @torch.no_grad()
    def set_part_deltas(self, deltas: torch.Tensor, reset_adam: bool = False) -> None:
        """Overwrite the pose state."""
        self.part_deltas.data.copy_(deltas)
        self.part_deltas.data[:, 3:] = quat_normalize(self.part_deltas.data[:, 3:])
        if reset_adam:
            self.part_optimizer = torch.optim.Adam([self.part_deltas], lr=self.config.pose_lr)

    @torch.no_grad()
    def render_full(self, camera: PinholeCamera) -> Dict[str, torch.Tensor]:
        return self._render(camera, self.part_deltas.data, None, want_dino=False)


# --------------------------------------------------------------------------- #
# Dual-GPU DINO prefetch (Sec. VI: "we will parallelize depth inference and
# DINO feature extraction for tracking speed optimization")
# --------------------------------------------------------------------------- #


class DinoPrefetcher:
    """Extracts the *next* frame's ROI DINO features on a second GPU while the
    current frame is being optimised.

    This matters because of where the cost actually is. During tracking POGS
    never uses the full-frame DINO map: ``get_optim_loss`` runs on per-object ROI
    crops, so the per-frame ViT cost is ``K`` extractions (one per tracked
    object), each of a crop that ``get_img_resolution`` upsamples to the DINO
    working resolution.  Prefetching the *full frame* would therefore save
    nothing.

    The catch is that frame t+1's ROIs are estimated from frame t's pose.
    Sec. VI lists this parallelisation as future work.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._thread = None
        self._obs: Optional[PosedObservation] = None
        self._key = None

    def submit(self, key, obs: PosedObservation) -> None:
        """Warm every ROI frame's lazy DINO in a background thread."""
        if not self.enabled:
            self._obs, self._key = obs, key
            return
        import threading

        def _work():
            for rf in obs.roi_frames:
                _ = rf.dino_feats            # forces the extraction on the feature device

        self._key, self._obs = key, obs
        self._thread = threading.Thread(target=_work, daemon=True)
        self._thread.start()

    def take(self, key) -> Optional[PosedObservation]:
        if self._key != key or self._obs is None:
            return None
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        obs, self._obs, self._key = self._obs, None, None
        return obs