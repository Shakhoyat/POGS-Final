"""
POGS training loop -- Phase 2 of Sec. IV.

Reproduces the release (``pogs_config.py`` + ``FullImageDatamanager`` +
nerfstudio's ``Trainer`` / splatfacto):

* 4000 iterations, one random training view per step (without replacement
  until every view has been seen, like ``train_unseen_cameras``);
* one Adam per parameter group with the release's exponential schedules; a
  group's scheduler advances only when that group steps (nerfstudio);
* gradient accumulation: only ``lerf`` (3) -- see ``TrainConfig.grad_accum_lerf``;
* splatfacto densification through gsplat's ``DefaultStrategy`` (the release's
  code path) when the gsplat backend is live; an approximate pure-torch fallback
  otherwise;
* random background colour per step (splatfacto ``background_color='random'``);
* staged supervision:
      step >  dino_step (100)          DINO distillation
      step >  lerf_step (2500)         CLIP + grouping distillation
      step >  dino_step + 1000         DINO nearest-neighbour variance term
* DINO rendered at full resolution before ``lerf_step`` and at
  ``dino_rescale_factor * (h // 14)`` after it, as the release does.

The 2D masks that supervise grouping come from Detic in the paper; here they
come from the simulator's instance channel, filtered the way the Detic loader
filters (``build_masks``).
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from . import losses as L
from .config import POGSConfig
from .encoders import ClipPyramid, DinoExtractor, OpenCLIPNetwork, get_img_resolution
from .model import POGSModel


# --------------------------------------------------------------------------- #
# Optimiser with per-group schedules + gradient accumulation
# --------------------------------------------------------------------------- #


class POGSOptimizer:
    """One Adam per parameter group, mirroring nerfstudio's ``Optimizers``."""

    GAUSS_GROUPS = ("means", "features_dc", "features_rest", "opacities", "scales", "quats", "dino_feats")

    def __init__(self, model: POGSModel, cfg: POGSConfig):
        t = cfg.train
        self.groups: Dict[str, torch.optim.Adam] = {}
        self.scheds: Dict[str, Callable[[int], float]] = {}
        self.base_lr: Dict[str, float] = {}
        self.n_steps: Dict[str, int] = {}
        self.accum: Dict[str, int] = {"lerf": t.grad_accum_lerf}

        def add(name, params, lr, lr_final=None, max_steps=30000):
            self.groups[name] = torch.optim.Adam(params, lr=lr, eps=1e-15)
            self.base_lr[name] = lr
            self.n_steps[name] = 0
            if lr_final is None:
                self.scheds[name] = lambda s: 1.0
            else:
                ratio = lr_final / lr
                self.scheds[name] = lambda s, r=ratio, m=max_steps: r ** min(s / m, 1.0)

        gp = model.gauss_params
        add("means", [gp["means"]], t.lr_means, t.lr_means_final, 30000)
        add("features_dc", [gp["features_dc"]], t.lr_features_dc)
        add("features_rest", [gp["features_rest"]], t.lr_features_rest)
        add("opacities", [gp["opacities"]], t.lr_opacities)
        add("scales", [gp["scales"]], t.lr_scales, t.lr_scales_final, 30000)
        add("quats", [gp["quats"]], t.lr_quats)
        add("dino_feats", [gp["dino_feats"]], t.lr_dino_feats, t.lr_dino_feats_final, 6000)
        add("lerf", list(model.gaussian_field.parameters()), t.lr_lerf, t.lr_lerf_final, 15000)
        add("nn_projection", list(model.nn.parameters()), t.lr_nn_projection, t.lr_nn_projection_final, 6000)

    @property
    def param_groups(self):
        out = []
        for name, opt in self.groups.items():
            for g in opt.param_groups:
                g["name"] = name
                out.append(g)
        return out

    def state_for(self, name: str):
        return self.groups[name].state

    def gauss_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        return {k: self.groups[k] for k in self.GAUSS_GROUPS}

    def zero_grad(self, step: int) -> None:
        for name, opt in self.groups.items():
            if step % self.accum.get(name, 1) == 0:
                opt.zero_grad(set_to_none=True)

    def step(self, step: int) -> None:
        for name, opt in self.groups.items():
            if (step + 1) % self.accum.get(name, 1) != 0:
                continue
            lr = self.base_lr[name] * self.scheds[name](self.n_steps[name])
            for g in opt.param_groups:
                g["lr"] = lr
            opt.step()
            self.n_steps[name] += 1

    def rebind(self, model: POGSModel) -> None:
        """Re-point the per-name Adams at the (possibly resized) parameters
        (torch fallback densifier only; gsplat's strategy does this itself)."""
        for name in self.GAUSS_GROUPS:
            self.groups[name].param_groups[0]["params"] = [model.gauss_params[name]]


# --------------------------------------------------------------------------- #
# Pre-computed 2D supervision
# --------------------------------------------------------------------------- #


@dataclass
class FeatureCache:
    dino: torch.Tensor                       # (V, h, w, 64) on CPU, PCA-projected
    masks: List[List[torch.Tensor]]          # per view: list of (H, W) bool, last = negative mask
    dino_pca: torch.Tensor                   # (384, 64) PCA basis -- saved with the checkpoint
    clip: Optional[ClipPyramid] = None


def build_masks(
    obj_ids: torch.Tensor,
    max_masks: int = 15,
    background_ids: Sequence[int] = (0,),
    max_area_frac: float = 1.0 / 3.5,
    min_pixels: int = 16,
) -> List[torch.Tensor]:
    """Detic stand-in. One mask per visible object instance (a detector would
    not report floor/support surfaces, and the release drops any mask larger
    than H*W/3.5), followed by the paper's negative mask -- every pixel not
    covered by an object mask -- which is always LAST."""
    H, W = obj_ids.shape
    masks: List[torch.Tensor] = []
    covered = torch.zeros_like(obj_ids, dtype=torch.bool)
    bg = set(int(b) for b in background_ids)
    for i in torch.unique(obj_ids).tolist():
        if i < 0 or i in bg:
            continue
        m = obj_ids == i
        n = int(m.sum())
        if n < min_pixels or n > max_area_frac * H * W:
            continue
        masks.append(m)
        covered |= m
        if len(masks) >= max_masks - 1:
            break
    masks.append(~covered)
    return masks


@torch.no_grad()
def precompute_features(
    capture: Any,
    cfg: POGSConfig,
    dino: DinoExtractor,
    clip: Optional[OpenCLIPNetwork] = None,
    verbose: bool = True,
) -> FeatureCache:
    """DINOv2 (PCA fitted on the training views), grouping masks, CLIP pyramid."""
    imgs = capture.images.permute(0, 3, 1, 2).contiguous()   # (V, 3, H, W)
    t0 = time.time()
    if verbose:
        print(f"[dino] backend={dino.backend} pretrained={dino.is_pretrained}")
    if not dino.is_pretrained:
        raise RuntimeError("DINOv2 weights unavailable -- refusing to train on random features")
    dino.fit_pca(imgs, batch=1)
    feats = [dino.pca_feats(imgs[v : v + 1]).squeeze(0).cpu() for v in range(imgs.shape[0])]
    dino_feats = torch.stack(feats)
    if verbose:
        print(f"[dino] {tuple(dino_feats.shape)} in {time.time() - t0:.1f}s")

    t = cfg.train
    masks = [
        build_masks(capture.obj_masks[v], t.num_random_masks, t.mask_background_ids, t.mask_max_area_frac)
        for v in range(imgs.shape[0])
    ]
    if verbose:
        n = [len(m) - 1 for m in masks]
        print(f"[masks] object masks per view: min={min(n)} mean={np.mean(n):.1f} max={max(n)}")

    pyramid = None
    if clip is not None:
        if not clip.is_pretrained:
            raise RuntimeError("OpenCLIP weights unavailable -- refusing to distil random language features")
        t0 = time.time()
        pyramid = ClipPyramid(
            clip, (capture.images.shape[1], capture.images.shape[2]),
            tile_size_range=(t.clip_scale_min, t.clip_scale_max),
            n_scales=t.num_clip_scales, stride_scaler=t.patch_stride_scaler, device=dino.device,
        )
        pyramid.build(imgs, verbose=verbose)
        if verbose:
            print(f"[clip] pyramid built in {time.time() - t0:.1f}s")

    return FeatureCache(dino=dino_feats, masks=masks, dino_pca=dino.get_pca(), clip=pyramid)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


@dataclass
class TrainLog:
    step: List[int] = field(default_factory=list)
    total: List[float] = field(default_factory=list)
    rgb: List[float] = field(default_factory=list)
    depth: List[float] = field(default_factory=list)
    dino: List[float] = field(default_factory=list)
    clip: List[float] = field(default_factory=list)
    group: List[float] = field(default_factory=list)
    dino_nn: List[float] = field(default_factory=list)
    psnr: List[float] = field(default_factory=list)
    n_gauss: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, list]:
        return {k: list(v) for k, v in self.__dict__.items()}


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def _make_strategy(model: POGSModel, n_views: int):
    """gsplat ``DefaultStrategy`` configured exactly like splatfacto does."""
    from gsplat.strategy import DefaultStrategy

    mc = model.config
    return DefaultStrategy(
        prune_opa=mc.cull_alpha_thresh,
        grow_grad2d=mc.densify_grad_thresh,
        grow_scale3d=mc.densify_size_thresh,
        grow_scale2d=mc.split_screen_size,
        prune_scale3d=mc.cull_scale_thresh,
        prune_scale2d=mc.cull_screen_size,
        refine_scale2d_stop_iter=mc.stop_screen_size_at,
        refine_start_iter=mc.warmup_length,
        refine_stop_iter=mc.stop_split_at,
        reset_every=mc.reset_alpha_every * mc.refine_every,
        refine_every=mc.refine_every,
        pause_refine_after_reset=n_views + mc.refine_every,
        absgrad=mc.use_absgrad,
        revised_opacity=False,
        verbose=False,
    )


def train_pogs(
    model: POGSModel,
    capture: Any,
    features: FeatureCache,
    cfg: POGSConfig,
    verbose: bool = True,
    progress: Optional[Callable[[int, int], None]] = None,
) -> TrainLog:
    device = cfg.runtime.device
    t, mc = cfg.train, cfg.model
    opt = POGSOptimizer(model, cfg)
    log = TrainLog()
    rng = np.random.default_rng(t.seed)
    torch.manual_seed(t.seed)
    random.seed(t.seed)

    V = len(capture.cameras)
    use_strategy = model.backend == "gsplat"
    strategy = strategy_state = None
    if use_strategy:
        strategy = _make_strategy(model, V)
        strategy.check_sanity(model.gauss_params, opt.gauss_optimizers())
        strategy_state = strategy.initialize_state(scene_scale=1.0)
    elif verbose:
        print("  [train] gsplat unavailable -> approximate torch densification")
    raster_kwargs = dict(
        max_per_tile=cfg.runtime.torch_raster_max_per_tile,
        tile_chunk=cfg.runtime.torch_raster_tile_chunk,
    ) if model.backend == "torch" else {}

    H, W = capture.images.shape[1], capture.images.shape[2]
    dh, dw = get_img_resolution(H, W, 14, cfg.runtime.dino_max_size)
    dino_hw_late = (mc.dino_rescale_factor * (dh // 14), mc.dino_rescale_factor * (dw // 14))

    unseen: List[int] = list(range(V))
    model.train()
    t_start = time.time()
    for step in range(t.max_num_iterations):
        model.step = step
        v = unseen.pop(int(rng.integers(0, len(unseen))))
        if not unseen:
            unseen = list(range(V))
        cam = capture.cameras[v].to(device)
        gt_rgb = capture.images[v].to(device)
        gt_depth = capture.depths[v].to(device).unsqueeze(-1)

        use_dino = step > t.dino_step
        use_lang = step > t.lerf_step and features.clip is not None
        opt.zero_grad(step)

        bg = torch.rand(3, device=device) if mc.random_background else None
        out = model.get_outputs(
            cam, render_dino=use_dino, render_hash=use_lang,
            hash_downscale=t.clip_downscale_factor,
            dino_hw=dino_hw_late if use_lang else None,
            background=bg, raster_kwargs=raster_kwargs,
        )
        if use_strategy:
            strategy.step_pre_backward(model.gauss_params, opt.gauss_optimizers(), strategy_state, step, out["info"])

        # ---- photometric (splatfacto) ------------------------------------ #
        loss = L.rgb_loss(out["rgb"], gt_rgb, mc.ssim_lambda)
        l_rgb = float(loss.detach())

        # ---- depth (Pearson, Sec. IV-B ref [48]) ------------------------- #
        l_depth = 0.0
        if mc.depth_loss_type != "none":
            dl = L.depth_loss(gt_depth, out["depth"], mc.depth_loss_type)
            loss = loss + mc.depth_loss_mult * dl
            l_depth = float(dl.detach())

        # ---- DINO distillation ------------------------------------------- #
        l_dino, l_nnv = 0.0, 0.0
        if use_dino:
            dl = L.dino_loss(out["dino"], features.dino[v].to(device))
            loss = loss + dl
            l_dino = float(dl.detach())
            if step > t.dino_step + t.dino_nn_delay:
                nnv = L.dino_nn_variance_loss(model.gauss_params["dino_feats"], model.ensure_nearest_ids())
                loss = loss + nnv
                l_nnv = float(nnv.detach())

        # ---- language + grouping ----------------------------------------- #
        l_clip, l_group = 0.0, 0.0
        if use_lang:
            hs = out["hashgrid"]
            clip_H, clip_W = hs.shape[:2]
            hs_ds = hs.reshape(-1, hs.shape[-1])
            n_pix = hs_ds.shape[0]
            sel = torch.randperm(n_pix, device=device)[: max(1, int(n_pix * 0.9))]

            scale = float(rng.uniform(t.clip_scale_min, t.clip_scale_max))
            # clip_scale = curr_scale * clip_H * (depth / fy)   (pogs.py)
            depth_ds = F.interpolate(
                out["depth"].permute(2, 0, 1)[None], size=(clip_H, clip_W), mode="nearest"
            )[0].permute(1, 2, 0).reshape(-1, 1)[sel].detach()
            clip_scale = scale * clip_H * depth_ds / cam.fy

            clip_pred = model.gaussian_field.clip_from_hash(hs_ds[sel], clip_scale)
            gt_clip = features.clip.sample(v, scale, (clip_H, clip_W)).to(device)
            gt_clip = gt_clip.reshape(-1, gt_clip.shape[-1])[sel]
            cl = L.clip_loss(clip_pred, gt_clip, mc.clip_loss_weight)
            loss = loss + cl
            l_clip = float(cl.detach())

            inst = model.gaussian_field.instance_from_hash(hs_ds[sel])
            mk = []
            for m in features.masks[v]:
                mm = F.interpolate(m[None, None].float().to(device), size=(clip_H, clip_W), mode="nearest")[0, 0]
                mk.append(mm.reshape(-1)[sel] > 0.5)
            gl = L.instance_contrastive_loss(inst, mk, mc.min_mask_screensize, n_dims=mc.instance_n_dims)
            if gl is not None:
                loss = loss + gl
                l_group = float(gl.detach())

        # ---- regularisation ---------------------------------------------- #
        if mc.use_scale_regularization and step % 10 == 0:
            loss = loss + L.scale_regularization(model.gauss_params["scales"], mc.max_gauss_ratio)

        loss.backward()
        if not use_strategy:
            model.accumulate_grad_stats(out)
        opt.step(step)

        # ---- densify / prune --------------------------------------------- #
        n_before = model.num_points
        if use_strategy:
            strategy.step_post_backward(model.gauss_params, opt.gauss_optimizers(), strategy_state, step, out["info"], packed=False)
        elif (
            step > mc.warmup_length and step % mc.refine_every == 0 and step < mc.stop_split_at
            and step % (mc.reset_alpha_every * mc.refine_every) >= V + mc.refine_every
        ):
            model.refinement_step(opt)
            opt.rebind(model)
        if not use_strategy and step < mc.stop_split_at and step > 0 and step % (mc.reset_alpha_every * mc.refine_every) == 0:
            model.reset_opacities(opt)
        if model.num_points != n_before or (step % mc.refine_every == 0):
            model.nearest_ids = None

        if step % t.log_every == 0 or step == t.max_num_iterations - 1:
            with torch.no_grad():
                p = psnr(out["rgb"].detach(), gt_rgb) if bg is None else float("nan")
            log.step.append(step)
            log.total.append(float(loss.detach()))
            log.rgb.append(l_rgb)
            log.depth.append(l_depth)
            log.dino.append(l_dino)
            log.clip.append(l_clip)
            log.group.append(l_group)
            log.dino_nn.append(l_nnv)
            log.psnr.append(p)
            log.n_gauss.append(model.num_points)
            if verbose:
                print(
                    f"  [{step:5d}/{t.max_num_iterations}] loss={float(loss.detach()):.4f} "
                    f"rgb={l_rgb:.4f} depth={l_depth:.4f} dino={l_dino:.4f} "
                    f"clip={l_clip:.3f} grp={l_group:.3f} nnv={l_nnv:.3f} "
                    f"N={model.num_points} ({time.time() - t_start:.0f}s)"
                )
        if progress is not None:
            progress(step, t.max_num_iterations)

    model.eval()
    if verbose:
        print(f"  training done in {time.time() - t_start:.1f}s, {model.num_points} gaussians")
    return log


@torch.no_grad()
def evaluate_views(model: POGSModel, capture: Any, cfg: POGSConfig) -> Dict[str, float]:
    """PSNR / depth error over the training views with the eval background."""
    model.eval()
    ps, de = [], []
    rk = dict(max_per_tile=cfg.runtime.torch_raster_max_per_tile,
              tile_chunk=cfg.runtime.torch_raster_tile_chunk) if model.backend == "torch" else {}
    for v in range(len(capture.cameras)):
        out = model.get_outputs(capture.cameras[v].to(cfg.runtime.device), rgb_only=True, raster_kwargs=rk)
        gt = capture.images[v].to(cfg.runtime.device)
        ps.append(psnr(out["rgb"], gt))
        gd = capture.depths[v].to(cfg.runtime.device)
        m = (gd > 0) & (out["accumulation"][..., 0] > 0.5)
        if m.any():
            de.append(float((out["depth"][..., 0][m] - gd[m]).abs().mean()))
    return {"psnr_mean": float(np.mean(ps)), "depth_mae_m": float(np.mean(de)) if de else float("nan")}
