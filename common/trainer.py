"""
POGS training loop -- Phase 2 of Sec. IV.

Reproduces ``pogs_config.py``: 4000 iterations, per-parameter-group Adam with
exponential-decay schedules, and nerfstudio's ``gradient_accumulation_steps``
({'camera_opt': 100, 'color': 10, 'shs': 10, 'lerf': 3}).

Supervision is staged exactly as in the official datamanager:
    step >  dino_step (100)   -> DINO distillation switches on
    step >  lerf_step (2500)  -> CLIP + grouping distillation switch on
    step >  dino_step + 1000  -> the DINO nearest-neighbour variance term

The 2D masks that supervise the grouping features come from Detic in the paper;
here they come from the simulator's instance channel (or from any external
segmenter you plug into ``FeatureCache.masks``), which is the same signal.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from . import losses as L
from .config import POGSConfig, TrainConfig, ModelConfig, RuntimeConfig
from .encoders import ClipPyramid, DinoExtractor, OpenCLIPNetwork
from .geometry import PinholeCamera
from .model import POGSModel
from .simscene import CaptureData


# --------------------------------------------------------------------------- #
# Optimiser with per-group schedules + gradient accumulation
# --------------------------------------------------------------------------- #


class POGSOptimizer:
    """One Adam per parameter group, mirroring nerfstudio's ``Optimizers``."""

    def __init__(self, model: POGSModel, cfg: POGSConfig):
        t = cfg.train
        self.groups: Dict[str, torch.optim.Adam] = {}
        self.scheds: Dict[str, Callable[[int], float]] = {}
        self.base_lr: Dict[str, float] = {}
        self.accum: Dict[str, int] = {
            "features_dc": t.grad_accum_color,
            "features_rest": t.grad_accum_shs,
            "lerf": t.grad_accum_lerf,
        }

        def add(name, params, lr, lr_final=None, max_steps=30000):
            self.groups[name] = torch.optim.Adam(params, lr=lr, eps=1e-15)
            self.base_lr[name] = lr
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
        add("lerf", list(model.gaussian_field.parameters()), t.lr_lerf, t.lr_lerf_final, 15000)
        add("dino_feats", [gp["dino_feats"]], t.lr_dino_feats, t.lr_dino_feats_final, 6000)
        add("nn_projection", list(model.nn.parameters()), t.lr_nn_projection, t.lr_dino_feats_final, 6000)

    # -- torch-like API used by the refinement code ------------------------ #

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

    def zero_grad(self, step: int) -> None:
        for name, opt in self.groups.items():
            if step % self.accum.get(name, 1) == 0:
                opt.zero_grad(set_to_none=True)

    def step(self, step: int) -> None:
        for name, opt in self.groups.items():
            if (step + 1) % self.accum.get(name, 1) != 0:
                continue
            lr = self.base_lr[name] * self.scheds[name](step)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.step()

    def rebind(self, model: POGSModel) -> None:
        """Re-point the per-name Adams at the (possibly resized) parameters."""
        gp = model.gauss_params
        mapping = {
            "means": [gp["means"]], "features_dc": [gp["features_dc"]],
            "features_rest": [gp["features_rest"]], "opacities": [gp["opacities"]],
            "scales": [gp["scales"]], "quats": [gp["quats"]], "dino_feats": [gp["dino_feats"]],
        }
        for name, params in mapping.items():
            self.groups[name].param_groups[0]["params"] = params


# --------------------------------------------------------------------------- #
# Pre-computed 2D supervision
# --------------------------------------------------------------------------- #


@dataclass
class FeatureCache:
    dino: torch.Tensor                       # (V, h, w, 64) on CPU
    masks: List[List[torch.Tensor]]          # per view: list of (H, W) bool, last = negative mask
    clip: Optional[ClipPyramid] = None
    dino_extractor: Optional[DinoExtractor] = None
    clip_encoder: Optional[OpenCLIPNetwork] = None


def build_masks_from_instance_ids(obj_ids: torch.Tensor, max_masks: int = 15) -> List[torch.Tensor]:
    """Detic stand-in: one boolean mask per visible object instance, plus the
    'negative mask' (background) that POGS appends last."""
    ids = torch.unique(obj_ids)
    masks = []
    covered = torch.zeros_like(obj_ids, dtype=torch.bool)
    for i in ids.tolist():
        if i < 0:
            continue
        m = obj_ids == i
        if m.sum() < 16:
            continue
        masks.append(m)
        covered |= m
        if len(masks) >= max_masks - 1:
            break
    masks.append(~covered)          # negative mask, always last
    return masks


@torch.no_grad()
def precompute_features(
    capture: CaptureData,
    cfg: POGSConfig,
    dino_device: Optional[str] = None,
    build_clip: bool = True,
    allow_download: bool = True,
    verbose: bool = True,
) -> FeatureCache:
    """Runs DINOv2 (and optionally the CLIP pyramid) over every training view.

    ``dino_device`` lets these run on the *second* T4 while the splat trains on
    the first -- the parallelisation the paper lists as future work (Sec. VI).
    """
    dev = dino_device or cfg.runtime.device
    imgs = capture.images.permute(0, 3, 1, 2).contiguous()   # (V, 3, H, W)

    t0 = time.time()
    dino = DinoExtractor(device=dev, pca_dim=cfg.model.gaussian_dim,
                         max_size=cfg.runtime.dino_max_size, allow_download=allow_download)
    if verbose:
        print(f"[dino] backend={dino.backend} pretrained={dino.is_pretrained}")
    dino.fit_pca(imgs, batch=1)
    feats = []
    for v in range(imgs.shape[0]):
        feats.append(dino.pca_feats(imgs[v : v + 1]).squeeze(0).cpu())
    dino_feats = torch.stack(feats)
    if verbose:
        print(f"[dino] {tuple(dino_feats.shape)} in {time.time() - t0:.1f}s")

    masks = [build_masks_from_instance_ids(capture.obj_ids[v], cfg.train.num_random_masks)
             for v in range(imgs.shape[0])]

    clip_enc, pyramid = None, None
    if build_clip:
        t0 = time.time()
        clip_enc = OpenCLIPNetwork(device=dev, clip_n_dims=cfg.model.clip_n_dims,
                                   allow_download=allow_download)
        if verbose:
            print(f"[clip] backend={clip_enc.backend} pretrained={clip_enc.is_pretrained}")
        pyramid = ClipPyramid(
            clip_enc, (capture.images.shape[1], capture.images.shape[2]),
            tile_size_range=(cfg.train.clip_scale_min, cfg.train.clip_scale_max),
            n_scales=cfg.train.num_clip_scales,
            stride_scaler=cfg.train.patch_stride_scaler, device=dev,
        )
        pyramid.build(imgs, verbose=verbose)
        if verbose:
            print(f"[clip] pyramid built in {time.time() - t0:.1f}s")

    return FeatureCache(dino=dino_feats, masks=masks, clip=pyramid,
                        dino_extractor=dino, clip_encoder=clip_enc)


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

    def __getitem__(self, key: str):
        if key == "loss":
            return self.total
        return getattr(self, key)

    def get(self, key: str, default=None):
        try:
            return self[key]
        except (AttributeError, KeyError):
            return default


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def train_pogs(
    model: POGSModel,
    capture: Any,
    *args,
    **kwargs,
) -> TrainLog:
    if len(args) > 0 and isinstance(args[0], FeatureCache):
        features = args[0]
        cfg = args[1] if len(args) > 1 else kwargs.get("cfg", POGSConfig())
        verbose = kwargs.get("verbose", args[2] if len(args) > 2 else True)
        progress = kwargs.get("progress", args[3] if len(args) > 3 else None)
    else:
        dino_fn = args[0] if len(args) > 0 else kwargs.get("dino_fn", None)
        clip_fn = args[1] if len(args) > 1 else kwargs.get("clip_fn", None)
        train_cfg = args[2] if len(args) > 2 else kwargs.get("cfg", None)
        dev = kwargs.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
        verbose = kwargs.get("verbose", True)
        progress = kwargs.get("progress", None)

        full_cfg = POGSConfig()
        full_cfg.runtime.device = dev
        if isinstance(train_cfg, TrainConfig):
            full_cfg.train = train_cfg
        elif hasattr(train_cfg, "train"):
            full_cfg = train_cfg
        elif hasattr(train_cfg, "max_num_iterations") or hasattr(train_cfg, "max_steps"):
            full_cfg.train = train_cfg

        cfg = full_cfg

        # Process capture views
        imgs = capture.images.permute(0, 3, 1, 2).contiguous()
        if hasattr(dino_fn, "fit_pca") and getattr(dino_fn, "pca_matrix", None) is None:
            dino_fn.fit_pca(imgs, batch=1)

        feats = []
        for v in range(imgs.shape[0]):
            f = dino_fn(imgs[v : v + 1])
            if f.dim() == 4:
                f = f.squeeze(0)
            feats.append(f.cpu())
        dino_feats = torch.stack(feats)

        obj_masks = getattr(capture, "obj_ids", getattr(capture, "obj_masks", None))
        if obj_masks is not None:
            masks = [build_masks_from_instance_ids(obj_masks[v], cfg.train.num_random_masks)
                     for v in range(imgs.shape[0])]
        else:
            masks = [[] for _ in range(imgs.shape[0])]

        pyramid = None
        if clip_fn is not None and isinstance(clip_fn, OpenCLIPNetwork) and getattr(clip_fn, "is_pretrained", False):
            pyramid = ClipPyramid(
                clip_fn, (capture.images.shape[1], capture.images.shape[2]),
                tile_size_range=(cfg.train.clip_scale_min, cfg.train.clip_scale_max),
                n_scales=min(cfg.train.num_clip_scales, 5),
                stride_scaler=cfg.train.patch_stride_scaler, device=dev,
            )
            pyramid.build(imgs, verbose=verbose)

        features = FeatureCache(
            dino=dino_feats,
            masks=masks,
            clip=pyramid,
            dino_extractor=dino_fn if isinstance(dino_fn, DinoExtractor) else None,
            clip_encoder=clip_fn if isinstance(clip_fn, OpenCLIPNetwork) else None,
        )

    device = cfg.runtime.device if hasattr(cfg, "runtime") else "cuda:0"
    t = cfg.train if hasattr(cfg, "train") else cfg
    mc = cfg.model if hasattr(cfg, "model") else ModelConfig()
    opt = POGSOptimizer(model, cfg)
    log = TrainLog()
    rng = np.random.default_rng(getattr(t, "seed", 42))
    torch.manual_seed(getattr(t, "seed", 42))

    V = capture.n_views() if hasattr(capture, "n_views") else len(capture.images)
    render_scale = min(1.0, t.render_width / capture.images.shape[2])
    raster_kwargs = dict(
        max_per_tile=cfg.runtime.torch_raster_max_per_tile,
        tile_chunk=cfg.runtime.torch_raster_tile_chunk,
    ) if model.backend == "torch" else {}

    model.train()
    t_start = time.time()
    for step in range(t.max_num_iterations):
        model.step = step
        v = int(rng.integers(0, V))
        cam = capture.cameras[v].to(device)
        if render_scale < 1.0:
            cam = cam.rescale(render_scale)
        gt_rgb = capture.images[v].to(device)
        gt_depth = capture.depths[v].to(device).unsqueeze(-1)
        if render_scale < 1.0:
            gt_rgb = F.interpolate(gt_rgb.permute(2, 0, 1)[None], size=(cam.height, cam.width),
                                   mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
            gt_depth = F.interpolate(gt_depth.permute(2, 0, 1)[None], size=(cam.height, cam.width),
                                     mode="nearest")[0].permute(1, 2, 0)

        use_dino = step > t.dino_step
        use_lang = step > t.lerf_step and features.clip is not None
        opt.zero_grad(step)

        out = model.get_outputs(
            cam, render_dino=use_dino, render_hash=use_lang,
            hash_downscale=t.clip_downscale_factor, raster_kwargs=raster_kwargs
        )

        # ---- photometric ------------------------------------------------- #
        loss = L.rgb_loss(out["rgb"], gt_rgb, mc.ssim_lambda)
        l_rgb = float(loss.detach())

        # ---- depth (Pearson) --------------------------------------------- #
        l_depth = 0.0
        if mc.depth_loss_type != "none":
            dl = L.depth_loss(gt_depth, out["depth"], mc.depth_loss_type)
            loss = loss + mc.depth_loss_mult * dl
            l_depth = float(dl.detach())

        # ---- DINO distillation ------------------------------------------- #
        l_dino, l_nnv = 0.0, 0.0
        if use_dino:
            gt_d = features.dino[v].to(device)
            dl = L.dino_loss(out["dino"], gt_d)
            loss = loss + dl
            l_dino = float(dl.detach())
            if step > t.dino_step + t.dino_nn_delay:
                nn_ids = model.ensure_nearest_ids(3)
                nnv = L.dino_nn_variance_loss(model.gauss_params["dino_feats"], nn_ids)
                loss = loss + nnv
                # summed over every gaussian, so this dominates the printed total
                # once it switches on -- exactly as in the official get_loss_dict
                l_nnv = float(nnv.detach())

        # ---- language + grouping ----------------------------------------- #
        l_clip, l_group = 0.0, 0.0
        if use_lang:
            # already rendered at 1/clip_downscale_factor, as in the official code
            hs = out["hashgrid"]
            clip_H, clip_W = hs.shape[:2]
            hs_ds = hs.reshape(-1, hs.shape[-1])
            n_pix = hs_ds.shape[0]
            sel = torch.randperm(n_pix, device=device)[: max(1, int(n_pix * 0.9))]

            scale = float(rng.uniform(t.clip_scale_min, t.clip_scale_max))
            # LERF scale conditioning, verbatim from ``pogs.py``:
            #   clip_scale = curr_scale * clip_H * (depth / camera.fy)
            # where ``clip_H`` is the *downscaled* render height and ``fy`` the
            # full-resolution focal length.
            depth_ds = F.interpolate(
                out["depth"].permute(2, 0, 1)[None], size=(clip_H, clip_W), mode="nearest"
            )[0].permute(1, 2, 0).reshape(-1, 1)[sel].detach()
            clip_scale = scale * clip_H * depth_ds / cam.fy

            clip_pred = model.gaussian_field.clip_from_hash(hs_ds[sel], clip_scale)
            gt_clip = features.clip.sample(v, scale, (clip_H, clip_W))
            gt_clip = gt_clip.reshape(-1, gt_clip.shape[-1])[sel.to(gt_clip.device)]
            cl = L.clip_loss(clip_pred, gt_clip, mc.clip_loss_weight)
            loss = loss + cl
            l_clip = float(cl.detach())

            inst = model.gaussian_field.instance_from_hash(hs_ds[sel]).cpu()
            mk = []
            for m in features.masks[v]:
                mm = F.interpolate(m[None, None].float(), size=(clip_H, clip_W), mode="nearest")[0, 0]
                mk.append(mm.reshape(-1)[sel.cpu()] > 0.5)
            gl = L.instance_contrastive_loss(
                inst, mk, mc.min_mask_screensize, n_dims=mc.instance_n_dims
            )
            if gl is not None:
                loss = loss + gl
                l_group = float(gl.detach())

        # ---- regularisation ---------------------------------------------- #
        if mc.use_scale_regularization and step % 10 == 0:
            loss = loss + L.scale_regularization(model.gauss_params["scales"], mc.max_gauss_ratio)

        loss.backward()
        model.accumulate_grad_stats(out)
        opt.step(step)

        # ---- densify / cull ---------------------------------------------- #
        if (
            step >= mc.warmup_length
            and step % mc.refine_every == 0
            and step <= mc.stop_split_at
        ):
            model.refinement_step(opt, verbose=verbose and cfg.train.log_every <= mc.refine_every)
            opt.rebind(model)
            if (step // mc.refine_every) % mc.reset_alpha_every == 0 and step > 0:
                model.reset_opacities(opt)

        if step % t.log_every == 0 or step == t.max_num_iterations - 1:
            with torch.no_grad():
                p = psnr(out["rgb"].detach(), gt_rgb)
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
                    f"clip={l_clip:.3f} grp={l_group:.3f} nnv={l_nnv:.2f} "
                    f"psnr={p:.2f} N={model.num_points}"
                )
        if progress is not None:
            progress(step, t.max_num_iterations)

    model.eval()
    if verbose:
        print(f"  training done in {time.time() - t_start:.1f}s, {model.num_points} gaussians")
    return log