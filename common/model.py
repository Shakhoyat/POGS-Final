"""
``POGSModel`` -- the unified representation of Fig. 2.

A single set of 3D Gaussians carries, per primitive:

    means, quats, scales, opacities, features_dc/features_rest   (3DGS, ref [1])
    dino_feats  (64-D)                                           Sec. IV-B

and a shared ``GaussianField`` maps position -> {CLIP-512, grouping-128}.
Rendering goes through the same rasteriser for every modality, which is what
lets the tracker compare *rendered* DINO features against *observed* ones.

Ported from ``pogs/pogs.py`` (which subclasses nerfstudio's ``SplatfactoModel``).
Densification uses gsplat's ``DefaultStrategy`` with splatfacto's settings when
the gsplat backend is available (exactly what the release does); the pure-torch
fallback re-implements the same thresholds approximately (``refinement_step``).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .fields import DinoProjection, GaussianField
from .geometry import PinholeCamera, quat_normalize
from .raster import rasterize, resolve_backend

SH_C0 = 0.28209479177387814


def _optimizer_state(optimizer, name: str):
    """Adam state dict for a named group. Supports both a plain ``torch.optim``
    optimizer and ``trainer.POGSOptimizer`` (one Adam per group)."""
    if hasattr(optimizer, "state_for"):
        return optimizer.state_for(name)
    return optimizer.state


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / SH_C0


def sh_to_rgb(sh: torch.Tensor) -> torch.Tensor:
    return sh * SH_C0 + 0.5


def random_quat_tensor(N: int, device="cpu") -> torch.Tensor:
    """Port of ``pogs/pogs.py::random_quat_tensor``."""
    u, v, w = torch.rand(N, device=device), torch.rand(N, device=device), torch.rand(N, device=device)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )


def k_nearest(x: torch.Tensor, k: int, include_self: bool = False):
    """``POGSModel.k_nearest_sklearn`` -- distances and indices of k neighbours."""
    from sklearn.neighbors import NearestNeighbors

    x_np = x.detach().cpu().numpy()
    nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)
    d, i = nn_model.kneighbors(x_np)
    if include_self:
        return d.astype(np.float32), i.astype(np.int64)
    return d[:, 1:].astype(np.float32), i[:, 1:].astype(np.int64)


class POGSModel(nn.Module):
    def __init__(
        self,
        seed_points: Optional[torch.Tensor] = None,
        seed_colors: Optional[torch.Tensor] = None,
        config: Optional[ModelConfig] = None,
        device: str = "cuda:0",
        raster_backend: str = "auto",
        **kwargs,
    ):
        super().__init__()
        if seed_points is None:
            seed_points = kwargs.get("points")
        if seed_colors is None:
            seed_colors = kwargs.get("colors")
        if config is None:
            config = kwargs.get("cfg", ModelConfig())
        self.config = config
        self.device_str = device
        self.backend = resolve_backend(raster_backend)
        self.step = 0

        N = seed_points.shape[0]
        means = seed_points.to(device).float()
        d, _ = k_nearest(means, 3)
        avg_d = torch.from_numpy(d).to(device).mean(dim=-1, keepdim=True)
        scales = torch.log(avg_d.repeat(1, 3).clamp(min=1e-5))
        dim_sh = (config.sh_degree + 1) ** 2

        fdc = rgb_to_sh(seed_colors.to(device).float().clamp(0, 1))
        frest = torch.zeros(N, dim_sh - 1, 3, device=device)

        self.gauss_params = nn.ParameterDict(
            {
                "means": nn.Parameter(means),
                "scales": nn.Parameter(scales),
                "quats": nn.Parameter(random_quat_tensor(N, device)),
                "features_dc": nn.Parameter(fdc),
                "features_rest": nn.Parameter(frest),
                "opacities": nn.Parameter(torch.logit(0.1 * torch.ones(N, 1, device=device))),
                "dino_feats": nn.Parameter(torch.randn(N, config.gaussian_dim, device=device) * config.dino_init_std),
            }
        )

        self.gaussian_field = GaussianField(
            clip_n_dims=config.clip_n_dims, instance_n_dims=config.instance_n_dims,
            log2_hashmap_size=config.log2_hashmap_size,
        ).to(device)
        self.nn = DinoProjection(config.gaussian_dim, config.dim).to(device)

        self.register_buffer(
            "background_color", torch.tensor(config.background_color, device=device), persistent=False
        )

        # densification bookkeeping
        self.xys_grad_norm: Optional[torch.Tensor] = None
        self.vis_counts: Optional[torch.Tensor] = None
        self.max_2d_size: Optional[torch.Tensor] = None
        self.nearest_ids: Optional[torch.Tensor] = None

        # populated after clustering
        self.cluster_labels: Optional[torch.Tensor] = None
        self.best_scale: Optional[float] = None

    # ------------------------------------------------------------------ #
    # accessors
    # ------------------------------------------------------------------ #

    @property
    def num_points(self) -> int:
        return self.gauss_params["means"].shape[0]

    @property
    def n_gaussians(self) -> int:
        return self.num_points

    @property
    def means(self):
        return self.gauss_params["means"]

    @property
    def scales(self):
        return self.gauss_params["scales"]

    @property
    def quats(self):
        return self.gauss_params["quats"]

    @property
    def opacities(self):
        return self.gauss_params["opacities"]

    @property
    def colors_sh(self) -> torch.Tensor:
        return torch.cat([self.gauss_params["features_dc"][:, None, :], self.gauss_params["features_rest"]], dim=1)

    def sh_degree_to_use(self) -> int:
        return min(self.step // self.config.sh_degree_interval, self.config.sh_degree)

    # ------------------------------------------------------------------ #
    # rendering
    # ------------------------------------------------------------------ #

    def _crop_ids(self, obj_id: Optional[int], invert: bool = False) -> Optional[torch.Tensor]:
        if obj_id is None or self.cluster_labels is None:
            return None
        if invert:
            return torch.nonzero(self.cluster_labels != obj_id, as_tuple=False).squeeze(-1)
        return torch.nonzero(self.cluster_labels == obj_id, as_tuple=False).squeeze(-1)

    def get_outputs(
        self,
        camera: PinholeCamera,
        tracking: bool = False,
        obj_id: Optional[int] = None,
        rgb_only: bool = False,
        render_dino: bool = True,
        render_hash: bool = False,
        hash_downscale: int = 1,
        dino_hw: Optional[Tuple[int, int]] = None,
        background: Optional[torch.Tensor] = None,
        tile_size: int = 16,
        raster_kwargs: Optional[dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """Mirror of ``POGSModel.get_outputs`` in ``pogs/pogs.py``.

        ``dino_hw``: resolution of the DINO render. ``None`` renders at the camera
        resolution (what the release does before ``lerf_step`` and while
        tracking); the trainer passes ``dino_rescale_factor * (h // 14)`` after
        ``lerf_step``, as the release does.
        """
        rk = raster_kwargs or {}
        crop = self._crop_ids(obj_id)
        gp = self.gauss_params
        aa = self.config.rasterize_mode == "antialiased"
        bg = self.background_color if background is None else background

        def _sel(t):
            return t if crop is None else t[crop]

        means = _sel(gp["means"])
        quats = quat_normalize(_sel(gp["quats"]))
        scales = torch.exp(_sel(gp["scales"]))
        opac = torch.sigmoid(_sel(gp["opacities"])).squeeze(-1)
        colors = _sel(self.colors_sh)

        out = rasterize(
            means, quats, scales, opac, colors, camera,
            backend=self.backend, sh_degree=self.sh_degree_to_use(),
            render_depth=True, background=bg, tile_size=tile_size,
            antialiased=aa, absgrad=self.training and self.config.use_absgrad, **rk,
        )
        alpha = out["alpha"]
        depth = out["depth"]
        # official: depth where nothing was hit is set to the max rendered depth
        depth = torch.where(alpha > 0, depth, depth.detach().max())
        outputs: Dict[str, torch.Tensor] = {
            "rgb": torch.clamp(out["render"], 0.0, 1.0),
            "depth": depth,
            "accumulation": alpha,
            "background": bg,
            "means2d": out["means2d"],
            "radii": out["radii"],
            "visible": out["visible"],
            "info": out.get("info", {}),
            "crop_ids": crop,
        }
        if rgb_only:
            return outputs

        # ---- DINO feature render (Sec. IV-B / IV-D) ---------------------- #
        # Geometry is detached during training so the DINO loss never moves the
        # Gaussians; while tracking the means stay attached (pose gradients).
        # quats / scales / opacities are always detached, as in the release.
        if render_dino:
            dino = _sel(gp["dino_feats"])
            g_means = means.detach() if self.training else means
            dcam = camera
            if dino_hw is not None and (dino_hw[0] != camera.height or dino_hw[1] != camera.width):
                dcam = camera.rescale_to(dino_hw[0], dino_hw[1])
            f = rasterize(
                g_means, quats.detach(), scales.detach(), opac.detach(), dino, dcam,
                backend=self.backend, sh_degree=None, render_depth=False,
                background=None, tile_size=tile_size, antialiased=aa, **rk,
            )
            feat, a = f["render"], f["alpha"]
            feat = torch.where(a > 0, feat / torch.clamp(a, min=1e-6).detach(), torch.zeros_like(feat))
            feat = self.nn(feat.reshape(-1, self.config.gaussian_dim)).reshape(*feat.shape[:-1], -1)
            if not self.training:
                feat = feat * (a >= 0.8).to(feat.dtype)
            outputs["dino"] = feat
            outputs["dino_alpha"] = a

        # ---- hash-grid render, consumed by the CLIP / grouping heads ------ #
        if render_hash:
            h = self.gaussian_field.get_hash(gp["means"])
            h = h if crop is None else h[crop]
            # rendered at 1/clip_downscale_factor, as in the release
            cam_h = camera if hash_downscale <= 1 else camera.rescale(1.0 / hash_downscale)
            hf = rasterize(
                means.detach(), quats.detach(), scales.detach(), opac.detach(), h, cam_h,
                backend=self.backend, sh_degree=None, render_depth=False,
                background=None, tile_size=tile_size, antialiased=aa, **rk,
            )
            outputs["hashgrid"] = hf["render"]
            outputs["hash_alpha"] = hf["alpha"]
        return outputs

    # ------------------------------------------------------------------ #
    # densification / pruning (splatfacto schedule)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def accumulate_grad_stats(self, outputs: Dict[str, torch.Tensor]) -> None:
        """Pure-torch fallback only (gsplat runs ``DefaultStrategy`` instead).
        Normalises the screen-space gradient the way gsplat / splatfacto do."""
        m2d = outputs.get("means2d")
        if m2d is None or m2d.grad is None:
            return
        H, W = outputs["rgb"].shape[:2]
        g = m2d.grad.detach().reshape(-1, 2).clone()
        g[:, 0] *= W / 2.0
        g[:, 1] *= H / 2.0
        grads = g.norm(dim=-1)
        vis = outputs["visible"]
        if vis is None:
            vis = grads > 0
        crop = outputs.get("crop_ids")
        if crop is not None:
            full = torch.zeros(self.num_points, device=grads.device)
            fullv = torch.zeros(self.num_points, device=grads.device, dtype=torch.bool)
            full[crop] = grads
            fullv[crop] = vis
            grads, vis = full, fullv
        if self.xys_grad_norm is None or self.xys_grad_norm.shape[0] != self.num_points:
            self.xys_grad_norm = torch.zeros(self.num_points, device=grads.device)
            self.vis_counts = torch.zeros(self.num_points, device=grads.device)
            self.max_2d_size = torch.zeros(self.num_points, device=grads.device)
        self.xys_grad_norm += grads * vis.float()
        self.vis_counts += vis.float()
        radii = outputs.get("radii")
        if radii is not None:
            r = radii.float()
            if crop is not None:
                fr = torch.zeros(self.num_points, device=r.device)
                fr[crop] = r
                r = fr
            self.max_2d_size = torch.maximum(self.max_2d_size, r / float(max(H, W)))

    def _resize_params(self, keep: Optional[torch.Tensor], extra: Optional[Dict[str, torch.Tensor]], optimizer) -> None:
        """Apply a keep-mask and/or append new Gaussians, keeping Adam state
        consistent (the part splatfacto does inside ``remove_from_all_optim``)."""
        new_params = {}
        for name, p in self.gauss_params.items():
            t = p.data
            if keep is not None:
                t = t[keep]
            if extra is not None and name in extra:
                t = torch.cat([t, extra[name]], dim=0)
            new_params[name] = t

        for group in optimizer.param_groups:
            name = group.get("name")
            if name not in new_params:
                continue
            opt_state = _optimizer_state(optimizer, name)
            old = group["params"][0]
            state = opt_state.get(old, None)
            new_p = nn.Parameter(new_params[name].contiguous())
            if state is not None and "exp_avg" in state:
                for key in ("exp_avg", "exp_avg_sq"):
                    s_ = state[key]
                    if keep is not None:
                        s_ = s_[keep]
                    if extra is not None and name in extra:
                        s_ = torch.cat([s_, torch.zeros_like(extra[name])], dim=0)
                    state[key] = s_
                del opt_state[old]
                opt_state[new_p] = state
            group["params"] = [new_p]
            self.gauss_params[name] = new_p

    @torch.no_grad()
    def refinement_step(self, optimizer, verbose: bool = False) -> Dict[str, int]:
        """Pure-torch fallback of splatfacto's grow + prune pass (gsplat's
        ``DefaultStrategy._grow_gs`` / ``_prune_gs``). Not bit-exact: the torch
        rasteriser exposes plain (not absolute) screen-space gradients."""
        cfg = self.config
        stats = {"split": 0, "dup": 0, "culled": 0}
        if self.xys_grad_norm is None:
            return stats

        avg = self.xys_grad_norm / self.vis_counts.clamp(min=1)
        n_new_budget = max(cfg.max_gaussians - self.num_points, 0)
        high_grad = avg > cfg.densify_grad_thresh
        smax = torch.exp(self.scales).max(dim=-1).values
        is_small = smax <= cfg.densify_size_thresh
        dup_mask = high_grad & is_small
        split_mask = high_grad & ~is_small
        if self.step < cfg.stop_screen_size_at and self.max_2d_size is not None:
            split_mask = split_mask | (high_grad & (self.max_2d_size > cfg.split_screen_size))
        n_split, n_dup = int(split_mask.sum()), int(dup_mask.sum())
        if n_split * cfg.n_split_samples + n_dup > n_new_budget:
            order = torch.argsort(avg * high_grad.float(), descending=True)
            allow = torch.zeros_like(high_grad)
            allow[order[: max(n_new_budget // (cfg.n_split_samples + 1), 0)]] = True
            split_mask &= allow
            dup_mask &= allow
            n_split, n_dup = int(split_mask.sum()), int(dup_mask.sum())

        if n_dup > 0:
            dup = {k: v.data[dup_mask].clone() for k, v in self.gauss_params.items()}
            self._resize_params(None, dup, optimizer)
            split_mask = torch.cat([split_mask, torch.zeros(n_dup, dtype=torch.bool, device=split_mask.device)])
            stats["dup"] = n_dup
        if n_split > 0:
            extra = self._split_gaussians(split_mask, cfg.n_split_samples)
            # splitting replaces the parent by its children
            self._resize_params(~split_mask, extra, optimizer)
            stats["split"] = n_split

        # ---- prune ------------------------------------------------------ #
        opac = torch.sigmoid(self.gauss_params["opacities"].data).squeeze(-1)
        cull = opac < cfg.cull_alpha_thresh
        if self.step > cfg.refine_every * cfg.reset_alpha_every:
            cull = cull | (torch.exp(self.gauss_params["scales"].data).max(dim=-1).values > cfg.cull_scale_thresh)
        n_cull = int(cull.sum().item())
        if 0 < n_cull < self.num_points - 128:
            self._resize_params(~cull, None, optimizer)
            stats["culled"] = n_cull

        self.xys_grad_norm = None
        self.vis_counts = None
        self.max_2d_size = None
        self.nearest_ids = None
        if verbose:
            print(f"    refine @{self.step}: {stats} -> {self.num_points} gaussians")
        return stats

    @torch.no_grad()
    def _split_gaussians(self, mask: torch.Tensor, samps: int) -> Dict[str, torch.Tensor]:
        from .geometry import quat_to_rotmat

        n = int(mask.sum().item())
        scales = torch.exp(self.gauss_params["scales"].data[mask])
        quats = quat_normalize(self.gauss_params["quats"].data[mask])
        R = quat_to_rotmat(quats)
        eps = torch.randn(samps, n, 3, device=scales.device) * scales.unsqueeze(0)
        offs = torch.einsum("nij,snj->sni", R, eps).reshape(-1, 3)

        out = {}
        for name, p in self.gauss_params.items():
            v = p.data[mask]
            rep = v.repeat(samps, *([1] * (v.dim() - 1)))
            if name == "means":
                rep = rep + offs
            elif name == "scales":
                rep = torch.log(torch.exp(rep) / 1.6)
            out[name] = rep
        return out

    @torch.no_grad()
    def reset_opacities(self, optimizer) -> None:
        cfg = self.config
        reset_val = torch.clamp(
            self.gauss_params["opacities"].data, max=torch.logit(torch.tensor(cfg.cull_alpha_thresh * 2.0)).item()
        )
        self.gauss_params["opacities"].data = reset_val
        for group in optimizer.param_groups:
            if group.get("name") == "opacities":
                st = _optimizer_state(optimizer, "opacities").get(group["params"][0], None)
                if st is not None and "exp_avg" in st:
                    st["exp_avg"].zero_()
                    st["exp_avg_sq"].zero_()

    # ------------------------------------------------------------------ #
    # helpers used by the loss / clustering
    # ------------------------------------------------------------------ #

    def ensure_nearest_ids(self, k: Optional[int] = None) -> torch.Tensor:
        """``k`` neighbours *including* the point itself (official: cuml
        ``NearestNeighbors(n_neighbors=3)``)."""
        k = self.config.dino_nn_k if k is None else k
        if self.nearest_ids is None or self.nearest_ids.shape[0] != self.num_points:
            _, idx = k_nearest(self.gauss_params["means"].data, k - 1, include_self=True)
            self.nearest_ids = torch.from_numpy(idx).to(self.gauss_params["means"].device)
        return self.nearest_ids

    def smoothed_hash(self, k: int = 3) -> torch.Tensor:
        """Opacity-weighted KNN-smoothed hash encoding.

        This is the ``cluster``/``localize_query_cb`` pre-processing in
        ``pogs/pogs.py``: average each Gaussian's hash encoding over itself plus
        its 3 nearest neighbours, weighted by softmax(sigmoid(opacity)).
        """
        means = self.gauss_params["means"].data
        _, idx = k_nearest(means, k, include_self=True)
        idx_t = torch.from_numpy(idx).to(means.device).view(-1)
        w = torch.sigmoid(self.gauss_params["opacities"].data[idx_t].view(-1, k + 1))
        w = torch.softmax(w, dim=-1)
        h = self.gaussian_field.get_hash(means[idx_t]).view(-1, k + 1, self.gaussian_field.hash_dim)
        return (h * w.unsqueeze(-1)).sum(dim=1)

    def state_snapshot(self) -> Dict[str, torch.Tensor]:
        return {k: v.data.detach().clone() for k, v in self.gauss_params.items()}

    def load_snapshot(self, snap: Dict[str, torch.Tensor]) -> None:
        for k, v in snap.items():
            self.gauss_params[k] = nn.Parameter(v.clone())