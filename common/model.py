"""
``POGSModel`` -- the unified representation of Fig. 2.

A single set of 3D Gaussians carries, per primitive:

    means, quats, scales, opacities, features_dc/features_rest   (3DGS, ref [1])
    dino_feats  (64-D)                                           Sec. IV-B

and a shared ``GaussianField`` maps position -> {CLIP-512, grouping-128}.
Rendering goes through the same rasteriser for every modality, which is what
lets the tracker compare *rendered* DINO features against *observed* ones.

Ported from ``pogs/pogs.py`` (which subclasses nerfstudio's ``SplatfactoModel``);
the densification/pruning schedule follows splatfacto with a backend-robust
percentile threshold in place of the raw gradient threshold, because the units
of ``means2d.grad`` differ between gsplat versions and the pure-torch backend.
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
                "dino_feats": nn.Parameter(torch.randn(N, config.gaussian_dim, device=device) * 0.1),
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
        detach_geometry_for_dino: bool = True,
        tile_size: int = 16,
        raster_kwargs: Optional[dict] = None,
    ) -> Dict[str, torch.Tensor]:
        rk = raster_kwargs or {}
        crop = self._crop_ids(obj_id)
        gp = self.gauss_params

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
            render_depth=True, background=self.background_color,
            tile_size=tile_size, antialiased=self.config.rasterize_mode == "antialiased", **rk,
        )
        outputs: Dict[str, torch.Tensor] = {
            "rgb": torch.clamp(out["render"], 0.0, 1.0),
            "depth": out["depth"],
            "accumulation": out["alpha"],
            "background": self.background_color,
            "means2d": out["means2d"],
            "radii": out["radii"],
            "visible": out["visible"],
            "crop_ids": crop,
        }
        if rgb_only:
            return outputs

        # ---- DINO feature render (Sec. IV-B / IV-D) ---------------------- #
        if render_dino:
            dino = _sel(gp["dino_feats"])
            g_means = means.detach() if detach_geometry_for_dino else means
            f = rasterize(
                g_means, quats.detach(), scales.detach(), opac.detach(), dino, camera,
                backend=self.backend, sh_degree=None, render_depth=False,
                background=None, tile_size=tile_size,
                antialiased=self.config.rasterize_mode == "antialiased", **rk,
            )
            feat, alpha = f["render"], f["alpha"]
            alpha_c = torch.clamp(alpha, min=1e-6)
            feat = torch.where(alpha > 0, feat / alpha_c.detach(), torch.zeros_like(feat))
            feat = self.nn(feat.reshape(-1, self.config.gaussian_dim)).reshape(*feat.shape[:-1], -1)
            if not self.training:
                feat = feat * (alpha >= 0.8).to(feat.dtype)
            outputs["dino"] = feat
            outputs["dino_alpha"] = alpha

        # ---- hash-grid render, consumed by the CLIP / grouping heads ------ #
        if render_hash:
            h = self.gaussian_field.get_hash(gp["means"])
            h = h if crop is None else h[crop]
            # The official code rescales the camera by 1/clip_downscale_factor
            # before this pass (``pogs.py``: ``camera.rescale_output_resolution``),
            # so the 96-channel hash is never rasterised at full resolution.
            cam_h = camera if hash_downscale <= 1 else camera.rescale(1.0 / hash_downscale)
            hf = rasterize(
                means.detach(), quats.detach(), scales.detach(), opac.detach(), h, cam_h,
                backend=self.backend, sh_degree=None, render_depth=False,
                background=None, tile_size=tile_size,
                antialiased=self.config.rasterize_mode == "antialiased", **rk,
            )
            outputs["hashgrid"] = hf["render"]
            outputs["hash_alpha"] = hf["alpha"]
        return outputs

    # ------------------------------------------------------------------ #
    # densification / pruning (splatfacto schedule)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def accumulate_grad_stats(self, outputs: Dict[str, torch.Tensor]) -> None:
        m2d = outputs.get("means2d")
        if m2d is None or m2d.grad is None:
            return
        grads = m2d.grad.detach().norm(dim=-1).squeeze(0)
        vis = outputs["visible"]
        if vis is None:
            vis = grads > 0
        crop = outputs.get("crop_ids")
        if crop is not None:
            full = torch.zeros(self.num_points, device=grads.device)
            fullv = torch.zeros(self.num_points, device=grads.device)
            full[crop] = grads
            fullv[crop] = vis.float()
            grads, vis = full, fullv > 0
        if self.xys_grad_norm is None or self.xys_grad_norm.shape[0] != self.num_points:
            self.xys_grad_norm = torch.zeros(self.num_points, device=grads.device)
            self.vis_counts = torch.zeros(self.num_points, device=grads.device)
            self.max_2d_size = torch.zeros(self.num_points, device=grads.device)
        self.xys_grad_norm += grads
        self.vis_counts += vis.float()
        radii = outputs.get("radii")
        if radii is not None:
            r = radii.float()
            if crop is not None:
                fr = torch.zeros(self.num_points, device=r.device)
                fr[crop] = r
                r = fr
            denom = max(outputs["rgb"].shape[0], outputs["rgb"].shape[1])
            self.max_2d_size = torch.maximum(self.max_2d_size, r / denom)

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
                    s = state[key]
                    if keep is not None:
                        s = s[keep]
                    if extra is not None and name in extra:
                        s = torch.cat([s, torch.zeros_like(extra[name])], dim=0)
                    state[key] = s
                del opt_state[old]
                opt_state[new_p] = state
            group["params"] = [new_p]
            self.gauss_params[name] = new_p

    @torch.no_grad()
    def refinement_step(self, optimizer, densify_percentile: float = 0.05, verbose: bool = False) -> Dict[str, int]:
        """One densify + cull pass. Called every ``refine_every`` steps between
        ``warmup_length`` and ``stop_split_at``."""
        cfg = self.config
        stats = {"split": 0, "dup": 0, "culled": 0}
        if self.xys_grad_norm is None:
            return stats

        avg = self.xys_grad_norm / self.vis_counts.clamp(min=1)
        seen = self.vis_counts > 0
        n_new_budget = max(cfg.max_gaussians - self.num_points, 0)

        if n_new_budget > 0 and bool(seen.any()):
            g = avg[seen]
            q = torch.quantile(g.float(), 1.0 - densify_percentile) if g.numel() > 1 else g.max()
            high_grad = seen & (avg >= q)
            big = torch.exp(self.scales).max(dim=-1).values > cfg.densify_size_thresh
            if self.max_2d_size is not None:
                big = big | (self.max_2d_size > cfg.split_screen_size)

            split_mask = high_grad & big
            dup_mask = high_grad & (~big)

            n_split = int(split_mask.sum().item())
            n_dup = int(dup_mask.sum().item())
            if n_split * cfg.n_split_samples + n_dup > n_new_budget:
                # trim to budget, keeping the highest-gradient candidates
                order = torch.argsort(avg * high_grad.float(), descending=True)
                allow = torch.zeros_like(high_grad)
                allow[order[: max(n_new_budget // (cfg.n_split_samples + 1), 0)]] = True
                split_mask &= allow
                dup_mask &= allow
                n_split = int(split_mask.sum().item())
                n_dup = int(dup_mask.sum().item())

            extra: Dict[str, torch.Tensor] = {}
            if n_split > 0:
                extra = self._split_gaussians(split_mask, cfg.n_split_samples)
                stats["split"] = n_split
            if n_dup > 0:
                dup = {k: v.data[dup_mask].clone() for k, v in self.gauss_params.items()}
                extra = {k: torch.cat([extra[k], dup[k]], 0) if k in extra else dup[k] for k in dup}
                stats["dup"] = n_dup

            if extra:
                # shrink the parents that were split (splatfacto: /1.6)
                self.gauss_params["scales"].data[split_mask] = torch.log(
                    torch.exp(self.gauss_params["scales"].data[split_mask]) / 1.6
                )
                self._resize_params(None, extra, optimizer)

        # ---- cull ------------------------------------------------------- #
        opac = torch.sigmoid(self.gauss_params["opacities"].data).squeeze(-1)
        cull = opac < cfg.cull_alpha_thresh
        if self.step > cfg.refine_every * cfg.reset_alpha_every:
            cull = cull | (torch.exp(self.gauss_params["scales"].data).max(dim=-1).values > cfg.cull_scale_thresh)
        n_cull = int(cull.sum().item())
        if n_cull > 0 and n_cull < self.num_points - 128:
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

    def ensure_nearest_ids(self, k: int = 3) -> torch.Tensor:
        if self.nearest_ids is None or self.nearest_ids.shape[0] != self.num_points:
            _, idx = k_nearest(self.gauss_params["means"].data, k, include_self=True)
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