"""
Differentiable Gaussian rasterisation with two interchangeable backends.

POGS renders **four** different things through the same rasteriser
(``pogs/pogs.py::get_outputs``):

  * RGB (+ expected depth)  -- spherical-harmonic colours
  * the 64-D per-Gaussian DINO feature      (``dino_crop``)
  * the multi-resolution hash encoding      (``clip_hash_encoding``)
  * the object grouping features

so the rasteriser must support an **arbitrary number of colour channels**.

Backends
--------
``gsplat``  the CUDA backend the paper uses (Nerfstudio's ``gsplat``).  Used
            automatically when importable.
``torch``   a self-contained tile-based EWA splatting implementation in pure
            PyTorch.  Slower, but needs no compiler, so the notebook still runs
            end-to-end if the gsplat JIT build fails on the Kaggle image.

Both return the same dict, so nothing downstream needs to know which ran.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch

from .geometry import PinholeCamera, quat_to_rotmat

_GSPLAT = None
_GSPLAT_CHECKED = False


def _ensure_cuda_env() -> None:
    """Point torch's JIT builder at the pip-installed CUDA toolkit in .venv
    (see ``tools/setup_gsplat.sh``) when the caller did not set one up."""
    import os
    import sys

    if os.environ.get("CUDA_HOME"):
        return
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sp = os.path.join(root, ".venv", "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages")
    cuda = os.path.join(sp, "nvidia", "cu13")
    if not os.path.exists(os.path.join(cuda, "bin", "nvcc")):
        return
    os.environ["CUDA_HOME"] = cuda
    os.environ["PATH"] = os.pathsep.join([os.path.join(root, ".venv", "bin"), os.path.join(cuda, "bin"), os.environ.get("PATH", "")])
    inc = os.path.join(root, ".venv", "sysinclude", "usr", "include")
    pyinc = os.path.join(inc, f"python{sys.version_info.major}.{sys.version_info.minor}")
    os.environ["CPATH"] = os.pathsep.join(p for p in [pyinc, inc, os.environ.get("CPATH", "")] if p)
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")


def gsplat_available() -> bool:
    global _GSPLAT, _GSPLAT_CHECKED
    if not _GSPLAT_CHECKED:
        _GSPLAT_CHECKED = True
        try:  # pragma: no cover - depends on the runtime image
            _ensure_cuda_env()
            from gsplat.rendering import rasterization  # noqa: F401
            from gsplat.cuda._backend import _C  # triggers / loads the JIT build
            if _C is None:
                raise RuntimeError("gsplat CUDA extension could not be built or loaded")
            _GSPLAT = rasterization
        except Exception as exc:  # noqa: BLE001
            print(f"[raster] gsplat unavailable ({type(exc).__name__}: {exc}); "
                  f"falling back to the pure-PyTorch rasteriser.")
            _GSPLAT = None
    return _GSPLAT is not None


def resolve_backend(preference: str) -> str:
    if preference == "torch":
        return "torch"
    if preference == "gsplat":
        if not gsplat_available():
            raise RuntimeError("raster_backend='gsplat' requested but gsplat is not importable")
        return "gsplat"
    return "gsplat" if gsplat_available() else "torch"


# --------------------------------------------------------------------------- #
# Shared front-end
# --------------------------------------------------------------------------- #


def rasterize(
    means: torch.Tensor,        # (N, 3) world space
    quats: torch.Tensor,        # (N, 4) wxyz, un-normalised ok
    scales: torch.Tensor,       # (N, 3) *linear* scales (already exp'd)
    opacities: torch.Tensor,    # (N,) in [0, 1]
    colors: torch.Tensor,       # (N, D) arbitrary D, or (N, K, 3) SH coeffs
    camera: PinholeCamera,
    backend: str = "auto",
    sh_degree: Optional[int] = None,
    render_depth: bool = True,
    background: Optional[torch.Tensor] = None,   # (D,)
    tile_size: int = 16,
    max_per_tile: int = 192,
    tile_chunk: int = 128,
    antialiased: bool = False,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    absgrad: bool = False,
) -> Dict[str, torch.Tensor]:
    """Returns ``{'render': (H,W,D), 'alpha': (H,W,1), 'depth': (H,W,1)|None,
    'means2d': (N,2), 'radii': (N,), 'visible': (N,) bool}``."""
    backend = resolve_backend(backend)
    if backend == "gsplat":
        return _rasterize_gsplat(
            means, quats, scales, opacities, colors, camera, sh_degree,
            render_depth, background, tile_size, antialiased, near_plane, far_plane,
            absgrad,
        )
    return _rasterize_torch(
        means, quats, scales, opacities, colors, camera, sh_degree,
        render_depth, background, tile_size, max_per_tile, tile_chunk,
        antialiased, near_plane, far_plane,
    )


def rasterize_gaussians(means, scales, quats, opacities, colors, camera, **kwargs):
    """Adapter function providing alternate parameter ordering."""
    if opacities.ndim > 1 and opacities.shape[-1] == 1:
        opacities = opacities.squeeze(-1)
    out = rasterize(
        means=means, quats=quats, scales=scales, opacities=opacities, colors=colors, camera=camera, **kwargs
    )
    if "render" in out and "rgb" not in out:
        out["rgb"] = out["render"]
    return out


# --------------------------------------------------------------------------- #
# gsplat backend
# --------------------------------------------------------------------------- #


def _rasterize_gsplat(
    means, quats, scales, opacities, colors, camera, sh_degree,
    render_depth, background, tile_size, antialiased, near_plane, far_plane,
    absgrad=False,
):
    rasterization = _GSPLAT
    viewmat = camera.w2c.unsqueeze(0)
    K = camera.K.unsqueeze(0)
    mode = "RGB+ED" if render_depth else "RGB"

    render, alpha, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat,
        Ks=K,
        width=camera.width,
        height=camera.height,
        tile_size=tile_size,
        packed=False,
        near_plane=near_plane,
        far_plane=far_plane,
        render_mode=mode,
        sh_degree=sh_degree,
        sparse_grad=False,
        absgrad=absgrad,
        rasterize_mode="antialiased" if antialiased else "classic",
    )
    alpha = alpha.squeeze(0)                       # (H, W, 1)
    if render_depth:
        rgb = render[0, ..., :-1]
        depth = render[0, ..., -1:]
    else:
        rgb = render[0]
        depth = None
    if background is not None:
        rgb = rgb + (1.0 - alpha) * background.to(rgb)

    means2d = info.get("means2d", None)
    if means2d is not None and means2d.requires_grad:
        means2d.retain_grad()
    radii = info.get("radii", None)
    if radii is not None and radii.ndim > 1:
        radii = radii.reshape(radii.shape[0], radii.shape[1], -1)[0]
        if radii.ndim > 1:
            radii = radii.max(dim=-1).values
    return {
        "render": rgb,
        "alpha": alpha,
        "depth": depth,
        "means2d": means2d,
        "radii": radii,
        "visible": (radii > 0) if radii is not None else None,
        "info": info,          # raw gsplat info, consumed by gsplat's DefaultStrategy
    }


# --------------------------------------------------------------------------- #
# Pure-PyTorch backend
# --------------------------------------------------------------------------- #


def _eval_sh(sh_coeffs: torch.Tensor, dirs: torch.Tensor, degree: int) -> torch.Tensor:
    """Real spherical harmonics up to degree 3 (same basis/order as gsplat)."""
    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
          -1.0925484305920792, 0.5462742152960396)
    C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
          0.3731763325901154, -0.4570457994644658, 1.445305721320277,
          -0.5900435899266435)

    result = C0 * sh_coeffs[:, 0]
    if degree >= 1:
        x, y, z = dirs[:, 0:1], dirs[:, 1:2], dirs[:, 2:3]
        result = result - C1 * y * sh_coeffs[:, 1] + C1 * z * sh_coeffs[:, 2] - C1 * x * sh_coeffs[:, 3]
        if degree >= 2:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + C2[0] * xy * sh_coeffs[:, 4]
                + C2[1] * yz * sh_coeffs[:, 5]
                + C2[2] * (2.0 * zz - xx - yy) * sh_coeffs[:, 6]
                + C2[3] * xz * sh_coeffs[:, 7]
                + C2[4] * (xx - yy) * sh_coeffs[:, 8]
            )
            if degree >= 3:
                result = (
                    result
                    + C3[0] * y * (3 * xx - yy) * sh_coeffs[:, 9]
                    + C3[1] * xy * z * sh_coeffs[:, 10]
                    + C3[2] * y * (4 * zz - xx - yy) * sh_coeffs[:, 11]
                    + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh_coeffs[:, 12]
                    + C3[4] * x * (4 * zz - xx - yy) * sh_coeffs[:, 13]
                    + C3[5] * z * (xx - yy) * sh_coeffs[:, 14]
                    + C3[6] * x * (xx - 3 * yy) * sh_coeffs[:, 15]
                )
    return result + 0.5



_TILE_OVERFLOW_STATE = {"dropped": 0, "total": 0, "warned": False}


def tile_overflow_stats() -> dict:
    """Fraction of gaussian-tile pairs discarded by ``max_per_tile`` so far.

    Anything above a few percent means the torch rasteriser's gradients are no
    longer trustworthy for pose optimisation -- raise ``max_per_tile`` or lower
    the gaussian count.  ``reset`` zeroes the counters.
    """
    st = _TILE_OVERFLOW_STATE
    return {
        "pairs_total": float(st["total"]),
        "pairs_dropped": float(st["dropped"]),
        "drop_rate": st["dropped"] / max(st["total"], 1),
    }


def reset_tile_overflow_stats() -> None:
    _TILE_OVERFLOW_STATE.update(dropped=0, total=0, warned=False)


def _report_tile_overflow(dropped: int, total: int, max_per_tile: int) -> None:
    st = _TILE_OVERFLOW_STATE
    st["dropped"] += dropped
    st["total"] += total
    if dropped and not st["warned"] and dropped / max(total, 1) > 0.02:
        st["warned"] = True
        print(
            f"[raster] WARNING: max_per_tile={max_per_tile} discarded "
            f"{dropped}/{total} ({dropped / max(total, 1):.1%}) gaussian-tile "
            f"pairs on this render. The truncation boundary moves with pose, so "
            f"pose gradients degrade with it -- raise "
            f"RuntimeConfig.torch_raster_max_per_tile or lower "
            f"ModelConfig.max_gaussians before trusting any tracking number."
        )


def _rasterize_torch(
    means, quats, scales, opacities, colors, camera, sh_degree,
    render_depth, background, tile_size, max_per_tile, tile_chunk,
    antialiased, near_plane, far_plane,
):
    device = means.device
    H, W = camera.height, camera.width
    w2c = camera.w2c.to(device)
    R_w2c, t_w2c = w2c[:3, :3], w2c[:3, 3]

    # ---- view transform -------------------------------------------------- #
    p_cam = means @ R_w2c.T + t_w2c                       # (N, 3)
    z = p_cam[:, 2]
    valid = (z > near_plane) & (z < far_plane)

    # ---- SH evaluation (needs world-space view dirs) --------------------- #
    if sh_degree is not None:
        cam_pos = camera.c2w[:3, 3].to(device)
        dirs = torch.nn.functional.normalize(means - cam_pos, dim=-1)
        rgb = _eval_sh(colors, dirs, sh_degree)
        colors_eval = torch.clamp(rgb, min=0.0)
    else:
        colors_eval = colors

    # ---- 2D projection --------------------------------------------------- #
    zc = z.clamp(min=near_plane)
    u = camera.fx * p_cam[:, 0] / zc + camera.cx
    v = camera.fy * p_cam[:, 1] / zc + camera.cy
    means2d = torch.stack([u, v], dim=-1)
    if means2d.requires_grad:
        means2d.retain_grad()

    # ---- 3D -> 2D covariance (EWA) --------------------------------------- #
    Rq = quat_to_rotmat(quats)                            # (N, 3, 3)
    S = scales.clamp(min=1e-7)
    M = Rq * S.unsqueeze(1)                               # R @ diag(s)
    Sigma = M @ M.transpose(1, 2)                         # (N, 3, 3)

    zero = torch.zeros_like(zc)
    J = torch.stack(
        [
            torch.stack([camera.fx / zc, zero, -camera.fx * p_cam[:, 0] / (zc * zc)], -1),
            torch.stack([zero, camera.fy / zc, -camera.fy * p_cam[:, 1] / (zc * zc)], -1),
        ],
        dim=1,
    )                                                     # (N, 2, 3)
    T = J @ R_w2c.unsqueeze(0)                            # (N, 2, 3)
    cov2d = T @ Sigma @ T.transpose(1, 2)                 # (N, 2, 2)

    det_raw = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] ** 2
    blur = 0.3
    cov2d = cov2d + blur * torch.eye(2, device=device, dtype=cov2d.dtype).unsqueeze(0)
    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] ** 2
    valid = valid & (det > 1e-12)
    det_safe = det.clamp(min=1e-12)

    opa = opacities
    if antialiased:
        compensation = torch.sqrt(torch.clamp(det_raw / det_safe, min=0.0))
        opa = opa * compensation

    # conic = inverse(cov2d)
    conic = torch.stack(
        [cov2d[:, 1, 1] / det_safe, -cov2d[:, 0, 1] / det_safe, cov2d[:, 0, 0] / det_safe],
        dim=-1,
    )                                                     # (a, b, c)

    # 3-sigma screen radius from the larger eigenvalue
    tr = 0.5 * (cov2d[:, 0, 0] + cov2d[:, 1, 1])
    disc = torch.sqrt(torch.clamp(tr * tr - det_safe, min=0.0))
    lam = tr + disc
    radii = torch.ceil(3.0 * torch.sqrt(torch.clamp(lam, min=0.0)))
    radii = torch.clamp(radii, max=float(max(H, W)))
    valid = valid & (radii >= 1.0) & (opa > 1.0 / 255.0)
    valid = valid & (u + radii > 0) & (u - radii < W) & (v + radii > 0) & (v - radii < H)

    D = colors_eval.shape[-1]
    out_render = torch.zeros(H, W, D, device=device, dtype=colors_eval.dtype)
    out_alpha = torch.zeros(H, W, 1, device=device, dtype=colors_eval.dtype)
    out_depth = torch.zeros(H, W, 1, device=device, dtype=colors_eval.dtype)

    idx_valid = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    if idx_valid.numel() == 0:
        if background is not None:
            out_render = out_render + background.to(out_render)
        return {
            "render": out_render, "alpha": out_alpha,
            "depth": out_depth if render_depth else None,
            "means2d": means2d, "radii": radii, "visible": valid, "info": {},
        }

    # ---- tile binning ---------------------------------------------------- #
    nx = (W + tile_size - 1) // tile_size
    ny = (H + tile_size - 1) // tile_size
    n_tiles = nx * ny

    uv = means2d[idx_valid]
    rr = radii[idx_valid]
    tx0 = torch.clamp(torch.floor((uv[:, 0] - rr) / tile_size).long(), 0, nx - 1)
    tx1 = torch.clamp(torch.floor((uv[:, 0] + rr) / tile_size).long(), 0, nx - 1)
    ty0 = torch.clamp(torch.floor((uv[:, 1] - rr) / tile_size).long(), 0, ny - 1)
    ty1 = torch.clamp(torch.floor((uv[:, 1] + rr) / tile_size).long(), 0, ny - 1)

    span_x = (tx1 - tx0 + 1)
    span_y = (ty1 - ty0 + 1)
    counts = span_x * span_y
    total = int(counts.sum().item())
    if total == 0:
        if background is not None:
            out_render = out_render + background.to(out_render)
        return {
            "render": out_render, "alpha": out_alpha,
            "depth": out_depth if render_depth else None,
            "means2d": means2d, "radii": radii, "visible": valid, "info": {},
        }

    # expand (gaussian -> covered tiles) without a python loop
    gauss_of_pair = torch.repeat_interleave(torch.arange(len(idx_valid), device=device), counts)
    offs = torch.cumsum(counts, 0) - counts
    within = torch.arange(total, device=device) - torch.repeat_interleave(offs, counts)
    sx = torch.repeat_interleave(span_x, counts)
    dy = within // sx
    dx = within - dy * sx
    tile_x = torch.repeat_interleave(tx0, counts) + dx
    tile_y = torch.repeat_interleave(ty0, counts) + dy
    tile_id = tile_y * nx + tile_x

    depth_of_pair = z[idx_valid][gauss_of_pair]

    # sort by (tile, depth) -- stable two-pass radix on the flattened key
    order = torch.argsort(depth_of_pair)
    tile_sorted = tile_id[order]
    order2 = torch.argsort(tile_sorted, stable=True)
    order = order[order2]
    tile_sorted = tile_id[order]
    gauss_sorted = gauss_of_pair[order]

    # per-tile rank -> keep the ``max_per_tile`` front-most gaussians
    tile_starts = torch.searchsorted(
        tile_sorted, torch.arange(n_tiles, device=device), right=False
    )
    rank = torch.arange(total, device=device) - tile_starts[tile_sorted]
    keep = rank < max_per_tile
    # Dropping the tail of a tile is not just a quality trade-off: which
    # gaussians survive depends on their projected depth, so the truncation
    # boundary MOVES as the pose changes.  That makes the rendered image a
    # discontinuous function of the pose, and the analytic gradient stops
    # agreeing with the real descent direction.  Measured on a 40k-gaussian
    # scene, cos(analytic, finite-difference) of the tracking loss was
    #   max_per_tile=  64 -> 0.04  (orthogonal: pose optimisation diverges)
    #   max_per_tile= 192 -> 0.70
    #   max_per_tile=2048 -> 0.90  (no truncation)
    # so a too-small value silently turns Sec. IV-D tracking into a random walk.
    # Report it rather than letting it corrupt gradients unnoticed.
    _report_tile_overflow(int(total - int(keep.sum())), int(total), max_per_tile)
    tile_sorted = tile_sorted[keep]
    gauss_sorted = gauss_sorted[keep]
    rank = rank[keep]

    L = int(rank.max().item()) + 1
    buf = torch.full((n_tiles, L), -1, dtype=torch.long, device=device)
    buf[tile_sorted, rank] = idx_valid[gauss_sorted]

    # ---- per-tile alpha compositing -------------------------------------- #
    py, px = torch.meshgrid(
        torch.arange(tile_size, device=device, dtype=means2d.dtype),
        torch.arange(tile_size, device=device, dtype=means2d.dtype),
        indexing="ij",
    )
    local = torch.stack([px.reshape(-1), py.reshape(-1)], dim=-1)   # (TT, 2)
    TT = tile_size * tile_size

    tile_ids_all = torch.arange(n_tiles, device=device)
    origin = torch.stack(
        [(tile_ids_all % nx) * tile_size, (tile_ids_all // nx) * tile_size], dim=-1
    ).to(means2d.dtype)                                             # (n_tiles, 2)

    render_flat = torch.zeros(n_tiles, TT, D, device=device, dtype=colors_eval.dtype)
    alpha_flat = torch.zeros(n_tiles, TT, 1, device=device, dtype=colors_eval.dtype)
    depth_flat = torch.zeros(n_tiles, TT, 1, device=device, dtype=colors_eval.dtype)

    for start in range(0, n_tiles, tile_chunk):
        stop = min(start + tile_chunk, n_tiles)
        ids = buf[start:stop]                                       # (C, L)
        occupied = ids >= 0
        if not bool(occupied.any()):
            continue
        # trim trailing all-empty columns for this chunk
        last = int(occupied.any(0).nonzero().max().item()) + 1
        ids = ids[:, :last]
        occupied = occupied[:, :last]
        safe = ids.clamp(min=0)

        g_uv = means2d[safe]                                        # (C, l, 2)
        g_conic = conic[safe]                                       # (C, l, 3)
        g_opa = opa[safe]                                           # (C, l)
        g_col = colors_eval[safe]                                   # (C, l, D)
        g_z = z[safe]                                               # (C, l)

        pix = origin[start:stop].unsqueeze(1) + local.unsqueeze(0)  # (C, TT, 2)
        dxp = pix[..., 0].unsqueeze(2) - g_uv[..., 0].unsqueeze(1)  # (C, TT, l)
        dyp = pix[..., 1].unsqueeze(2) - g_uv[..., 1].unsqueeze(1)
        power = -0.5 * (
            g_conic[:, None, :, 0] * dxp * dxp + g_conic[:, None, :, 2] * dyp * dyp
        ) - g_conic[:, None, :, 1] * dxp * dyp

        a = g_opa.unsqueeze(1) * torch.exp(torch.clamp(power, max=0.0))
        a = torch.clamp(a, max=0.99) * occupied.unsqueeze(1).to(a.dtype)

        # exclusive cumulative transmittance  T_i = prod_{j<i} (1 - a_j)
        one_minus = 1.0 - a
        trans = torch.cat(
            [torch.ones_like(one_minus[..., :1]), torch.cumprod(one_minus, dim=-1)[..., :-1]],
            dim=-1,
        )
        wgt = a * trans                                             # (C, TT, l)

        acc = wgt.sum(-1, keepdim=True)
        render_flat[start:stop] = torch.einsum("ctl,cld->ctd", wgt, g_col)
        alpha_flat[start:stop] = acc
        depth_flat[start:stop] = (
            (wgt * g_z.unsqueeze(1)).sum(-1, keepdim=True) / acc.clamp(min=1e-8)
        )

    # ---- untile ---------------------------------------------------------- #
    def _untile(flat, C):
        img = flat.view(ny, nx, tile_size, tile_size, C).permute(0, 2, 1, 3, 4)
        img = img.reshape(ny * tile_size, nx * tile_size, C)
        return img[:H, :W]

    out_render = _untile(render_flat, D)
    out_alpha = _untile(alpha_flat, 1)
    out_depth = _untile(depth_flat, 1)

    if background is not None:
        out_render = out_render + (1.0 - out_alpha) * background.to(out_render)

    return {
        "render": out_render,
        "alpha": out_alpha,
        "depth": out_depth if render_depth else None,
        "means2d": means2d,
        "radii": radii,
        "visible": valid,
        "info": {"means2d": means2d, "radii": radii},
    }