"""
POGS training losses.

Total training objective (Sec. IV-B) =
    L_rgb  (splatfacto: (1-lambda)*L1 + lambda*(1-SSIM))
  + depth_loss_mult * L_depth      (Pearson correlation, ref [48])
  + L_dino  (MSE on rendered vs observed DINO maps) + 0.01 * NN-variance
  + L_group (contrastive lift, ref [50], with the extra "negative mask")
  + clip_loss_weight * L_clip      (Huber, delta=1.25)
  + scale regularisation           (PhysGaussian)

Every constant matches ``pogs/pogs.py::get_loss_dict`` /
``pogs/model_components/losses.py``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# RGB
# --------------------------------------------------------------------------- #


def _gaussian_window(size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return g.outer(g)


def ssim(a: torch.Tensor, b: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """a, b: (H, W, C) in [0, 1]."""
    C = a.shape[-1]
    x = a.permute(2, 0, 1).unsqueeze(0)
    y = b.permute(2, 0, 1).unsqueeze(0)
    win = _gaussian_window(window_size, sigma, a.device, a.dtype).expand(C, 1, window_size, window_size)
    pad = window_size // 2
    mu_x = F.conv2d(x, win, padding=pad, groups=C)
    mu_y = F.conv2d(y, win, padding=pad, groups=C)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sx = F.conv2d(x * x, win, padding=pad, groups=C) - mu_x2
    sy = F.conv2d(y * y, win, padding=pad, groups=C) - mu_y2
    sxy = F.conv2d(x * y, win, padding=pad, groups=C) - mu_xy
    C1, C2 = 0.01**2, 0.03**2
    m = ((2 * mu_xy + C1) * (2 * sxy + C2)) / ((mu_x2 + mu_y2 + C1) * (sx + sy + C2))
    return m.mean()


def rgb_loss(pred: torch.Tensor, gt: torch.Tensor, ssim_lambda: float = 0.2) -> torch.Tensor:
    l1 = torch.abs(pred - gt).mean()
    return (1.0 - ssim_lambda) * l1 + ssim_lambda * (1.0 - ssim(pred, gt))


# --------------------------------------------------------------------------- #
# Depth (Sec. IV-B: "We additionally regularize the 3D geometry with a depth
# reconstruction objective to encourage Gaussian means to be positioned on
# object surfaces", ref [48] Chung et al.)
# --------------------------------------------------------------------------- #


def pearson_corrcoef(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.reshape(-1)
    y = y.reshape(-1)
    xm = x - x.mean()
    ym = y - y.mean()
    return (xm * ym).sum() / (xm.norm() * ym.norm() + 1e-8)


def pearson_correlation_depth_loss(termination_depth: torch.Tensor, predicted_depth: torch.Tensor) -> torch.Tensor:
    """Port of ``pogs/model_components/losses.py::pearson_correlation_depth_loss``."""
    return torch.mean(1.0 - pearson_corrcoef(predicted_depth.reshape(-1, 1), termination_depth.reshape(-1, 1)))


def mse_depth_loss(termination_depth: torch.Tensor, predicted_depth: torch.Tensor) -> torch.Tensor:
    mask = termination_depth > 0
    return torch.mean(((termination_depth - predicted_depth) ** 2) * mask)


def depth_loss(
    termination_depth: torch.Tensor, predicted_depth: torch.Tensor, kind: str = "pearson"
) -> torch.Tensor:
    if kind == "none":
        return torch.zeros((), device=predicted_depth.device)
    if kind == "mse":
        return mse_depth_loss(termination_depth, predicted_depth)
    valid = termination_depth > 0
    if valid.sum() < 16:
        return torch.zeros((), device=predicted_depth.device)
    return pearson_correlation_depth_loss(termination_depth[valid], predicted_depth[valid])


# --------------------------------------------------------------------------- #
# DINO distillation
# --------------------------------------------------------------------------- #


def dino_loss(rendered: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Official: resize the GT map to the rendered resolution, then MSE."""
    if rendered.shape[:2] != gt.shape[:2]:
        gt = F.interpolate(
            gt.permute(2, 0, 1).unsqueeze(0), size=rendered.shape[:2],
            mode="bilinear", align_corners=False,
        )[0].permute(1, 2, 0)
    return F.mse_loss(rendered, gt)


def dino_nn_variance_loss(dino_feats: torch.Tensor, nearest_ids: torch.Tensor, weight: float = 0.01) -> torch.Tensor:
    """``0.01 * dino_feats[nearest_ids].var(dim=1).sum()`` -- encourages spatial
    neighbours to share DINO features, which is what makes cluster-level rigid
    tracking well-posed."""
    return weight * dino_feats[nearest_ids].var(dim=1).sum()


# --------------------------------------------------------------------------- #
# Grouping (contrastive lift, ref [50])
# --------------------------------------------------------------------------- #


def instance_contrastive_loss(
    instance: torch.Tensor,          # (P, 128) rendered group features at sampled pixels
    masks: List[torch.Tensor],       # list of (P,) bool; the LAST one is the negative mask
    min_mask_screensize: float = 0.003,
    margin: float = 1.0,
    n_dims: int = 128,
) -> Optional[torch.Tensor]:
    """Verbatim port of the ``instance_loss`` block in
    ``pogs/pogs.py::get_loss_dict``.

    (1) repel the mean feature of randomly paired *different* masks up to
        ``margin``; (2) attract features inside a mask to that mask's mean;
    (3) push the "negative mask" (everything that is not a tracked object)
        toward the all-ones unit vector -- the paper notes this
        "is helpful in reducing group feature noise for the scene background".
    """
    if len(masks) <= 2:
        return None
    device = instance.device
    total = instance.shape[0]
    loss = torch.zeros((), device=device)
    count = 0
    idx = torch.randperm(len(masks) - 1)

    for i in range(len(masks) - 2):
        a, b = masks[idx[i]], masks[idx[i + 1]]
        if (a.sum() / total <= min_mask_screensize) or (b.sum() / total <= min_mask_screensize):
            continue
        loss = loss + F.relu(
            margin - torch.norm(instance[a].mean(0) - instance[b].mean(0), p=2, dim=-1)
        ).nansum()
        count += 1

    for i in range(len(masks) - 1):
        m = masks[i]
        if m.sum() / total <= min_mask_screensize:
            continue
        mi = masks[idx[i]]
        if mi.sum() == 0:
            continue
        mean = instance[mi].mean(0, keepdim=True)
        loss = loss + F.relu(torch.norm(instance[mi] - mean, p=2, dim=-1)).nanmean()
        count += 1

    neg = masks[-1]
    if neg.sum() > 0:
        ones = torch.ones(n_dims, device=device)
        ones = ones / ones.norm()
        loss = loss + 0.1 * F.relu(torch.norm(instance[neg] - ones, p=2, dim=-1)).nanmean()
        count += 1

    if count == 0:
        return None
    out = loss / count
    return out if torch.isfinite(out) and out != 0 else None


# --------------------------------------------------------------------------- #
# Language
# --------------------------------------------------------------------------- #


def clip_loss(rendered: torch.Tensor, gt: torch.Tensor, weight: float = 0.5) -> torch.Tensor:
    """``clip_loss_weight * huber(delta=1.25)`` summed over channels, mean over
    rays -- exactly the official formulation."""
    unreduced = weight * F.huber_loss(rendered, gt.to(rendered), delta=1.25, reduction="none")
    return unreduced.sum(dim=-1).nanmean()


# --------------------------------------------------------------------------- #
# Regularisation
# --------------------------------------------------------------------------- #


def scale_regularization(scales_log: torch.Tensor, max_gauss_ratio: float = 3.0) -> torch.Tensor:
    """PhysGaussian anisotropy penalty used by splatfacto/POGS."""
    s = torch.exp(scales_log)
    smax = s.amax(dim=-1)
    smin = s.amin(dim=-1).clamp(min=1e-8)
    return 0.1 * torch.mean(torch.clamp(smax / smin, min=max_gauss_ratio) - max_gauss_ratio)


# --------------------------------------------------------------------------- #
# Tracking loss (Sec. IV-D) -- used by RigidGroupOptimizer
# --------------------------------------------------------------------------- #


def tracking_feature_loss(real_dino: torch.Tensor, rendered_dino: torch.Tensor) -> torch.Tensor:
    """``(real - rendered).norm(dim=-1).nanmean()`` -- the DINO feature loss the
    optimiser minimises."""
    return (real_dino - rendered_dino).norm(dim=-1).nanmean()


def tracking_depth_loss(
    real_depth: torch.Tensor,
    rendered_depth: torch.Tensor,
    valids: torch.Tensor,
    ignore_threshold: float = 0.26,
) -> Optional[torch.Tensor]:
    """Official: clamp both to (1e-8, 1.0), squared error on valid pixels, then
    drop residuals above ``depth_ignore_threshold**2`` before averaging."""
    pred = torch.clamp(rendered_depth, min=1e-8, max=1.0)[valids]
    real = torch.clamp(real_depth, min=1e-8, max=1.0)[valids]
    if pred.numel() == 0:
        return None
    pix = (pred - real) ** 2
    pix = pix[pix < ignore_threshold**2]
    if pix.numel() == 0 or torch.isnan(pix.mean()):
        return None
    return pix.mean()


def gaussian_blur2d(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """kornia.filters.GaussianBlur2d equivalent; sigma follows the official
    ``0.3 * ((k - 1) * 0.5 - 1) + 0.8``. x: (H, W, C)."""
    k = kernel_size
    if k <= 1:
        return x
    s = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
    win = _gaussian_window(k, s, x.device, x.dtype)
    C = x.shape[-1]
    inp = x.permute(2, 0, 1).unsqueeze(0)
    w = win.expand(C, 1, k, k)
    return F.conv2d(inp, w, padding=k // 2, groups=C)[0].permute(1, 2, 0)


def erode(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """kornia.morphology.erosion with a square structuring element.
    mask: (H, W) bool."""
    if kernel_size <= 1:
        return mask
    m = mask.float().unsqueeze(0).unsqueeze(0)
    out = -F.max_pool2d(-m, kernel_size, stride=1, padding=kernel_size // 2)
    return out.squeeze(0).squeeze(0)[: mask.shape[0], : mask.shape[1]] > 0.5