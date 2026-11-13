"""
Gaussian-splat metrics (evaluation only).

* ``gaussian_covariances`` / ``rigid_transform_gaussians``: 3D covariances from
  POGS parameters and the rigid motion POGS applies to a group.
* ``kl_gaussians``: closed-form KL between paired 3D Gaussians, and
  ``weighted_kl`` for the opacity-weighted splat divergence of PLAN Phase 2:

      D_KL(x || y) = sum_i a_i KL(N(mu_i^x, S_i^x) || N(mu_i^y, S_i^y)) / sum_i a_i

  IMPORTANT (PLAN §2.1): POGS moves each group rigidly, so with matched indices
  this divergence is a deterministic function of the residual rigid transform.
  It is a covariance-aware view of pose error, not independent evidence.
* ``mask_metrics``: IoU / leakage / completeness of a rendered object alpha
  against a ground-truth object mask (PLAN Phase 1).
* ``image_psnr`` / ``image_ssim``: torchmetrics wrappers.

Conventions follow ``model.pogs``: quaternions are wxyz, ``scales`` are linear
(i.e. ``exp`` of the stored log-scales), means in metres.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from model.pogs.geometry import quat_to_rotmat

COV_EPS = 1e-8
"""Added to every covariance diagonal (m^2, i.e. (0.1 mm)^2). Trained splats
contain near-degenerate (flat) Gaussians whose covariance is numerically
singular; without a floor their KL is dominated by round-off. Stated in the
report wherever KL values appear."""


# --------------------------------------------------------------------------- #
# Gaussian parameters
# --------------------------------------------------------------------------- #


def gaussian_covariances(quats_wxyz: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """(N,4) wxyz + (N,3) linear scales -> (N,3,3) covariances R diag(s^2) R^T."""
    R = quat_to_rotmat(quats_wxyz)
    S2 = torch.diag_embed(scales * scales)
    return R @ S2 @ R.transpose(-1, -2)


def rigid_transform_gaussians(means: torch.Tensor, covs: torch.Tensor, T: torch.Tensor):
    """Apply a 4x4 rigid transform to Gaussians: mu' = R mu + t, S' = R S R^T."""
    T = T.to(means)
    R, t = T[:3, :3], T[:3, 3]
    return means @ R.T + t, R @ covs @ R.T


# --------------------------------------------------------------------------- #
# KL divergence
# --------------------------------------------------------------------------- #


def kl_gaussians(mu0: torch.Tensor, cov0: torch.Tensor, mu1: torch.Tensor, cov1: torch.Tensor,
                 eps: float = COV_EPS) -> torch.Tensor:
    """Per-pair KL( N(mu0, cov0) || N(mu1, cov1) ) in nats, shape (N,).

    KL = 1/2 [ tr(S1^-1 S0) + (mu1-mu0)^T S1^-1 (mu1-mu0) - k + ln det S1 - ln det S0 ]
    Computed in float64 with Cholesky factors.
    """
    k = mu0.shape[-1]
    I = torch.eye(k, dtype=torch.float64, device=mu0.device)
    S0 = cov0.to(torch.float64) + eps * I
    S1 = cov1.to(torch.float64) + eps * I
    L0 = torch.linalg.cholesky(S0)
    L1 = torch.linalg.cholesky(S1)
    # tr(S1^-1 S0) = || L1^-1 L0 ||_F^2
    A = torch.linalg.solve_triangular(L1, L0, upper=False)
    trace = (A * A).sum(dim=(-1, -2))
    d = (mu1 - mu0).to(torch.float64).unsqueeze(-1)
    z = torch.linalg.solve_triangular(L1, d, upper=False).squeeze(-1)
    maha = (z * z).sum(-1)
    logdet1 = 2.0 * torch.log(torch.diagonal(L1, dim1=-2, dim2=-1)).sum(-1)
    logdet0 = 2.0 * torch.log(torch.diagonal(L0, dim1=-2, dim2=-1)).sum(-1)
    return 0.5 * (trace + maha - k + logdet1 - logdet0)


def weighted_kl(mu_x: torch.Tensor, cov_x: torch.Tensor, mu_y: torch.Tensor, cov_y: torch.Tensor,
                weights: Optional[torch.Tensor] = None, symmetric: bool = False,
                eps: float = COV_EPS) -> Dict[str, float]:
    """Opacity-weighted splat divergence between matched Gaussians.

    Returns the weighted mean (the formula in the module docstring), plus the
    unweighted median and 95th percentile, because per-Gaussian KL is heavy-tailed
    (flat Gaussians react strongly to motion along their thin axis).
    ``symmetric=True`` uses 1/2 [KL(x||y) + KL(y||x)].
    """
    kl = kl_gaussians(mu_x, cov_x, mu_y, cov_y, eps)
    if symmetric:
        kl = 0.5 * (kl + kl_gaussians(mu_y, cov_y, mu_x, cov_x, eps))
    w = torch.ones_like(kl) if weights is None else weights.to(kl).reshape(-1)
    return {
        "kl_weighted_mean": float((w * kl).sum() / w.sum().clamp_min(1e-12)),
        "kl_median": float(kl.median()),
        "kl_p95": float(torch.quantile(kl, 0.95)),
        "n": int(kl.numel()),
    }


# --------------------------------------------------------------------------- #
# Rendered-mask and image metrics
# --------------------------------------------------------------------------- #


def mask_metrics(alpha: torch.Tensor, gt_mask: torch.Tensor, thresh: float = 0.5) -> Dict[str, float]:
    """Compare a rendered object-only alpha (H,W in [0,1]) with a GT mask (H,W bool).

    iou           |A & G| / |A | G| with A = alpha > thresh
    leakage       share of rendered opacity outside the GT mask (0 = none)
    completeness  share of GT pixels covered (A & G) / G
    """
    a = alpha.float()
    g = gt_mask.bool()
    A = a > thresh
    inter = (A & g).sum().item()
    union = (A | g).sum().item()
    total_alpha = a.sum().item()
    return {
        "iou": inter / union if union else float("nan"),
        "leakage": (a[~g].sum().item() / total_alpha) if total_alpha > 0 else float("nan"),
        "completeness": inter / g.sum().item() if g.any() else float("nan"),
    }


def image_psnr(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """PSNR (dB) of (H,W,3) images in [0,1], optionally inside a boolean mask."""
    from torchmetrics.functional.image import peak_signal_noise_ratio

    if mask is not None:
        m = mask.bool()
        return float(peak_signal_noise_ratio(pred[m], target[m], data_range=1.0))
    return float(peak_signal_noise_ratio(pred, target, data_range=1.0))


def image_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """SSIM of (H,W,3) images in [0,1]."""
    from torchmetrics.functional.image import structural_similarity_index_measure

    p = pred.permute(2, 0, 1).unsqueeze(0).float()
    t = target.permute(2, 0, 1).unsqueeze(0).float()
    return float(structural_similarity_index_measure(p, t, data_range=1.0))


# --------------------------------------------------------------------------- #
# Ground-truth-free consistency (PLAN 2.4)
# --------------------------------------------------------------------------- #


def kl_render(observed: torch.Tensor, rendered: torch.Tensor, mask: torch.Tensor,
              bins: int = 8, alpha: float = 0.01, min_pixels: int = 50) -> float:
    """KL( P_obs || P_render ) between colour histograms inside ``mask``.

    ``observed`` / ``rendered``: (H,W,3) in [0,1]; ``mask``: (H,W) bool, normally the
    rendered object region (accumulation > 0.5), so no ground truth is needed.
    Joint ``bins^3`` RGB histogram with add-``alpha`` smoothing (0.01 per bin, ~5
    pseudo-pixels in total, so the prior does not swamp small objects). Near 0
    when the rendered object explains the pixels it covers; rises when the
    estimate slides off the object. Returns NaN if fewer than ``min_pixels``.
    """
    m = mask.bool()
    if int(m.sum()) < min_pixels:
        return float("nan")

    def hist(img: torch.Tensor) -> torch.Tensor:
        q = (img[m].clamp(0, 1 - 1e-6) * bins).long()
        idx = (q[:, 0] * bins + q[:, 1]) * bins + q[:, 2]
        h = torch.bincount(idx, minlength=bins ** 3).double() + alpha
        return h / h.sum()

    p, q = hist(observed), hist(rendered)
    return float((p * (p.log() - q.log())).sum())
