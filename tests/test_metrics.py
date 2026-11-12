"""Tests for metrics/ (PLAN Phase 0.3, 2.3).

    .venv/bin/python -m pytest -q tests/test_metrics.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics.gaussian import (  # noqa: E402
    gaussian_covariances, kl_gaussians, mask_metrics, rigid_transform_gaussians, weighted_kl,
)
from metrics.pose import pose_success, rotation_error_deg, translation_error_cm  # noqa: E402
from metrics.stats import mean_std, target_verdict, wilson_interval  # noqa: E402


def _random_gaussians(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    means = torch.randn(n, 3, generator=g, dtype=torch.float64) * 0.05
    q = torch.randn(n, 4, generator=g, dtype=torch.float64)
    q = q / q.norm(dim=-1, keepdim=True)
    scales = torch.rand(n, 3, generator=g, dtype=torch.float64) * 0.01 + 1e-3
    return means, gaussian_covariances(q, scales)


def _rot_z(deg):
    a = math.radians(deg)
    T = torch.eye(4, dtype=torch.float64)
    T[:2, :2] = torch.tensor([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return T


def test_kl_identity_is_zero():
    mu, cov = _random_gaussians()
    kl = kl_gaussians(mu, cov, mu, cov)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-9)
    assert weighted_kl(mu, cov, mu, cov)["kl_weighted_mean"] == pytest.approx(0.0, abs=1e-9)


def test_kl_translation_closed_form():
    # Same covariance, shifted mean: KL = 1/2 d^T S^-1 d (with the eps floor in S).
    mu, cov = _random_gaussians()
    d = torch.tensor([0.003, -0.001, 0.002], dtype=torch.float64)
    kl = kl_gaussians(mu, cov, mu + d, cov, eps=0.0)
    expected = 0.5 * torch.einsum("i,nij,j->n", d, torch.linalg.inv(cov), d)
    assert torch.allclose(kl, expected, rtol=1e-8)


def test_kl_isotropic_closed_form():
    # 1D sanity in 3D: N(0, s0^2 I) vs N(0, s1^2 I): 3/2 [r - 1 - ln r], r = s0^2/s1^2.
    mu = torch.zeros(1, 3, dtype=torch.float64)
    s0, s1 = 0.02, 0.03
    cov0 = (s0 ** 2) * torch.eye(3, dtype=torch.float64)[None]
    cov1 = (s1 ** 2) * torch.eye(3, dtype=torch.float64)[None]
    r = s0 ** 2 / s1 ** 2
    assert kl_gaussians(mu, cov0, mu, cov1, eps=0.0).item() == pytest.approx(1.5 * (r - 1 - math.log(r)))


def test_kl_nonnegative_and_grows_with_rotation():
    mu, cov = _random_gaussians()
    prev = 0.0
    for deg in (5, 10, 20, 30):
        m2, c2 = rigid_transform_gaussians(mu, cov, _rot_z(deg))
        kl = kl_gaussians(mu, cov, m2, c2)
        assert (kl >= -1e-9).all()
        cur = weighted_kl(mu, cov, m2, c2)["kl_weighted_mean"]
        assert cur > prev
        prev = cur


def test_kl_round_trip_returns_to_zero():
    # O3 protocol: rotate +30 in 5-degree steps, then back; exact return => KL(x||y) = 0.
    mu, cov = _random_gaussians()
    m, c = mu, cov
    for _ in range(6):
        m, c = rigid_transform_gaussians(m, c, _rot_z(5))
    for _ in range(6):
        m, c = rigid_transform_gaussians(m, c, _rot_z(-5))
    assert weighted_kl(mu, cov, m, c, symmetric=True)["kl_weighted_mean"] == pytest.approx(0.0, abs=1e-6)


def test_mask_metrics():
    gt = torch.zeros(10, 10, dtype=torch.bool)
    gt[2:6, 2:6] = True  # 16 px
    alpha = torch.zeros(10, 10)
    alpha[2:6, 2:6] = 1.0
    m = mask_metrics(alpha, gt)
    assert m == {"iou": 1.0, "leakage": 0.0, "completeness": 1.0}
    alpha[0, 0] = 1.0  # one leaked pixel
    m = mask_metrics(alpha, gt)
    assert m["iou"] == pytest.approx(16 / 17)
    assert m["leakage"] == pytest.approx(1 / 17)
    assert m["completeness"] == 1.0


def test_pose_metrics_and_success():
    T = np.eye(4)
    T2 = _rot_z(4).numpy()
    T2[:3, 3] = [0.02, 0.0, 0.0]
    assert translation_error_cm(T2, T) == pytest.approx(2.0)
    assert rotation_error_deg(T2, T) == pytest.approx(4.0)
    assert pose_success(2.0, 4.0) and not pose_success(3.5, 1.0)


def test_wilson_and_verdict():
    lo, hi = wilson_interval(48, 50)  # paper Table II, 15 deg
    assert 0.86 < lo < 0.87 and 0.98 < hi < 0.99
    assert wilson_interval(0, 10)[0] == 0.0
    assert target_verdict(48, 50, 0.80) == "met"
    assert target_verdict(40, 50, 0.80) == "inconclusive (CI spans target)"
    assert target_verdict(20, 50, 0.80) == "not met"
    assert mean_std([1.0, 3.0, float("nan")]) == (2.0, pytest.approx(math.sqrt(2.0)))
