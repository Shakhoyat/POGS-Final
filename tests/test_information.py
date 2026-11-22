"""Tests for model.pogs.information (plan rev. 2, task T2): perturbation model, finite-difference
Gauss-Newton matrix, Hessian of a scalar loss, and the null direction of a symmetric object."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.pogs.geometry import posevec_to_matrix, quat_normalize, so3_exp  # noqa: E402
from model.pogs.information import (  # noqa: E402
    gauss_newton_info, loss_hessian, observability_summary, perturb_part_delta, perturbed_deltas,
)


def _rand_pd(seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(3, generator=g, dtype=torch.float64) * 0.1
    q = quat_normalize(torch.randn(4, generator=g, dtype=torch.float64))
    return torch.cat([t, q])


def _exp_se3_right(delta):
    """[Exp(omega) | v] as a 4x4 (right factor of the perturbation model)."""
    E = torch.eye(4, dtype=torch.float64)
    E[:3, :3] = posevec_to_matrix(torch.cat([torch.zeros(3, dtype=torch.float64), so3_exp(delta[:3])]).unsqueeze(0))[0, :3, :3]
    E[:3, 3] = delta[3:]
    return E


def test_perturbation_is_right_multiplication():
    pd = _rand_pd(1)
    delta = torch.tensor([0.05, -0.02, 0.03, 0.01, 0.0, -0.02], dtype=torch.float64)
    T = posevec_to_matrix(pd.unsqueeze(0))[0]
    T_new = posevec_to_matrix(perturb_part_delta(pd, delta).unsqueeze(0))[0]
    assert torch.allclose(T_new, T @ _exp_se3_right(delta), atol=1e-10)


def test_perturbed_deltas_only_touches_group_k():
    pds = torch.stack([_rand_pd(0), _rand_pd(1), _rand_pd(2)])
    out = perturbed_deltas(pds, 1, torch.tensor([0.1, 0, 0, 0, 0, 0], dtype=torch.float64))
    assert torch.equal(out[0], pds[0]) and torch.equal(out[2], pds[2]) and not torch.equal(out[1], pds[1])


def test_gauss_newton_exact_for_linear_residual():
    rng = np.random.default_rng(0)
    A = torch.tensor(rng.normal(size=(40, 6)))
    b = torch.tensor(rng.normal(size=40))
    info = gauss_newton_info(lambda d: A @ d + b)
    assert np.allclose(info["H"], (A.T @ A).numpy(), rtol=1e-8, atol=1e-8)
    assert abs(info["sigma2"] - float(b @ b) / 34) < 1e-10


def test_loss_hessian_matches_quadratic():
    rng = np.random.default_rng(1)
    A = torch.tensor(rng.normal(size=(30, 6)))
    b = torch.tensor(rng.normal(size=30))
    out = loss_hessian(lambda d: 0.5 * ((A @ d + b) ** 2).sum())
    assert np.allclose(out["H"], (A.T @ A).numpy(), rtol=1e-6, atol=1e-6)
    assert np.allclose(out["grad"], (A.T @ b).numpy(), atol=1e-8)
    assert out["asym"] < 1e-6


def _cylinder_points(n=400, radius=0.03, half_len=0.08, seed=0):
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi, n)
    x = rng.uniform(-half_len, half_len, n)
    return torch.tensor(np.stack([x, radius * np.cos(ang), radius * np.sin(ang)], 1))


def test_symmetric_object_has_roll_null_direction():
    """A textureless cylinder about object +x: the residual (radius and axial position of every
    surface point, as an image of the geometry would constrain them) is invariant to roll about x,
    so roll must be the (only) null direction of H."""
    P = _cylinder_points()
    r0 = 0.03
    pd = _rand_pd(3)
    pd[3:] = torch.tensor([1.0, 0, 0, 0], dtype=torch.float64)   # identity rotation, like the scan pose

    def res(delta):
        T = posevec_to_matrix(perturb_part_delta(pd, delta).unsqueeze(0))[0]
        T0 = posevec_to_matrix(pd.unsqueeze(0))[0]
        X = (torch.linalg.inv(T0) @ T)[:3, :3] @ P.T + (torch.linalg.inv(T0) @ T)[:3, 3:4]   # in the scan frame
        rad = torch.sqrt(X[1] ** 2 + X[2] ** 2) - r0
        axial = X[0] - P[:, 0]
        return torch.cat([rad, axial])

    info = gauss_newton_info(res)
    s = observability_summary(info["H"], radius_m=0.08, axis_local=np.array([1.0, 0.0, 0.0]))
    assert s["rel_min"] < 1e-8                      # one direction carries no information
    assert s["weak_rot_axis_cos"] > 0.999           # ... and it is roll about the cylinder axis
    assert s["roll_info_rel"] < 1e-8
    assert np.sort(s["eig"])[1] / s["eig"][-1] > 1e-3   # only one null direction


def test_symmetry_breaking_mark_restores_roll_information():
    """Add a small asymmetric 'decal' (points whose tangential position is observed): roll becomes
    observable, but weakly, as the theory section predicts (lambda_min grows with the mark)."""
    P = _cylinder_points()
    pd = _rand_pd(4)
    pd[3:] = torch.tensor([1.0, 0, 0, 0], dtype=torch.float64)

    def make_res(n_mark):
        M = P[:n_mark]

        def res(delta):
            T = posevec_to_matrix(perturb_part_delta(pd, delta).unsqueeze(0))[0]
            T0 = posevec_to_matrix(pd.unsqueeze(0))[0]
            D = torch.linalg.inv(T0) @ T
            X = D[:3, :3] @ P.T + D[:3, 3:4]
            Y = D[:3, :3] @ M.T + D[:3, 3:4]
            return torch.cat([torch.sqrt(X[1] ** 2 + X[2] ** 2) - 0.03, X[0] - P[:, 0],
                              (Y - M.T).reshape(-1)])
        return res

    rel = [observability_summary(gauss_newton_info(make_res(n))["H"], 0.08,
                                 np.array([1.0, 0, 0]))["roll_info_rel"] for n in (4, 40)]
    assert 0 < rel[0] < rel[1]


def test_pose_to_part_delta_round_trip():
    from model.pogs.information import pose_to_part_delta
    pd = _rand_pd(5).float()
    c0 = np.array([0.1, -0.2, 0.05])
    T = posevec_to_matrix(pd.unsqueeze(0).double())[0].numpy()
    T[:3, 3] += c0                                   # part2world translation = c0 + t
    back = pose_to_part_delta(T, c0)
    assert torch.allclose(back[:3], pd[:3], atol=1e-6)
    assert torch.allclose(back[3:] * torch.sign(back[3] * pd[3]), pd[3:], atol=1e-5)


def test_regression_hessian_quadratic_and_smooths_noise():
    from model.pogs.information import loss_hessian_regression
    rng = np.random.default_rng(3)
    A = torch.tensor(rng.normal(size=(30, 6)))
    b = torch.tensor(rng.normal(size=30))
    Hq = (A.T @ A).numpy()
    out = loss_hessian_regression(lambda d: 0.5 * ((A @ d + b) ** 2).sum(), 0.02, 0.003, n=12)
    assert np.allclose(out["H"], Hq, rtol=1e-6, atol=1e-6) and out["fit_resid"] < 1e-8
    # a small high-frequency ripple (sub-pixel non-smoothness) barely moves the fitted curvature
    rip = lambda d: 0.5 * ((A @ d + b) ** 2).sum() + 1e-4 * torch.sin(4000.0 * d).sum()
    out2 = loss_hessian_regression(rip, 0.02, 0.003, n=24)
    assert np.abs(out2["H"] - Hq).max() / np.abs(Hq).max() < 0.25
