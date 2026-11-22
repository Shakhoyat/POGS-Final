"""Tests for model.pogs.kalman_info (plan rev. 2, task T4): frame conversion, ladder modes, and that an
anisotropic R makes the filter keep its prior along a weakly observed direction."""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.pogs.kalman_info import InfoKalmanPrior, body_cov_to_world, information_to_cov  # noqa: E402


def _rz(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])


def test_body_to_world_rotates_the_weak_axis():
    S = np.diag([1.0, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6])          # weak rotation about body x
    Rr, _ = body_cov_to_world(S, _rz(90))                     # body x -> world y
    w, V = np.linalg.eigh(Rr)
    assert abs(abs(V[:, -1] @ np.array([0, 1.0, 0])) - 1) < 1e-9


def test_information_to_cov_null_direction_gets_large_variance():
    H = np.diag([0.0, 10.0, 10.0, 100.0, 100.0, 100.0])
    S = information_to_cov(H, scale=1.0)
    assert S[0, 0] > 100 * S[1, 1] and np.all(np.isfinite(S))


def _prior(mode):
    p = InfoKalmanPrior(1, mode=mode, warm_start=False, fuse=True)
    p.initialize(torch.tensor([[0.0, 0, 0, 1, 0, 0, 0]]))
    return p


def _rot_meas(rotvec):
    from model.pogs.geometry import so3_exp
    q = so3_exp(torch.tensor(rotvec, dtype=torch.float64)).float()
    return torch.cat([torch.zeros(3), q]).unsqueeze(0)


def test_full_mode_keeps_prior_along_weak_axis_trace_does_not():
    S = np.diag([1.0, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6])          # roll about x unobserved
    meas = _rot_meas([0.1, 0.1, 0.0])                          # 0.1 rad error on x and y
    out = {}
    for mode in ("full", "trace"):
        p = _prior(mode)
        p.predict()
        p.set_measurement_cov(0, S, np.eye(3))
        q = p.update(meas)[0, 3:].double()
        from model.pogs.geometry import so3_log
        out[mode] = so3_log(q).numpy()
    assert abs(out["full"][0]) < 0.01 and out["full"][1] > 0.08   # ignores x, accepts y
    assert abs(out["trace"][0] - out["trace"][1]) < 1e-6          # isotropic: treats both alike


def test_oracle_mode_requires_gt_fn_and_sets_diag():
    p = _prior("oracle")
    p.set_oracle_cov(0, np.array([0.1, 0.0, 0.0]), np.array([0.01, 0.0, 0.0]))
    assert np.isclose(p.filters[0].R_r[0, 0], 0.01 + 1e-6)
    try:
        p.observe(None, None)
        raise AssertionError("oracle without oracle_err_fn must fail")
    except ValueError:
        pass


def test_fixed_mode_is_the_v2_filter():
    p = _prior("fixed")
    R0 = p.filters[0].R_r.copy()
    p.observe(None, None)                                      # no-op
    assert np.array_equal(p.filters[0].R_r, R0)


def test_step_damping_term_small_angle_and_translation():
    from model.pogs.geometry import so3_exp
    from model.pogs.tracking import step_damping_term
    pd0 = torch.tensor([[0.0, 0, 0, 1, 0, 0, 0]])
    th = 0.05
    pd = torch.cat([torch.tensor([[0.01, 0.0, 0.0]]), so3_exp(torch.tensor([[0.0, 0.0, th]]))], dim=1)
    v = float(step_damping_term(pd, pd0, lam_rot=1.0, lam_trans=0.0))
    assert abs(v - th ** 2) < 1e-4                       # ~ theta^2
    assert abs(float(step_damping_term(pd, pd0, 0.0, 2.0)) - 2 * 0.01 ** 2) < 1e-9
    assert float(step_damping_term(pd0, pd0, 5.0, 5.0)) == 0.0


def test_damping_defaults_are_release_behaviour():
    from model.pogs.config import TrackingConfig
    c = TrackingConfig()
    assert c.step_damping_rot == 0.0 and c.step_damping_trans == 0.0
