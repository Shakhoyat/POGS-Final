import numpy as np

from metrics.recovery_report import format_report, recovery_report, rot_about

AX, TIP = np.array([1.0, 0, 0]), np.array([0.1, 0, 0])


def _trial(roll_deg, swing_deg=0.0, roll_pre=None):
    T = rot_about([0, 0, 1], swing_deg) @ rot_about(AX, roll_deg)
    t = {"T_pred": T, "T_gt": np.eye(4)}
    if roll_pre is not None:
        t["T_pred_pre"], t["T_gt_pre"] = rot_about(AX, roll_pre), np.eye(4)
    return t


def test_constant_roll_bias_passes_roll_free_only():
    rep = recovery_report([_trial(10.0) for _ in range(10)], [_trial(10.0, 20.0) for _ in range(10)], AX, TIP)
    assert rep["rules"]["roll_free"]["method"]["k"] == 10
    assert rep["rules"]["full"]["method"]["k"] == 0
    assert rep["rules"]["roll_free"]["static"]["k"] == 0
    assert abs(rep["error"]["roll_deg"]["mean"] - 10.0) < 1e-6
    assert rep["roll_share_of_rotation_error"] > 0.99


def test_relative_rule_is_passed_by_static_when_bias_aligns():
    # Static never moves: after a +15 deg perturbation its error is |pre - 15|; with a +10 pre-bias that is 5 < 10
    static = [_trial(10.0 - 15.0, roll_pre=10.0) for _ in range(5)]
    rep = recovery_report(static, static, AX, TIP, perturbation_sign=+1)
    assert rep["rules"]["relative"]["static"]["k"] == 5          # the rule credits a tracker that did nothing
    assert rep["relative_by_sign"]["pre-error same sign as perturbation"]["k"] == 5
    assert "Static" in format_report(rep)
