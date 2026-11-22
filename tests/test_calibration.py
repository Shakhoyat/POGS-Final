"""Tests for metrics.calibration (plan rev. 2, task T6)."""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics.calibration import (  # noqa: E402
    UNIFORM_CIRCLE_NLL, circular_nll, coverage, event_recovered, gaussian_nll, hierarchical_bootstrap,
    kaplan_meier, nees, nees_consistency, reliability_curve,
)


def _samples(scale_true, scale_pred, n=4000, d=6, seed=0):
    rng = np.random.default_rng(seed)
    err = rng.normal(size=(n, d)) * scale_true
    cov = np.broadcast_to(np.eye(d) * scale_pred ** 2, (n, d, d))
    return err, cov


def test_consistent_gaussian_is_judged_consistent():
    err, cov = _samples(0.01, 0.01)
    v = nees(err, cov)
    assert nees_consistency(v, 6)["verdict"] == "consistent"
    assert abs(coverage(v, 6, 0.9) - 0.9) < 0.02


def test_overconfident_belief_is_flagged():
    err, cov = _samples(0.03, 0.01)               # true error 3x the claimed sigma
    v = nees(err, cov)
    assert nees_consistency(v, 6)["verdict"] == "overconfident"
    assert coverage(v, 6, 0.9) < 0.3
    # NLL prefers the calibrated belief
    _, cov_ok = _samples(0.03, 0.03)
    assert gaussian_nll(err, cov_ok).mean() < gaussian_nll(err, cov).mean()


def test_reliability_curve_tracks_sigma_when_calibrated():
    rng = np.random.default_rng(1)
    s = rng.uniform(0.5, 5.0, 20000)
    e = np.abs(rng.normal(size=s.size) * s)
    rc = reliability_curve(s, e, 8)
    assert np.allclose(rc["rms_err"] / rc["pred_sigma"], 1.0, atol=0.1)


def test_circular_nll_uniform_and_peaked():
    g = np.linspace(-math.pi, math.pi, 360, endpoint=False)
    assert abs(circular_nll(g, np.zeros_like(g), 0.3) - UNIFORM_CIRCLE_NLL) < 1e-9
    peaked = -0.5 * (np.angle(np.exp(1j * (g - 0.3))) / 0.1) ** 2
    assert circular_nll(g, peaked, 0.3) < UNIFORM_CIRCLE_NLL - 1.0      # truth at the mode: much better
    assert circular_nll(g, peaked, 0.3 + math.pi) > UNIFORM_CIRCLE_NLL  # truth at the antipode: worse
    two = np.logaddexp(peaked, -0.5 * (np.angle(np.exp(1j * (g - 0.3 - math.pi))) / 0.1) ** 2)
    assert circular_nll(g, two, 0.3 + math.pi) < UNIFORM_CIRCLE_NLL     # bimodal keeps the flip


def test_event_metric_and_survival():
    s = [False] * 5 + [False, False, True, True] + [False] * 10
    r = event_recovered(s, event_frame=4, window=10)
    assert r == {"recovered": True, "delay": 3}
    assert event_recovered([False] * 20, 4)["recovered"] is False
    km = kaplan_meier([1, 2, None, None], horizon=10)
    assert km["survival"][0] == 1.0 and abs(km["survival"][1] - 0.75) < 1e-12
    assert abs(km["survival"][2] - 0.5) < 1e-12 and km["survival"][-1] == 0.5


def test_hierarchical_bootstrap_widens_with_clustering():
    """Trials sharing a splat seed are correlated; the hierarchical interval must be wider than an
    i.i.d. bootstrap of the same values."""
    rng = np.random.default_rng(2)
    seeds = np.repeat(np.arange(3), 50)
    vals = rng.normal(size=3)[seeds] + 0.1 * rng.normal(size=seeds.size)
    h = hierarchical_bootstrap(vals, [seeds], n_boot=1000)
    iid = hierarchical_bootstrap(vals, [], n_boot=1000)
    assert (h["hi"] - h["lo"]) > 3 * (iid["hi"] - iid["lo"])
