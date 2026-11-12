"""
v2 configuration: identical to v1 (same checkpoint, same grouping, same
tracking settings); the ONLY change is a per-object constant-velocity SE(3)
Kalman filter (``model.pogs.kalman``) plugged into the shared tracking loop.

Claim under test: the motion prior lets POGS keep its accuracy with fewer
optimiser steps per frame (speed) and on faster motion (``speed`` factor).

Methods (all filters: velocity damping tau = 1 s, |v| <= 0.5 m/s, |w| <= 3 rad/s)
-------
POGS                              v1 base POGS (shared run cache: not re-run)
POGS + KF                         each frame starts from the filter's prediction
POGS + KF + check                 ... only where it lowers the tracking loss
N-iteration sweep                 the same at 10 and 5 optimiser steps per frame,
                                  next to plain POGS at the same budget, so a
                                  speed claim is made against an equally cheap
                                  baseline

History: an undamped filter diverged by metres once tracking slipped (the
optimiser's output is not independent of a warm-start prediction); its runs
were discarded. The "fuse" variant was dropped for the same double-counting.
"""

from __future__ import annotations

from model.pogs.kalman import KalmanConfig, KalmanMotionPrior
from experiments.pipeline import MethodSpec
from experiments.v1_base_pogs.config import get_v1_config

RESULTS_DIR = "data_results/v2_kalman"


def get_v2_config(device: str = "cuda:0"):
    return get_v1_config(device)


def _kf(check_loss: bool = False):
    """Factory ``(K, dt) -> prior``; ``dt`` follows the case's frame rate."""
    return lambda K, dt: KalmanMotionPrior(K, KalmanConfig(dt=dt), warm_start=True, check_loss=check_loss)


def v2_methods():
    methods = [
        MethodSpec("POGS", "POGS"),
        MethodSpec("POGS + KF", "POGS", _kf()),
        MethodSpec("POGS + KF + check", "POGS", _kf(True)),
    ]
    for n in (10, 5):
        methods.append(MethodSpec(f"POGS, {n} iters", f"{n} iters"))
        methods.append(MethodSpec(f"POGS + KF, {n} iters", f"{n} iters", _kf()))
        methods.append(MethodSpec(f"POGS + KF + check, {n} iters", f"{n} iters", _kf(True)))
    return methods
