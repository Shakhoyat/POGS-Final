"""
Real-time protocol: does a faster tracker win once the camera does not wait?

Offline benchmarks track every frame however long it takes. Here the camera
streams at 30 Hz (``REALTIME_FPS``); the tracker always takes the newest frame
once it is free, skipping the rest, and is scored at EVERY camera frame on the
pose it had published by then (``model.pogs.evaluate.track_realtime``). Compute
time is measured, so a slow tracker pays twice: larger motion between the
frames it sees, and staler output. Run on an otherwise idle GPU.

Methods
-------
POGS                       base POGS, 25 optimiser steps per frame
POGS, 10 iters             equally cheap baseline for the 10-step KF
POGS + KF + check          v2 (loss-checked Kalman warm start)
POGS + KF + check, 10 iters
Every KF method also yields "<name> + latency comp.": its last estimate moved
forward to the current time with the filter's velocity (no extra compute).
"""

from __future__ import annotations

from experiments.pipeline import MethodSpec
from experiments.v1_base_pogs.config import get_v1_config
from experiments.v2_kalman_pogs.config import _kf

RESULTS_DIR = "data_results/realtime"


def get_realtime_config(device: str = "cuda:0"):
    return get_v1_config(device)


def realtime_methods():
    return [
        MethodSpec("POGS", "POGS"),
        MethodSpec("POGS, 10 iters", "10 iters"),
        MethodSpec("POGS + KF + check", "POGS", _kf(True)),
        MethodSpec("POGS + KF + check, 10 iters", "10 iters", _kf(True)),
    ]
