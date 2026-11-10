"""
v3 configuration: other target objects. Same scene, checkpoint and grouping as
v1/v2; v3 reports the benchmark's ``object`` (mug / mustard bottle / cereal box)
and ``multi-object`` factors for base POGS and the best v2 variant. Runs come
from the shared cache, so anything v1/v2 already tracked is not repeated.
"""

from __future__ import annotations

from experiments.pipeline import MethodSpec
from experiments.v1_base_pogs.config import get_v1_config
from experiments.v2_kalman_pogs.config import _kf

RESULTS_DIR = "data_results/v3_objects"
FACTORS_V3 = ("object", "multi-object")


def get_v3_config(device: str = "cuda:0"):
    return get_v1_config(device)


def v3_methods():
    return [
        MethodSpec("POGS", "POGS"),
        MethodSpec("POGS (release: no per-frame depth)", "POGS (release: no per-frame depth)"),
        MethodSpec("POGS + KF", "POGS", _kf()),
    ]
