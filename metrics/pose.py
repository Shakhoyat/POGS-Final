"""
Pose metrics (evaluation only; inputs are 4x4 homogeneous transforms in metres).

Moved here from ``model.pogs.evaluate`` (which re-exports them) so that the
repository follows the Model / Data / Metrics layout asked for in the brief.
The maths is unchanged.
"""

from __future__ import annotations

import math

import numpy as np

from model.pogs.geometry import geodesic_rotation_error

SUCCESS_POS_CM = 3.0
SUCCESS_ROT_DEG = 5.0
"""Sec. V-B: 'locate the drill tip within 3cm and 5 deg of the target'."""

ADDS_SUCCESS_FRAC = 0.10
"""Standard ADD(-S) criterion: ADD-S below 10% of the object diameter
(Hinterstoisser et al. 2012; PoseCNN). Symmetry-aware, unlike 3 cm / 5 deg."""


def translation_error_cm(T_pred: np.ndarray, T_gt: np.ndarray) -> float:
    return float(np.linalg.norm(T_pred[:3, 3] - T_gt[:3, 3]) * 100.0)


def rotation_error_deg(T_pred: np.ndarray, T_gt: np.ndarray) -> float:
    return geodesic_rotation_error(T_pred[:3, :3], T_gt[:3, :3])


def yaw_deg(R: np.ndarray) -> float:
    return math.degrees(math.atan2(R[1, 0], R[0, 0]))


def yaw_error_deg(T_pred: np.ndarray, T_gt: np.ndarray) -> float:
    d = yaw_deg(T_pred[:3, :3]) - yaw_deg(T_gt[:3, :3])
    return abs((d + 180.0) % 360.0 - 180.0)


def add_metric(points: np.ndarray, T_pred: np.ndarray, T_gt: np.ndarray) -> float:
    """Average Distance of model points (Hinterstoisser et al.), metres.
    ``points`` are object-local (relative to the scan-time centroid)."""
    a = points @ T_pred[:3, :3].T + T_pred[:3, 3]
    b = points @ T_gt[:3, :3].T + T_gt[:3, 3]
    return float(np.linalg.norm(a - b, axis=1).mean())


def adds_metric(points: np.ndarray, T_pred: np.ndarray, T_gt: np.ndarray, max_pts: int = 1500, chunk: int = 256) -> float:
    """ADD-S: symmetry-aware nearest-neighbour variant (chunked, float64)."""
    if len(points) > max_pts:
        idx = np.random.default_rng(0).choice(len(points), max_pts, replace=False)
        points = points[idx]
    a = (points @ T_pred[:3, :3].T + T_pred[:3, 3]).astype(np.float64)
    b = (points @ T_gt[:3, :3].T + T_gt[:3, 3]).astype(np.float64)
    b2 = (b * b).sum(axis=1)
    total = 0.0
    for i in range(0, len(a), chunk):
        block = a[i : i + chunk]
        d2 = (block * block).sum(axis=1)[:, None] - 2.0 * (block @ b.T) + b2[None, :]
        total += float(np.sqrt(np.maximum(d2.min(axis=1), 0.0)).sum())
    return total / max(len(a), 1)


def object_diameter(points: np.ndarray) -> float:
    """Robust object diameter (m): diagonal of the 1-99 percentile box of the
    model points, so a few floater Gaussians do not inflate it."""
    lo, hi = np.percentile(points, 1, axis=0), np.percentile(points, 99, axis=0)
    return float(np.linalg.norm(hi - lo))


def adds_success_rate(adds_m: np.ndarray, diameter_m: float) -> float:
    v = adds_m[~np.isnan(adds_m)]
    return float((v < ADDS_SUCCESS_FRAC * diameter_m).mean()) if v.size else float("nan")


def pose_success(trans_err_cm: float, rot_err_deg: float,
                 pos_cm: float = SUCCESS_POS_CM, rot_deg: float = SUCCESS_ROT_DEG) -> bool:
    """The paper's servoing criterion (Sec. V-B): within 3 cm AND 5 deg."""
    return bool(trans_err_cm < pos_cm and rot_err_deg < rot_deg)
