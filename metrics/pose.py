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


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def swing_twist_deg(R_err: np.ndarray, axis: np.ndarray) -> tuple:
    """Split a rotation into twist about ``axis`` (signed, deg) and swing of the axis (deg).

    R_err = R_swing @ R_twist, where R_twist fixes ``axis`` and R_swing is the smallest rotation
    taking ``axis`` to R_err @ axis. The swing angle equals the angle between ``axis`` and
    R_err @ axis, so it is independent of the twist.
    """
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = R_err @ a
    c = float(np.clip(a @ b, -1.0, 1.0))
    swing = math.acos(c)
    n = np.cross(a, b)
    R_swing = np.eye(3) if np.linalg.norm(n) < 1e-12 else _rodrigues(n, swing)
    R_twist = R_swing.T @ R_err
    twist = math.atan2(float(a @ np.array([R_twist[2, 1] - R_twist[1, 2],
                                            R_twist[0, 2] - R_twist[2, 0],
                                            R_twist[1, 0] - R_twist[0, 1]])),
                       float(np.trace(R_twist) - 1.0))
    return math.degrees(twist), math.degrees(swing)


def roll_swing_error_deg(T_pred: np.ndarray, T_gt: np.ndarray, axis_local: np.ndarray) -> tuple:
    """Signed roll about ``axis_local`` and swing of that axis, for the error R_gt^T R_pred.

    Both poses map the object's local frame to the world, so ``axis_local`` (e.g. the drill-bit
    axis) is expressed in the local frame. Roll is what POGS's servoing rule leaves free (Sec. V-B:
    'We leave tool rotation about its target axis (i.e. drill bit axis) unconstrained').
    """
    R_err = T_gt[:3, :3].T @ T_pred[:3, :3]
    return swing_twist_deg(R_err, axis_local)


def tool_success_roll_free(tip_err_cm: float, swing_deg: float,
                           pos_cm: float = SUCCESS_POS_CM, rot_deg: float = SUCCESS_ROT_DEG) -> bool:
    """Paper-faithful tool rule: tip within 3 cm and tool axis within 5 deg, roll about it free."""
    return bool(tip_err_cm < pos_cm and swing_deg < rot_deg)
