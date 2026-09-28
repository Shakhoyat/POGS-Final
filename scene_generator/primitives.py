"""
Analytic 3D primitives and closed-form ray intersection geometry.
Provides Box, Ellipsoid, and Cylinder primitives ray-cast in closed form
so depth, normals, and instance masks are exact and hole-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import torch

BOX, ELLIPSOID, CYLINDER = 0, 1, 2


@dataclass
class Primitive:
    kind: int
    scale: Tuple[float, float, float]
    # local pose inside the *object* frame
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rot_euler: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    color: Tuple[float, float, float] = (0.6, 0.6, 0.6)
    texture_freq: float = 22.0
    texture_amp: float = 0.16
    texture_phase: Tuple[float, float, float] = (0.0, 0.0, 0.0)


def euler_to_R(rx: float, ry: float, rz: float) -> np.ndarray:
    """Euler angles (XYZ order in radians) to 3x3 rotation matrix."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Construct 4x4 SE(3) matrix from R and t."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def smoothstep(x: float) -> float:
    """Hermite smoothstep interpolation on [0, 1]."""
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)
