"""
Tabletop object library and primitive assemblies matching ICRA 2025 paper objects.
Shapes mirror the paper's irregular objects:
- clothes iron
- power drill (tool object for servoing)
- shoe
- shoebox
- jigsaw
- wooden shelf
- human hand (for dynamic occlusions)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING
import numpy as np
import torch

from .primitives import Primitive, BOX, ELLIPSOID, CYLINDER

if TYPE_CHECKING:
    from .scene_builder import FlatScene


@dataclass
class SceneObject:
    name: str
    prims: List[Primitive]
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw: float = 0.0
    movable: bool = True
    obj_id: int = 0


def clothes_iron() -> List[Primitive]:
    return [
        Primitive(BOX, (0.085, 0.052, 0.022), (0.0, 0.0, 0.022),
                  color=(0.78, 0.79, 0.82), texture_freq=34.0, texture_amp=0.10),
        Primitive(ELLIPSOID, (0.070, 0.040, 0.028), (0.0, 0.0, 0.062),
                  color=(0.20, 0.24, 0.34), texture_freq=26.0, texture_amp=0.14),
        Primitive(CYLINDER, (0.013, 0.013, 0.034), (-0.055, 0.0, 0.058),
                  rot_euler=(0.0, math.pi / 2.4, 0.0),
                  color=(0.14, 0.15, 0.18), texture_freq=48.0, texture_amp=0.09),
    ]


def power_drill() -> List[Primitive]:
    return [
        Primitive(BOX, (0.055, 0.030, 0.036), (0.0, 0.0, 0.086),
                  color=(0.85, 0.42, 0.10), texture_freq=30.0, texture_amp=0.15),
        Primitive(CYLINDER, (0.019, 0.019, 0.048), (0.070, 0.0, 0.092),
                  rot_euler=(0.0, math.pi / 2, 0.0),
                  color=(0.30, 0.31, 0.34), texture_freq=52.0, texture_amp=0.12),
        Primitive(BOX, (0.024, 0.026, 0.050), (-0.010, 0.0, 0.036),
                  rot_euler=(0.0, -0.16, 0.0),
                  color=(0.16, 0.17, 0.20), texture_freq=44.0, texture_amp=0.18),
        Primitive(BOX, (0.038, 0.032, 0.012), (-0.012, 0.0, 0.010),
                  color=(0.86, 0.44, 0.12), texture_freq=28.0, texture_amp=0.12),
    ]


def shoe() -> List[Primitive]:
    return [
        Primitive(ELLIPSOID, (0.125, 0.045, 0.040), (0.0, 0.0, 0.046),
                  color=(0.28, 0.25, 0.23), texture_freq=36.0, texture_amp=0.14),
        Primitive(BOX, (0.115, 0.042, 0.012), (0.0, 0.0, 0.012),
                  color=(0.92, 0.90, 0.86), texture_freq=50.0, texture_amp=0.08),
        Primitive(CYLINDER, (0.035, 0.035, 0.030), (-0.040, 0.0, 0.080),
                  color=(0.22, 0.20, 0.19), texture_freq=32.0, texture_amp=0.10),
    ]


def shoebox() -> List[Primitive]:
    return [
        Primitive(BOX, (0.16, 0.10, 0.055), (0.0, 0.0, 0.055),
                  color=(0.68, 0.45, 0.28), texture_freq=22.0, texture_amp=0.18),
    ]


def jigsaw() -> List[Primitive]:
    return [
        Primitive(BOX, (0.080, 0.035, 0.050), (0.0, 0.0, 0.055),
                  color=(0.12, 0.50, 0.25), texture_freq=38.0, texture_amp=0.14),
        Primitive(CYLINDER, (0.016, 0.016, 0.055), (-0.02, 0.0, 0.110),
                  rot_euler=(0.0, math.pi / 2, 0.0),
                  color=(0.15, 0.15, 0.15), texture_freq=45.0, texture_amp=0.10),
        Primitive(BOX, (0.090, 0.045, 0.006), (0.005, 0.0, 0.006),
                  color=(0.75, 0.75, 0.78), texture_freq=60.0, texture_amp=0.08),
    ]


def wooden_shelf() -> List[Primitive]:
    return [
        Primitive(BOX, (0.24, 0.14, 0.010), (0.0, 0.0, 0.010),
                  color=(0.65, 0.46, 0.30), texture_freq=18.0, texture_amp=0.22),
        Primitive(BOX, (0.24, 0.14, 0.010), (0.0, 0.0, 0.140),
                  color=(0.65, 0.46, 0.30), texture_freq=18.0, texture_amp=0.22),
        Primitive(BOX, (0.010, 0.14, 0.065), (-0.23, 0.0, 0.075),
                  color=(0.60, 0.42, 0.26), texture_freq=20.0, texture_amp=0.18),
        Primitive(BOX, (0.010, 0.14, 0.065), (0.23, 0.0, 0.075),
                  color=(0.60, 0.42, 0.26), texture_freq=20.0, texture_amp=0.18),
    ]


def make_hand(scale: float = 1.0) -> SceneObject:
    """Human hand / arm model for realistic dynamic occlusions."""
    s = scale
    skin = (0.78, 0.60, 0.49)
    prims = [
        Primitive(BOX, (0.040 * s, 0.035 * s, 0.012 * s), (0.0, 0.0, 0.0), color=skin),
        Primitive(CYLINDER, (0.032 * s, 0.028 * s, 0.110 * s), (-0.02 * s, -0.12 * s, 0.02 * s),
                  rot_euler=(math.pi / 3.2, 0.0, 0.0), color=skin),
    ]
    for i, xo in enumerate([-0.025, -0.008, 0.008, 0.025]):
        flen = (0.030 + 0.005 * (1 - abs(i - 1.5))) * s
        prims.append(
            Primitive(CYLINDER, (0.005 * s, 0.005 * s, flen * 0.5),
                      (xo * s, 0.035 * s + flen * 0.5, 0.0),
                      rot_euler=(math.pi / 2, 0.0, 0.0), color=skin)
        )
    prims.append(
        Primitive(CYLINDER, (0.006 * s, 0.006 * s, 0.018 * s),
                  (-0.035 * s, 0.010 * s, 0.0),
                  rot_euler=(math.pi / 3, -math.pi / 4, 0.0), color=skin)
    )
    for p in prims:
        p.pattern, p.texture_freq, p.texture_amp, p.roughness = "noise", 250.0, 0.08, 0.6
    return SceneObject(name="hand", prims=prims, movable=False, obj_id=99)


def object_frame(obj: SceneObject) -> Tuple[np.ndarray, np.ndarray]:
    c, s = math.cos(obj.yaw), math.sin(obj.yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return R, np.asarray(obj.position, dtype=np.float64)


def tool_tip_frame(obj: SceneObject) -> Tuple[np.ndarray, np.ndarray]:
    """Frame attached to the tool's operational tip (Sec. V-B)."""
    R_obj, t_obj = object_frame(obj)
    local_tip = np.array([0.125, 0.0, 0.092])
    t_tip = t_obj + R_obj @ local_tip
    R_tip = R_obj @ np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    return R_tip, t_tip


def target_marker_frame(obj: SceneObject) -> Tuple[np.ndarray, np.ndarray]:
    """Frame of the ArUco-style target on the target object."""
    R_obj, t_obj = object_frame(obj)
    local_target = np.array([0.0, 0.0, 0.115])
    t_target = t_obj + R_obj @ local_target
    return R_obj, t_target


def sample_object_model_points(
    scene: "FlatScene", obj_id: int, n: int = 2000, seed: int = 0, device: str = "cpu"
) -> torch.Tensor:
    """Surface samples of one object in the scan world frame.
    Used for computing ADD and ADD-S tracking metrics.
    """
    rng = np.random.default_rng(seed)
    mask = (scene.obj_id == obj_id).cpu().numpy()
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        return torch.zeros(0, 3)
    per = max(n // len(idx), 1)
    pts = []
    for i in idx:
        kind = int(scene.kind[i].item())
        s = scene.scale[i].cpu().numpy()
        R = scene.R[i].cpu().numpy()
        t = scene.t[i].cpu().numpy()
        u = rng.normal(size=(per, 3))
        if kind == ELLIPSOID:
            u /= np.linalg.norm(u, axis=1, keepdims=True)
        elif kind == BOX:
            ax = rng.integers(0, 3, size=per)
            u = np.clip(u, -1, 1)
            u[np.arange(per), ax] = rng.choice([-1.0, 1.0], size=per)
        else:  # cylinder side surface
            th = rng.uniform(0, 2 * math.pi, size=per)
            u = np.stack([np.cos(th), np.sin(th), rng.uniform(-1, 1, size=per)], axis=-1)
        pts.append(u * s[None] @ R.T + t[None])
    P = np.concatenate(pts, axis=0)
    if len(P) > n:
        P = P[rng.choice(len(P), n, replace=False)]
    return torch.from_numpy(P).float().to(device)
