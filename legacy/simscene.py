"""
POGS-Sim: a self-contained tabletop scene that emits **exactly** the data
contract the POGS pipeline consumes.

Why this exists
---------------
The scenes used in the ICRA paper were captured with a wrist-mounted ZED Mini
on a UR5 and are not distributed with the code release, and the online demo
needs a live stereo camera plus a robot.  POGS-Sim replaces the *hardware*, not
the *algorithm*: it produces the same tensors the real rig would
(multi-view RGB, metric depth, camera extrinsics, 2D instance masks, and a
manipulation video), so every downstream module is the real thing.

It also gives us something the real rig cannot: **ground-truth 6-DoF object
trajectories**, which is what makes the tracking numbers in this notebook
quantitative rather than qualitative.

Phase 1 (Sec. IV-A "Scene Capture")
    ``capture_scene``  -> 35 hemisphere views, RGB + noisy metric depth +
                          instance masks + poses, then fuse + DBSCAN-filter
                          the point cloud exactly as the paper does.

Phase 3 (Sec. IV-D "Tracking with POGS")
    ``make_tracking_sequence`` -> an RGB-D video in which a human hand
                          perturbs the objects, with scripted "object reset"
                          events matching the Sec. V-A benchmark protocol.

Geometry is a union of analytic primitives (box / ellipsoid / cylinder) ray-cast
in closed form, so depth and masks are exact and hole-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import CaptureConfig
from .geometry import PinholeCamera, look_at, quat_to_rotmat

# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #

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


@dataclass
class SceneObject:
    name: str
    prims: List[Primitive]
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw: float = 0.0
    movable: bool = True
    obj_id: int = 0


def _euler_to_R(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# --------------------------------------------------------------------------- #
# The object library -- shapes chosen to mirror the paper's irregular objects
# (jigsaw, clothes iron, shoe, shelf, shoebox; Fig. 4).
# --------------------------------------------------------------------------- #


def _clothes_iron() -> List[Primitive]:
    return [
        Primitive(BOX, (0.085, 0.052, 0.022), (0.0, 0.0, 0.022),
                  color=(0.78, 0.79, 0.82), texture_freq=34.0, texture_amp=0.10),
        Primitive(ELLIPSOID, (0.070, 0.040, 0.028), (0.0, 0.0, 0.062),
                  color=(0.20, 0.24, 0.34), texture_freq=26.0, texture_amp=0.14),
        Primitive(CYLINDER, (0.013, 0.013, 0.034), (-0.055, 0.0, 0.058),
                  rot_euler=(0.0, math.pi / 2.4, 0.0),
                  color=(0.14, 0.15, 0.18), texture_freq=48.0, texture_amp=0.09),
    ]


def _power_drill() -> List[Primitive]:
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


def _shoe() -> List[Primitive]:
    return [
        Primitive(ELLIPSOID, (0.125, 0.045, 0.040), (0.0, 0.0, 0.046),
                  color=(0.42, 0.45, 0.50), texture_freq=38.0, texture_amp=0.20),
        Primitive(BOX, (0.130, 0.048, 0.012), (0.0, 0.0, 0.012),
                  color=(0.90, 0.90, 0.88), texture_freq=60.0, texture_amp=0.10),
        Primitive(ELLIPSOID, (0.048, 0.043, 0.046), (-0.070, 0.0, 0.056),
                  color=(0.34, 0.37, 0.43), texture_freq=30.0, texture_amp=0.16),
    ]


def _shoebox() -> List[Primitive]:
    return [
        Primitive(BOX, (0.115, 0.072, 0.045), (0.0, 0.0, 0.045),
                  color=(0.16, 0.16, 0.19), texture_freq=18.0, texture_amp=0.10),
    ]


def _jigsaw() -> List[Primitive]:
    return [
        Primitive(BOX, (0.080, 0.042, 0.040), (0.0, 0.0, 0.078),
                  color=(0.88, 0.36, 0.09), texture_freq=26.0, texture_amp=0.16),
        Primitive(BOX, (0.052, 0.046, 0.038), (0.0, 0.0, 0.038),
                  color=(0.20, 0.21, 0.24), texture_freq=40.0, texture_amp=0.14),
        Primitive(BOX, (0.060, 0.006, 0.026), (0.070, 0.0, 0.026),
                  color=(0.72, 0.74, 0.78), texture_freq=70.0, texture_amp=0.08),
    ]


def _wooden_shelf() -> List[Primitive]:
    legs = [
        Primitive(BOX, (0.012, 0.012, 0.048), (sx * 0.13, sy * 0.075, 0.048),
                  color=(0.55, 0.38, 0.20), texture_freq=44.0, texture_amp=0.14)
        for sx in (-1, 1) for sy in (-1, 1)
    ]
    return [
        Primitive(BOX, (0.150, 0.092, 0.010), (0.0, 0.0, 0.106),
                  color=(0.64, 0.45, 0.24), texture_freq=20.0, texture_amp=0.18),
    ] + legs


OBJECT_LIBRARY = {
    "clothes iron": _clothes_iron,
    "power drill": _power_drill,
    "shoe": _shoe,
    "shoebox": _shoebox,
    "jigsaw": _jigsaw,
    "wooden shelf": _wooden_shelf,
}


# --------------------------------------------------------------------------- #
# Scene assembly
# --------------------------------------------------------------------------- #


TABLE_ID = 0
"""Object id 0 is the tabletop; it is static background, never tracked."""


def build_scene(
    object_names: Sequence[str] = ("clothes iron", "power drill"),
    layout_radius: float = 0.205,
    seed: int = 0,
) -> List[SceneObject]:
    rng = np.random.default_rng(seed)
    table = SceneObject(
        name="table",
        prims=[
            Primitive(BOX, (0.52, 0.42, 0.02), (0.0, 0.0, -0.02),
                      color=(0.60, 0.57, 0.52), texture_freq=6.0, texture_amp=0.16),
        ],
        movable=False,
        obj_id=TABLE_ID,
    )
    objs = [table]
    n = len(object_names)
    for i, name in enumerate(object_names):
        ang = 2 * math.pi * i / max(n, 1) + 0.35
        r = layout_radius * (0.85 + 0.3 * rng.random())
        prims = OBJECT_LIBRARY[name]()
        for pr in prims:
            pr.texture_phase = tuple(rng.uniform(0, 2 * math.pi, size=3).tolist())
            pr.texture_amp = float(pr.texture_amp * rng.uniform(1.3, 1.8))
        objs.append(
            SceneObject(
                name=name,
                prims=prims,
                position=(r * math.cos(ang), r * math.sin(ang), 0.0),
                yaw=float(rng.uniform(-math.pi, math.pi)),
                movable=True,
                obj_id=i + 1,
            )
        )
    return objs


def make_hand(scale: float = 1.0) -> SceneObject:
    """A crude human hand used as an occluder (Sec. V-A: 'a human randomly
    perturbs the positions and orientations of all objects')."""
    fingers = [
        Primitive(CYLINDER, (0.010, 0.010, 0.038 * scale),
                  (0.050 * scale, (j - 1.5) * 0.021 * scale, 0.0),
                  rot_euler=(0.0, math.pi / 2, 0.0),
                  color=(0.76, 0.60, 0.50), texture_freq=40.0, texture_amp=0.08)
        for j in range(4)
    ]
    return SceneObject(
        name="hand",
        prims=[
            Primitive(ELLIPSOID, (0.052 * scale, 0.042 * scale, 0.020 * scale),
                      (0.0, 0.0, 0.0),
                      color=(0.78, 0.62, 0.52), texture_freq=22.0, texture_amp=0.10),
            Primitive(CYLINDER, (0.013, 0.013, 0.030 * scale),
                      (0.012 * scale, -0.048 * scale, 0.0),
                      rot_euler=(math.pi / 2, 0.0, 0.4),
                      color=(0.76, 0.60, 0.50), texture_freq=40.0, texture_amp=0.08),
        ] + fingers,
        movable=True,
        obj_id=99,
    )


# --------------------------------------------------------------------------- #
# Tool / target annotations (Sec. V-B, "Tool Servoing")
# --------------------------------------------------------------------------- #

TOOL_TIP_LOCAL: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = {
    # name -> (tip position in the object frame, tip axis in the object frame)
    "power drill": ((0.118, 0.0, 0.092), (1.0, 0.0, 0.0)),
    "jigsaw": ((0.100, 0.0, 0.026), (1.0, 0.0, 0.0)),
}
"""Paper Sec. V-B: 'The tool is manually annotated with a coordinate frame at
the tool tip to indicate which component to align with the target (i.e. align
the drill tip perpendicularly).'  These are those annotations: the drill tip is
the far end of the chuck cylinder, pointing along the object's +x axis."""

TARGET_MARKER_LOCAL: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = {
    # name -> (marker position in the object frame, outward surface normal)
    "wooden shelf": ((0.0, 0.0, 0.116), (0.0, 0.0, 1.0)),
    "shoebox": ((0.0, 0.0, 0.090), (0.0, 0.0, 1.0)),
}
"""Paper Sec. V-B: 'an ArUco marker fixed to a target object surface'.  The
marker sits at the centre of the top face; its normal is the surface normal the
tool tip must approach perpendicularly."""


def _object_frame(obj: SceneObject) -> Tuple[np.ndarray, np.ndarray]:
    """(R, t) of the object frame at scan time -- must match ``FlatScene``."""
    return _euler_to_R(0.0, 0.0, obj.yaw), np.asarray(obj.position, dtype=np.float64)


def tool_tip_frame(obj: SceneObject) -> Tuple[np.ndarray, np.ndarray]:
    """Scan-time world tool-tip position and unit tip axis."""
    if obj.name not in TOOL_TIP_LOCAL:
        raise KeyError(f"no tool-tip annotation for '{obj.name}'; "
                       f"known tools: {sorted(TOOL_TIP_LOCAL)}")
    p_local, a_local = TOOL_TIP_LOCAL[obj.name]
    R, t = _object_frame(obj)
    axis = R @ np.asarray(a_local, dtype=np.float64)
    return R @ np.asarray(p_local, dtype=np.float64) + t, axis / np.linalg.norm(axis)


def target_marker_frame(obj: SceneObject) -> Tuple[np.ndarray, np.ndarray]:
    """Scan-time world marker position and outward surface normal."""
    if obj.name not in TARGET_MARKER_LOCAL:
        raise KeyError(f"no marker annotation for '{obj.name}'; "
                       f"known targets: {sorted(TARGET_MARKER_LOCAL)}")
    p_local, n_local = TARGET_MARKER_LOCAL[obj.name]
    R, t = _object_frame(obj)
    n = R @ np.asarray(n_local, dtype=np.float64)
    return R @ np.asarray(p_local, dtype=np.float64) + t, n / np.linalg.norm(n)


# --------------------------------------------------------------------------- #
# Flattened GPU representation
# --------------------------------------------------------------------------- #


class FlatScene:
    """All primitives of all objects packed into tensors for batched ray casting."""

    def __init__(self, objects: Sequence[SceneObject], device="cpu"):
        kinds, scales, R_list, t_list, colors, freqs, amps, phases, oids = (
            [], [], [], [], [], [], [], [], []
        )
        for obj in objects:
            R_obj = _euler_to_R(0.0, 0.0, obj.yaw)
            t_obj = np.asarray(obj.position, dtype=np.float64)
            for p in obj.prims:
                R_p = _euler_to_R(*p.rot_euler)
                kinds.append(p.kind)
                scales.append(p.scale)
                R_list.append(R_obj @ R_p)
                t_list.append(t_obj + R_obj @ np.asarray(p.offset))
                colors.append(p.color)
                freqs.append(p.texture_freq)
                amps.append(p.texture_amp)
                phases.append(p.texture_phase)
                oids.append(obj.obj_id)

        f = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=device)  # noqa: E731
        self.device = device
        self.kind = f(kinds, torch.long)
        self.scale = f(scales)
        self.R = f(np.asarray(R_list))
        self.t = f(np.asarray(t_list))
        self.color = f(colors)
        self.freq = f(freqs)
        self.amp = f(amps)
        self.phase = f(phases)
        self.obj_id = f(oids, torch.long)
        self.objects = list(objects)
        # per-object base transform, used to apply tracking-time deltas
        self.obj_index = {o.obj_id: i for i, o in enumerate(objects)}

    def n_prims(self) -> int:
        return int(self.kind.shape[0])

    def clone_with_transforms(
        self, deltas: Dict[int, np.ndarray]
    ) -> "FlatScene":
        """Return a copy with a 4x4 world-frame delta applied to the given
        object ids (used to animate the tracking sequence)."""
        out = object.__new__(FlatScene)
        out.__dict__.update({k: v for k, v in self.__dict__.items()})
        R = self.R.clone()
        t = self.t.clone()
        for oid, T in deltas.items():
            m = self.obj_id == oid
            if not bool(m.any()):
                continue
            Tt = torch.as_tensor(T, dtype=torch.float32, device=self.device)
            R[m] = Tt[:3, :3] @ R[m]
            t[m] = (Tt[:3, :3] @ t[m].T).T + Tt[:3, 3]
        out.R, out.t = R, t
        return out


# --------------------------------------------------------------------------- #
# Analytic ray casting
# --------------------------------------------------------------------------- #


def _ray_dirs(camera: PinholeCamera) -> torch.Tensor:
    """(H*W, 3) world-space unit ray directions, OpenCV convention."""
    device = camera.c2w.device
    ys, xs = torch.meshgrid(
        torch.arange(camera.height, device=device, dtype=torch.float32) + 0.5,
        torch.arange(camera.width, device=device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    x = (xs - camera.cx) / camera.fx
    y = (ys - camera.cy) / camera.fy
    d_cam = torch.stack([x, y, torch.ones_like(x)], dim=-1).reshape(-1, 3)
    d_world = d_cam @ camera.c2w[:3, :3].T
    return torch.nn.functional.normalize(d_world, dim=-1)


def _intersect_batch(
    o_l: torch.Tensor, d_l: torch.Tensor, kind: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ray/unit-primitive intersection.

    o_l, d_l: (P, R, 3) rays already mapped into each primitive's unit frame.
    Returns (t, normal_local) with t = +inf where there is no hit.
    """
    if o_l.shape[1] == 1 and d_l.shape[1] != 1:
        o_l = o_l.expand(-1, d_l.shape[1], -1)
    INF = torch.full(o_l.shape[:2], float("inf"), device=o_l.device)
    t_out = INF.clone()
    n_out = torch.zeros_like(o_l)

    # ---- box: slab method ------------------------------------------------ #
    m_box = (kind == BOX).unsqueeze(-1).unsqueeze(-1)
    inv = 1.0 / torch.where(d_l.abs() < 1e-9, torch.full_like(d_l, 1e-9), d_l)
    t1 = (-1.0 - o_l) * inv
    t2 = (1.0 - o_l) * inv
    tmin = torch.minimum(t1, t2).max(dim=-1)
    tmax = torch.maximum(t1, t2).min(dim=-1).values
    t_near, axis = tmin.values, tmin.indices
    hit_box = (t_near <= tmax) & (tmax > 1e-6)
    t_near_pos = torch.where(t_near > 1e-6, t_near, tmax)
    hit_box = hit_box & (t_near_pos > 1e-6)
    n_box = torch.zeros_like(o_l)
    n_box.scatter_(-1, axis.unsqueeze(-1), 1.0)
    sgn = -torch.sign(torch.gather(d_l, -1, axis.unsqueeze(-1)))
    n_box = n_box * sgn

    sel = m_box.squeeze(-1).expand_as(t_out) & hit_box
    t_out = torch.where(sel, t_near_pos, t_out)
    n_out = torch.where(sel.unsqueeze(-1), n_box, n_out)

    # ---- ellipsoid: unit sphere in local frame ---------------------------- #
    a = (d_l * d_l).sum(-1)
    b = (o_l * d_l).sum(-1)
    c = (o_l * o_l).sum(-1) - 1.0
    disc = b * b - a * c
    ok = disc > 0
    sq = torch.sqrt(disc.clamp(min=0))
    ta = (-b - sq) / a.clamp(min=1e-12)
    tb = (-b + sq) / a.clamp(min=1e-12)
    t_sph = torch.where(ta > 1e-6, ta, tb)
    hit_sph = ok & (t_sph > 1e-6)
    p_sph = o_l + t_sph.unsqueeze(-1) * d_l
    sel = (kind == ELLIPSOID).unsqueeze(-1).expand_as(t_out) & hit_sph
    t_out = torch.where(sel, t_sph, t_out)
    n_out = torch.where(sel.unsqueeze(-1), p_sph, n_out)

    # ---- cylinder: x^2 + y^2 = 1, |z| <= 1, with caps --------------------- #
    a2 = (d_l[..., 0] ** 2 + d_l[..., 1] ** 2)
    b2 = (o_l[..., 0] * d_l[..., 0] + o_l[..., 1] * d_l[..., 1])
    c2 = (o_l[..., 0] ** 2 + o_l[..., 1] ** 2) - 1.0
    disc2 = b2 * b2 - a2 * c2
    ok2 = (disc2 > 0) & (a2 > 1e-12)
    sq2 = torch.sqrt(disc2.clamp(min=0))
    a2s = a2.clamp(min=1e-12)
    tc1 = (-b2 - sq2) / a2s
    tc2 = (-b2 + sq2) / a2s
    z1 = o_l[..., 2] + tc1 * d_l[..., 2]
    z2 = o_l[..., 2] + tc2 * d_l[..., 2]
    v1 = ok2 & (tc1 > 1e-6) & (z1.abs() <= 1.0)
    v2 = ok2 & (tc2 > 1e-6) & (z2.abs() <= 1.0)
    t_side = torch.where(v1, tc1, torch.where(v2, tc2, INF))

    dz = torch.where(d_l[..., 2].abs() < 1e-9, torch.full_like(d_l[..., 2], 1e-9), d_l[..., 2])
    tcap1 = (1.0 - o_l[..., 2]) / dz
    tcap2 = (-1.0 - o_l[..., 2]) / dz
    for tcap, nz in ((tcap1, 1.0), (tcap2, -1.0)):
        pc = o_l + tcap.unsqueeze(-1) * d_l
        good = (tcap > 1e-6) & ((pc[..., 0] ** 2 + pc[..., 1] ** 2) <= 1.0) & (tcap < t_side)
        t_side = torch.where(good, tcap, t_side)

    hit_cyl = torch.isfinite(t_side)
    p_cyl = o_l + torch.where(hit_cyl, t_side, torch.zeros_like(t_side)).unsqueeze(-1) * d_l
    on_cap = p_cyl[..., 2].abs() > 0.999
    n_cyl = torch.stack(
        [
            torch.where(on_cap, torch.zeros_like(p_cyl[..., 0]), p_cyl[..., 0]),
            torch.where(on_cap, torch.zeros_like(p_cyl[..., 1]), p_cyl[..., 1]),
            torch.where(on_cap, torch.sign(p_cyl[..., 2]), torch.zeros_like(p_cyl[..., 2])),
        ],
        dim=-1,
    )
    sel = (kind == CYLINDER).unsqueeze(-1).expand_as(t_out) & hit_cyl
    t_out = torch.where(sel, t_side, t_out)
    n_out = torch.where(sel.unsqueeze(-1), n_cyl, n_out)

    return t_out, n_out


TEXTURE_SCALE = 3.2
"""Global multiplier on the procedural texture frequency. Surfaces need
sub-centimetre structure or the distilled DINO features become degenerate."""

LIGHT_DIR = np.array([0.42, -0.36, 0.83])
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)


@torch.no_grad()
def render_scene(
    scene: FlatScene,
    camera: PinholeCamera,
    supersample: int = 2,
    ray_chunk: int = 65536,
    ambient: float = 0.50,
    background: Tuple[float, float, float] = (0.10, 0.11, 0.13),
) -> Dict[str, torch.Tensor]:
    """Exact RGB / metric depth / instance-id render.

    Returns ``rgb (H,W,3) in [0,1]``, ``depth (H,W) metres`` (0 = miss),
    ``obj_id (H,W) int64`` (-1 = background).
    """
    device = scene.device
    cam_hi = camera.rescale(supersample) if supersample != 1 else camera
    dirs = _ray_dirs(cam_hi).to(device)                       # (R, 3)
    origin = cam_hi.c2w[:3, 3].to(device)
    R_tot = dirs.shape[0]

    P = scene.n_prims()
    inv_scale = 1.0 / scene.scale                              # (P, 3)
    Rt = scene.R.transpose(1, 2)                               # (P, 3, 3)

    depth_all = torch.zeros(R_tot, device=device)
    rgb_all = torch.zeros(R_tot, 3, device=device)
    oid_all = torch.full((R_tot,), -1, dtype=torch.long, device=device)

    light = torch.as_tensor(LIGHT_DIR, dtype=torch.float32, device=device)

    for s in range(0, R_tot, ray_chunk):
        e = min(s + ray_chunk, R_tot)
        d = dirs[s:e]                                          # (r, 3)
        # map into every primitive's unit frame
        o_rel = (origin.unsqueeze(0) - scene.t).unsqueeze(1)   # (P, 1, 3)
        o_l = torch.einsum("pij,prj->pri", Rt, o_rel) * inv_scale.unsqueeze(1)
        d_l = torch.einsum("pij,rj->pri", Rt, d) * inv_scale.unsqueeze(1)

        t_hit, n_l = _intersect_batch(o_l, d_l, scene.kind)    # (P, r), (P, r, 3)
        best_t, best_p = t_hit.min(dim=0)                      # (r,)
        hit = torch.isfinite(best_t)

        idx = best_p.clamp(min=0)
        n_local = n_l[best_p, torch.arange(e - s, device=device)]
        # normal transform: n_world ~ R * S^-1 * n_local
        n_world = torch.einsum(
            "rij,rj->ri", scene.R[idx], n_local * inv_scale[idx]
        )
        n_world = torch.nn.functional.normalize(n_world, dim=-1)

        p_world = origin.unsqueeze(0) + torch.where(hit, best_t, torch.zeros_like(best_t)).unsqueeze(-1) * d
        # procedural 3-D texture so the surfaces carry DINO-visible structure
        f = scene.freq[idx].unsqueeze(-1) * TEXTURE_SCALE
        ph = scene.phase[idx]
        q = f * p_world + ph
        tex = (
            torch.sin(q).prod(dim=-1)
            + 0.6 * torch.sin(1.9 * q[..., 0] + 1.3 * q[..., 1])
            + 0.4 * torch.sin(3.7 * q[..., 2] - 2.1 * q[..., 0])
            + 0.3 * torch.sin(7.3 * q[..., 1] + 0.9 * q[..., 2])
        ) / 2.3
        shade = ambient + (1.0 - ambient) * torch.clamp((n_world * light).sum(-1), min=0.0)
        base = scene.color[idx] * (1.0 + scene.amp[idx].unsqueeze(-1) * tex.unsqueeze(-1))
        col = torch.clamp(base * shade.unsqueeze(-1), 0.0, 1.0)

        bg = torch.as_tensor(background, dtype=torch.float32, device=device)
        rgb_all[s:e] = torch.where(hit.unsqueeze(-1), col, bg.unsqueeze(0))
        # metric depth = z along the optical axis, not ray length
        z_axis = (d @ cam_hi.c2w[:3, 2].to(device))
        depth_all[s:e] = torch.where(hit, best_t * z_axis, torch.zeros_like(best_t))
        oid_all[s:e] = torch.where(hit, scene.obj_id[idx], torch.full_like(idx, -1))

    Hh, Wh = cam_hi.height, cam_hi.width
    rgb = rgb_all.view(Hh, Wh, 3)
    depth = depth_all.view(Hh, Wh)
    oid = oid_all.view(Hh, Wh)

    if supersample != 1:
        k = supersample
        rgb = rgb.view(Hh // k, k, Wh // k, k, 3).mean(dim=(1, 3))
        depth = depth.view(Hh // k, k, Wh // k, k)[:, 0, :, 0]
        oid = oid.view(Hh // k, k, Wh // k, k)[:, 0, :, 0]
    return {"rgb": rgb, "depth": depth, "obj_id": oid}


# --------------------------------------------------------------------------- #
# Camera trajectory (Sec. IV-A: hemisphere sweep with the wrist camera)
# --------------------------------------------------------------------------- #


def hemisphere_trajectory(cfg: CaptureConfig, center=(0.0, 0.0, 0.05)) -> List[np.ndarray]:
    poses = []
    n = cfg.num_views
    for i in range(n):
        u = i / max(n - 1, 1)
        az = 2 * math.pi * cfg.azimuth_turns * u + 0.7
        el = math.radians(cfg.elevation_min_deg + (cfg.elevation_max_deg - cfg.elevation_min_deg) * (0.5 - 0.5 * math.cos(math.pi * u)))
        r = cfg.radius_min + (cfg.radius_max - cfg.radius_min) * (0.5 + 0.5 * math.sin(3.0 * math.pi * u))
        eye = np.array(
            [center[0] + r * math.cos(el) * math.cos(az),
             center[1] + r * math.cos(el) * math.sin(az),
             center[2] + r * math.sin(el)]
        )
        poses.append(look_at(eye, np.asarray(center)))
    return poses


# --------------------------------------------------------------------------- #
# Capture phase
# --------------------------------------------------------------------------- #


@dataclass
class CaptureData:
    images: torch.Tensor            # (V, H, W, 3) float32 in [0, 1]
    depths: torch.Tensor            # (V, H, W)    float32 metres, 0 = invalid
    obj_ids: torch.Tensor           # (V, H, W)    int64,  -1 = background
    cameras: List[PinholeCamera]
    seed_points: torch.Tensor       # (S, 3)   DBSCAN-filtered fused cloud
    seed_colors: torch.Tensor       # (S, 3)
    seed_obj_ids: torch.Tensor      # (S,)     GT ids, only used for diagnostics
    object_names: List[str]         # index == obj_id
    scene: Optional[FlatScene] = None
    """Only POGS-Sim provides this (needed for ADD/ADD-S model points).
    Real-data loaders in :mod:`pogs_kaggle.datasets` leave it None."""

    def n_views(self) -> int:
        return int(self.images.shape[0])


def deproject(depth: torch.Tensor, camera: PinholeCamera) -> torch.Tensor:
    """(H, W) metric depth -> (H, W, 3) world-space points."""
    device = depth.device
    ys, xs = torch.meshgrid(
        torch.arange(camera.height, device=device, dtype=torch.float32) + 0.5,
        torch.arange(camera.width, device=device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    x = (xs - camera.cx) / camera.fx * depth
    y = (ys - camera.cy) / camera.fy * depth
    p_cam = torch.stack([x, y, depth], dim=-1)
    c2w = camera.c2w.to(device)
    return p_cam @ c2w[:3, :3].T + c2w[:3, 3]


def _dbscan_filter(
    points: np.ndarray, eps: float, min_samples: int, min_keep_frac: float = 0.5
) -> np.ndarray:
    """Paper Sec. IV-A: 'We run DBSCAN on the fused pointcloud to filter noise
    and floater points.' Returns a boolean keep-mask.

    ``eps`` is a metric radius, so its correct value depends on how densely the
    fused cloud was sampled.  If the paper's default would discard most of the
    cloud (which happens whenever the capture is sub-sampled) the radius is
    relaxed rather than silently returning an empty scene.
    """
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    try:
        from sklearn.cluster import DBSCAN
    except Exception:  # noqa: BLE001
        return np.ones(len(points), dtype=bool)

    for factor in (1.0, 2.0, 4.0, 8.0):
        labels = DBSCAN(eps=eps * factor, min_samples=min_samples, n_jobs=-1).fit_predict(points)
        keep = labels >= 0
        if keep.mean() >= min_keep_frac:
            if factor > 1.0:
                print(f"  [dbscan] relaxed eps {eps:.4f} -> {eps * factor:.4f} "
                      f"for this cloud density")
            return keep
    print("  [dbscan] cloud too sparse for radius filtering; keeping all points")
    return np.ones(len(points), dtype=bool)


@torch.no_grad()
def capture_scene(
    objects: Sequence[SceneObject],
    cfg: CaptureConfig,
    device: str = "cuda:0",
    supersample: int = 2,
    verbose: bool = True,
) -> CaptureData:
    """Phase 1. Emits the multi-view RGB-D + poses and the DBSCAN-filtered
    fused point cloud used to seed the Gaussian Splat."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    scene = FlatScene(objects, device=device)
    poses = hemisphere_trajectory(cfg)

    imgs, deps, oids, cams = [], [], [], []
    for i, c2w in enumerate(poses):
        cam = PinholeCamera.from_fov(
            torch.as_tensor(c2w, dtype=torch.float32, device=device),
            cfg.fov_x_deg, cfg.image_width, cfg.image_height,
        )
        out = render_scene(scene, cam, supersample=supersample)
        d = out["depth"]
        # simulate RAFT-Stereo error: depth-proportional noise + dropout
        noise = torch.randn_like(d) * cfg.depth_noise_std * (d / 0.7).clamp(min=0.2)
        d_noisy = torch.where(d > 0, d + noise, d)
        drop = torch.rand_like(d) < cfg.depth_dropout
        d_noisy = torch.where(drop, torch.zeros_like(d_noisy), d_noisy)
        imgs.append(out["rgb"].cpu())
        deps.append(d_noisy.cpu())
        oids.append(out["obj_id"].cpu())
        cams.append(cam)
        if verbose and (i % 10 == 0 or i == len(poses) - 1):
            print(f"  capture view {i + 1}/{len(poses)}")

    images = torch.stack(imgs)
    depths = torch.stack(deps)
    obj_ids = torch.stack(oids)

    # ---- fuse into a single cloud ---------------------------------------- #
    stride = max(1, int(math.sqrt(images[0].numel() / 3 * len(cams) / max(cfg.max_seed_points * 4, 1))))
    pts, cols, ids = [], [], []
    for v, cam in enumerate(cams):
        d = depths[v].to(device)
        p = deproject(d, cam)
        m = (d > 0.05)[..., None].squeeze(-1)
        p = p[m][::stride]
        c = images[v].to(device)[m][::stride]
        o = obj_ids[v].to(device)[m][::stride]
        pts.append(p.cpu())
        cols.append(c.cpu())
        ids.append(o.cpu())
    P = torch.cat(pts).numpy()
    C = torch.cat(cols).numpy()
    I = torch.cat(ids).numpy()

    if len(P) > cfg.max_seed_points:
        sel = rng.choice(len(P), cfg.max_seed_points, replace=False)
        P, C, I = P[sel], C[sel], I[sel]

    keep = _dbscan_filter(P, cfg.dbscan_eps, cfg.dbscan_min_samples)
    if verbose:
        print(f"  fused cloud {len(P)} pts -> DBSCAN kept {int(keep.sum())}")
    P, C, I = P[keep], C[keep], I[keep]

    names = ["table"] + [o.name for o in objects if o.movable]
    return CaptureData(
        images=images,
        depths=depths,
        obj_ids=obj_ids,
        cameras=cams,
        seed_points=torch.from_numpy(P).float(),
        seed_colors=torch.from_numpy(C).float(),
        seed_obj_ids=torch.from_numpy(I).long(),
        object_names=names,
        scene=scene,
    )


# --------------------------------------------------------------------------- #
# Tracking sequence (Phase 3)
# --------------------------------------------------------------------------- #


@dataclass
class TrackingFrame:
    rgb: torch.Tensor          # (H, W, 3)
    depth: torch.Tensor        # (H, W)
    obj_id: torch.Tensor       # (H, W)
    gt_delta: Dict[int, np.ndarray]   # obj_id -> 4x4 world-frame delta from scan pose
    hand_visible: bool
    reset_event: bool


@dataclass
class TrackingSequence:
    frames: List[TrackingFrame]
    camera: PinholeCamera
    object_ids: List[int]
    fps: float = 5.0


def _se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def plan_object_trajectory(
    n_frames: int,
    n_resets: int,
    max_translation: float,
    max_yaw_deg: float,
    tier: int,
    rng: np.random.Generator,
    pivot: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """Piecewise pick-and-place motion matching the Sec. V-A reset protocol.

    Tier 1: 'objects could be translated anywhere within the ZED 2 camera
            frustum but rotated only up to 90 deg around the vertical axis'.
    Tier 2: '... rotated to any magnitude around the vertical axis'.

    ``pivot`` is the point the yaw turns about, in world coordinates -- pass the
    object's own origin.  The paper perturbs each object about *its own* vertical
    axis and translates it independently; pivoting at the world origin instead
    (``pivot=None``) drags the object along a circular arc of radius ``|pivot|``,
    which for a 0.2 m tabletop layout injects 25-43 cm of translation that the
    protocol never asked for and makes tier 2 strictly further-travelling than
    tier 1 for reasons unrelated to the yaw limit.

    Keyframes are stored as (yaw, translation) and only composed into a world
    delta at the end, so interpolating them interpolates the *intended* motion
    rather than the pivot-induced arc.
    """
    yaw_limit = math.radians(max_yaw_deg)
    segs = max(n_resets, 1)
    p = np.zeros(3) if pivot is None else np.asarray(pivot, dtype=np.float64)

    def world_delta(yaw: float, t: np.ndarray) -> np.ndarray:
        """SE(3) that rotates by ``yaw`` about the vertical through ``p``, then
        translates by ``t`` -- expressed in the world frame, which is what
        ``FlatScene.clone_with_transforms`` applies."""
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return _se3(R, p - R @ p + t)

    keys = [(0.0, np.zeros(3))]
    for _ in range(segs):
        t = rng.uniform(-max_translation, max_translation, size=3)
        t[2] = abs(rng.normal(0.0, 0.012))          # stays roughly on the table
        keys.append((float(rng.uniform(-yaw_limit, yaw_limit)), t))

    out = []
    per = max(n_frames // segs, 1)
    for f in range(n_frames):
        seg = min(f // per, segs - 1)
        a = _smoothstep((f - seg * per) / per)
        (y0, t0), (y1, t1) = keys[seg], keys[seg + 1]
        # yaw-only rotations commute, so shortest-arc scalar interpolation is exact
        dy = (y1 - y0 + math.pi) % (2 * math.pi) - math.pi
        out.append(world_delta(y0 + a * dy, (1 - a) * t0 + a * t1))
    return out


@torch.no_grad()
def make_tracking_sequence(
    capture: CaptureData,
    n_frames: int = 60,
    n_resets: int = 3,
    tier: int = 1,
    max_translation: float = 0.09,
    with_hand: bool = True,
    hand_occlusion_frames: Tuple[float, float] = (0.30, 0.62),
    camera: Optional[PinholeCamera] = None,
    device: str = "cuda:0",
    supersample: int = 2,
    depth_noise_std: float = 0.0035,
    seed: int = 1,
    verbose: bool = True,
) -> TrackingSequence:
    """Phase 3 input: a static third-person ZED-2-style view (Sec. V) watching a
    human perturb the objects."""
    rng = np.random.default_rng(seed)
    if camera is None:
        eye = np.array([0.30, -0.52, 0.42])
        camera = PinholeCamera.from_fov(
            torch.as_tensor(look_at(eye, np.array([0.0, 0.0, 0.045])),
                            dtype=torch.float32, device=device),
            62.0,
            capture.images.shape[2],
            capture.images.shape[1],
        )

    movable = [o for o in capture.scene.objects if o.movable]
    trajs = {
        o.obj_id: plan_object_trajectory(
            n_frames, n_resets, max_translation, 90.0 if tier == 1 else 180.0, tier, rng,
            pivot=np.asarray(o.position, dtype=np.float64),
        )
        for o in movable
    }
    reset_frames = {int(round((k + 1) * n_frames / max(n_resets, 1))) - 1 for k in range(n_resets)}

    hand = make_hand()
    f0, f1 = hand_occlusion_frames
    hand_lo, hand_hi = int(f0 * n_frames), int(f1 * n_frames)

    frames: List[TrackingFrame] = []
    for f in range(n_frames):
        deltas = {oid: trajs[oid][f] for oid in trajs}
        objs = list(capture.scene.objects)
        if with_hand and hand_lo <= f <= hand_hi:
            u = (f - hand_lo) / max(hand_hi - hand_lo, 1)
            target = movable[0]
            base = np.asarray(target.position) + deltas[target.obj_id][:3, 3]
            h = make_hand()
            reach = 0.22 * (1 - math.sin(math.pi * u))
            h.position = (float(base[0] - reach * 0.4), float(base[1] - reach), float(base[2] + 0.10))
            h.yaw = float(math.pi / 2 + 0.5 * math.sin(2 * math.pi * u))
            objs = objs + [h]
        scene_f = FlatScene(objs, device=device).clone_with_transforms(deltas)
        out = render_scene(scene_f, camera, supersample=supersample)
        d = out["depth"]
        noise = torch.randn_like(d) * depth_noise_std * (d / 0.7).clamp(min=0.2)
        d = torch.where(d > 0, d + noise, d)
        frames.append(
            TrackingFrame(
                rgb=out["rgb"].cpu(),
                depth=d.cpu(),
                obj_id=out["obj_id"].cpu(),
                gt_delta={k: v.copy() for k, v in deltas.items()},
                hand_visible=with_hand and hand_lo <= f <= hand_hi,
                reset_event=f in reset_frames,
            )
        )
        if verbose and (f % 20 == 0 or f == n_frames - 1):
            print(f"  tracking frame {f + 1}/{n_frames}")

    return TrackingSequence(frames=frames, camera=camera,
                            object_ids=[o.obj_id for o in movable])


@torch.no_grad()
def sample_object_model_points(
    scene: FlatScene, obj_id: int, n: int = 2000, seed: int = 0, device="cpu"
) -> torch.Tensor:
    """Surface samples of one object in the *scan* world frame.
    Used for ADD / ADD-S."""
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