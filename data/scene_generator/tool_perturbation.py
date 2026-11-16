"""
Tool scene and perturbation sequences for O3 / O4 (PLAN Phase 2-3,
``experiments/protocols/o3.md``, ``o4.md``).

Scene ``tool_table`` = the pick-place scene (floor, low table, five household
objects) plus a cordless drill lying upright on the table. The drill uses scene-v2
procedural materials (textures in each primitive's frame, so they move with it).

Sequences (static tracking camera of the benchmark, 5 Hz, same file format as the
pick-place benchmark so ``load_sequence`` reads them):

    step_p{peak}_d{step}_{axis}_s{seed}   step-wise: rest -> +step ... +peak (state p)
                                          -> -step ... 0 (state y) -> rest      [supervisor]
    jump_{+/-}{deg}_{axis}_s{seed}        single jump to +/-deg, then hold      [paper Table II]

``axis``: ``z`` = the drill's vertical axis through its bounding-box centre,
``x`` = horizontal axis along the drill body (barrel direction). Per seed the
drill first moves smoothly from its scanned pose to a random start pose
(<= 5 cm, <= 45 deg yaw; see ``SETTLE_FRAMES``).
In-grasp sequences (O4) are added separately.
"""

from __future__ import annotations

import json
import math
import os
import time
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .objects import SceneObject
from .photoreal_scene import render_photoreal_scene
from .pick_place_benchmark import (
    FLOOR_ID, OBJECT_NAMES, TABLE_ID, TABLE_TOP_Z, BenchmarkConfig, _write_png_depth_mm,
    generate_capture, tracking_camera,
)
from .pick_place_scene import build_pick_place_scene
from .primitives import BOX, CYLINDER, Primitive
from .scene_builder import FlatScene

TOOL_SCENE_VERSION = 1
DRILL_ID = 6
TOOL_OBJECT_NAMES: Dict[int, str] = {**OBJECT_NAMES, DRILL_ID: "power_drill"}
TOOL_DATASET_DIR = "datasets/synthetic_scenes/tool_table"

DRILL_HEIGHT = 0.140      # bit axis at z = 0.095 + 0.020 radius ... housing top 0.130; incl. margin
DRILL_TIP_LOCAL = np.array([0.170, 0.0, 0.095])
"""Bit tip in the drill frame (bit cylinder centre 0.125 + half-length 0.045)."""

SETTLE_FRAMES = 10
SETTLE_MAX_SHIFT = 0.05   # m
SETTLE_MAX_YAW = 45.0     # deg
"""Each trial starts at the SCANNED pose (where POGS initialises) and moves the
drill smoothly to a per-seed start pose over ``SETTLE_FRAMES`` (tracked), then
rests. Starting at a random pose instead would mix initialisation failures into
the recovery result."""

REST_FRAMES = 5
HOLD_PER_STEP = 3
JUMP_HOLD = 15


def cordless_drill() -> List[Primitive]:
    """Asymmetric cordless drill, ~0.21 m long, scene-v2 materials."""
    plastic = dict(pattern="noise", texture_freq=160.0, texture_amp=0.05, roughness=0.35)
    return [
        Primitive(BOX, (0.058, 0.034, 0.038), (0.0, 0.0, 0.092), color=(0.08, 0.52, 0.58),
                  color2=(0.06, 0.40, 0.45), **plastic),                                   # motor housing
        Primitive(BOX, (0.020, 0.0345, 0.010), (0.030, 0.0, 0.118), color=(0.95, 0.95, 0.95),
                  pattern="stripes", color2=(0.10, 0.10, 0.10), texture_freq=90.0,
                  texture_amp=0.5, roughness=0.4),                                          # label band
        Primitive(CYLINDER, (0.020, 0.020, 0.035), (0.068, 0.0, 0.095), rot_euler=(0.0, math.pi / 2, 0.0),
                  color=(0.28, 0.29, 0.32), pattern="brushed", texture_freq=400.0, texture_amp=0.10,
                  roughness=0.25, metallic=0.8),                                            # chuck
        Primitive(CYLINDER, (0.005, 0.005, 0.045), (0.125, 0.0, 0.095), rot_euler=(0.0, math.pi / 2, 0.0),
                  color=(0.85, 0.85, 0.88), pattern="brushed", texture_freq=600.0, texture_amp=0.05,
                  roughness=0.2, metallic=1.0),                                             # bit
        Primitive(BOX, (0.026, 0.028, 0.055), (-0.012, 0.0, 0.042), rot_euler=(0.0, -0.18, 0.0),
                  color=(0.12, 0.13, 0.15), pattern="noise", texture_freq=220.0, texture_amp=0.12,
                  roughness=0.8),                                                           # rubber grip
        Primitive(BOX, (0.042, 0.036, 0.018), (-0.018, 0.0, 0.009), color=(0.14, 0.15, 0.18),
                  color2=(0.85, 0.42, 0.10), pattern="split", texture_freq=1.0, texture_amp=0.0,
                  roughness=0.5),                                                           # battery
    ]


def build_tool_scene(position: Tuple[float, float] = (0.18, 0.12), yaw: float = 0.0) -> List[SceneObject]:
    objs = build_pick_place_scene()
    objs.append(SceneObject(name="power_drill", prims=cordless_drill(),
                            position=(position[0], position[1], TABLE_TOP_Z), yaw=yaw, obj_id=DRILL_ID))
    return objs


def drill_center(obj: SceneObject) -> np.ndarray:
    """Pivot of the perturbation: drill bounding-box centre (body, not bit)."""
    c, s = math.cos(obj.yaw), math.sin(obj.yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return np.asarray(obj.position) + R @ np.array([0.02, 0.0, 0.5 * 0.13])


def _axis_vector(obj: SceneObject, axis: str) -> np.ndarray:
    if axis == "z":
        return np.array([0.0, 0.0, 1.0])
    if axis == "x":
        return np.array([math.cos(obj.yaw), math.sin(obj.yaw), 0.0])
    raise ValueError(axis)


def _rodrigues(axis: np.ndarray, deg: float) -> np.ndarray:
    a = math.radians(deg)
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def rotation_about(center: np.ndarray, axis: np.ndarray, deg: float) -> np.ndarray:
    """4x4 world transform rotating by ``deg`` about ``axis`` through ``center``."""
    T = np.eye(4)
    R = _rodrigues(axis, deg)
    T[:3, :3] = R
    T[:3, 3] = center - R @ center
    return T


# --------------------------------------------------------------------------- #
# Angle schedules
# --------------------------------------------------------------------------- #


def step_schedule(peak: float, step: float, hold: int = HOLD_PER_STEP, rest: int = REST_FRAMES) -> List[Tuple[float, str]]:
    """(angle, phase) per frame: rest, steps out to +peak, steps back to 0, rest.
    Each step is applied in one frame and then held ``hold`` frames in total."""
    n = int(round(peak / step))
    assert abs(n * step - peak) < 1e-9, "peak must be a multiple of step"
    out = [(0.0, "rest_x")] * rest
    for i in range(1, n + 1):
        out += [(i * step, "out" if i < n else "peak_p")] * hold
    for i in range(n - 1, -1, -1):
        out += [(i * step, "back" if i > 0 else "return_y")] * hold
    out += [(0.0, "rest_y")] * rest
    return out


def jump_schedule(deg: float, hold: int = JUMP_HOLD, rest: int = REST_FRAMES) -> List[Tuple[float, str]]:
    return [(0.0, "rest_x")] * rest + [(deg, "perturbed")] * hold


# --------------------------------------------------------------------------- #
# Sequences
# --------------------------------------------------------------------------- #


@dataclass
class ToolCase:
    kind: str                 # "step" | "jump"
    angle: float              # step: peak (deg); jump: signed angle (deg)
    step: float = 5.0         # step only
    axis: str = "z"
    seed: int = 0

    @property
    def name(self) -> str:
        if self.kind == "step":
            return f"step_p{self.angle:g}_d{self.step:g}_{self.axis}_s{self.seed}"
        return f"jump_{self.angle:+g}_{self.axis}_s{self.seed}"

    def schedule(self) -> List[Tuple[float, str]]:
        return step_schedule(self.angle, self.step) if self.kind == "step" else jump_schedule(self.angle)


def sample_start(seed: int) -> Tuple[np.ndarray, float]:
    """Per-seed start offset from the scanned pose: (dx, dy, 0) and yaw (deg)."""
    rng = np.random.default_rng(zlib.crc32(f"tool_s{seed}".encode()))
    r = SETTLE_MAX_SHIFT * math.sqrt(rng.uniform())
    a = rng.uniform(0, 2 * math.pi)
    return np.array([r * math.cos(a), r * math.sin(a), 0.0]), float(rng.uniform(-SETTLE_MAX_YAW, SETTLE_MAX_YAW))


def _settle(c0: np.ndarray, shift: np.ndarray, yaw_deg: float, u: float) -> np.ndarray:
    """Smooth (cosine) interpolation from identity (u=0) to the start pose (u=1)."""
    w = 0.5 - 0.5 * math.cos(math.pi * u)
    T = rotation_about(c0, np.array([0.0, 0.0, 1.0]), w * yaw_deg)
    T[:3, 3] += w * shift
    return T


def generate_tool_capture(out_dir: str = TOOL_DATASET_DIR, cfg: Optional[BenchmarkConfig] = None,
                          device: str = "cuda:0") -> None:
    """Scan of the tool scene with the drill at the canonical placement.
    Tracking sequences re-place the drill per seed by a GT transform from it."""
    generate_capture(out_dir, cfg or BenchmarkConfig(), device, objects=build_tool_scene(),
                     object_names=TOOL_OBJECT_NAMES, scene_version=TOOL_SCENE_VERSION)


def generate_tool_sequence(case: ToolCase, out_dir: str = TOOL_DATASET_DIR, cfg: Optional[BenchmarkConfig] = None,
                           device: str = "cuda:0", verbose: bool = True) -> str:
    """Renders ``case``: settle (scanned pose -> per-seed start pose), then the
    angle schedule about ``case.axis`` through the drill centre at the start pose."""
    cfg = cfg or BenchmarkConfig()
    objects = build_tool_scene()
    drill = next(o for o in objects if o.obj_id == DRILL_ID)
    cam = tracking_camera(cfg, device)
    shift, yaw0 = sample_start(case.seed)
    c0 = drill_center(drill)
    T_start = _settle(c0, shift, yaw0, 1.0)
    c_start = (T_start @ np.append(c0, 1.0))[:3]
    axis_local = {"z": np.array([0.0, 0.0, 1.0]), "x": np.array([1.0, 0.0, 0.0])}[case.axis]
    R_scan = np.array([[math.cos(drill.yaw), -math.sin(drill.yaw), 0],
                       [math.sin(drill.yaw), math.cos(drill.yaw), 0], [0, 0, 1]])
    axis_world = T_start[:3, :3] @ R_scan @ axis_local

    sched = [(None, "settle")] * SETTLE_FRAMES + case.schedule()
    sc_dir = os.path.join(out_dir, case.name)
    for sub in ("rgb", "depth_mm", "masks"):
        os.makedirs(os.path.join(sc_dir, sub), exist_ok=True)
    movable_ids = [o.obj_id for o in objects if o.obj_id not in (FLOOR_ID, TABLE_ID)]
    frames_meta = []
    t0 = time.time()
    for f, (ang, phase) in enumerate(sched):
        if ang is None:                              # settle: scanned pose -> start pose
            delta = _settle(c0, shift, yaw0, (f + 1) / SETTLE_FRAMES)
            ang = 0.0
        else:
            delta = rotation_about(c_start, axis_world, ang) @ T_start
        out = render_photoreal_scene(FlatScene(objects, device=device).clone_with_transforms({DRILL_ID: delta}),
                                     cam, cast_shadows=True)
        rgb = (out["rgb"].cpu().numpy() * 255).astype(np.uint8)
        fname = f"frame_{f:04d}.png"
        cv2.imwrite(os.path.join(sc_dir, "rgb", fname), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        _write_png_depth_mm(os.path.join(sc_dir, "depth_mm", fname), out["depth"].cpu().numpy())
        cv2.imwrite(os.path.join(sc_dir, "masks", fname), out["obj_id"].cpu().numpy().astype(np.uint8))
        frames_meta.append({
            "frame": f, "timestamp_sec": f / cfg.fps, "phase": phase, "rot_deg": abs(ang),
            "commanded_deg": ang, "yaw_deg": ang if case.axis == "z" else 0.0,
            "tilt_deg": ang if case.axis == "x" else 0.0, "translation_cm": 0.0,
            "hand_visible": False, "occlusion_frac": 0.0, "occlusion_by_obj": {str(DRILL_ID): 0.0},
            "gt_deltas": {str(oid): (delta if oid == DRILL_ID else np.eye(4)).tolist() for oid in movable_ids},
        })
    meta = {
        "name": case.name, "scene_version": TOOL_SCENE_VERSION, "scene": "tool_table",
        "condition": f"{case.kind}_{case.axis}", "seed": case.seed, "case": case.__dict__,
        "target_object_id": DRILL_ID, "target_object": TOOL_OBJECT_NAMES[DRILL_ID],
        "moving_object_ids": [DRILL_ID], "start_shift_m": shift.tolist(), "start_yaw_deg": yaw0,
        "settle_frames": SETTLE_FRAMES,
        "pivot_world": c_start.tolist(), "axis_world": axis_world.tolist(),
        "tool_tip_local": DRILL_TIP_LOCAL.tolist(),
        "fps": cfg.fps, "n_frames": len(sched),
        "depth_format": "uint16 PNG, millimetres, noise-free (noise is added at load time)",
        "camera": {"c2w": cam.c2w.cpu().tolist(), "fx": cam.fx, "fy": cam.fy, "cx": cam.cx, "cy": cam.cy,
                   "width": cam.width, "height": cam.height},
        "frames": frames_meta,
    }
    with open(os.path.join(sc_dir, "tracking_metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    if verbose:
        print(f"[tool] {case.name}: {len(sched)} frames in {time.time() - t0:.1f}s")
    return sc_dir


def o3_cases(seeds: Sequence[int] = range(5), axes: Sequence[str] = ("z",)) -> List[ToolCase]:
    """O3 matrix (draft protocol): step-wise 30 deg with 5/10/15 deg steps, and
    single jumps of +/-15 and +/-30 deg."""
    out = []
    for ax in axes:
        for s in seeds:
            out += [ToolCase("step", 30.0, st, ax, s) for st in (5.0, 10.0, 15.0)]
            out += [ToolCase("jump", a, axis=ax, seed=s) for a in (15.0, -15.0, 30.0, -30.0)]
    return out
