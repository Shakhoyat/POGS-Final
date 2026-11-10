"""
POGS tracking benchmark: one scanned scene, many controlled tracking conditions.

Goal: let a modified tracker be compared with base POGS *per condition*, so a
claim like "faster at equal accuracy" or "more robust under occlusion" can be
backed by numbers. Conditions follow the limitations the paper itself lists
(Sec. VI): slow motion only (~5 Hz), gripper / hand occlusion, and the
assumptions of Sec. III (mostly-visible, rigid objects).

Scene and scan (Phase 1)
------------------------
Floor scene with a low staging table and five household objects; 35 hemisphere
views with stereo-like depth noise; seed points balanced per object.

Rendered conditions (Phase 3) -- one factor changed at a time from ``base``
--------------------------------------------------------------------------
``base``          mug, picked by the gripper, carried to the table, yawed 30 deg
``yaw15/yaw90``   net yaw 15 / 90 deg  (Table II perturbations / reset Tier 1)
``tilt30/tilt60`` peak tilt about a random horizontal axis mid-carry, upright
                  again at placement -- exposes the unscanned underside
``combined45``    yaw ramp 45 deg + peak tilt 45 deg
``occ_none``      no gripper rendered: same motion, no occluder
``occ_hand``      gripper + a human hand reaching down between camera and object,
                  sweeping across it and covering about half of it (~1.4 s)
``occ_full``      same sweep, lower: >70% hidden for ~1 s, 100% at the peak
``obj_mustard``   mustard bottle instead of the mug (thin, near-symmetric)
``obj_cereal``    cereal box instead of the mug (large, flat faces)
``multi``         mug and soup can moved at the same time by two grippers

Every condition has ``N_SEEDS`` random trajectories (placement point, carry
height, path wobble, rotation sign, tilt axis), named ``<condition>_s<seed>``.

Load-time conditions (no extra rendering)
-----------------------------------------
Speed: ``load_sequence(..., speed=k)`` keeps every k-th frame at the same
camera rate, i.e. the object moves k times faster per tracked frame (equivalently,
a tracker k times slower than the camera). Depth: depth is stored clean and
``depth`` picks a noise profile (``DEPTH_PROFILES``) applied deterministically
at load time.

``FACTORS`` lists, per factor, the (level, condition, speed, depth) cases a
results table compares; ``benchmark_cases()`` expands them over seeds.

Ground-truth convention: ``gt_deltas[obj]`` is the world-frame rigid motion
applied to the object's geometry (x' = R x + t), identical to what
``FlatScene.clone_with_transforms`` renders.
"""

from __future__ import annotations

import json
import math
import os
import time
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from common.geometry import PinholeCamera, look_at
from .capture_generator import CaptureData, dbscan_filter, hemisphere_trajectory, save_pointcloud_ply
from .objects import make_hand
from .photoreal_scene import make_robot_gripper, render_photoreal_scene
from .pick_place_scene import build_pick_place_scene
from .primitives import euler_to_R, se3, smoothstep
from .scene_builder import FlatScene, deproject

FLOOR_ID, TABLE_ID = 0, 9
GRIPPER_ID, GRIPPER2_ID, HAND_ID = 99, 98, 97
OCCLUDER_IDS = (GRIPPER_ID, GRIPPER2_ID, HAND_ID)
TABLE_TOP_Z = 0.060
OBJECT_NAMES = {1: "ceramic_mug", 2: "mustard_bottle", 3: "soup_can", 4: "red_apple", 5: "cereal_box"}
OBJECT_QUERIES = {1: "coffee mug", 2: "mustard bottle", 3: "soup can", 4: "apple", 5: "cereal box"}
TARGETS: Dict[str, int] = {"mug": 1, "mustard": 2, "soup": 3, "cereal": 5}

# Placement regions (x range, y range) on the table top (x in [-0.02, 0.38],
# y in [-0.04, 0.28]); the second object of ``multi`` uses its own region.
PLACE_REGION = {
    "mug": ((0.14, 0.30), (0.02, 0.12)),
    "mustard": ((0.14, 0.30), (0.04, 0.14)),
    "cereal": ((0.14, 0.26), (0.06, 0.16)),
    "soup": ((0.04, 0.12), (0.19, 0.23)),
}


@dataclass(frozen=True)
class Condition:
    name: str
    target: str = "mug"
    rot_axis: str = "yaw"          # "yaw" | "tilt" | "combined"
    rot_deg: float = 30.0
    occlusion: str = "gripper"     # "none" | "gripper" | "hand"
    hand_lift: float = 0.045       # hand height above the line of sight (m): 0.045 ~ half, -0.01 ~ full
    second: Optional[str] = None   # second object moved simultaneously


CONDITIONS: Dict[str, Condition] = {c.name: c for c in [
    Condition("base"),
    Condition("yaw15", rot_deg=15.0),
    Condition("yaw90", rot_deg=90.0),
    Condition("tilt30", rot_axis="tilt", rot_deg=30.0),
    Condition("tilt60", rot_axis="tilt", rot_deg=60.0),
    Condition("combined45", rot_axis="combined", rot_deg=45.0),
    Condition("occ_none", occlusion="none"),
    Condition("occ_hand", occlusion="hand", hand_lift=0.045),
    Condition("occ_full", occlusion="hand", hand_lift=-0.01),
    Condition("obj_mustard", target="mustard"),
    Condition("obj_cereal", target="cereal"),
    Condition("multi", second="soup"),
]}
N_SEEDS = 3

# name -> (depth std in metres at 0.7 m range, fraction of pixels lost in blobs)
DEPTH_PROFILES: Dict[str, Tuple[float, float]] = {
    "clean": (0.0, 0.0),
    "stereo": (0.003, 0.0),
    "noisy": (0.010, 0.05),
}

# factor -> [(level label, condition, speed, depth profile)]; the base case
# (``base``, 1x, stereo) appears in every factor as the reference level.
FACTORS: Dict[str, List[Tuple[str, str, int, str]]] = {
    "rotation": [("yaw 15", "yaw15", 1, "stereo"), ("yaw 30", "base", 1, "stereo"),
                 ("yaw 90", "yaw90", 1, "stereo"), ("tilt 30", "tilt30", 1, "stereo"),
                 ("tilt 60", "tilt60", 1, "stereo"), ("yaw+tilt 45", "combined45", 1, "stereo")],
    "speed": [("1x", "base", 1, "stereo"), ("2x", "base", 2, "stereo"), ("4x", "base", 4, "stereo"),
              ("yaw 90, 1x", "yaw90", 1, "stereo"), ("yaw 90, 2x", "yaw90", 2, "stereo"),
              ("yaw 90, 4x", "yaw90", 4, "stereo")],
    "occlusion": [("none", "occ_none", 1, "stereo"), ("gripper", "base", 1, "stereo"),
                  ("gripper + hand (partial)", "occ_hand", 1, "stereo"),
                  ("gripper + hand (heavy)", "occ_full", 1, "stereo")],
    "depth": [("clean", "base", 1, "clean"), ("stereo 3 mm", "base", 1, "stereo"),
              ("noisy 10 mm + holes", "base", 1, "noisy")],
    "object": [("mug", "base", 1, "stereo"), ("mustard bottle", "obj_mustard", 1, "stereo"),
               ("cereal box", "obj_cereal", 1, "stereo")],
    "multi-object": [("one moving", "base", 1, "stereo"), ("two moving", "multi", 1, "stereo")],
}


@dataclass(frozen=True)
class Case:
    """One tracking run: a rendered sequence plus load-time settings."""
    condition: str
    seed: int
    speed: int = 1
    depth: str = "stereo"

    @property
    def sequence(self) -> str:
        return f"{self.condition}_s{self.seed}"

    @property
    def name(self) -> str:
        return f"{self.sequence}__{self.speed}x_{self.depth}"


def benchmark_cases(factors: Optional[Sequence[str]] = None, seeds: Optional[Sequence[int]] = None) -> List[Case]:
    """Unique cases needed by the requested factors (all by default)."""
    seeds = list(range(N_SEEDS)) if seeds is None else list(seeds)
    out: List[Case] = []
    for f in (factors or list(FACTORS)):
        for _, cond, speed, depth in FACTORS[f]:
            for s in seeds:
                c = Case(cond, s, speed, depth)
                if c not in out:
                    out.append(c)
    return out


def rendered_sequences(seeds: Optional[Sequence[int]] = None) -> List[str]:
    seeds = list(range(N_SEEDS)) if seeds is None else list(seeds)
    return [f"{c}_s{s}" for c in CONDITIONS for s in seeds]


@dataclass
class BenchmarkConfig:
    n_views: int = 35
    capture_width: int = 640
    capture_height: int = 360
    capture_fov_deg: float = 62.0
    capture_radius: float = 0.68
    capture_elevation_deg: Tuple[float, float] = (30.0, 75.0)
    capture_center: Tuple[float, float, float] = (0.02, 0.02, 0.06)
    track_width: int = 889
    track_height: int = 500
    track_fx: float = 700.0
    track_eye: Tuple[float, float, float] = (0.24, -0.70, 0.54)
    track_target: Tuple[float, float, float] = (0.02, 0.02, 0.10)
    n_frames: int = 60
    fps: float = 5.0
    clearance_z: Tuple[float, float] = (0.09, 0.12)
    """Carry height range (m, of the object's base); sampled per trajectory."""
    gripper_scale: float = 0.7
    """The stock gripper model is sized for drills; 0.7 suits 6-15 cm objects."""
    hand_scale: float = 1.0
    depth_noise: float = 0.003
    """Scan depth noise: std (m) at 0.7 m range, growing linearly with depth."""
    seed_stride: int = 3
    seed_cap_object: int = 2500
    seed_cap_floor: int = 4000
    seed_cap_table: int = 2500
    seed: int = 0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _object_by_id(objects, obj_id):
    return next(o for o in objects if o.obj_id == obj_id)


def _object_height(obj) -> float:
    return max(p.offset[2] + p.scale[2] for p in obj.prims)


def _object_center(obj) -> np.ndarray:
    """Pivot for the GT motion: the object's bounding-box centre at scan time."""
    h = _object_height(obj)
    return np.array([obj.position[0], obj.position[1], obj.position[2] + 0.5 * h])


def _noisy_depth(depth: torch.Tensor, std: float, gen: torch.Generator) -> torch.Tensor:
    noise = torch.randn(depth.shape, generator=gen, device="cpu").to(depth.device)
    noise = noise * std * (depth / 0.7).clamp(min=0.2)
    return torch.where(depth > 0, depth + noise, depth)


def degrade_depth(depth: np.ndarray, profile: str, key: str) -> np.ndarray:
    """Deterministic depth degradation (same ``key`` -> same noise): Gaussian
    noise growing linearly with range, plus blob-shaped missing regions."""
    std, holes = DEPTH_PROFILES[profile]
    if std == 0.0 and holes == 0.0:
        return depth
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    out = depth.copy()
    valid = depth > 0
    if std > 0:
        out = out + rng.standard_normal(depth.shape).astype(np.float32) * std * np.clip(depth / 0.7, 0.2, None)
    if holes > 0:
        h, w = depth.shape
        low = rng.standard_normal((max(h // 24, 2), max(w // 24, 2))).astype(np.float32)
        field_ = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
        out[field_ > np.quantile(field_, 1.0 - holes)] = 0.0
    out[~valid] = 0.0
    return np.clip(out, 0.0, None)


def tracking_camera(cfg: BenchmarkConfig, device="cpu") -> PinholeCamera:
    c2w = torch.from_numpy(look_at(np.array(cfg.track_eye), np.array(cfg.track_target))).float().to(device)
    return PinholeCamera(c2w, cfg.track_fx, cfg.track_fx, cfg.track_width / 2.0, cfg.track_height / 2.0,
                         cfg.track_width, cfg.track_height)


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def _rot_angle_deg(R: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip((np.trace(R) - 1) / 2, -1, 1))))


# --------------------------------------------------------------------------- #
# Trajectory
# --------------------------------------------------------------------------- #


@dataclass
class TrajectoryParams:
    place_xy: Tuple[float, float]
    clearance: float
    wobble: float           # lateral path offset amplitude (m)
    rot_sign: float
    tilt_axis_deg: float    # direction of the horizontal tilt axis
    t_offset: float = 0.0   # phase delay (fraction of the sequence)


def sample_params(rng: np.random.Generator, target: str, cfg: BenchmarkConfig, t_offset: float = 0.0) -> TrajectoryParams:
    (x0, x1), (y0, y1) = PLACE_REGION[target]
    return TrajectoryParams(
        place_xy=(float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1))),
        clearance=float(rng.uniform(*cfg.clearance_z)),
        wobble=float(rng.uniform(-0.04, 0.04)),
        rot_sign=float(rng.choice([-1.0, 1.0])),
        tilt_axis_deg=float(rng.uniform(0.0, 180.0)),
        t_offset=t_offset,
    )


def plan_pick_and_place(objects, obj_id: int, cond: Condition, prm: TrajectoryParams,
                        cfg: BenchmarkConfig) -> Tuple[List[Dict[str, object]], np.ndarray]:
    """Phases (fraction of the object's active window):
        0.00-0.12 approach   gripper descends onto the object (object static)
        0.12-0.25 lift       to carry height
        0.25-0.75 carry      transit to the table + rotation profile
        0.75-0.88 place      descend onto the table top (upright, final yaw)
        0.88-1.00 release    gripper rises out of view (object static)
    Rotation profile over the carry (s in [0, 1]):
        yaw       yaw = sign * rot * s
        tilt      tilt = rot * sin(pi s) about a random horizontal axis
        combined  both, with the tilt peak equal to the yaw total
    """
    obj = _object_by_id(objects, obj_id)
    start = np.array(obj.position, dtype=np.float64)
    place = np.array([prm.place_xy[0], prm.place_xy[1], TABLE_TOP_Z])
    c_scan = _object_center(obj)
    h = _object_height(obj)
    grip_pos0 = np.array([start[0], start[1], start[2] + h + 0.008])
    approach_h = 0.18
    path = place[:2] - start[:2]
    lateral = np.array([-path[1], path[0], 0.0]) / max(np.linalg.norm(path), 1e-9)
    tilt_axis = np.array([math.cos(math.radians(prm.tilt_axis_deg)), math.sin(math.radians(prm.tilt_axis_deg)), 0.0])
    rot = math.radians(cond.rot_deg)
    final_yaw = prm.rot_sign * rot if cond.rot_axis in ("yaw", "combined") else 0.0

    plan = []
    for f in range(cfg.n_frames):
        u_glob = f / max(cfg.n_frames - 1, 1)
        u = min(max((u_glob - prm.t_offset) / (1.0 - prm.t_offset), 0.0), 1.0)
        pos, yaw, tilt, phase = start.copy(), 0.0, 0.0, ""
        grip_offset = np.zeros(3)
        if u < 0.12:
            phase = "approach" if u_glob >= prm.t_offset else "wait"
            grip_offset[2] = approach_h * (1.0 - smoothstep(u / 0.12))
        elif u < 0.25:
            phase = "lift"
            pos[2] = start[2] + (prm.clearance - start[2]) * smoothstep((u - 0.12) / 0.13)
        elif u < 0.75:
            phase = "carry"
            s = smoothstep((u - 0.25) / 0.50)
            pos = start + (place - start) * s + lateral * prm.wobble * math.sin(math.pi * s)
            pos[2] = prm.clearance + 0.012 * math.sin(math.pi * s)
            if cond.rot_axis in ("yaw", "combined"):
                yaw = prm.rot_sign * rot * s
            if cond.rot_axis in ("tilt", "combined"):
                tilt = rot * math.sin(math.pi * s)
        elif u < 0.88:
            phase = "place"
            s = smoothstep((u - 0.75) / 0.13)
            pos = place.copy()
            pos[2] = prm.clearance + (place[2] - prm.clearance) * s
            yaw = final_yaw
        else:
            phase = "release"
            pos = place.copy()
            yaw = final_yaw
            grip_offset[2] = approach_h * smoothstep((u - 0.88) / 0.12)

        R = _axis_angle(tilt_axis, tilt) @ euler_to_R(0.0, 0.0, yaw)
        t = pos - start
        delta = se3(R, c_scan - R @ c_scan + t)            # rotate about c_scan, then translate
        grip = se3(np.eye(3), grip_offset) @ delta          # gripper rigidly holds the object
        plan.append({
            "frame": f, "phase": phase, "rot_deg": _rot_angle_deg(R), "yaw_deg": math.degrees(yaw),
            "tilt_deg": math.degrees(tilt), "translation_cm": float(np.linalg.norm(t) * 100.0),
            "center": R @ c_scan + c_scan - R @ c_scan + t,
            "delta": delta, "gripper_delta": grip,
            "gripper_visible": phase != "release" or grip_offset[2] < approach_h * 0.98,
        })
    return plan, grip_pos0


def _hand_pose(center: np.ndarray, eye: np.ndarray, u: float, lift: float,
               window=(0.33, 0.67), sweep: float = 0.14):
    """A hand reaching down from above (fingers down, palm facing the camera),
    30% of the way from the object to the camera, sweeping sideways across
    the line of sight while ``u`` is inside ``window``; ``lift`` raises it
    above the line of sight (less coverage). Returns None outside the window."""
    if not (window[0] <= u <= window[1]):
        return None
    k = (u - window[0]) / (window[1] - window[0])
    view = eye - center
    z = view / np.linalg.norm(view)
    up = np.array([0.0, 0.0, 1.0])
    up = up - up.dot(z) * z
    up /= np.linalg.norm(up)
    y = -up                                   # hand-local +y (fingers) points down
    x = np.cross(y, z)
    p = center + 0.3 * view + up * lift + x * sweep * (2.0 * k - 1.0)
    return se3(np.stack([x, y, z], 1), p)


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def _write_png_depth_mm(path: str, depth: np.ndarray) -> None:
    cv2.imwrite(path, np.clip(depth * 1000.0, 0, 65535).astype(np.uint16))


def generate_capture(out_dir: str, cfg: BenchmarkConfig, device: str = "cuda:0", verbose: bool = True) -> None:
    capture_dir = os.path.join(out_dir, "capture")
    os.makedirs(capture_dir, exist_ok=True)
    objects = build_pick_place_scene()
    scene = FlatScene(objects, device=device)
    cams = hemisphere_trajectory(
        n_views=cfg.n_views, radius=cfg.capture_radius, elevation_deg=cfg.capture_elevation_deg,
        center=cfg.capture_center, width=cfg.capture_width, height=cfg.capture_height,
        fov_deg=cfg.capture_fov_deg, device=device,
    )
    gen = torch.Generator().manual_seed(cfg.seed)
    rgbs, depths, masks, pts, cols, ids = [], [], [], [], [], []
    t0 = time.time()
    for i, cam in enumerate(cams):
        out = render_photoreal_scene(scene, cam, cast_shadows=True)
        d = _noisy_depth(out["depth"], cfg.depth_noise, gen)
        rgbs.append((out["rgb"].cpu().numpy() * 255).astype(np.uint8))
        depths.append(d.cpu().numpy().astype(np.float32))
        masks.append(out["obj_id"].cpu().numpy().astype(np.uint8))
        s = cfg.seed_stride
        p = deproject(d, cam).cpu().numpy()[::s, ::s].reshape(-1, 3)
        dd = d.cpu().numpy()[::s, ::s].reshape(-1)
        v = (dd > 0.15) & (dd < 1.5)
        pts.append(p[v])
        cols.append(out["rgb"].cpu().numpy()[::s, ::s].reshape(-1, 3)[v])
        ids.append(out["obj_id"].cpu().numpy()[::s, ::s].reshape(-1)[v])
    pts, cols, ids = np.concatenate(pts), np.concatenate(cols), np.concatenate(ids)
    pts, cols, keep = dbscan_filter(pts, cols, eps=0.012, min_samples=8, return_mask=True)
    ids = ids[keep]

    rng = np.random.default_rng(cfg.seed)
    sel = []
    for oid in np.unique(ids):
        idx = np.where(ids == oid)[0]
        cap = cfg.seed_cap_floor if oid == FLOOR_ID else cfg.seed_cap_table if oid == TABLE_ID else cfg.seed_cap_object
        if len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        sel.append(idx)
    sel = np.sort(np.concatenate(sel))
    pts, cols, ids = pts[sel], cols[sel], ids[sel]

    np.save(os.path.join(capture_dir, "images.npy"), np.stack(rgbs))
    np.save(os.path.join(capture_dir, "depths.npy"), np.stack(depths))
    np.save(os.path.join(capture_dir, "obj_masks.npy"), np.stack(masks))
    np.save(os.path.join(capture_dir, "seed_obj_ids.npy"), ids.astype(np.int64))
    save_pointcloud_ply(os.path.join(capture_dir, "fused_pointcloud.ply"), pts, cols)
    with open(os.path.join(capture_dir, "cameras.json"), "w") as f:
        json.dump([{"view_idx": i, "c2w": c.c2w.cpu().tolist(), "fx": c.fx, "fy": c.fy, "cx": c.cx,
                    "cy": c.cy, "width": c.width, "height": c.height} for i, c in enumerate(cams)], f, indent=1)
    counts = {int(k): int((ids == k).sum()) for k in np.unique(ids)}
    with open(os.path.join(capture_dir, "capture_metadata.json"), "w") as f:
        json.dump({"n_views": cfg.n_views, "seed_points_per_object": counts,
                   "object_names": OBJECT_NAMES, "depth_noise_m_at_0.7m": cfg.depth_noise}, f, indent=1)
    if verbose:
        print(f"[capture] {cfg.n_views} views in {time.time() - t0:.1f}s; seed points per id: {counts}")


def generate_sequence(out_dir: str, condition: str, seed: int, cfg: BenchmarkConfig,
                      device: str = "cuda:0", verbose: bool = True) -> str:
    cond = CONDITIONS[condition]
    name = f"{condition}_s{seed}"
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    objects = build_pick_place_scene()
    cam = tracking_camera(cfg, device)
    eye = np.array(cfg.track_eye)

    movers = [(TARGETS[cond.target], sample_params(rng, cond.target, cfg), GRIPPER_ID)]
    if cond.second:
        movers.append((TARGETS[cond.second], sample_params(rng, cond.second, cfg, t_offset=0.08), GRIPPER2_ID))
    plans, grippers = [], []
    for oid, prm, gid in movers:
        plan, gpos = plan_pick_and_place(objects, oid, cond, prm, cfg)
        plans.append(plan)
        g = make_robot_gripper(scale=cfg.gripper_scale)
        g.position, g.obj_id, g.movable = tuple(gpos), gid, True
        grippers.append(g)
    hand = make_hand(scale=cfg.hand_scale)
    hand.position, hand.obj_id, hand.movable = (0.0, 0.0, 0.0), HAND_ID, True
    tgt = movers[0][0]

    sc_dir = os.path.join(out_dir, name)
    for sub in ("rgb", "depth_mm", "masks"):
        os.makedirs(os.path.join(sc_dir, sub), exist_ok=True)
    movable_ids = [o.obj_id for o in objects if o.obj_id not in (FLOOR_ID, TABLE_ID)]
    frames_meta, previews = [], []
    t0 = time.time()
    for f in range(cfg.n_frames):
        obj_deltas = {oid: plans[i][f]["delta"] for i, (oid, _, _) in enumerate(movers)}
        objs, deltas = list(objects), dict(obj_deltas)
        if cond.occlusion in ("gripper", "hand"):
            for i, g in enumerate(grippers):
                if plans[i][f]["gripper_visible"]:
                    objs.append(g)
                    deltas[g.obj_id] = plans[i][f]["gripper_delta"]
        hand_T = (_hand_pose(plans[0][f]["center"], eye, f / max(cfg.n_frames - 1, 1), cond.hand_lift)
                  if cond.occlusion == "hand" else None)
        if hand_T is not None:
            objs.append(hand)
            deltas[HAND_ID] = hand_T
        out = render_photoreal_scene(FlatScene(objs, device=device).clone_with_transforms(deltas), cam, cast_shadows=True)
        ids = out["obj_id"]
        occl = {}
        if len(objs) > len(objects):
            ref = render_photoreal_scene(FlatScene(list(objects), device=device).clone_with_transforms(obj_deltas),
                                         cam, cast_shadows=False)["obj_id"]
            for oid, _, _ in movers:
                occl[oid] = 1.0 - int((ids == oid).sum()) / max(int((ref == oid).sum()), 1)
        else:
            occl = {oid: 0.0 for oid, _, _ in movers}
        d = out["depth"].cpu().numpy()
        rgb = (out["rgb"].cpu().numpy() * 255).astype(np.uint8)
        fname = f"frame_{f:04d}.png"
        cv2.imwrite(os.path.join(sc_dir, "rgb", fname), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        _write_png_depth_mm(os.path.join(sc_dir, "depth_mm", fname), d)
        cv2.imwrite(os.path.join(sc_dir, "masks", fname), ids.cpu().numpy().astype(np.uint8))
        p0 = plans[0][f]
        frames_meta.append({
            "frame": f, "timestamp_sec": f / cfg.fps, "phase": p0["phase"],
            "rot_deg": p0["rot_deg"], "yaw_deg": p0["yaw_deg"], "tilt_deg": p0["tilt_deg"],
            "translation_cm": p0["translation_cm"], "hand_visible": hand_T is not None,
            "occlusion_frac": float(occl[tgt]), "occlusion_by_obj": {str(k): float(v) for k, v in occl.items()},
            "gt_deltas": {str(oid): obj_deltas.get(oid, np.eye(4)).tolist() for oid in movable_ids},
        })
        previews.append(_annotate(rgb, f, cfg, p0, occl[tgt], name))
    meta = {
        "name": name, "condition": condition, "seed": seed, "condition_spec": cond.__dict__,
        "target_object_id": tgt, "target_object": OBJECT_NAMES[tgt],
        "moving_object_ids": [oid for oid, _, _ in movers],
        "trajectories": [{"obj_id": oid, **prm.__dict__} for oid, prm, _ in movers],
        "fps": cfg.fps, "n_frames": cfg.n_frames,
        "depth_format": "uint16 PNG, millimetres, noise-free (noise is added at load time)",
        "camera": {"c2w": cam.c2w.cpu().tolist(), "fx": cam.fx, "fy": cam.fy, "cx": cam.cx, "cy": cam.cy,
                   "width": cam.width, "height": cam.height},
        "frames": frames_meta,
    }
    with open(os.path.join(sc_dir, "tracking_metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    try:
        import imageio.v2 as imageio

        imageio.mimsave(os.path.join(sc_dir, "preview.mp4"), [im[: im.shape[0] // 2 * 2, : im.shape[1] // 2 * 2]
                                                              for im in previews], fps=cfg.fps)
    except Exception as exc:  # noqa: BLE001
        print(f"  (preview video skipped: {exc})")
    if verbose:
        oc = [m["occlusion_frac"] for m in frames_meta]
        print(f"[track] {name}: {cfg.n_frames} frames in {time.time() - t0:.1f}s, "
              f"max target occlusion {max(oc):.0%}, max rotation {max(m['rot_deg'] for m in frames_meta):.0f} deg")
    return sc_dir


def _annotate(rgb, f, cfg, p, occ, name):
    img = rgb.copy()
    txt = (f"{name} | frame {f:02d} t={f / cfg.fps:4.1f}s | {p['phase']:8s} | "
           f"rot {p['rot_deg']:5.1f} deg | occl {occ:3.0%}")
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, txt, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def generate_benchmark(
    out_dir: str = "datasets/synthetic_scenes/pick_place_floor_table",
    sequences: Optional[Sequence[str]] = None,
    cfg: Optional[BenchmarkConfig] = None,
    device: str = "cuda:0",
    capture: Optional[bool] = None,
    overwrite: bool = False,
) -> str:
    """Renders the capture (if missing or ``capture=True``) and every sequence
    that is not on disk yet (all of ``rendered_sequences()`` by default)."""
    cfg = cfg or BenchmarkConfig()
    os.makedirs(out_dir, exist_ok=True)
    have_capture = os.path.exists(os.path.join(out_dir, "capture", "images.npy"))
    if capture or (capture is None and not have_capture):
        generate_capture(out_dir, cfg, device)
    for name in (sequences or rendered_sequences()):
        if not overwrite and os.path.exists(os.path.join(out_dir, name, "tracking_metadata.json")):
            continue
        cond, seed = name.rsplit("_s", 1)
        generate_sequence(out_dir, cond, int(seed), cfg, device)
    return out_dir


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _read_ply(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path) as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == "end_header") + 1
    arr = np.array([l.split() for l in lines[start:] if l.strip()], dtype=np.float64)
    return arr[:, :3].astype(np.float32), (arr[:, 3:6] / 255.0).astype(np.float32)


def load_capture(dataset_dir: str, device: str = "cuda:0") -> CaptureData:
    cdir = os.path.join(dataset_dir, "capture")
    with open(os.path.join(cdir, "cameras.json")) as f:
        cams = [PinholeCamera(torch.tensor(c["c2w"], dtype=torch.float32, device=device), c["fx"], c["fy"],
                              c["cx"], c["cy"], c["width"], c["height"]) for c in json.load(f)]
    pts, cols = _read_ply(os.path.join(cdir, "fused_pointcloud.ply"))
    cap = CaptureData(
        scene_objects=build_pick_place_scene(),
        images=torch.from_numpy(np.load(os.path.join(cdir, "images.npy"))).float() / 255.0,
        depths=torch.from_numpy(np.load(os.path.join(cdir, "depths.npy"))).float(),
        obj_masks=torch.from_numpy(np.load(os.path.join(cdir, "obj_masks.npy")).astype(np.int64)),
        cameras=cams,
        seed_points=torch.from_numpy(pts),
        seed_colors=torch.from_numpy(cols),
        metadata={"dataset_dir": dataset_dir},
    )
    cap.seed_obj_ids = torch.from_numpy(np.load(os.path.join(cdir, "seed_obj_ids.npy"))).long()
    return cap



@dataclass
class BenchmarkFrame:
    frame_idx: int
    timestamp_sec: float
    rgb: torch.Tensor           # (H, W, 3) in [0, 1]
    depth: torch.Tensor         # (H, W) metres, 0 = missing
    obj_id: torch.Tensor        # (H, W)
    gt_delta: Dict[int, np.ndarray]
    phase: str
    rot_deg: float
    occlusion_frac: float

    @property
    def occluded(self) -> bool:
        return self.occlusion_frac > 0.05


@dataclass
class BenchmarkSequence:
    name: str
    target_obj_id: int
    frames: List[BenchmarkFrame]
    camera: PinholeCamera
    fps: float
    condition: str = ""
    seed: int = 0
    speed: int = 1
    depth_profile: str = "stereo"
    moving_obj_ids: List[int] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)


def load_case(dataset_dir: str, case: Case, device: str = "cpu") -> BenchmarkSequence:
    return load_sequence(dataset_dir, case.sequence, speed=case.speed, depth=case.depth, device=device)


def load_sequence(dataset_dir: str, name: str, speed: int = 1, depth: str = "stereo",
                  device: str = "cpu") -> BenchmarkSequence:
    """Loads every ``speed``-th frame (the camera rate is unchanged, so ``fps``
    drops by ``speed``) and degrades depth with profile ``depth``."""
    sdir = os.path.join(dataset_dir, name)
    with open(os.path.join(sdir, "tracking_metadata.json")) as f:
        meta = json.load(f)
    c = meta["camera"]
    cam = PinholeCamera(torch.tensor(c["c2w"], dtype=torch.float32, device=device), c["fx"], c["fy"],
                        c["cx"], c["cy"], c["width"], c["height"])
    frames = []
    for fm in meta["frames"][::speed]:
        n = f"frame_{fm['frame']:04d}.png"
        rgb = cv2.cvtColor(cv2.imread(os.path.join(sdir, "rgb", n)), cv2.COLOR_BGR2RGB)
        d = cv2.imread(os.path.join(sdir, "depth_mm", n), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        d = degrade_depth(d, depth, f"{name}/{fm['frame']}/{depth}")
        mask = cv2.imread(os.path.join(sdir, "masks", n), cv2.IMREAD_UNCHANGED).astype(np.int64)
        frames.append(BenchmarkFrame(
            frame_idx=fm["frame"], timestamp_sec=fm["timestamp_sec"],
            rgb=torch.from_numpy(rgb).float() / 255.0, depth=torch.from_numpy(d),
            obj_id=torch.from_numpy(mask),
            gt_delta={int(k): np.array(v, dtype=np.float64) for k, v in fm["gt_deltas"].items()},
            phase=fm["phase"], rot_deg=fm["rot_deg"], occlusion_frac=fm["occlusion_frac"],
        ))
    return BenchmarkSequence(
        name=name, target_obj_id=meta["target_object_id"], frames=frames, camera=cam,
        fps=meta["fps"] / speed, condition=meta["condition"], seed=meta["seed"], speed=speed,
        depth_profile=depth, moving_obj_ids=meta["moving_object_ids"], meta=meta,
    )
