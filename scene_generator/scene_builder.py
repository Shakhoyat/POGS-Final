"""
Scene composition and vectorized closed-form raycasting engine.
Constructs multi-object tabletop scenes and renders hole-free RGB, depth,
and instance masks from arbitrary camera poses.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch

from .primitives import Primitive, BOX, ELLIPSOID, CYLINDER, euler_to_R
from .objects import (
    SceneObject,
    clothes_iron,
    power_drill,
    shoe,
    shoebox,
    jigsaw,
    wooden_shelf,
)
from common.geometry import PinholeCamera


def build_scene(
    name: str = "tabletop_drill_iron",
    table_color: Tuple[float, float, float] = (0.35, 0.38, 0.42),
    table_size: Tuple[float, float, float] = (0.50, 0.40, 0.02),
) -> List[SceneObject]:
    """Factory creating tabletop configurations from the paper."""
    table = SceneObject(
        name="table",
        prims=[Primitive(BOX, table_size, (0.0, 0.0, -table_size[2]),
                         color=table_color, texture_freq=12.0, texture_amp=0.08)],
        movable=False,
        obj_id=0,
    )

    if name in ("tabletop_drill_iron", "default"):
        return [
            table,
            SceneObject(name="drill", prims=power_drill(), position=(-0.10, -0.04, 0.0), yaw=0.25, obj_id=1),
            SceneObject(name="iron", prims=clothes_iron(), position=(0.10, 0.02, 0.0), yaw=-0.50, obj_id=2),
        ]
    elif name == "tabletop_shoe_box":
        return [
            table,
            SceneObject(name="shoe", prims=shoe(), position=(-0.10, -0.02, 0.0), yaw=0.30, obj_id=1),
            SceneObject(name="shoebox", prims=shoebox(), position=(0.11, 0.02, 0.0), yaw=-0.20, obj_id=2),
        ]
    elif name == "tabletop_jigsaw_shelf":
        return [
            table,
            SceneObject(name="jigsaw", prims=jigsaw(), position=(-0.08, -0.03, 0.0), yaw=0.40, obj_id=1),
            SceneObject(name="shelf", prims=wooden_shelf(), position=(0.12, 0.05, 0.0), yaw=-0.10, obj_id=2),
        ]
    else:
        raise ValueError(f"Unknown scene preset '{name}'. Choose from: tabletop_drill_iron, tabletop_shoe_box, tabletop_jigsaw_shelf")


class FlatScene:
    """Vectorised collection of primitive instances for raycasting."""

    def __init__(self, objects: Sequence[SceneObject], device: str = "cpu"):
        self.objects = list(objects)
        self.device = device
        kinds, scales, Rs, ts, colors, targs = [], [], [], [], [], []
        obj_ids = []

        for obj in objects:
            c, s = math.cos(obj.yaw), math.sin(obj.yaw)
            R_obj = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            t_obj = np.asarray(obj.position, dtype=np.float64)

            for prim in obj.prims:
                R_prim = euler_to_R(*prim.rot_euler)
                t_prim = np.asarray(prim.offset, dtype=np.float64)
                R_world = R_obj @ R_prim
                t_world = t_obj + R_obj @ t_prim

                kinds.append(prim.kind)
                scales.append(prim.scale)
                Rs.append(R_world)
                ts.append(t_world)
                colors.append(prim.color)
                targs.append((prim.texture_freq, prim.texture_amp, *prim.texture_phase))
                obj_ids.append(obj.obj_id)

        self.kind = torch.tensor(kinds, dtype=torch.long, device=device)
        self.scale = torch.tensor(np.array(scales), dtype=torch.float32, device=device)
        self.R = torch.tensor(np.array(Rs), dtype=torch.float32, device=device)
        self.t = torch.tensor(np.array(ts), dtype=torch.float32, device=device)
        self.color = torch.tensor(np.array(colors), dtype=torch.float32, device=device)
        self.tex_args = torch.tensor(np.array(targs), dtype=torch.float32, device=device)
        self.obj_id = torch.tensor(obj_ids, dtype=torch.long, device=device)

    def n_prims(self) -> int:
        return self.kind.shape[0]

    def clone_with_transforms(self, object_deltas: Dict[int, np.ndarray]) -> "FlatScene":
        """Return a copy with per-object rigid deltas applied."""
        out = FlatScene(self.objects, device=self.device)
        R_list = self.R.clone()
        t_list = self.t.clone()

        for obj_id, delta in object_deltas.items():
            mask = self.obj_id == obj_id
            if not mask.any():
                continue
            R_d = torch.as_tensor(delta[:3, :3], dtype=torch.float32, device=self.device)
            t_d = torch.as_tensor(delta[:3, 3], dtype=torch.float32, device=self.device)
            R_list[mask] = R_d @ R_list[mask]
            t_list[mask] = (R_d @ t_list[mask].unsqueeze(-1)).squeeze(-1) + t_d

        out.R = R_list
        out.t = t_list
        return out


def _ray_dirs(camera: PinholeCamera) -> torch.Tensor:
    H, W = camera.height, camera.width
    dev = camera.c2w.device
    v, u = torch.meshgrid(
        torch.arange(H, device=dev, dtype=torch.float32) + 0.5,
        torch.arange(W, device=dev, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    dirs_cam = torch.stack([(u - camera.cx) / camera.fx,
                            (v - camera.cy) / camera.fy,
                            torch.ones_like(u)], dim=-1)
    dirs_cam = dirs_cam / torch.linalg.norm(dirs_cam, dim=-1, keepdim=True)
    c2w_rot = camera.c2w[:3, :3]
    dirs_world = (c2w_rot @ dirs_cam.reshape(-1, 3).T).T.reshape(H, W, 3)
    return dirs_world


def _intersect_batch(
    o: torch.Tensor,
    d: torch.Tensor,
    kind: torch.Tensor,
    scale: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed-form ray-primitive intersection."""
    R_t = R.permute(0, 2, 1)
    inv_s = 1.0 / scale
    o_rel = o.unsqueeze(1) - t.unsqueeze(0)
    o_loc = (torch.einsum("npk,pjk->npj", o_rel, R_t) * inv_s.unsqueeze(0))
    d_loc = (torch.einsum("nk,pjk->npj", d, R_t) * inv_s.unsqueeze(0))

    N, P, _ = o_loc.shape
    LARGE = 1e6
    t_hits = torch.full((N, P), LARGE, dtype=torch.float32, device=o.device)
    normal_loc = torch.zeros((N, P, 3), dtype=torch.float32, device=o.device)

    is_box = kind == BOX
    is_ell = kind == ELLIPSOID
    is_cyl = kind == CYLINDER

    # 1) Ellipsoid: |o + t*d|^2 = 1
    if is_ell.any():
        idx = is_ell.nonzero().squeeze(-1)
        oe, de = o_loc[:, idx], d_loc[:, idx]
        a = (de * de).sum(-1)
        b = 2.0 * (oe * de).sum(-1)
        c = (oe * oe).sum(-1) - 1.0
        disc = b * b - 4 * a * c
        real = disc >= 0
        sqrt_disc = torch.where(real, torch.sqrt(disc.clamp(min=0)), torch.zeros_like(disc))
        t0 = (-b - sqrt_disc) / (2 * a + 1e-9)
        t1 = (-b + sqrt_disc) / (2 * a + 1e-9)
        hit_e = real & (t1 > 1e-4)
        t_ell = torch.where(t0 > 1e-4, t0, t1)
        t_ell = torch.where(hit_e, t_ell, torch.full_like(t_ell, LARGE))
        t_hits[:, idx] = t_ell
        hit_pt = oe + t_ell.unsqueeze(-1) * de
        normal_loc[:, idx] = hit_pt

    # 2) Box: axis-aligned slab test on [-1, 1]^3
    if is_box.any():
        idx = is_box.nonzero().squeeze(-1)
        ob, db = o_loc[:, idx], d_loc[:, idx]
        db_safe = torch.where(db.abs() < 1e-6, torch.sign(db) * 1e-6 + 1e-6, db)
        t_lo = (-1.0 - ob) / db_safe
        t_hi = (1.0 - ob) / db_safe
        t_near = torch.minimum(t_lo, t_hi).max(dim=-1).values
        t_far = torch.maximum(t_lo, t_hi).min(dim=-1).values
        hit_b = (t_near <= t_far) & (t_far > 1e-4)
        t_box = torch.where(t_near > 1e-4, t_near, t_far)
        t_box = torch.where(hit_b, t_box, torch.full_like(t_box, LARGE))
        t_hits[:, idx] = t_box
        hit_pt = ob + t_box.unsqueeze(-1) * db
        eps = 1e-3
        nx = (hit_pt[..., 0].abs() >= 1.0 - eps).float() * hit_pt[..., 0].sign()
        ny = (hit_pt[..., 1].abs() >= 1.0 - eps).float() * hit_pt[..., 1].sign()
        nz = (hit_pt[..., 2].abs() >= 1.0 - eps).float() * hit_pt[..., 2].sign()
        normal_loc[:, idx] = torch.stack([nx, ny, nz], dim=-1)

    # 3) Cylinder: x^2 + y^2 = 1, z in [-1, 1]
    if is_cyl.any():
        idx = is_cyl.nonzero().squeeze(-1)
        oc, dc = o_loc[:, idx], d_loc[:, idx]
        a = dc[..., 0] ** 2 + dc[..., 1] ** 2
        b = 2.0 * (oc[..., 0] * dc[..., 0] + oc[..., 1] * dc[..., 1])
        c = oc[..., 0] ** 2 + oc[..., 1] ** 2 - 1.0
        disc = b * b - 4 * a * c
        real = (disc >= 0) & (a > 1e-8)
        sqrt_disc = torch.where(real, torch.sqrt(disc.clamp(min=0)), torch.zeros_like(disc))
        t0 = (-b - sqrt_disc) / (2 * a + 1e-9)
        t1 = (-b + sqrt_disc) / (2 * a + 1e-9)
        z0 = oc[..., 2] + t0 * dc[..., 2]
        z1 = oc[..., 2] + t1 * dc[..., 2]
        h0 = real & (t0 > 1e-4) & (z0.abs() <= 1.0)
        h1 = real & (t1 > 1e-4) & (z1.abs() <= 1.0)
        t_side = torch.where(h0, t0, torch.where(h1, t1, torch.full_like(t0, LARGE)))

        # Caps at z = +1 and z = -1
        dz = torch.where(dc[..., 2].abs() < 1e-6, torch.sign(dc[..., 2]) * 1e-6 + 1e-6, dc[..., 2])
        t_cap_top = (1.0 - oc[..., 2]) / dz
        t_cap_bot = (-1.0 - oc[..., 2]) / dz
        x_top = oc[..., 0] + t_cap_top * dc[..., 0]
        y_top = oc[..., 1] + t_cap_top * dc[..., 1]
        x_bot = oc[..., 0] + t_cap_bot * dc[..., 0]
        y_bot = oc[..., 1] + t_cap_bot * dc[..., 1]
        h_top = (t_cap_top > 1e-4) & ((x_top ** 2 + y_top ** 2) <= 1.0)
        h_bot = (t_cap_bot > 1e-4) & ((x_bot ** 2 + y_bot ** 2) <= 1.0)
        t_cap_top = torch.where(h_top, t_cap_top, torch.full_like(t_cap_top, LARGE))
        t_cap_bot = torch.where(h_bot, t_cap_bot, torch.full_like(t_cap_bot, LARGE))

        t_min = torch.minimum(t_side, torch.minimum(t_cap_top, t_cap_bot))
        t_hits[:, idx] = t_min

        hit_pt = oc + t_min.unsqueeze(-1) * dc
        is_top = (t_min == t_cap_top).unsqueeze(-1)
        is_bot = (t_min == t_cap_bot).unsqueeze(-1)
        side_n = torch.stack([hit_pt[..., 0], hit_pt[..., 1], torch.zeros_like(hit_pt[..., 0])], dim=-1)
        top_n = torch.tensor([0.0, 0.0, 1.0], device=o.device).view(1, 1, 3)
        bot_n = torch.tensor([0.0, 0.0, -1.0], device=o.device).view(1, 1, 3)
        normal_loc[:, idx] = torch.where(is_top, top_n, torch.where(is_bot, bot_n, side_n))

    # Scale back to world t: |d_world| * t_world = |d_loc * scale| * t_loc
    d_mag = torch.linalg.norm(d_loc * scale.unsqueeze(0), dim=-1).clamp(min=1e-6)
    t_world = t_hits * d_mag

    # Transform normals back to world: N_world = normalize(R @ (N_loc / scale))
    norm_w = normal_loc * inv_s.unsqueeze(0)
    norm_w = torch.einsum("npj,pjk->npk", norm_w, R)
    norm_w = norm_w / torch.linalg.norm(norm_w, dim=-1, keepdim=True).clamp(min=1e-6)

    return t_world, norm_w, t_hits < (LARGE - 10)


@torch.no_grad()
def render_scene(
    scene: FlatScene,
    camera: PinholeCamera,
    light_dir: Tuple[float, float, float] = (0.35, -0.45, 0.82),
    supersample: int = 1,
) -> Dict[str, torch.Tensor]:
    """Render hole-free RGB, depth, normal, and instance mask maps."""
    H, W = camera.height, camera.width
    dev = scene.device

    if supersample > 1:
        ss_cam = PinholeCamera(
            c2w=camera.c2w,
            fx=camera.fx * supersample,
            fy=camera.fy * supersample,
            cx=camera.cx * supersample,
            cy=camera.cy * supersample,
            width=W * supersample,
            height=H * supersample,
        )
        out = render_scene(scene, ss_cam, light_dir=light_dir, supersample=1)
        rgb = (out["rgb"].permute(2, 0, 1).unsqueeze(0))
        rgb = torch.nn.functional.avg_pool2d(rgb, supersample).squeeze(0).permute(1, 2, 0)
        depth = out["depth"].unsqueeze(0).unsqueeze(0)
        depth = -torch.nn.functional.max_pool2d(-depth, supersample).squeeze()
        obj_id = out["obj_id"][::supersample, ::supersample]
        return {"rgb": rgb, "depth": depth, "obj_id": obj_id}

    dirs = _ray_dirs(camera).reshape(-1, 3)
    origin = camera.c2w[:3, 3].unsqueeze(0).expand(dirs.shape[0], -1)

    chunk = 16384
    all_t, all_norm, all_prim = [], [], []
    for s in range(0, dirs.shape[0], chunk):
        e = min(s + chunk, dirs.shape[0])
        t_w, n_w, hit = _intersect_batch(
            origin[s:e], dirs[s:e], scene.kind, scene.scale, scene.R, scene.t
        )
        t_min, prim_idx = t_w.min(dim=-1)
        valid = t_min < 1e5
        prim_idx = torch.where(valid, prim_idx, torch.tensor(-1, device=dev))
        p_clamped = prim_idx.clamp(min=0)
        n_hit = torch.gather(n_w, 1, p_clamped.unsqueeze(1).unsqueeze(2).expand(-1, 1, 3)).squeeze(1)
        n_hit = torch.where(valid.unsqueeze(-1), n_hit, torch.zeros_like(n_hit))
        t_min = torch.where(valid, t_min, torch.zeros_like(t_min))

        all_t.append(t_min)
        all_norm.append(n_hit)
        all_prim.append(prim_idx)

    t_flat = torch.cat(all_t, dim=0)
    norm_flat = torch.cat(all_norm, dim=0)
    prim_flat = torch.cat(all_prim, dim=0)

    # Lighting
    L = torch.tensor(light_dir, dtype=torch.float32, device=dev)
    L = L / torch.linalg.norm(L)
    diffuse = (norm_flat @ L).clamp(min=0.0)
    ambient = 0.28
    shade = ambient + (1.0 - ambient) * diffuse

    # Object color & procedural textures
    valid = prim_flat >= 0
    p_safe = prim_flat.clamp(min=0)
    base_color = scene.color[p_safe]

    pts = origin + t_flat.unsqueeze(-1) * dirs
    targs = scene.tex_args[p_safe]
    freq = targs[:, 0:1]
    amp = targs[:, 1:2]
    pattern = 0.5 + 0.5 * torch.sin(pts[:, 0:1] * freq) * torch.cos(pts[:, 1:2] * freq)
    tex_color = base_color * (1.0 + amp * (pattern - 0.5))

    rgb = tex_color * shade.unsqueeze(-1)
    rgb = torch.where(valid.unsqueeze(-1), rgb, torch.tensor([0.94, 0.94, 0.95], device=dev))
    rgb = rgb.clamp(0.0, 1.0).reshape(H, W, 3)

    # Metric depth: z-component along optical axis
    c2w_z = camera.c2w[:3, 2]
    cos_angle = dirs @ c2w_z
    depth = (t_flat * cos_angle).reshape(H, W)
    depth = torch.where(depth > 0, depth, torch.zeros_like(depth))

    # Instance object mask
    obj_id = torch.where(valid, scene.obj_id[p_safe], torch.tensor(0, device=dev)).reshape(H, W)

    return {"rgb": rgb, "depth": depth, "normal": norm_flat.reshape(H, W, 3), "obj_id": obj_id}


def deproject(depth: torch.Tensor, camera: PinholeCamera) -> torch.Tensor:
    """Unproject a (H, W) depth map into (H, W, 3) world coordinates."""
    H, W = depth.shape
    dev = depth.device
    v, u = torch.meshgrid(
        torch.arange(H, device=dev, dtype=torch.float32),
        torch.arange(W, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    z = depth
    x = (u - camera.cx) * z / camera.fx
    y = (v - camera.cy) * z / camera.fy
    cam_pts = torch.stack([x, y, z], dim=-1)

    c2w_rot = camera.c2w[:3, :3]
    c2w_t = camera.c2w[:3, 3]
    world_pts = (c2w_rot @ cam_pts.reshape(-1, 3).T).T + c2w_t
    return world_pts.reshape(H, W, 3)
