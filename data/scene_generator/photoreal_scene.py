"""
Photorealistic Tabletop Scene Engine (v3).

Features:
1. Closed-form ray-traced cast shadows and ambient contact occlusion.
2. Dual-light Blinn-Phong shading (Key Light + Soft Fill + Specular Highlights).
3. Realistic Franka/Robotiq industrial parallel-jaw robot gripper.
4. Real-world objects: Industrial Cordless Drill, Domestic Steam Iron, Ceramic Mug, YCB Mustard Bottle, Soup Can.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch

from .materials import specular_params, surface_color
from .primitives import Primitive, BOX, ELLIPSOID, CYLINDER, euler_to_R
from .objects import SceneObject
from model.pogs.geometry import PinholeCamera


# --------------------------------------------------------------------------- #
# Realistic Robot Gripper (Robotiq 2F-85 / Franka Panda parallel-jaw)
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Realistic Robot Gripper (Robotiq 2F-85 / Franka Panda parallel-jaw)
# --------------------------------------------------------------------------- #

def make_robot_gripper(
    scale: float = 1.0,
    finger_span: float = 0.085,
    grip_color: Tuple[float, float, float] = (0.94, 0.45, 0.08),  # Industrial safety orange
    metal_color: Tuple[float, float, float] = (0.20, 0.21, 0.24), # Matte gunmetal aluminum
) -> SceneObject:
    """Detailed industrial parallel-jaw robotic gripper oriented downwards for natural grasp occlusions."""
    prims: List[Primitive] = []
    s = scale
    span_half = (finger_span / 2.0) * s

    # 1. Wrist Mounting Flange (Cylinder - top)
    prims.append(
        Primitive(
            CYLINDER,
            scale=(0.038 * s, 0.038 * s, 0.022 * s),
            offset=(0.0, 0.0, 0.065 * s),
            color=(0.32, 0.34, 0.38),
            texture_freq=30.0, texture_amp=0.05,
        )
    )

    # 2. Main Actuator Housing (Box - middle)
    prims.append(
        Primitive(
            BOX,
            scale=(0.068 * s, 0.046 * s, 0.036 * s),
            offset=(0.0, 0.0, 0.028 * s),
            color=metal_color,
            texture_freq=20.0, texture_amp=0.06,
        )
    )

    # Status LED Strip (Glowing cyan)
    prims.append(
        Primitive(
            BOX,
            scale=(0.050 * s, 0.003 * s, 0.006 * s),
            offset=(0.0, 0.024 * s, 0.028 * s),
            color=(0.0, 0.88, 0.98),
            texture_freq=1.0, texture_amp=0.0,
        )
    )

    # 3. Left Finger Assembly (pointing DOWN towards the object)
    # Hinge bracket
    prims.append(
        Primitive(
            BOX,
            scale=(0.012 * s, 0.022 * s, 0.024 * s),
            offset=(-span_half - 0.010 * s, 0.0, -0.005 * s),
            color=(0.34, 0.35, 0.38),
            texture_freq=25.0, texture_amp=0.04,
        )
    )
    # Distal finger bar
    prims.append(
        Primitive(
            BOX,
            scale=(0.010 * s, 0.018 * s, 0.042 * s),
            offset=(-span_half, 0.0, -0.036 * s),
            color=metal_color,
            texture_freq=30.0, texture_amp=0.05,
        )
    )
    # High-friction textured elastomer finger pad
    prims.append(
        Primitive(
            BOX,
            scale=(0.005 * s, 0.016 * s, 0.032 * s),
            offset=(-span_half + 0.006 * s, 0.0, -0.036 * s),
            color=grip_color,
            texture_freq=60.0, texture_amp=0.12,
        )
    )

    # 4. Right Finger Assembly (pointing DOWN)
    # Hinge bracket
    prims.append(
        Primitive(
            BOX,
            scale=(0.012 * s, 0.022 * s, 0.024 * s),
            offset=(span_half + 0.010 * s, 0.0, -0.005 * s),
            color=(0.34, 0.35, 0.38),
            texture_freq=25.0, texture_amp=0.04,
        )
    )
    # Distal finger bar
    prims.append(
        Primitive(
            BOX,
            scale=(0.010 * s, 0.018 * s, 0.042 * s),
            offset=(span_half, 0.0, -0.036 * s),
            color=metal_color,
            texture_freq=30.0, texture_amp=0.05,
        )
    )
    # High-friction textured elastomer finger pad
    prims.append(
        Primitive(
            BOX,
            scale=(0.005 * s, 0.016 * s, 0.032 * s),
            offset=(span_half - 0.006 * s, 0.0, -0.036 * s),
            color=grip_color,
            texture_freq=60.0, texture_amp=0.12,
        )
    )

    for p in prims:
        if p.color == grip_color:
            p.pattern, p.texture_freq, p.texture_amp, p.roughness = "noise", 400.0, 0.12, 0.9
        elif p.color == (0.0, 0.88, 0.98):
            p.pattern, p.texture_amp, p.roughness = "noise", 0.0, 0.1
        else:
            p.pattern, p.texture_freq, p.texture_amp, p.roughness, p.metallic = "brushed", 300.0, 0.08, 0.35, 0.25
    return SceneObject(name="robot_gripper", prims=prims, movable=False, obj_id=99)


# --------------------------------------------------------------------------- #
# Realistic Real-Life Objects
# --------------------------------------------------------------------------- #

def ceramic_coffee_mug(color: Tuple[float, float, float] = (0.86, 0.20, 0.16)) -> List[Primitive]:
    """Glossy ceramic coffee mug with inner cream glaze and espresso crema."""
    return [
        # Mug main body cylinder
        Primitive(
            CYLINDER,
            scale=(0.038, 0.038, 0.048),
            offset=(0.0, 0.0, 0.048),
            color=color,
            texture_freq=12.0, texture_amp=0.03,
        ),
        # Inner hollow rim
        Primitive(
            CYLINDER,
            scale=(0.033, 0.033, 0.046),
            offset=(0.0, 0.0, 0.051),
            color=(0.95, 0.94, 0.90),
            texture_freq=8.0, texture_amp=0.01,
        ),
        # Espresso liquid surface
        Primitive(
            CYLINDER,
            scale=(0.032, 0.032, 0.004),
            offset=(0.0, 0.0, 0.075),
            color=(0.18, 0.10, 0.06),
            texture_freq=20.0, texture_amp=0.02,
        ),
        # Handle upper horizontal bar
        Primitive(
            BOX,
            scale=(0.006, 0.018, 0.006),
            offset=(0.045, 0.0, 0.072),
            color=color,
            texture_freq=10.0, texture_amp=0.02,
        ),
        # Handle outer vertical grip
        Primitive(
            BOX,
            scale=(0.006, 0.007, 0.030),
            offset=(0.060, 0.0, 0.050),
            color=color,
            texture_freq=10.0, texture_amp=0.02,
        ),
        # Handle lower horizontal bar
        Primitive(
            BOX,
            scale=(0.006, 0.018, 0.006),
            offset=(0.045, 0.0, 0.028),
            color=color,
            texture_freq=10.0, texture_amp=0.02,
        ),
    ]


def mustard_squeeze_bottle() -> List[Primitive]:
    """YCB-style French's mustard squeeze bottle with contoured curves and red nozzle."""
    return [
        # Curved main bottle body (French's golden yellow)
        Primitive(
            ELLIPSOID,
            scale=(0.038, 0.026, 0.065),
            offset=(0.0, 0.0, 0.065),
            color=(0.96, 0.78, 0.08),
            texture_freq=15.0, texture_amp=0.04,
        ),
        # Cylindrical neck
        Primitive(
            CYLINDER,
            scale=(0.013, 0.013, 0.014),
            offset=(0.0, 0.0, 0.134),
            color=(0.94, 0.76, 0.08),
            texture_freq=20.0, texture_amp=0.02,
        ),
        # Red twist nozzle cap
        Primitive(
            CYLINDER,
            scale=(0.009, 0.009, 0.018),
            offset=(0.0, 0.0, 0.152),
            color=(0.85, 0.12, 0.10),
            texture_freq=30.0, texture_amp=0.03,
        ),
        # Front label band (flush on the bottle face)
        Primitive(
            BOX,
            scale=(0.026, 0.002, 0.022),
            offset=(0.0, 0.023, 0.062),
            color=(0.98, 0.97, 0.92),
            texture_freq=35.0, texture_amp=0.06,
        ),
    ]


def tomato_soup_can() -> List[Primitive]:
    """Campbell's style tomato soup can with red & white label and shiny tin lids."""
    return [
        # Main cylinder body (Campbell Red)
        Primitive(
            CYLINDER,
            scale=(0.033, 0.033, 0.048),
            offset=(0.0, 0.0, 0.050),
            color=(0.82, 0.14, 0.12),
            texture_freq=20.0, texture_amp=0.05,
        ),
        # Top metallic tin lid
        Primitive(
            CYLINDER,
            scale=(0.034, 0.034, 0.003),
            offset=(0.0, 0.0, 0.100),
            color=(0.82, 0.84, 0.88),
            texture_freq=50.0, texture_amp=0.03,
        ),
        # Bottom metallic tin rim
        Primitive(
            CYLINDER,
            scale=(0.034, 0.034, 0.003),
            offset=(0.0, 0.0, 0.003),
            color=(0.82, 0.84, 0.88),
            texture_freq=50.0, texture_amp=0.03,
        ),
        # Gold medal seal on label
        Primitive(
            CYLINDER,
            scale=(0.010, 0.010, 0.003),
            offset=(0.0, 0.032, 0.050),
            rot_euler=(math.pi / 2, 0.0, 0.0),
            color=(0.94, 0.78, 0.18),
            texture_freq=40.0, texture_amp=0.05,
        ),
    ]


def power_drill_v3() -> List[Primitive]:
    """Detailed industrial cordless hammer drill."""
    return [
        # Motor housing (teal/cyan Makita style)
        Primitive(BOX, (0.058, 0.034, 0.038), (0.0, 0.0, 0.092),
                  color=(0.08, 0.52, 0.58), texture_freq=32.0, texture_amp=0.10),
        # Chuck assembly (steel)
        Primitive(CYLINDER, (0.020, 0.020, 0.035), (0.068, 0.0, 0.095),
                  rot_euler=(0.0, math.pi / 2, 0.0),
                  color=(0.28, 0.29, 0.32), texture_freq=60.0, texture_amp=0.08),
        # Drill bit tip (carbide steel)
        Primitive(CYLINDER, (0.005, 0.005, 0.045), (0.125, 0.0, 0.095),
                  rot_euler=(0.0, math.pi / 2, 0.0),
                  color=(0.85, 0.85, 0.88), texture_freq=90.0, texture_amp=0.05),
        # Ergonomic rubberized grip handle
        Primitive(BOX, (0.026, 0.028, 0.055), (-0.012, 0.0, 0.042),
                  rot_euler=(0.0, -0.18, 0.0),
                  color=(0.12, 0.13, 0.15), texture_freq=55.0, texture_amp=0.22),
        # Heavy lithium battery pack base
        Primitive(BOX, (0.042, 0.036, 0.018), (-0.018, 0.0, 0.009),
                  color=(0.14, 0.15, 0.18), texture_freq=30.0, texture_amp=0.10),
    ]


def clothes_iron_v3() -> List[Primitive]:
    """Modern steam iron with stainless soleplate and translucent reservoir."""
    return [
        # Polished stainless steel soleplate
        Primitive(BOX, (0.090, 0.055, 0.012), (0.0, 0.0, 0.006),
                  color=(0.86, 0.88, 0.90), texture_freq=50.0, texture_amp=0.04),
        # Transparent blue water reservoir
        Primitive(ELLIPSOID, (0.075, 0.042, 0.032), (0.0, 0.0, 0.045),
                  color=(0.15, 0.45, 0.82), texture_freq=25.0, texture_amp=0.08),
        # Handle bridge (dark matte charcoal)
        Primitive(CYLINDER, (0.014, 0.014, 0.075), (-0.020, 0.0, 0.085),
                  rot_euler=(0.0, math.pi / 2.2, 0.0),
                  color=(0.12, 0.13, 0.15), texture_freq=35.0, texture_amp=0.06),
        # Temperature control dial
        Primitive(CYLINDER, (0.018, 0.018, 0.006), (-0.010, 0.0, 0.062),
                  color=(0.95, 0.35, 0.10), texture_freq=40.0, texture_amp=0.08),
    ]


# --------------------------------------------------------------------------- #
# Scene Builder with Multi-Object Presets
# --------------------------------------------------------------------------- #

def build_photoreal_scene(
    name: str = "photoreal_robot_tabletop",
    table_color: Tuple[float, float, float] = (0.24, 0.18, 0.14), # Rich mahogany/walnut wood
    table_size: Tuple[float, float, float] = (0.60, 0.45, 0.025),
) -> List[SceneObject]:
    """Builds a rich, photorealistic scene with real-world objects."""
    # Sturdy wooden workbench tabletop
    table = SceneObject(
        name="table",
        prims=[
            Primitive(
                BOX, table_size, (0.0, 0.0, -table_size[2]),
                color=table_color, texture_freq=18.0, texture_amp=0.18
            )
        ],
        movable=False,
        obj_id=0,
    )

    if name in ("photoreal_robot_tabletop", "default"):
        return [
            table,
            # Object 1: Power Drill (industrial tool)
            SceneObject(name="drill", prims=power_drill_v3(), position=(-0.11, -0.06, 0.0), yaw=0.20, obj_id=1),
            # Object 2: Steam Iron (domestic appliance)
            SceneObject(name="iron", prims=clothes_iron_v3(), position=(0.10, 0.03, 0.0), yaw=-0.45, obj_id=2),
            # Object 3: Ceramic Coffee Mug (glossy ceramic kitchenware)
            SceneObject(name="mug", prims=ceramic_coffee_mug(), position=(-0.01, 0.10, 0.0), yaw=0.80, obj_id=3),
            # Object 4: Mustard Squeeze Bottle (YCB condiments)
            SceneObject(name="mustard", prims=mustard_squeeze_bottle(), position=(0.04, -0.10, 0.0), yaw=-0.15, obj_id=4),
            # Object 5: Tomato Soup Can (canned food with tin rims)
            SceneObject(name="soup_can", prims=tomato_soup_can(), position=(-0.15, 0.07, 0.0), yaw=0.0, obj_id=5),
        ]
    elif name == "kitchen_tabletop":
        return [
            table,
            SceneObject(name="mug", prims=ceramic_coffee_mug(), position=(-0.08, 0.0, 0.0), yaw=0.0, obj_id=1),
            SceneObject(name="mustard", prims=mustard_squeeze_bottle(), position=(0.05, 0.05, 0.0), yaw=0.2, obj_id=2),
            SceneObject(name="soup", prims=tomato_soup_can(), position=(0.12, -0.05, 0.0), yaw=-0.4, obj_id=3),
        ]
    else:
        raise ValueError(f"Unknown scene preset '{name}'. Choose: photoreal_robot_tabletop, kitchen_tabletop")


# --------------------------------------------------------------------------- #
# Photorealistic Raycasting Engine with Cast Shadows and Blinn-Phong Shading
# --------------------------------------------------------------------------- #

@torch.no_grad()
def render_photoreal_scene(
    scene: "FlatScene",
    camera: PinholeCamera,
    key_light: Tuple[float, float, float] = (0.35, -0.45, 0.85),
    fill_light: Tuple[float, float, float] = (-0.50, 0.20, 0.60),
    cast_shadows: bool = True,
    supersample: int = 1,
) -> Dict[str, torch.Tensor]:
    """Renders physically-plausible RGB with ray-traced shadows and specular highlights."""
    from .scene_builder import _ray_dirs, _intersect_batch, FlatScene

    H, W = camera.height, camera.width
    dev = scene.device
    camera = camera.to(dev)

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
    valid = prim_flat >= 0
    p_safe = prim_flat.clamp(min=0)

    # Surface 3D world positions
    pts = origin + t_flat.unsqueeze(-1) * dirs

    # 1. Primary Key Light & Ray-Traced Cast Shadows
    L_key = torch.tensor(key_light, dtype=torch.float32, device=dev)
    L_key = L_key / torch.linalg.norm(L_key)
    diffuse_key = (norm_flat @ L_key).clamp(min=0.0)

    shadow_factor = torch.ones_like(diffuse_key)
    if cast_shadows:
        # Trace shadow rays towards key light
        shadow_origin = pts + norm_flat * 1.5e-3 # Bias offset
        shadow_dir = L_key.unsqueeze(0).expand(shadow_origin.shape[0], -1)

        # Trace in chunks to conserve memory
        all_sh = []
        for s in range(0, shadow_origin.shape[0], chunk):
            e = min(s + chunk, shadow_origin.shape[0])
            t_sh, _, _ = _intersect_batch(
                shadow_origin[s:e], shadow_dir[s:e], scene.kind, scene.scale, scene.R, scene.t
            )
            # If hit another primitive, it is in shadow
            t_min_sh, _ = t_sh.min(dim=-1)
            in_sh = (t_min_sh > 1e-4) & (t_min_sh < 1e5)
            all_sh.append(in_sh)
        is_shadowed = torch.cat(all_sh, dim=0)
        shadow_factor = torch.where(is_shadowed, torch.tensor(0.22, device=dev), torch.tensor(1.0, device=dev))

    # 2. Secondary Soft Fill Light (Simulating ambient bounce)
    L_fill = torch.tensor(fill_light, dtype=torch.float32, device=dev)
    L_fill = L_fill / torch.linalg.norm(L_fill)
    diffuse_fill = (norm_flat @ L_fill).clamp(min=0.0) * 0.35

    # 3. Blinn-Phong specular from the material's roughness / metallic
    tex_color = surface_color(scene, p_safe, pts)
    exponent, ks, metal = specular_params(scene, p_safe)
    V = -dirs # View direction
    H_key = L_key.unsqueeze(0) + V
    H_key = H_key / torch.linalg.norm(H_key, dim=-1, keepdim=True).clamp(min=1e-6)
    n_dot_h = (norm_flat * H_key).sum(dim=-1).clamp(min=0.0)
    spec = shadow_factor * ks * n_dot_h ** exponent * (diffuse_key > 0).float()
    H_fill = L_fill.unsqueeze(0) + V
    H_fill = H_fill / torch.linalg.norm(H_fill, dim=-1, keepdim=True).clamp(min=1e-6)
    spec = spec + 0.3 * ks * (norm_flat * H_fill).sum(dim=-1).clamp(min=0.0) ** exponent
    # grazing-angle sheen on smooth surfaces (Schlick-style Fresnel)
    n_dot_v = (norm_flat * V).sum(dim=-1).clamp(min=0.0)
    spec = spec + 0.5 * ks * (1.0 - n_dot_v) ** 5
    spec_col = torch.lerp(torch.ones_like(tex_color), tex_color, metal.unsqueeze(-1))

    # Ambient Contact Occlusion (darkens tabletop immediately beneath objects)
    ao_factor = torch.clamp((pts[:, 2] + 0.005) / 0.04, 0.40, 1.0)

    # Total illumination shade
    ambient = 0.22
    total_diffuse = shadow_factor * (diffuse_key * 0.75) + diffuse_fill
    shade = ((ambient * ao_factor) + total_diffuse) * (1.0 - 0.75 * metal)

    # Combined radiance: Texture * Diffuse + Specular highlights
    rgb = (tex_color * shade.unsqueeze(-1)) + spec.unsqueeze(-1) * spec_col
    # Background color (subtle neutral studio gradient)
    bg_gradient = 0.92 - 0.05 * (dirs[:, 2:3] + 0.5)
    rgb = torch.where(valid.unsqueeze(-1), rgb, bg_gradient.expand(-1, 3))
    rgb = rgb.clamp(0.0, 1.0).reshape(H, W, 3)

    # Metric depth
    c2w_z = camera.c2w[:3, 2]
    cos_angle = dirs @ c2w_z
    depth = (t_flat * cos_angle).reshape(H, W)
    depth = torch.where(depth > 0, depth, torch.zeros_like(depth))

    # Instance object ID mask
    obj_id = torch.where(valid, scene.obj_id[p_safe], torch.tensor(0, device=dev)).reshape(H, W)

    return {
        "rgb": rgb,
        "depth": depth,
        "normal": norm_flat.reshape(H, W, 3),
        "obj_id": obj_id,
        "shadow": shadow_factor.reshape(H, W),
    }
