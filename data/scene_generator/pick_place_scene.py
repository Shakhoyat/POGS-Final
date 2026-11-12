"""
Photorealistic Pick-and-Place Tabletop Scene Engine (v4).

Scene Layout:
1. Large floor plane.
2. Small height table (raised staging platform) on the floor.
3. 4-5 household YCB-style objects resting on the floor:
   - Target Object: Red Ceramic Coffee Mug (with espresso crema)
   - Distractor 1: Yellow Mustard Squeeze Bottle
   - Distractor 2: Campbell's Tomato Soup Can
   - Distractor 3: Fresh Red Apple
   - Distractor 4: Breakfast Cereal Box
4. Vertical articulated industrial robot arm (always present in the scene):
   - Floor pedestal, vertical column, shoulder joint, upper arm, elbow, forearm, wrist, and parallel-jaw gripper.
   - Smoothly picks the target object from the floor, lifts it, transits across, rotates, and places it on the small table.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch

from .primitives import Primitive, BOX, ELLIPSOID, CYLINDER, euler_to_R
from .objects import SceneObject
from model.pogs.geometry import PinholeCamera


# --------------------------------------------------------------------------- #
# Real-World Household Objects (Optimized for OpenCLIP Recognition)
#
# Primitive ``scale`` is a HALF-extent (boxes span [-s, s], cylinders have
# radius s_x and half-height s_z). Materials: see ``materials.py``; texture
# frequencies are features per metre in the primitive's own frame.
# --------------------------------------------------------------------------- #

def ceramic_coffee_mug(color: Tuple[float, float, float] = (0.80, 0.12, 0.10)) -> List[Primitive]:
    """Glossy red glazed ceramic mug with cream interior, handle and coffee."""
    glaze = dict(color=color, pattern="noise", texture_freq=180.0, texture_amp=0.05, roughness=0.12)
    return [
        Primitive(CYLINDER, scale=(0.038, 0.038, 0.048), offset=(0.0, 0.0, 0.048), **glaze),
        # inner glaze (a thin cream ring shows at the rim) and the coffee surface
        # just above it -- the primitives are solid, so the coffee must be on top
        Primitive(CYLINDER, scale=(0.033, 0.033, 0.0482), offset=(0.0, 0.0, 0.0482),
                  color=(0.93, 0.91, 0.86), pattern="noise", texture_freq=120.0, texture_amp=0.03, roughness=0.15),
        Primitive(CYLINDER, scale=(0.029, 0.029, 0.0010), offset=(0.0, 0.0, 0.0960),
                  color=(0.22, 0.12, 0.06), color2=(0.36, 0.21, 0.11), pattern="blotch",
                  texture_freq=260.0, texture_amp=0.05, roughness=0.08),
        # handle: upper bar, grip, lower bar
        Primitive(BOX, scale=(0.006, 0.018, 0.006), offset=(0.045, 0.0, 0.070), **glaze),
        Primitive(BOX, scale=(0.006, 0.007, 0.030), offset=(0.060, 0.0, 0.050), **glaze),
        Primitive(BOX, scale=(0.006, 0.018, 0.006), offset=(0.045, 0.0, 0.030), **glaze),
    ]


def mustard_squeeze_bottle() -> List[Primitive]:
    """YCB-style yellow mustard bottle: glossy plastic, red cap, printed label."""
    return [
        Primitive(ELLIPSOID, scale=(0.036, 0.024, 0.062), offset=(0.0, 0.0, 0.062),
                  color=(0.95, 0.76, 0.07), pattern="noise", texture_freq=150.0, texture_amp=0.05, roughness=0.25),
        Primitive(CYLINDER, scale=(0.012, 0.012, 0.012), offset=(0.0, 0.0, 0.128),
                  color=(0.93, 0.74, 0.07), pattern="noise", texture_freq=150.0, texture_amp=0.04, roughness=0.25),
        Primitive(CYLINDER, scale=(0.008, 0.008, 0.018), offset=(0.0, 0.0, 0.145),
                  color=(0.85, 0.10, 0.08), pattern="noise", texture_freq=200.0, texture_amp=0.05, roughness=0.35),
        # front label: white with red bands
        Primitive(BOX, scale=(0.026, 0.002, 0.022), offset=(0.0, 0.022, 0.060),
                  color=(0.97, 0.96, 0.91), color2=(0.80, 0.12, 0.10), pattern="stripes",
                  texture_freq=180.0, texture_amp=0.05, roughness=0.6),
    ]


def tomato_soup_can() -> List[Primitive]:
    """Two-tone (red over white) soup can with brushed tin lids and gold seal."""
    tin = dict(color=(0.78, 0.80, 0.84), pattern="brushed", texture_freq=400.0, texture_amp=0.10,
               roughness=0.30, metallic=0.85)
    return [
        Primitive(CYLINDER, scale=(0.032, 0.032, 0.046), offset=(0.0, 0.0, 0.048),
                  color=(0.78, 0.10, 0.09), color2=(0.95, 0.94, 0.90), pattern="split",
                  texture_freq=200.0, texture_amp=0.06, roughness=0.40),
        Primitive(CYLINDER, scale=(0.033, 0.033, 0.003), offset=(0.0, 0.0, 0.096), **tin),
        Primitive(CYLINDER, scale=(0.033, 0.033, 0.003), offset=(0.0, 0.0, 0.003), **tin),
        Primitive(CYLINDER, scale=(0.010, 0.010, 0.002), offset=(0.0, 0.031, 0.048),
                  rot_euler=(math.pi / 2, 0.0, 0.0), color=(0.92, 0.74, 0.20), pattern="noise",
                  texture_freq=300.0, texture_amp=0.08, roughness=0.25, metallic=0.9),
    ]


def red_apple() -> List[Primitive]:
    """Glossy red apple with yellow-green streaks and a dark stem."""
    return [
        Primitive(ELLIPSOID, scale=(0.038, 0.038, 0.034), offset=(0.0, 0.0, 0.034),
                  color=(0.72, 0.07, 0.08), color2=(0.82, 0.40, 0.14), pattern="blotch",
                  texture_freq=420.0, texture_amp=0.12, roughness=0.18),
        Primitive(CYLINDER, scale=(0.002, 0.002, 0.010), offset=(0.0, 0.0, 0.072), rot_euler=(0.15, 0.15, 0.0),
                  color=(0.28, 0.18, 0.10), pattern="noise", texture_freq=300.0, texture_amp=0.10, roughness=0.8),
    ]


def cereal_box() -> List[Primitive]:
    """Cardboard cereal box: blue print with yellow patches and a front band."""
    return [
        Primitive(BOX, scale=(0.035, 0.055, 0.075), offset=(0.0, 0.0, 0.075),
                  color=(0.16, 0.45, 0.78), color2=(0.96, 0.82, 0.18), pattern="blotch",
                  texture_freq=60.0, texture_amp=0.10, roughness=0.85),
        Primitive(BOX, scale=(0.036, 0.002, 0.035), offset=(0.0, 0.056, 0.075),
                  color=(0.96, 0.84, 0.15), color2=(0.85, 0.15, 0.12), pattern="stripes",
                  texture_freq=160.0, texture_amp=0.08, roughness=0.85),
    ]


# --------------------------------------------------------------------------- #
# Floor & Small Table Construction
# --------------------------------------------------------------------------- #

def make_floor(
    size: Tuple[float, float, float] = (1.00, 0.80, 0.02),
    color: Tuple[float, float, float] = (0.70, 0.71, 0.72),
) -> SceneObject:
    """Tiled concrete-look floor; top surface at z = 0 (``size`` = half-extents)."""
    return SceneObject(
        name="floor",
        prims=[Primitive(BOX, scale=size, offset=(0.0, 0.0, -size[2]), color=color,
                         color2=(0.50, 0.50, 0.51), pattern="tiles", texture_freq=6.0,
                         texture_amp=0.10, roughness=0.75)],
        position=(0.0, 0.0, 0.0),
        movable=False,
        obj_id=0,
    )


def make_small_table(
    position: Tuple[float, float, float] = (0.18, 0.12, 0.0),
    top_size: Tuple[float, float, float] = (0.20, 0.16, 0.012),
    height: float = 0.060,
    wood_color: Tuple[float, float, float] = (0.42, 0.27, 0.16),
    leg_color: Tuple[float, float, float] = (0.30, 0.31, 0.33),
) -> SceneObject:
    """Low walnut staging table. ``top_size`` are HALF-extents; the top surface
    is exactly at z = ``height`` (objects are placed there)."""
    w, l, th = top_size
    prims = [Primitive(BOX, scale=(w, l, th), offset=(0.0, 0.0, height - th), color=wood_color,
                       color2=(0.26, 0.15, 0.08), pattern="wood", texture_freq=40.0,
                       texture_amp=0.10, roughness=0.45)]
    leg_r, leg_half = 0.008, (height - 2.0 * th) / 2.0
    dx, dy = w - 0.020, l - 0.020
    for lx in (-dx, dx):
        for ly in (-dy, dy):
            prims.append(Primitive(CYLINDER, scale=(leg_r, leg_r, leg_half), offset=(lx, ly, leg_half),
                                   color=leg_color, pattern="brushed", texture_freq=300.0,
                                   texture_amp=0.08, roughness=0.35, metallic=0.7))
    return SceneObject(name="small_table", prims=prims, position=position, movable=False, obj_id=9)


# --------------------------------------------------------------------------- #
# Vertical Articulated Robotic Arm (Always in Scene)
# --------------------------------------------------------------------------- #

def build_robot_arm_primitives(
    base_pos: Tuple[float, float, float] = (-0.25, -0.06, 0.0),
    end_effector_pos: Tuple[float, float, float] = (-0.08, -0.06, 0.075),
    gripper_yaw: float = 0.0,
    gripper_pitch: float = 0.0,
    gripper_roll: float = 0.0,
    gripper_span: float = 0.082,
    arm_color: Tuple[float, float, float] = (0.22, 0.24, 0.28), # Gunmetal grey
    joint_color: Tuple[float, float, float] = (0.35, 0.37, 0.42), # Steel pivot
    orange_pad: Tuple[float, float, float] = (0.94, 0.45, 0.08),  # High-vis grip pad
) -> SceneObject:
    """Builds a multi-link vertical robot arm whose gripper connects to end_effector_pos."""
    bx, by, bz = base_pos
    ex, ey, ez = end_effector_pos

    prims: List[Primitive] = []

    # 1. Base Mounting Flange on Floor
    prims.append(Primitive(CYLINDER, (0.055, 0.055, 0.015), (bx, by, 0.015), color=(0.16, 0.17, 0.19)))

    # 2. Rotating Turret
    prims.append(Primitive(CYLINDER, (0.045, 0.045, 0.035), (bx, by, 0.050), color=arm_color))

    # 3. Vertical Column (Torso)
    torso_top_z = 0.30
    prims.append(Primitive(CYLINDER, (0.030, 0.030, 0.105), (bx, by, 0.185), color=arm_color))

    # 4. Shoulder Joint Pivot
    shoulder_pos = (bx, by, torso_top_z)
    prims.append(Primitive(ELLIPSOID, (0.036, 0.036, 0.036), shoulder_pos, color=joint_color))

    # 5. Elbow Joint Pivot (midpoint between shoulder and wrist)
    elbow_x = (bx + ex) / 2.0 - 0.02
    elbow_y = (by + ey) / 2.0 - 0.03
    elbow_z = max(torso_top_z + 0.05, ez + 0.15)
    elbow_pos = (elbow_x, elbow_y, elbow_z)
    prims.append(Primitive(ELLIPSOID, (0.030, 0.030, 0.030), elbow_pos, color=joint_color))

    # 6. Upper Arm Link (Connecting shoulder to elbow)
    u_mid = ((shoulder_pos[0] + elbow_pos[0]) / 2.0, (shoulder_pos[1] + elbow_pos[1]) / 2.0, (shoulder_pos[2] + elbow_pos[2]) / 2.0)
    u_len = math.sqrt((elbow_pos[0] - shoulder_pos[0])**2 + (elbow_pos[1] - shoulder_pos[1])**2 + (elbow_pos[2] - shoulder_pos[2])**2)
    prims.append(Primitive(BOX, (0.024, 0.024, u_len / 2.0), u_mid, color=arm_color))

    # 7. Forearm Link (Connecting elbow to wrist)
    wrist_pos = (ex, ey, ez + 0.075)
    f_mid = ((elbow_pos[0] + wrist_pos[0]) / 2.0, (elbow_pos[1] + wrist_pos[1]) / 2.0, (elbow_pos[2] + wrist_pos[2]) / 2.0)
    f_len = math.sqrt((wrist_pos[0] - elbow_pos[0])**2 + (wrist_pos[1] - elbow_pos[1])**2 + (wrist_pos[2] - elbow_pos[2])**2)
    prims.append(Primitive(BOX, (0.022, 0.022, f_len / 2.0), f_mid, color=arm_color))

    # 8. Wrist Joint Assembly
    prims.append(Primitive(CYLINDER, (0.026, 0.026, 0.016), wrist_pos, color=joint_color))

    # 9. Industrial Parallel-Jaw Gripper (Positioned above object and reaching down)
    span_half = gripper_span / 2.0
    rot = (gripper_roll, gripper_pitch, gripper_yaw)

    # Gripper Actuator Body
    act_offset = (ex, ey, ez + 0.042)
    prims.append(Primitive(BOX, (0.062, 0.040, 0.028), act_offset, rot_euler=rot, color=(0.18, 0.19, 0.22)))

    # Cyan Status LED Strip
    led_offset = (ex, ey + 0.021, ez + 0.042)
    prims.append(Primitive(BOX, (0.042, 0.003, 0.005), led_offset, rot_euler=rot, color=(0.0, 0.88, 0.98)))

    # Left Finger Bar & Pad
    f_rot_mat = euler_to_R(*rot)
    l_pos = np.array([ex, ey, ez + 0.012]) + f_rot_mat @ np.array([-span_half, 0.0, 0.0])
    prims.append(Primitive(BOX, (0.008, 0.015, 0.030), tuple(l_pos), rot_euler=rot, color=(0.25, 0.26, 0.29)))
    l_pad = np.array([ex, ey, ez + 0.012]) + f_rot_mat @ np.array([-span_half + 0.005, 0.0, 0.0])
    prims.append(Primitive(BOX, (0.005, 0.013, 0.022), tuple(l_pad), rot_euler=rot, color=orange_pad))

    # Right Finger Bar & Pad
    r_pos = np.array([ex, ey, ez + 0.012]) + f_rot_mat @ np.array([span_half, 0.0, 0.0])
    prims.append(Primitive(BOX, (0.008, 0.015, 0.030), tuple(r_pos), rot_euler=rot, color=(0.25, 0.26, 0.29)))
    r_pad = np.array([ex, ey, ez + 0.012]) + f_rot_mat @ np.array([span_half - 0.005, 0.0, 0.0])
    prims.append(Primitive(BOX, (0.005, 0.013, 0.022), tuple(r_pad), rot_euler=rot, color=orange_pad))

    return SceneObject(name="robot_arm", prims=prims, movable=False, obj_id=99)


# --------------------------------------------------------------------------- #
# Master Scene Builder
# --------------------------------------------------------------------------- #

def build_pick_place_scene() -> List[SceneObject]:
    """Floor, low table (top x in [-0.02, 0.38], y in [-0.04, 0.28], z = 0.06)
    and five household objects on the floor, each >= 2 cm clear of the table
    footprint and of each other."""
    return [
        make_floor(),
        make_small_table(position=(0.18, 0.12, 0.0), top_size=(0.20, 0.16, 0.012), height=0.060),
        SceneObject(name="ceramic_mug", prims=ceramic_coffee_mug(), position=(-0.10, -0.08, 0.0), yaw=0.0, obj_id=1),
        SceneObject(name="mustard_bottle", prims=mustard_squeeze_bottle(), position=(-0.18, 0.07, 0.0), yaw=-0.15, obj_id=2),
        SceneObject(name="soup_can", prims=tomato_soup_can(), position=(-0.11, 0.20, 0.0), yaw=0.0, obj_id=3),
        SceneObject(name="red_apple", prims=red_apple(), position=(0.06, -0.14, 0.0), yaw=0.45, obj_id=4),
        SceneObject(name="cereal_box", prims=cereal_box(), position=(-0.17, -0.19, 0.0), yaw=-0.30, obj_id=5),
    ]
