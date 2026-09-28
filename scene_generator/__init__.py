"""
Scene Generator Package
Detached, modular synthetic scene & tracking dataset generator for POGS.
Generates multi-view hemispherical captures, fused point clouds, and tracking sequences
with explicit angle constraint scenarios (respected, threshold, violated, servoing).
"""

from .primitives import Primitive, BOX, ELLIPSOID, CYLINDER
from .objects import (
    clothes_iron,
    power_drill,
    shoe,
    shoebox,
    jigsaw,
    wooden_shelf,
    make_hand,
    sample_object_model_points,
)
from .scene_builder import SceneObject, FlatScene, build_scene, deproject
from .trajectory_planner import (
    TrajectoryPlanner,
    TrajectoryScenario,
    AngleConstraintStatus,
)
from .capture_generator import CaptureGenerator
from .tracking_generator import TrackingSequenceGenerator
from .video_annotator import VideoAnnotator

__all__ = [
    "Primitive",
    "BOX",
    "ELLIPSOID",
    "CYLINDER",
    "clothes_iron",
    "power_drill",
    "shoe",
    "shoebox",
    "jigsaw",
    "wooden_shelf",
    "make_hand",
    "sample_object_model_points",
    "SceneObject",
    "FlatScene",
    "build_scene",
    "deproject",
    "TrajectoryPlanner",
    "TrajectoryScenario",
    "AngleConstraintStatus",
    "CaptureGenerator",
    "TrackingSequenceGenerator",
    "VideoAnnotator",
]
