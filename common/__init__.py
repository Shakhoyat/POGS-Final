"""
POGS Common Library
Core building blocks for Persistent Object Gaussian Splats.
Includes configuration, geometry, rasterization, fields, encoders, losses,
models, trainers, grouping, tracking, Kalman filtering, evaluation, grasp,
servoing, and visualization.
"""

from .config import (
    CaptureConfig,
    ModelConfig,
    GroupingConfig,
    TrackingConfig,
    TrainingConfig,
    POGSConfig,
)
from .geometry import PinholeCamera, look_at, quat_to_rotmat, rotmat_to_quat
from .model import POGSModel
from .grouping import ObjectGroups, cluster_scene
from .tracking import RigidGroupOptimizer
from .evaluate import TrackingMetrics, track_sequence, results_table

__all__ = [
    "CaptureConfig",
    "ModelConfig",
    "GroupingConfig",
    "TrackingConfig",
    "TrainingConfig",
    "POGSConfig",
    "PinholeCamera",
    "look_at",
    "quat_to_rotmat",
    "rotmat_to_quat",
    "POGSModel",
    "ObjectGroups",
    "cluster_scene",
    "RigidGroupOptimizer",
    "TrackingMetrics",
    "track_sequence",
    "results_table",
]
