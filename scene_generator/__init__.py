"""
Synthetic scene + dataset generator for the POGS benchmark (detached from the
POGS library). Entry point: ``scene_generator.pick_place_benchmark``.
"""

from .primitives import BOX, CYLINDER, ELLIPSOID, Primitive
from .objects import SceneObject
from .scene_builder import FlatScene, deproject

__all__ = ["BOX", "CYLINDER", "ELLIPSOID", "Primitive", "SceneObject", "FlatScene", "deproject"]
