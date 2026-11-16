"""Schedules and geometry of the tool-perturbation scene (no rendering)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.scene_generator.tool_perturbation import (  # noqa: E402
    GraspCase, ToolCase, jump_schedule, o3_cases, o4_cases, rotation_about, step_schedule,
)


def test_step_schedule_goes_out_and_back():
    s = step_schedule(30, 5)
    ang = [a for a, _ in s]
    assert max(ang) == 30 and ang[0] == 0 and ang[-1] == 0
    peak = ang.index(30)
    assert ang[peak:peak + 3] == [30, 30, 30]                      # held 3 frames
    assert [a for a, _ in s if a not in (0.0,)][:3] == [5, 5, 5]
    assert sorted(set(ang)) == [0, 5, 10, 15, 20, 25, 30]
    assert any(p == "return_y" for _, p in s) and any(p == "peak_p" for _, p in s)


def test_step_schedule_rejects_non_multiple():
    with pytest.raises(AssertionError):
        step_schedule(30, 7)


def test_jump_and_cases():
    assert [a for a, _ in jump_schedule(-30)][-1] == -30
    names = [c.name for c in o3_cases(range(2))]
    assert len(names) == len(set(names)) == 2 * (3 + 7)
    assert len(o4_cases()) == 200 and GraspCase(-15, 3).name == "grasp_-15_s3"
    assert ToolCase("step", 30, 10).name == "step_p30_d10_z_s0"


def test_rotation_about_fixes_pivot_and_axis():
    c = np.array([0.3, -0.1, 0.2])
    ax = np.array([1.0, 2.0, 0.5])
    T = rotation_about(c, ax, 40.0)
    assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3)) and np.isclose(np.linalg.det(T[:3, :3]), 1)
    assert np.allclose((T @ np.append(c, 1))[:3], c)
    k = ax / np.linalg.norm(ax)
    assert np.allclose(T[:3, :3] @ k, k)
    assert np.isclose(np.degrees(np.arccos((np.trace(T[:3, :3]) - 1) / 2)), 40.0)
