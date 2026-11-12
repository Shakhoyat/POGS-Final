"""
Trajectory planning with explicit angle constraint modes and event logging.
Scenarios:
1. RESPECTED: Max rotation <= 30 deg (e.g. 20-25 deg), respecting the tracking threshold.
2. THRESHOLD: Rotation hovers close to the 30 deg boundary (26-32 deg).
3. VIOLATED: Rotation violates the 30 deg constraint (45-90 deg or 180 deg tier 2).
4. SERVOING: Step-wise CW / CCW perturbations (15 deg and 30 deg) matching Sec. V-B.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from .primitives import se3, smoothstep


class TrajectoryScenario(str, enum.Enum):
    RESPECTED = "respected"      # Rotation <= 30 deg
    THRESHOLD = "threshold"      # Rotation reaches 26 - 32 deg
    VIOLATED = "violated"        # Rotation exceeds 32 deg (45 - 90 deg)
    SERVOING = "servoing"        # Tool servoing 15 deg / 30 deg pulses


class AngleConstraintStatus(str, enum.Enum):
    RESPECTED = "RESPECTED"          # |yaw| <= 25 deg
    NEAR_THRESHOLD = "NEAR_THRESHOLD" # 25 deg < |yaw| <= 32 deg
    VIOLATED = "VIOLATED"            # |yaw| > 32 deg


@dataclass
class FrameEvent:
    frame_idx: int
    timestamp_sec: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    translation_cm: float
    constraint_status: AngleConstraintStatus
    hand_occlusion: bool
    reset_event: bool
    event_type: str
    description: str


@dataclass
class PlannedTrajectory:
    scenario: TrajectoryScenario
    object_id: int
    object_name: str
    deltas: List[np.ndarray]               # 4x4 SE(3) matrix per frame
    events: List[FrameEvent]               # detailed per-frame events
    summary_metadata: Dict                 # stats: max_yaw, min_yaw, max_trans, etc.


class TrajectoryPlanner:
    """Plans 6-DoF object trajectories with explicit angle constraints."""

    def __init__(self, fps: float = 5.0, seed: int = 42):
        self.fps = fps
        self.seed = seed

    def plan(
        self,
        scenario: TrajectoryScenario,
        n_frames: int = 60,
        n_resets: int = 3,
        pivot: Optional[np.ndarray] = None,
        max_translation_m: float = 0.08,
        hand_frames: Tuple[int, int] = (18, 38),
        obj_id: int = 1,
        obj_name: str = "object",
    ) -> PlannedTrajectory:
        rng = np.random.default_rng(self.seed + obj_id * 17)
        p = np.zeros(3) if pivot is None else np.asarray(pivot, dtype=np.float64)

        if scenario == TrajectoryScenario.RESPECTED:
            # Constrained rotation: max 22 degrees
            max_yaw_deg = 22.0
            min_yaw_deg = -22.0
        elif scenario == TrajectoryScenario.THRESHOLD:
            # Hovers near 30 degrees (26 to 32 degrees)
            max_yaw_deg = 31.5
            min_yaw_deg = 26.0
        elif scenario == TrajectoryScenario.VIOLATED:
            # Substantially exceeds 30 degrees: up to 60-90 degrees
            max_yaw_deg = 75.0
            min_yaw_deg = -75.0
        elif scenario == TrajectoryScenario.SERVOING:
            # Discrete 15 deg and 30 deg pulses
            max_yaw_deg = 30.0
            min_yaw_deg = -30.0
        else:
            raise ValueError(f"Unknown scenario {scenario}")

        yaw_limit_rad = math.radians(max_yaw_deg)
        segs = max(n_resets, 1)

        # Plan keyframe waypoints
        keys: List[Tuple[float, np.ndarray]] = [(0.0, np.zeros(3))]
        for s_idx in range(segs):
            t = rng.uniform(-max_translation_m, max_translation_m, size=3)
            t[2] = abs(rng.normal(0.0, 0.010))  # stays on table surface

            if scenario == TrajectoryScenario.THRESHOLD:
                # Force yaw to hit 28-32 degrees
                sign = 1.0 if (s_idx % 2 == 0) else -1.0
                yaw = sign * math.radians(rng.uniform(27.0, 31.8))
            elif scenario == TrajectoryScenario.SERVOING:
                # Alternating 15 deg then 30 deg
                angles = [15.0, -15.0, 30.0, -30.0]
                yaw = math.radians(angles[s_idx % len(angles)])
            else:
                yaw = float(rng.uniform(math.radians(min_yaw_deg), yaw_limit_rad))
            keys.append((yaw, t))

        def world_delta(yaw: float, t_trans: np.ndarray) -> np.ndarray:
            c, s = math.cos(yaw), math.sin(yaw)
            R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            return se3(R, p - R @ p + t_trans)

        per_seg = max(n_frames // segs, 1)
        reset_frame_indices = {min((k + 1) * per_seg - 1, n_frames - 1) for k in range(segs)}

        deltas: List[np.ndarray] = []
        events: List[FrameEvent] = []

        hand_lo, hand_hi = hand_frames
        yaws_recorded = []
        trans_recorded = []

        for f in range(n_frames):
            seg = min(f // per_seg, segs - 1)
            frac = (f - seg * per_seg) / max(per_seg, 1)
            alpha = smoothstep(frac)

            (y0, t0), (y1, t1) = keys[seg], keys[seg + 1]
            dy = (y1 - y0 + math.pi) % (2.0 * math.pi) - math.pi
            current_yaw = y0 + alpha * dy
            current_trans = (1.0 - alpha) * t0 + alpha * t1

            T_world = world_delta(current_yaw, current_trans)
            deltas.append(T_world)

            yaw_deg = math.degrees(current_yaw)
            trans_cm = float(np.linalg.norm(current_trans) * 100.0)
            yaws_recorded.append(abs(yaw_deg))
            trans_recorded.append(trans_cm)

            # Determine constraint status
            abs_yaw = abs(yaw_deg)
            if abs_yaw <= 25.0:
                c_status = AngleConstraintStatus.RESPECTED
            elif abs_yaw <= 32.0:
                c_status = AngleConstraintStatus.NEAR_THRESHOLD
            else:
                c_status = AngleConstraintStatus.VIOLATED

            hand_active = hand_lo <= f <= hand_hi
            is_reset = f in reset_frame_indices

            # Event description
            event_type = "NORMAL"
            desc_parts = []
            if is_reset:
                event_type = "RESET"
                desc_parts.append(f"Object reset event {seg + 1}/{segs}")
            if hand_active:
                if f == hand_lo:
                    desc_parts.append("Hand entered workspace")
                elif f == hand_hi:
                    desc_parts.append("Hand leaving workspace")
                else:
                    desc_parts.append("Hand manipulating object")
            if c_status == AngleConstraintStatus.VIOLATED:
                desc_parts.append(f"Angle {yaw_deg:.1f}° exceeds 30° threshold!")
            elif c_status == AngleConstraintStatus.NEAR_THRESHOLD:
                desc_parts.append(f"Angle {yaw_deg:.1f}° near 30° threshold")

            desc = "; ".join(desc_parts) if desc_parts else "Smooth tracking"

            events.append(
                FrameEvent(
                    frame_idx=f,
                    timestamp_sec=f / self.fps,
                    yaw_deg=yaw_deg,
                    pitch_deg=0.0,
                    roll_deg=0.0,
                    translation_cm=trans_cm,
                    constraint_status=c_status,
                    hand_occlusion=hand_active,
                    reset_event=is_reset,
                    event_type=event_type,
                    description=desc,
                )
            )

        summary = {
            "scenario": scenario.value,
            "object_id": obj_id,
            "object_name": obj_name,
            "n_frames": n_frames,
            "fps": self.fps,
            "duration_sec": n_frames / self.fps,
            "max_yaw_deg": float(np.max(yaws_recorded)),
            "mean_yaw_deg": float(np.mean(yaws_recorded)),
            "max_translation_cm": float(np.max(trans_recorded)),
            "mean_translation_cm": float(np.mean(trans_recorded)),
            "n_resets": len(reset_frame_indices),
            "reset_frames": sorted(list(reset_frame_indices)),
            "hand_occlusion_frames": [hand_lo, hand_hi],
            "num_respected_frames": sum(1 for e in events if e.constraint_status == AngleConstraintStatus.RESPECTED),
            "num_threshold_frames": sum(1 for e in events if e.constraint_status == AngleConstraintStatus.NEAR_THRESHOLD),
            "num_violated_frames": sum(1 for e in events if e.constraint_status == AngleConstraintStatus.VIOLATED),
        }

        return PlannedTrajectory(
            scenario=scenario,
            object_id=obj_id,
            object_name=obj_name,
            deltas=deltas,
            events=events,
            summary_metadata=summary,
        )
