"""
Phase 3 Tracking Sequence Generator.
Synthesizes dynamic RGB-D manipulation sequences observed from a static 3rd-person
camera (ZED-2 viewpoint), simulating realistic human resets, angle perturbations,
and hand occlusions.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import cv2
import numpy as np
import torch

from .scene_builder import FlatScene, render_scene
from .objects import SceneObject, make_hand
from .trajectory_planner import TrajectoryPlanner, TrajectoryScenario, PlannedTrajectory, FrameEvent
from common.geometry import PinholeCamera, look_at


@dataclass
class TrackingFrame:
    frame_idx: int
    timestamp_sec: float
    rgb: torch.Tensor             # (H, W, 3) in [0, 1]
    depth: torch.Tensor           # (H, W) in meters
    obj_id: torch.Tensor          # (H, W) integer mask
    gt_delta: Dict[int, np.ndarray]  # {obj_id: 4x4 SE(3)}
    event: FrameEvent


@dataclass
class TrackingSequence:
    scenario: TrajectoryScenario
    frames: List[TrackingFrame]
    camera: PinholeCamera
    object_ids: List[int]
    trajectories: Dict[int, PlannedTrajectory]
    fps: float = 5.0


class TrackingSequenceGenerator:
    """Generates synthetic tracking sequences for a specific scenario."""

    def __init__(self, device: str = "cpu", fps: float = 5.0, seed: int = 42):
        self.device = device
        self.fps = fps
        self.seed = seed
        self.planner = TrajectoryPlanner(fps=fps, seed=seed)

    @torch.no_grad()
    def generate(
        self,
        scene_objects: List[SceneObject],
        scenario: TrajectoryScenario,
        n_frames: int = 60,
        n_resets: int = 3,
        width: int = 400,
        height: int = 300,
        with_hand: bool = True,
        hand_occlusion_fraction: Tuple[float, float] = (0.28, 0.62),
        depth_noise_std: float = 0.0035,
        supersample: int = 1,
        verbose: bool = True,
    ) -> TrackingSequence:
        # Static third-person camera looking at tabletop from angled front-right
        eye = np.array([0.30, -0.52, 0.42])
        center = np.array([0.0, 0.0, 0.045])
        c2w = look_at(eye, center, up=np.array([0.0, 0.0, 1.0]))
        camera = PinholeCamera.from_fov(
            torch.as_tensor(c2w, dtype=torch.float32, device=self.device),
            62.0, width, height
        )

        movable = [obj for obj in scene_objects if obj.movable]
        hand_lo = int(hand_occlusion_fraction[0] * n_frames)
        hand_hi = int(hand_occlusion_fraction[1] * n_frames)

        # Plan trajectory for each movable object
        trajectories: Dict[int, PlannedTrajectory] = {}
        for obj in movable:
            traj = self.planner.plan(
                scenario=scenario,
                n_frames=n_frames,
                n_resets=n_resets,
                pivot=np.asarray(obj.position, dtype=np.float64),
                max_translation_m=0.08,
                hand_frames=(hand_lo, hand_hi),
                obj_id=obj.obj_id,
                obj_name=obj.name,
            )
            trajectories[obj.obj_id] = traj

        frames: List[TrackingFrame] = []

        for f in range(n_frames):
            deltas = {oid: trajectories[oid].deltas[f] for oid in trajectories}
            current_objs = list(scene_objects)

            # Insert hand if active in this frame window
            if with_hand and hand_lo <= f <= hand_hi:
                u = (f - hand_lo) / max(hand_hi - hand_lo, 1)
                primary_target = movable[0]
                base = np.asarray(primary_target.position) + deltas[primary_target.obj_id][:3, 3]
                h = make_hand()
                reach = 0.22 * (1.0 - math.sin(math.pi * u))
                h.position = (float(base[0] - reach * 0.4), float(base[1] - reach), float(base[2] + 0.10))
                h.yaw = float(math.pi / 2.0 + 0.5 * math.sin(2.0 * math.pi * u))
                current_objs = current_objs + [h]

            scene_f = FlatScene(current_objs, device=self.device).clone_with_transforms(deltas)
            out = render_scene(scene_f, camera, supersample=supersample)

            d = out["depth"]
            noise = torch.randn_like(d) * depth_noise_std * (d / 0.6).clamp(min=0.2)
            noisy_d = torch.where(d > 0, d + noise, d)

            # Primary event from first object
            primary_oid = movable[0].obj_id
            ev = trajectories[primary_oid].events[f]

            frames.append(
                TrackingFrame(
                    frame_idx=f,
                    timestamp_sec=f / self.fps,
                    rgb=out["rgb"].cpu(),
                    depth=noisy_d.cpu(),
                    obj_id=out["obj_id"].cpu(),
                    gt_delta={k: v.copy() for k, v in deltas.items()},
                    event=ev,
                )
            )

            if verbose and (f % 20 == 0 or f == n_frames - 1):
                print(f"  [Tracking {scenario.value}] Rendered frame {f + 1}/{n_frames} "
                      f"(yaw={ev.yaw_deg:.1f}°, {ev.constraint_status.value})")

        return TrackingSequence(
            scenario=scenario,
            frames=frames,
            camera=camera,
            object_ids=[obj.obj_id for obj in movable],
            trajectories=trajectories,
            fps=self.fps,
        )

    def export(self, seq: TrackingSequence, output_dir: str):
        """Saves frame images, depth maps, metadata and timelines to disk."""
        rgb_dir = os.path.join(output_dir, "rgb")
        depth_npy_dir = os.path.join(output_dir, "depth_npy")
        depth_vis_dir = os.path.join(output_dir, "depth_vis")
        masks_dir = os.path.join(output_dir, "masks")

        for d in [rgb_dir, depth_npy_dir, depth_vis_dir, masks_dir]:
            os.makedirs(d, exist_ok=True)

        timeline = []
        frames_meta = []

        for f_idx, fr in enumerate(seq.frames):
            prefix = f"frame_{f_idx:04d}"

            # RGB
            rgb_np = (fr.rgb.numpy() * 255.0).astype(np.uint8)
            cv2.imwrite(os.path.join(rgb_dir, f"{prefix}.png"), cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))

            # Depth
            d_np = fr.depth.numpy().astype(np.float32)
            np.save(os.path.join(depth_npy_dir, f"{prefix}.npy"), d_np)

            # Colormapped depth
            d_norm = cv2.normalize(d_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_colored = cv2.applyColorMap(d_norm, cv2.COLORMAP_TURBO)
            cv2.imwrite(os.path.join(depth_vis_dir, f"{prefix}.png"), depth_colored)

            # Mask
            mask_np = (fr.obj_id.numpy() * 60).clip(0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(masks_dir, f"{prefix}.png"), mask_np)

            # Frame metadata
            frames_meta.append({
                "frame": f_idx,
                "timestamp_sec": fr.timestamp_sec,
                "yaw_deg": fr.event.yaw_deg,
                "translation_cm": fr.event.translation_cm,
                "constraint_status": fr.event.constraint_status.value,
                "hand_occlusion": fr.event.hand_occlusion,
                "reset_event": fr.event.reset_event,
                "event_type": fr.event.event_type,
                "description": fr.event.description,
                "gt_deltas": {str(k): v.tolist() for k, v in fr.gt_delta.items()},
            })

            if fr.event.reset_event or fr.event.event_type != "NORMAL":
                timeline.append({
                    "frame": f_idx,
                    "timestamp_sec": fr.timestamp_sec,
                    "event": fr.event.event_type,
                    "yaw_deg": fr.event.yaw_deg,
                    "constraint_status": fr.event.constraint_status.value,
                    "description": fr.event.description,
                })

        # Save metadata and timeline JSONs
        meta = {
            "scenario": seq.scenario.value,
            "fps": seq.fps,
            "n_frames": len(seq.frames),
            "duration_sec": len(seq.frames) / seq.fps,
            "object_ids": seq.object_ids,
            "camera": {
                "c2w": seq.camera.c2w.cpu().numpy().tolist(),
                "fx": seq.camera.fx,
                "fy": seq.camera.fy,
                "cx": seq.camera.cx,
                "cy": seq.camera.cy,
                "width": seq.camera.width,
                "height": seq.camera.height,
            },
            "summaries": {str(oid): t.summary_metadata for oid, t in seq.trajectories.items()},
            "frames": frames_meta,
        }

        with open(os.path.join(output_dir, "tracking_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        with open(os.path.join(output_dir, "events_timeline.json"), "w", encoding="utf-8") as f:
            json.dump(timeline, f, indent=2)

        print(f"  [Tracking {seq.scenario.value}] Exported {len(seq.frames)} frames to: {output_dir}")
