"""
Video annotation and visualization engine for tracking sequences.
Renders side-by-side RGB and Depth videos with professional HUD overlay
showing real-time angle perturbation, constraint adherence status,
translation distance, hand occlusion status, and timeline events.
"""

from __future__ import annotations

import os
from typing import List, Optional
import cv2
import numpy as np
from PIL import Image

from .trajectory_planner import AngleConstraintStatus
from .tracking_generator import TrackingSequence


class VideoAnnotator:
    """Renders annotated MP4 and GIF videos from tracking sequences."""

    @staticmethod
    def render_hud_frame(
        rgb_hwc: np.ndarray,
        depth_hw: np.ndarray,
        scenario_name: str,
        frame_idx: int,
        total_frames: int,
        time_sec: float,
        yaw_deg: float,
        trans_cm: float,
        constraint_status: AngleConstraintStatus,
        hand_occlusion: bool,
        reset_event: bool,
        description: str,
    ) -> np.ndarray:
        """Compose a side-by-side annotated video frame with HUD overlay."""
        H, W, _ = rgb_hwc.shape

        # Format RGB
        frame_bgr = cv2.cvtColor((rgb_hwc * 255.0).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Format Depth
        d_norm = cv2.normalize(depth_hw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_bgr = cv2.applyColorMap(d_norm, cv2.COLORMAP_TURBO)

        # Create side-by-side canvas: Left = RGB, Right = Depth
        canvas = np.hstack([frame_bgr, depth_bgr])
        canvas_h, canvas_w, _ = canvas.shape

        # Add top banner for HUD
        banner_h = 75
        canvas_with_hud = np.zeros((canvas_h + banner_h, canvas_w, 3), dtype=np.uint8)
        canvas_with_hud[banner_h:, :] = canvas

        # Dark HUD background
        cv2.rectangle(canvas_with_hud, (0, 0), (canvas_w, banner_h), (20, 24, 28), -1)
        cv2.line(canvas_with_hud, (0, banner_h), (canvas_w, banner_h), (60, 68, 76), 2)
        cv2.line(canvas_with_hud, (W, banner_h), (W, canvas_h + banner_h), (80, 88, 96), 2)

        # Title & Frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        title_text = f"POGS Tracking -- Scenario: {scenario_name.upper()}"
        cv2.putText(canvas_with_hud, title_text, (15, 24), font, 0.65, (240, 240, 240), 2, cv2.LINE_AA)

        info_text = f"Frame: {frame_idx + 1:03d}/{total_frames:03d}  |  Time: {time_sec:5.2f}s  |  Trans: {trans_cm:4.1f} cm"
        cv2.putText(canvas_with_hud, info_text, (15, 52), font, 0.52, (180, 190, 200), 1, cv2.LINE_AA)

        # Angle Constraint Badge
        abs_yaw = abs(yaw_deg)
        if constraint_status == AngleConstraintStatus.RESPECTED:
            badge_color = (40, 180, 60)      # Green
            status_text = f"YAW: {yaw_deg:+5.1f} deg [30 deg OK]"
        elif constraint_status == AngleConstraintStatus.NEAR_THRESHOLD:
            badge_color = (30, 200, 240)     # Amber/Yellow
            status_text = f"YAW: {yaw_deg:+5.1f} deg [NEAR THRESHOLD]"
        else:
            badge_color = (40, 40, 230)      # Red
            status_text = f"YAW: {yaw_deg:+5.1f} deg [VIOLATED >30 deg]"

        badge_x = canvas_w - 380
        cv2.rectangle(canvas_with_hud, (badge_x, 10), (canvas_w - 15, 42), badge_color, -1)
        cv2.putText(canvas_with_hud, status_text, (badge_x + 12, 32), font, 0.54, (255, 255, 255), 2, cv2.LINE_AA)

        # Alerts (Hand Occlusion & Reset)
        alert_y = 65
        if hand_occlusion:
            cv2.rectangle(canvas_with_hud, (badge_x, 48), (badge_x + 190, 70), (0, 120, 255), -1)
            cv2.putText(canvas_with_hud, "HAND OCCLUSION", (badge_x + 8, 64), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        if reset_event:
            cv2.rectangle(canvas_with_hud, (badge_x + 200, 48), (canvas_w - 15, 70), (180, 60, 180), -1)
            cv2.putText(canvas_with_hud, "RESET EVENT", (badge_x + 210, 64), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Sub-panel labels
        cv2.putText(canvas_with_hud, "RGB Camera (ZED-2 View)", (20, banner_h + 30), font, 0.55, (255, 255, 255), 2)
        cv2.putText(canvas_with_hud, "Metric Depth (Sensor Simulation)", (W + 20, banner_h + 30), font, 0.55, (255, 255, 255), 2)

        return canvas_with_hud

    @classmethod
    def export_video(
        cls,
        seq: TrackingSequence,
        output_path: str,
        fps: Optional[float] = None,
        save_gif: bool = True,
    ) -> str:
        """Renders complete tracking sequence into MP4 and optional GIF."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        video_fps = fps or seq.fps
        N = len(seq.frames)

        annotated_frames_bgr = []
        for f_idx, fr in enumerate(seq.frames):
            frame = cls.render_hud_frame(
                rgb_hwc=fr.rgb.numpy(),
                depth_hw=fr.depth.numpy(),
                scenario_name=seq.scenario.value,
                frame_idx=f_idx,
                total_frames=N,
                time_sec=fr.timestamp_sec,
                yaw_deg=fr.event.yaw_deg,
                trans_cm=fr.event.translation_cm,
                constraint_status=fr.event.constraint_status,
                hand_occlusion=fr.event.hand_occlusion,
                reset_event=fr.event.reset_event,
                description=fr.event.description,
            )
            annotated_frames_bgr.append(frame)

        if not annotated_frames_bgr:
            return ""

        H_out, W_out, _ = annotated_frames_bgr[0].shape

        # Write MP4 video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, video_fps, (W_out, H_out))
        for f in annotated_frames_bgr:
            writer.write(f)
        writer.release()
        print(f"  [Video] Wrote annotated tracking video to: {output_path}")

        # Also write GIF for immediate inline web / notebook viewing
        if save_gif:
            gif_path = os.path.splitext(output_path)[0] + ".gif"
            rgb_frames_pil = [
                Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).resize((W_out // 2, H_out // 2))
                for f in annotated_frames_bgr[::2]  # subsample to keep GIF size light
            ]
            if rgb_frames_pil:
                rgb_frames_pil[0].save(
                    gif_path,
                    save_all=True,
                    append_images=rgb_frames_pil[1:],
                    duration=int(1000 / (video_fps / 2)),
                    loop=0,
                )
                print(f"  [Video] Wrote animated GIF to: {gif_path}")

        return output_path
