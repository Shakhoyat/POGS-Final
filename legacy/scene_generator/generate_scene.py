"""
Standalone CLI entrypoint for generating artificial scenes and tracking sequences.
Totally detached from POGS model/tracking training.

Usage:
    python -m scene_generator.generate_scene --scene tabletop_drill_iron --out datasets/synthetic_scenes
    python -m scene_generator.generate_scene --lightweight --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .scene_builder import build_scene
from .capture_generator import CaptureGenerator
from .tracking_generator import TrackingSequenceGenerator
from .trajectory_planner import TrajectoryScenario
from .video_annotator import VideoAnnotator


def generate_dataset_suite(
    scene_name: str = "tabletop_drill_iron",
    output_base: str = "datasets/synthetic_scenes",
    scenarios: list[str] = None,
    n_views: int = 35,
    n_frames: int = 60,
    width: int = 400,
    height: int = 300,
    device: str = "cpu",
    seed: int = 42,
    verbose: bool = True,
):
    scenarios = scenarios or ["respected", "threshold", "violated", "servoing"]
    scene_out_dir = os.path.join(output_base, scene_name)
    os.makedirs(scene_out_dir, exist_ok=True)

    print("=" * 72)
    print(f"POGS SYNTHETIC DATASET GENERATION: {scene_name}")
    print(f"Target Output Directory: {scene_out_dir}")
    print(f"Device: {device} | Seed: {seed} | Resolution: {width}x{height}")
    print(f"Scenarios: {', '.join(scenarios)}")
    print("=" * 72)

    # 1. Build Scene Objects
    scene_objects = build_scene(scene_name)
    movable = [obj for obj in scene_objects if obj.movable]
    print(f"Scene objects: {[obj.name for obj in scene_objects]}")
    print(f"Tracked targets: {[obj.name for obj in movable]}")

    # 2. Phase 1: Scene Capture (Multi-view hemisphere for POGS training)
    print("\n--- PHASE 1: Multi-View Hemisphere Capture (35 views) ---")
    t0 = time.time()
    capture_gen = CaptureGenerator(device=device)
    capture_data = capture_gen.generate(
        scene_objects=scene_objects,
        n_views=n_views,
        width=width,
        height=height,
        verbose=verbose,
    )
    capture_dir = os.path.join(scene_out_dir, "capture")
    capture_gen.export(capture_data, capture_dir)
    print(f"Phase 1 completed in {time.time() - t0:.1f}s. Initial point cloud: {len(capture_data.seed_points)} pts")

    # 3. Phase 3: Tracking Sequences across multiple angle-constraint scenarios
    print("\n--- PHASE 3: Tracking Sequences & Video Synthesis ---")
    track_gen = TrackingSequenceGenerator(device=device, fps=5.0, seed=seed)
    scenario_catalog = {}

    for sc_name in scenarios:
        sc_enum = TrajectoryScenario(sc_name)
        sc_dir = os.path.join(scene_out_dir, f"tracking_{sc_name}")
        print(f"\n[Scenario: {sc_name.upper()}] Generating {n_frames} frames...")
        t_sc = time.time()

        seq = track_gen.generate(
            scene_objects=scene_objects,
            scenario=sc_enum,
            n_frames=n_frames,
            n_resets=3,
            width=width,
            height=height,
            verbose=verbose,
        )

        track_gen.export(seq, sc_dir)

        # Render annotated video with HUD
        video_path = os.path.join(sc_dir, "tracking_annotated.mp4")
        VideoAnnotator.export_video(seq, video_path, fps=5.0, save_gif=True)

        sc_summary = {
            "scenario": sc_name,
            "n_frames": len(seq.frames),
            "video_mp4": os.path.relpath(video_path, scene_out_dir),
            "video_gif": os.path.relpath(os.path.splitext(video_path)[0] + ".gif", scene_out_dir),
            "tracking_metadata": os.path.relpath(os.path.join(sc_dir, "tracking_metadata.json"), scene_out_dir),
            "events_timeline": os.path.relpath(os.path.join(sc_dir, "events_timeline.json"), scene_out_dir),
            "summaries": {str(oid): t.summary_metadata for oid, t in seq.trajectories.items()},
        }
        scenario_catalog[sc_name] = sc_summary
        print(f"[Scenario: {sc_name.upper()}] Done in {time.time() - t_sc:.1f}s.")

    # 4. Master Catalog JSON and Readme
    master_catalog = {
        "scene_name": scene_name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "resolution": [width, height],
        "objects": [{"name": o.name, "id": o.obj_id, "movable": o.movable} for o in scene_objects],
        "capture": {
            "views": n_views,
            "path": os.path.relpath(capture_dir, scene_out_dir),
            "fused_pointcloud": os.path.relpath(os.path.join(capture_dir, "fused_pointcloud.ply"), scene_out_dir),
        },
        "scenarios": scenario_catalog,
    }

    catalog_path = os.path.join(scene_out_dir, "scene_catalog.json")
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(master_catalog, f, indent=2)

    # Markdown Summary
    readme_md = f"""# Synthetic Scene Dataset: `{scene_name}`

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Overview
- **Tracked Objects**: {', '.join([o.name for o in movable])}
- **Capture Views**: {n_views} hemispherical RGB-D views (`capture/`)
- **Initial Point Cloud**: `capture/fused_pointcloud.ply`

## Tracking Scenarios
| Scenario | Condition | Max Yaw (deg) | Status | Video Preview |
|---|---|---|---|---|
"""
    for sc_name, sc_info in scenario_catalog.items():
        first_obj_summary = list(sc_info["summaries"].values())[0]
        max_yaw = first_obj_summary.get("max_yaw_deg", 0.0)
        cond = (
            "Rotations <= 30 deg" if sc_name == "respected" else
            "Rotations near 30 deg boundary" if sc_name == "threshold" else
            "Rotations exceeding 30 deg" if sc_name == "violated" else
            "15 deg / 30 deg Servoing pulses"
        )
        readme_md += f"| **{sc_name}** | {cond} | {max_yaw:.1f}° | `{first_obj_summary.get('num_respected_frames', 0)} ok / {first_obj_summary.get('num_violated_frames', 0)} viol` | [{sc_info['video_mp4']}]({sc_info['video_mp4']}) |\n"

    readme_md += """
## Directory Structure
- `capture/`: Phase 1 multi-view images (`rgb/`), depth (`depth_npy/`, `depth_vis/`), instance masks, `poses.json`, and `fused_pointcloud.ply`.
- `tracking_<scenario>/`: Phase 3 tracking RGB (`rgb/`), depth (`depth_npy/`, `depth_vis/`), `tracking_annotated.mp4`, `tracking_annotated.gif`, and `events_timeline.json`.
"""

    with open(os.path.join(scene_out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_md)

    print("\n" + "=" * 72)
    print(f"DATASET GENERATION COMPLETE!")
    print(f"Catalog saved to: {catalog_path}")
    print(f"Readme saved to: {os.path.join(scene_out_dir, 'README.md')}")
    print("=" * 72)
    return master_catalog


def main():
    parser = argparse.ArgumentParser(description="Standalone Synthetic Scene & Dataset Generator for POGS")
    parser.add_argument("--scene", type=str, default="tabletop_drill_iron",
                        choices=["tabletop_drill_iron", "tabletop_shoe_box", "tabletop_jigsaw_shelf"],
                        help="Scene configuration preset")
    parser.add_argument("--out", type=str, default="datasets/synthetic_scenes",
                        help="Base output directory for generated datasets")
    parser.add_argument("--scenarios", nargs="+", default=["respected", "threshold", "violated", "servoing"],
                        help="Scenarios to generate: respected, threshold, violated, servoing")
    parser.add_argument("--n_views", type=int, default=35, help="Number of hemisphere capture views")
    parser.add_argument("--n_frames", type=int, default=60, help="Number of tracking frames per scenario")
    parser.add_argument("--width", type=int, default=400, help="Image width")
    parser.add_argument("--height", type=int, default=300, help="Image height")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device (cpu, cuda:0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lightweight", action="store_true",
                        help="Quick test run with fewer views (10) and frames (20)")
    args = parser.parse_args()

    n_views = 10 if args.lightweight else args.n_views
    n_frames = 20 if args.lightweight else args.n_frames
    width = 240 if args.lightweight else args.width
    height = 180 if args.lightweight else args.height

    generate_dataset_suite(
        scene_name=args.scene,
        output_base=args.out,
        scenarios=args.scenarios,
        n_views=n_views,
        n_frames=n_frames,
        width=width,
        height=height,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
