# POGS: Persistent Object Gaussian Splatting (Research Architecture)

This guide documents the modular architecture, experiment framework, synthetic scene generation, and execution workflows for POGS.

---

## 1. Project Background & Motivation

### The Paper
- **Paper**: *Persistent Object Gaussian Splat (POGS) for Tracking Human and Robot Manipulation of Irregularly Shaped Objects* (ICRA 2025).
- **Core Principle**: Builds a persistent 3D Gaussian Splatting representation of tabletop scenes from an initial hemispherical multi-view scan (Phase 1). Clusters Gaussians into 3D persistent objects with shared geometric and foundation model (DINOv2, OpenCLIP) feature fields (Phase 2). Tracks 6-DoF object poses online under dynamic human/robot manipulation and heavy hand occlusions without needing CAD models (Phase 3).
- **Key Experimental Findings**:
  - POGS recovers from in-grasp tool perturbations of up to **30°** (Section V-B, Table II). At 15°, recovery success is 96% (avg 6.01s). At 30°, higher object deltas cause tracking degradation, dropping recovery to 80% (avg 12.66s).
  - Robust tracking despite severe human hand occlusions.
  - Full occluded grasp sampling directly from the persistent 3D Gaussian representation (Section IV-F, Fig. 3).

### Why Modular Decomposition Was Required
The official authors did not publish their real robot dataset. An initial exploratory notebook was drafted (`base_notebooks/pogs-draft.ipynb`) with PyTorch-native Gaussian rasterization, synthetic raycasting, foundation feature fields, and tracking. However, running multi-hour experiments, conducting ablation studies, and modifying tracking dynamics in a notebook is error-prone. This repository decomposes the entire pipeline into clean, modular Python modules with shared core libraries, versioned experiment suites (`experiments/`), and detached dataset generation.

---

## 2. Repository Architecture

```
f:/pogs/
├── 2025-ICRA-POGS-CRv6.pdf          # Official ICRA 2025 Paper
├── POGS-author-github/              # Official Reference Codebase
├── base_notebooks/                  # Base exploratory notebooks & references
│   └── pogs-draft.ipynb
├── common/                          # Shared core POGS library
│   ├── config.py                    # Dataclasses for all pipeline configs
│   ├── geometry.py                  # SE(3), SO(3), quaternions, pinhole camera
│   ├── raster.py                   # Differentiable Gaussian rasterizer (gsplat & PyTorch fallback)
│   ├── fields.py                    # GaussianField: language & DINO grouping heads
│   ├── encoders.py                  # DINOv2 & OpenCLIP foundation feature extractors
│   ├── losses.py                    # RGB, depth, semantic, and densification loss terms
│   ├── model.py                     # Unified POGSModel (Gaussians + feature fields)
│   ├── trainer.py                   # 2-Phase training loop with densification & alpha reset
│   ├── grouping.py                  # DBSCAN & persistent object representation
│   ├── tracking.py                  # RigidGroupOptimizer for online 6-DoF tracking
│   ├── evaluate.py                  # ADD/ADD-S, translation/rotation error evaluation
│   ├── grasp.py                     # Occluded grasp sampling from persistent Gaussians
│   ├── servoing.py                  # Closed-loop tool visual servoing simulation
│   └── viz.py                       # Visualization & trajectory plotting utilities
│
├── scene_generator/                 # Standalone, detached synthetic dataset generator
│   ├── primitives.py                # Analytic 3D shapes: box, ellipsoid, cylinder, raycasting
│   ├── objects.py                   # Tabletop objects library (mug, table, floor, robot arm)
│   ├── scene_builder.py             # FlatScene vectorized raycaster
│   ├── trajectory_planner.py        # Multi-scenario trajectory planner (30° angle constraint modes)
│   ├── capture_generator.py         # Phase 1: 35 hemisphere views, depth, fused point cloud (.ply)
│   ├── tracking_generator.py        # Phase 3: RGB-D tracking sequence with hand occlusion
│   ├── video_annotator.py           # Annotated side-by-side RGB/Depth MP4/GIF videos with HUD
│   └── generate_scene.py            # Standalone CLI entrypoint
│
├── experiments/                     # Versioned experiment suites
│   └── v1_base_pogs/                # Baseline reproduction of ICRA 2025 POGS
│       ├── config.py                # Baseline config (25 iters, no motion prior)
│       ├── generate_dataset.py      # Benchmark dataset generator
│       ├── train.py                 # POGS model training
│       ├── cluster.py               # Object clustering
│       ├── track.py                 # 3-scenario tracking evaluation
│       ├── generate_visuals.py      # Multi-panel telemetry dashboard videos
│       └── run_all.py               # End-to-end execution
│
├── datasets/                        # Shared datasets directory
│   └── synthetic_scenes/            # Generated synthetic scenes with images, depth, videos, metadata
│       └── pick_place_floor_table/  # Standardized pick-and-place tabletop benchmark
│
├── model_checkpoints/               # Shared model weights and checkpoints
│   └── v1_base_pogs/                # cluster_labels.npy, object_clusters.json, train_summary.json
│
├── data_results/                    # Master tables, experiment logs, and metrics
│   └── v1_base_pogs/                # tracking metrics, CSVs, and videos
│
├── RESULTS.md                       # Master benchmark tables & findings
├── JOURNEY.md                       # Research milestones and validation log
└── README.md                        # High-level overview
```

---

## 3. Synthetic Scene & Dataset Generation

The `scene_generator` package is **completely detached** from the rest of the codebase. It can generate complete datasets without training a POGS model.

### Output Structure
When executed, `scene_generator` produces:
1. **Phase 1 Multi-View Hemisphere Capture (`capture/`)**:
   - 35 RGB images (`rgb/view_000.png` ... `view_034.png`)
   - 35 metric depth maps (`depth_npy/view_000.npy`)
   - 35 colormapped depth images (`depth_vis/view_000.png`) using the Turbo colormap
   - 35 instance segmentation masks (`masks/view_000.png`)
   - Camera extrinsics and intrinsics (`poses.json`)
   - Initial 3D fused point cloud (`fused_pointcloud.ply`) after DBSCAN spatial filtering
   - Metadata (`capture_metadata.json`)
2. **Phase 3 Tracking Sequences across 3 Scenarios**:
   - `tracking_respected/`: Rotation angles strictly respect $|\theta| \le 20^\circ$.
   - `tracking_threshold/`: Rotation angles approach/hover at the 30° threshold (20°–30°).
   - `tracking_violated/`: Rotation angles significantly exceed the threshold (>30°).
3. **Annotated Tracking Videos (`tracking_annotated.mp4` and `.gif`)**:
   - Side-by-side layout: Left = RGB with crosshairs, Right = Turbo Depth.
   - Real-time HUD banner displaying:
     - Frame count and elapsed time
     - Real-time yaw rotation angle $\theta$
     - Color-coded badge:
       - **GREEN**: `YAW: +18.4° [30° OK]`
       - **AMBER**: `YAW: +29.1° [NEAR THRESHOLD]`
       - **RED**: `YAW: +54.8° [VIOLATED >30°]`
4. **Structured Event Logs**:
   - `tracking_metadata.json`: Per-frame 4x4 SE(3) ground-truth matrices, translation norms, and constraint flags.

---

## 4. Experiment Structure & Baseline Comparison

- **`experiments/v1_base_pogs`**:
  - The canonical baseline implementation matching ICRA 2025.
  - Uses 25 Levenberg-Marquardt / GD tracking iterations per frame.
  - Pure causal decoupled tracking without Kalman filtering.
  - Serves as the ground-truth benchmark against which all future versions are evaluated.
- **Creating Future Versions (`v2`, `v3`, ...)**:
  1. Create `experiments/v<version>_<name>/`.
  2. Implement `config.py` inheriting/modifying `common.config.POGSConfig`.
  3. Implement tracking and evaluation scripts logging to `data_results/v<version>_<name>/`.
  4. Append results to `RESULTS.md`.

---

## 5. Critical Operational Guidelines

1. **Reproducible Pipeline Execution**:
   - Each experiment module is self-contained with a `run_all.py` entrypoint.
2. **Never Break Baseline Compatibility**:
   - `common/` is the shared library. Changes to `common/` must preserve backward compatibility with `experiments/v1_base_pogs/`.
3. **Artifact Logging**:
   - All experiments must write machine-readable `.csv` and human-readable `.md` tables to `data_results/<version>/`.
