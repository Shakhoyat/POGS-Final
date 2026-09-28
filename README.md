# Persistent Object Gaussian Splatting (POGS)

Implementation and research codebase for **Persistent Object Gaussian Splat (POGS) for Tracking Human and Robot Manipulation of Irregularly Shaped Objects** (ICRA 2025).

Modularized, decoupled from monolithic notebooks, and structured for reproducible 6-DoF tracking and evaluation.

---

## Key Features

1. **Standalone Synthetic Scene & Dataset Generator (`scene_generator/`)**:
   - Closed-form analytic 3D scene raycaster generating hole-free RGB, depth, and instance masks.
   - Generates Phase 1 multi-view hemisphere scans (35 views) and fused 3D point clouds (`.ply`).
   - Generates Phase 3 tracking sequences under 4 explicit angle constraint scenarios:
     - **Respected**: Rotations $\le 30^\circ$ (sub-centimeter tracking).
     - **Threshold**: Rotations near the $30^\circ$ boundary ($26^\circ - 32^\circ$).
     - **Violated**: Rotations exceeding $30^\circ$ ($45^\circ - 90^\circ$).
     - **Servoing**: Tool servoing rotation pulses ($15^\circ$ and $30^\circ$).
   - Renders side-by-side RGB and Turbo-Depth videos (`.mp4` / `.gif`) with real-time HUD showing rotation angles, translation, hand occlusions, and reset events.
   - Saves frame-by-frame metadata (`tracking_metadata.json`) and event timelines (`events_timeline.json`).

2. **Common Shared Core (`common/`)**:
   - Differentiable Gaussian Splatting rasterizer (`gsplat` with pure PyTorch fallback).
   - Foundation models: DINOv2 tracking features and OpenCLIP language queries.
   - 2-Phase trainer with densification and opacity resets.
   - Object clustering into persistent representations.
   - 6-DoF tracking optimizer (`RigidGroupOptimizer`).
   - Occluded grasp sampling directly on persistent 3D geometry.
   - Closed-loop tool visual servoing.

3. **Versioned Experiments (`experiments/`)**:
   - **`v1_base_pogs`**: Faithful reproduction of the ICRA 2025 baseline on the pick-and-place tabletop benchmark.

---

## Directory Structure

```
├── common/                  # Core shared POGS algorithms & models
├── scene_generator/         # Standalone synthetic dataset generator
├── experiments/             # Versioned experiment suites
│   └── v1_base_pogs/        # Canonical POGS baseline
├── datasets/                # Shared datasets (synthetic & real)
├── model_checkpoints/       # Shared model checkpoints
├── data_results/            # Experiment CSVs, metrics, and logs
├── base_notebooks/          # Base exploratory notebooks & references
├── Claude.md                # Detailed project instructions for AI / humans
├── RESULTS.md               # Master quantitative results & findings
└── README.md                # This file
```

---

## Quick Start

### 1. Generate Synthetic Pick-and-Place Benchmark
```bash
python -m experiments.v1_base_pogs.generate_dataset
```

### 2. Train 3D Gaussian Representation
```bash
python -m experiments.v1_base_pogs.train
```

### 3. Singulate Target Object via OpenCLIP
```bash
python -m experiments.v1_base_pogs.cluster
```

### 4. Run Online 6-DoF Tracking Benchmark
```bash
python -m experiments.v1_base_pogs.track
```

### 5. Generate Multi-Panel Telemetry Dashboard Videos
```bash
python -m experiments.v1_base_pogs.generate_visuals
```

For complete instructions and benchmark results, consult [Claude.md](Claude.md), [JOURNEY.md](JOURNEY.md), and [RESULTS.md](RESULTS.md).
