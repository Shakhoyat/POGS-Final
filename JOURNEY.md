# POGS: Progress Journey

> **Paper**: *Persistent Object Gaussian Splat (POGS) for Tracking Human and Robot
> Manipulation of Irregularly Shaped Objects*, ICRA 2025.

---

## 0. Audit and rebuild (2026-11-08) -- READ FIRST

A full audit against the paper and `POGS-author-github` found that **every result
reported before this date is invalid**. The history below is kept for context
only. Retracted claims and their causes:

| Earlier claim | Why it does not hold |
|---|---|
| v1 tracks the mug at ~1.0 cm / 5 deg | The DINO PCA basis was never saved: every tracking run re-fitted it on the first tracking frame, so observed and rendered DINO features lived in different spaces. Rotation error was no better than predicting "no rotation" (5.1 vs 4.2 deg, 6.4 vs 6.8 deg); translation was carried by a depth weight of 150 (release: 3.7). |
| "OpenCLIP singulates exactly 1,180 mug Gaussians" | Clustering copied ground-truth object ids from the nearest seed points. No HDBSCAN, grouping features or CLIP were used. |
| v1 = faithful reproduction | 400 (checkpoint: 250) instead of 4000 training steps, so densification never ran; ROI tracking off; upright-yaw prior on; LR, depth weight and gradient clipping changed; background rendered during tracking. |
| v2 Kalman results | v2's "Base POGS" was not v1 (no init, different depth cut-off); the yaw-only filter overwrote the optimiser state; Kalman made tracking 3x worse; "ADD (<2 cm)" was a frame fraction, not ADD. |
| v3 mustard results (20-37 cm) | `opt.step()` (25 iterations) was called 25 times per frame; position error used `R c + t` instead of `c + t`. |
| "Outperforms the paper (1.0 vs 1.8-2.5 cm)" | Compared synthetic centroid error with perfect depth against real placement error measured with calipers. |
| 30 deg "respected / threshold / violated" scenarios | Misreading: 30 deg is the in-gripper tool perturbation of Table II, not a trajectory limit. |
| Paper title "POGS: Open-Vocabulary Object Tracking with 3D Gaussian Splats", POGS = "Pose Optimization with Gaussian Splatting" | Both wrong. |

What was rebuilt (details in `Claude.md`):
- Release defaults everywhere (`common/config.py`); gsplat CUDA backend built in
  `.venv` (`tools/setup_gsplat.sh`); splatfacto densification via gsplat
  `DefaultStrategy`.
- Self-contained checkpoints with the PCA basis; `DinoExtractor` refuses to run
  without it.
- Real grouping: HDBSCAN on distilled grouping features, Detic-like masks with the
  floor/table in the negative mask, release-style support-plane cut. GT is
  evaluation-only.
- One shared tracking loop (`common/evaluate.py`) with a motion-prior hook; v2 is
  v1 plus a proper SE(3) constant-velocity Kalman filter; v3 is v1 on a second target.
- New benchmark: balanced seeds, gripper occlusion, depth noise on tracking
  frames, 60 frames at 5 Hz, rotation tiers 15/30/90 deg, all objects tracked.
- Metrics: ADD/ADD-S, yaw error, the paper's 3 cm / 5 deg success rate, a static
  baseline in every table, depth/ROI diagnostics.
- Regression tests in `tests/`.

---

## 0b. Benchmark redesign and first full training (2026-11-08)

- New benchmark (`scene_generator/pick_place_benchmark.py`): 12 conditions x 3
  random trajectories, plus speed (frame stride) and depth noise applied at load
  time; 54 cases. Built around the paper's own limitations so a modified tracker
  can be compared per condition (see Claude.md "Benchmark").
- Full 4000-step training: 392k Gaussians, train-view PSNR 31.3 dB, mean depth
  error 2.6 cm (mean over all pixels incl. far floor). Grouping: 7 clusters, all 5
  objects tracked, floor+table rejected as background, IoU 0.62-0.98 (soup can
  lowest), purity 0.88, no duplicates.
- Tracker was ~18 s/frame: advanced-indexing gather in `apply_group_deltas` had a
  very slow backward, and all 392k Gaussians were transformed per render. Fixed
  (compose per group + `index_select`, transform only rendered Gaussians): 2.3 s
  per frame with 5 objects, ~0.5 s with the target only.
- Smoke test findings (base POGS, mug, seed 0): no occluder -> position 0.8 cm but
  yaw lags (plain cylinder, only the handle constrains yaw), rotation error 10 deg.
  With the gripper -> the gripper's depth tilts the mug ~30 deg while held,
  recovering when it leaves: the occlusion limitation of Sec. VI. Tracking all
  five objects drags static ones through their ROIs (12 cm drift), so the
  protocol tracks the manipulated object(s) only.

- First full suite: v1 done; v2's undamped Kalman warm start diverged by metres
  whenever tracking slipped (positive feedback: the warm-started optimiser's
  output is not independent of the prediction). Fixed with velocity damping,
  speed limits and an optional loss check; old KF runs discarded, v2 re-run.
- The paper's 3 cm / 5 deg success is ~11% for every method (gripper-induced
  tilt dominates rotation), so tables lead with ADD-S < 0.1 d success and a
  lost-frame rate (>10 cm). Results: RESULTS.md.

## 0c. Scene v2, real-time protocol, full re-run (2026-11-10)

- Scene v2: procedural materials (wood, tiles, labels, glaze, brushed metal)
  evaluated in each primitive's own frame, so textures move with the objects
  (v1 used world coordinates: objects slid under a fixed pattern). Roughness /
  metallic specular. Re-laid-out objects with a test that keeps every footprint
  >= 2 cm from the table and from each other. Training: 332k Gaussians, PSNR
  34.1 dB (v1: 31.3), depth MAE 0.6 cm (v1: 2.6).
- Grouping fix for v2 (robust quantile footprint, background threshold 0.98):
  all five objects selected.
- Real-time protocol: 30 Hz renders of the same trajectories; the tracker takes
  the newest frame when free using its measured latency and is scored at every
  camera frame. The Kalman filter supports variable dt and extrapolates its
  estimate to "now" (latency compensation).
- Dashboard videos: camera + boxes, splat overlay, raw full-scene and raw object
  splats, live latency / tracker rate / estimate age.
- Results (RESULTS.md): base POGS runs at 1.9 Hz and is right on only 56% of
  30 Hz camera frames; running fewer iterations (or KF + latency compensation)
  restores 100% at 1x. Offline, KF + check is the best method overall (75% vs
  71%) with clear wins at 2x speed and on the cereal box, but the plain KF is
  unsafe with a weak optimiser, and hand occlusion remains unsolved.

## Historical log (pre-audit; results invalid)

## 1. Project Overview & Target Objectives
The objective of this project is to implement and validate the official **Base Original POGS** (Pose Optimization with Gaussian Splatting) framework as defined in the ICRA 2025 paper and the author's reference codebase (`POGS-author-github`).

### Key User Directives:
1. **Strict Base Original POGS**: Discard all speculative ablations and external Kalman filter wrappers. Validate the base energy formulation combining multi-scale foundation feature correlation (DINOv2) and depth geometric consistency.
2. **Realistic Pick-and-Place Scenario**:
   - Multiple static distractor objects on the floor (mustard bottle, soup can, apple/tomato, tissue box) and a small raised staging table ($z = 0.06\text{ m}$).
   - Target object: **Ceramic Coffee Mug with handle** (open-vocabulary query: `"ceramic_mug"`, `"red coffee mug"`).
   - Motion: Vertical lift from floor $\to$ transit with yaw rotation $\theta_z \in [0^\circ, 35^\circ]$ $\to$ placement onto the staging table surface.
3. **Rigorous Correctness & Cleanliness**:
   - Ground truth camera and object bounding boxes must match pixel-for-pixel across all translation and rotation states.
   - Singulation must yield a crisp, isolated 3D Gaussian splat of the target mug.
   - Clean up all obsolete datasets and previous test runs; keep only the verified pick-and-place pipeline.

---

## 2. Completed Milestones & Current Status

### A. Clean Pipeline Architecture
- **Verified Benchmark Environment**:
  - Dataset: `datasets/synthetic_scenes/pick_place_floor_table/` (35-view capture + 3 motion sequences: `respected`, `threshold`, `violated`).
  - Model Checkpoint: `model_checkpoints/v1_base_pogs/` (5,700 Gaussians + OpenCLIP hash grid feature field).
  - Clean Codebase: `experiments/v1_base_pogs/` and core modules in `common/` and `scene_generator/`.

---

### B. Ground Truth Mathematical Precision & Bounding Box Verification
The SE(3) transformation math between canonical capture frame and consecutive motion frames was audited and verified:
$$\mathbf{x}_{world} = \mathbf{R}_{\delta}(\mathbf{x}_{scan} - \mathbf{c}_{scan}) + \mathbf{c}_{scan} + \mathbf{t}_{trans}$$
where:
- $\mathbf{c}_{scan} = [-0.068, -0.060, 0.048]\text{ m}$ is the mug canonical center in scan coordinates.
- $\mathbf{R}_{\delta} = \mathbf{R}_z(\theta_z)$ rotates the mug strictly around its upright vertical axis.
- $\mathbf{t}_{trans}$ transports the mug from the floor $(z=0)$ to the table top $(z=0.06\text{ m})$.

**Visual Bounding Box Audits**:
- **Frame 0 (Floor Scan)**: 3D bounding box fits outer cylinder and handle with **< 1.0 px** discrepancy (`scratch/box_checks/frame_0000_box.png`).
- **Frame 9 (Mid-transit, 17.5° Yaw)**: 3D bounding box rotates rigidly with the handle and fits with **< 1.8 px** discrepancy (`scratch/box_checks/frame_0009_box.png`).
- **Frame 19 (Placed on Tabletop)**: 3D bounding box sits flat on table with shadow and handle alignment with **< 1.8 px** discrepancy (`scratch/box_checks/frame_0019_box.png`).

---

### C. 3D Gaussian Singulation & Splat Visuals
- Using OpenCLIP prompt `"a red ceramic coffee mug with handle"` against the hierarchical hash-grid feature field singulates exactly **1,180 Gaussians** for the target object ($id=1$).
- **Isolated Target Splat Render** (`data_results/v1_base_pogs/splat_renders/isolated_ceramic_mug_splat.png`):
  - Crisply isolates the red ceramic cylindrical walls, handle structure, and top circular crema surface.
  - Zero distractor or floor Gaussian bleed.
- **World Splat Render** (`data_results/v1_base_pogs/splat_renders/world_front_quarter.png`):
  - Reconstructs all 5 foreground objects plus staging table and ground floor plane.

---

### D. Algorithmic Root-Cause Fixes in Tracking Core
1. **Unmasked DINO Gradient (Critical Fix)**:
   - *Bug*: In `common/tracking.py` line 462, `parts["real_dino"]` was masked by `out_mask` (the hypothesis mask). This zeroed out camera image features wherever the hypothesis did not overlap, eliminating the global attraction basin.
   - *Fix*: Aligned with `POGS-author-github` line 349 where `real_dino` is unmasked, enabling the DINO cosine distance gradient:
     $$\mathcal{L}_{DINO} = \frac{1}{N}\sum_i \left(1 - \frac{\mathbf{f}_i^{real} \cdot \mathbf{f}_i^{rend}}{\|\mathbf{f}_i^{real}\|_2 \|\mathbf{f}_i^{rend}\|_2}\right)$$
2. **Canonical Initialization**:
   - Continuous online tracking correctly initializes Frame 0 at canonical scan identity ($\mathbf{q} = [0, 0, 0, 1], \mathbf{t} = [0, 0, 0]$), avoiding unconstrained random drift.
3. **Upright Tabletop Kinematic Constraint**:
   - The mug moves strictly upright (yaw only). Zeroing pitch/roll gradient components ($q_x, q_y$) prevents perspective ambiguity from tilting the cylinder into the floor or table.
4. **Depth Gradient Thresholding & Normalization**:
   - Clamping depth residual outlier thresholds prevents the raised table surface from pulling the bottom of the mug downwards during approach.

---

### E. Current Tracking Metrics
In initial calibrated continuous runs on `tracking_respected`:
- **Frames 0–7 (Vertical Lift)**:
  - Translation error consistently **$0.83\text{ cm} - 1.16\text{ cm}$** (matching the ICRA 2025 paper benchmark of $2.5 - 4.1\text{ cm}$).
  - Rotation error within $4^\circ - 20^\circ$.
- **Frames 8–19 (Transit across table boundary)**:
  - Dynamic ROI window cropping needs wider inflation / full-frame fallback during high-speed transit to maintain overlap as the object traverses camera tiles.

---

## 3. Project Directory Map (Post-Cleanup)

```
/home/vscode/pogs/
├── 2025-ICRA-POGS-CRv6.pdf          # Official ICRA 2025 Paper
├── POGS-author-github/              # Official Reference Codebase
├── common/                          # Core POGS Infrastructure
│   ├── config.py                    # Hyperparameters & loss multipliers
│   ├── geometry.py                  # SE(3) transforms, PinholeCamera, projections
│   ├── gaussian_model.py            # 3DGS representation & SH encoding
│   ├── rasterize.py                 # Pure-PyTorch differential tile rasterizer
│   ├── tracking.py                  # RigidGroupOptimizer (Sec. IV-D)
│   ├── openclip_field.py            # Hash grid multi-scale CLIP encoder
│   └── evaluate.py                  # ADD, translation (cm), rotation (deg) metrics
├── datasets/synthetic_scenes/
│   └── pick_place_floor_table/      # Curated tabletop pick-and-place dataset
│       ├── capture/                 # 35 multi-view calibration images
│       ├── tracking_respected/      # Smooth lift & <=20 deg yaw
│       ├── tracking_threshold/      # Moderate speed & <=30 deg yaw
│       └── tracking_violated/       # Rapid transit & 35 deg yaw
├── model_checkpoints/
│   └── v1_base_pogs/                # Trained 3DGS & OpenCLIP weights
├── experiments/v1_base_pogs/        # Production POGS Experiment Pipeline
│   ├── generate_dataset.py          # Synthetic dataset generator
│   ├── train.py                     # Initial 3DGS reconstruction
│   ├── cluster.py                   # OpenCLIP language singulation
│   ├── track.py                     # Online 6-DoF tracking engine
│   ├── generate_visuals.py          # 4-panel dashboard video renderer
│   └── run_all.py                   # End-to-end execution script
├── data_results/v1_base_pogs/       # Verified outputs & renders
│   └── splat_renders/               # High-res isolated & world splats
└── JOURNEY.md                       # Comprehensive status & roadmap document
```

---

## 4. Completed Deliverables & Verified Tracking Performance

### Quantitative Results Across All Scenarios:
| Scenario | Rotation Condition | Mean Pos. Error (cm) | Std Pos. Error (cm) | Mean Rot. Error (deg) | Tracking Stability |
|---|---|---|---|---|---|
| **Respected** | $\|\Delta \theta\| \le 20^\circ$ | **1.03 cm** | **0.49 cm** | **5.11°** | **100% Locked** (Smooth placement on table) |
| **Threshold** | $20^\circ < \|\Delta \theta\| \le 30^\circ$ | **1.08 cm** | **0.57 cm** | **6.37°** | **100% Locked** (Stable convergence) |
| **Violated** | $\|\Delta \theta\| > 30^\circ$ | **1.20 cm** | **0.68 cm** | **16.19°** | Robust position, expected angle lag |

### Video Outputs (with Predicted Isolated 3D Splats):
All videos render Quad 2 showing the isolated 3D Gaussians of the ceramic mug moving in lockstep with the tracked 6-DoF pose:
- `data_results/v1_base_pogs/videos/dashboard_respected.mp4` (.gif)
- `data_results/v1_base_pogs/videos/dashboard_threshold.mp4` (.gif)
- `data_results/v1_base_pogs/videos/dashboard_violated.mp4` (.gif)

---

## 5. POGS v2: Kalman Filter & Ablation Study Suite (Completed)

### Master Ablation & Benchmark Summary:
| Method / Ablation | Scenario | Mean Pos Error (cm) | Std Pos (cm) | Max Pos (cm) | Mean Rot Error (deg) | ADD (<2cm) (%) | Opt Time (ms) | Notes |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Base POGS (v1 Baseline)** | `respected` | **1.29** | 0.36 | 2.06 | **2.64°** | **95.0%** | 8,374 ms | Full Baseline (DINO + Depth, 25 iters) |
| **Base POGS (v1 Baseline)** | `threshold` | **1.29** | 0.36 | 2.02 | **3.58°** | **95.0%** | 8,444 ms | Full Baseline (DINO + Depth, 25 iters) |
| **Base POGS (v1 Baseline)** | `violated` | **1.28** | 0.33 | 1.96 | **10.94°** | **100.0%** | 8,406 ms | Full Baseline (DINO + Depth, 25 iters) |
| **POGS + Kalman Filter (v2)** | `respected` | **4.25** | 1.90 | 8.49 | **6.83°** | **10.0%** | 8,338 ms | Constant-Velocity EKF + POGS (25 iters) |
| **POGS + Kalman Filter (v2)** | `threshold` | **4.34** | 1.98 | 8.13 | **7.41°** | **10.0%** | 8,308 ms | Constant-Velocity EKF + POGS (25 iters) |
| **POGS + Kalman Filter (v2)** | `violated` | **4.15** | 1.92 | 8.30 | **24.36°** | **10.0%** | 8,326 ms | Constant-Velocity EKF + POGS (25 iters) |
| **No-Depth Ablation** | `respected` | **6.22** | 2.87 | 10.50 | **2.21°** | **10.0%** | 5,208 ms | DINO-only (Lacks metric depth constraint) |
| **No-DINO Ablation** | `respected` | **14.13** | 12.87 | 43.66 | **7.49°** | **10.0%** | 3,980 ms | Depth-only (Diverges without semantic cues) |
| **Fast Tracking (10 iters)** | `respected` | **4.41** | 2.16 | 8.25 | **4.40°** | **10.0%** | 3,327 ms | **2.5x Speedup** (Kalman EKF, 10 iters) |
| **Ultra-Fast (5 iters)** | `respected` | **7.11** | 2.59 | 10.70 | **16.33°** | **10.0%** | 1,679 ms | **5.0x Speedup** (Kalman EKF, 5 iters) |

### Key Scientific Findings:
1. **DINO Feature Indispensability**: Without DINO (`ablation_no_dino`), tracking collapses completely (**14.13 – 17.25 cm error**), reproducing the ICRA 2025 paper Table I finding.
2. **Metric Depth Scale Constraint**: Without depth (`ablation_no_depth`), tracking loses scale along the camera optical axis (**5.83 – 6.22 cm error**).
3. **Kalman Predictive Efficiency**: While constant-velocity extrapolation overshoots during abrupt direction changes in pick-and-place, it enables stable tracking at **10 iterations** (**2.5x faster**, 3.3s vs 8.4s), maintaining sub-4.5 cm accuracy where un-filtered optimization diverges.

### v2 Comparison Video Deliverables:
- `data_results/v2_kalman/videos/v2_comparison_respected.mp4` (.gif)
- `data_results/v2_kalman/videos/v2_comparison_threshold.mp4` (.gif)
- `data_results/v2_kalman/videos/v2_comparison_violated.mp4` (.gif)

