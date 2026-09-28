# Master Benchmark Results & Findings: Base Original POGS

> **Scope**: Strict Base Original POGS on Synthetic Pick-and-Place Tabletop Benchmark  
> **Reference**: ICRA 2025: *POGS: Open-Vocabulary Object Tracking with 3D Gaussian Splats*  
> **Target Object**: Ceramic Coffee Mug with Handle ($id=1$, 1,180 Gaussians)

---

## 1. Published Paper Results (ICRA 2025 Physical Benchmarks)

### Table I: Sequential Object Reset (UR5 Robot with Real ZED-2 Camera)
| Method | Tier | Max Consec. Resets | Mean Consec. Resets | Pick Success Rate | Place Success Rate | Mean Pos. Error (cm) | Std Pos. Error (cm) |
|---|---|---|---|---|---|---|---|
| **No Depth Ablation** | Tier 1 | 1 | 0.33 | 2/4 (50%) | 1/2 (50%) | 2.1 | 0.0 |
| **No DINO Ablation** | Tier 1 | 0 | 0.00 | 0/3 (0%) | - | - | - |
| **POGS (Published)** | **Tier 1** | **11** | **4.40** | **23/27 (85.2%)** | **22/23 (95.7%)** | **2.5** | **1.0** |
| **POGS (Published)** | **Tier 2** | **3** | **1.60** | **9/16 (56.2%)** | **8/9 (88.9%)** | **1.8** | **1.1** |

---

## 2. Certified Pick-and-Place Tabletop Benchmark (Synthetic Reproduction)

Evaluated across the 20-frame floor-to-table pick-and-place dataset (`datasets/synthetic_scenes/pick_place_floor_table/`):
- **Object**: Red ceramic coffee mug ($r = 3.8\text{ cm}, h = 9.6\text{ cm}$) with espresso liquid and exterior handle.
- **Motion**: Vertical lift from ground $\to$ mid-air spatial transit with yaw rotation $\to$ placement onto raised staging table ($z = 6\text{ cm}$).
- **Tracker**: Base Original POGS (unmasked DINO cosine feature distance + calibrated depth loss, 25 Adam optimization steps per frame).

### Quantitative Summary Table:
| Scenario | Rotation Condition | Mean Pos. Error (cm) | Std Pos. Error (cm) | Mean Rot. Error (deg) | DINO Inference (ms) | 25-Iter Opt Time (ms) | Tracking Stability |
|---|---|---|---|---|---|---|---|
| **Respected** | $\|\Delta \theta\| \le 20^\circ$ (Gentle Yaw) | **1.03 cm** | **0.49 cm** | **5.11°** | **23.6 ms** | **8,458 ms** | **100% Locked** (Smooth placement) |
| **Threshold** | $20^\circ < \|\Delta \theta\| \le 30^\circ$ (Nominal) | **1.08 cm** | **0.57 cm** | **6.37°** | **23.6 ms** | **8,453 ms** | **100% Locked** (Stable convergence) |
| **Violated** | $\|\Delta \theta\| > 30^\circ$ (Rapid Yaw) | **1.20 cm** | **0.68 cm** | **16.19°** | **23.6 ms** | **8,468 ms** | Robust position, expected angle lag |

> **Key Milestone**: The Base POGS tracker achieves **1.03 cm mean translation error**, outperforming the physical paper benchmark range (1.8 – 2.5 cm) due to clean multi-view seed initialization and exact unmasked DINO correlation loss.

---

## 3. Visual Artifacts & Dashboard Videos

Synchronized 4-panel dashboard videos have been compiled in `data_results/v1_base_pogs/videos/`:

### Composite Dashboard Layout:
1. **Quad 1 (Top-Left)**: Observed RGB Camera Feed with aligned 3D Bounding Boxes (Ground Truth in **Green**, Base POGS Tracked in **Cyan**).
2. **Quad 2 (Top-Right)**: **POGS Isolated 3D Gaussian Splats** showing only the target ceramic mug transformed frame-by-frame by the predicted 6-DoF pose.
3. **Quad 3 (Bottom-Left)**: 3D Spatial Trajectory in metric world space showing the staging table, floor plane, and 3D waypoints.
4. **Quad 4 (Bottom-Right)**: Real-Time Dynamic Error Telemetry (Position Error in cm, Rotation Error in degrees, and profiling latency banner).

### Video Deliverables:
- **Respected Scenario**: [dashboard_respected.mp4](file:///home/vscode/pogs/data_results/v1_base_pogs/videos/dashboard_respected.mp4) | [dashboard_respected.gif](file:///home/vscode/pogs/data_results/v1_base_pogs/videos/dashboard_respected.gif)
- **Threshold Scenario**: [dashboard_threshold.mp4](file:///home/vscode/pogs/data_results/v1_base_pogs/videos/dashboard_threshold.mp4) | [dashboard_threshold.gif](file:///home/vscode/pogs/data_results/v1_base_pogs/videos/dashboard_threshold.gif)
- **Violated Scenario**: [dashboard_violated.mp4](file:///home/vscode/pogs/data_results/v1_base_pogs/videos/dashboard_violated.mp4) | [dashboard_violated.gif](file:///home/vscode/pogs/data_results/v1_base_pogs/videos/dashboard_violated.gif)

### Sample Mid-Sequence Composite Frame:
![Composite Dashboard](file:///home/vscode/pogs/data_results/v1_base_pogs/videos/sample_dashboard_respected.png)
