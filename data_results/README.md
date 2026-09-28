# Data Results Directory

Stores tracking outputs, ablation tables, trajectory logs, and speed benchmark results.

## Subdirectories:
- `v1_baseline/`:
  - `tracking_results.csv`: Benchmark across respected, threshold, and violated scenarios.
  - `ablations.csv`: No Depth, No DINO, and full POGS comparison.
  - `part_deltas_traj__*.npy`: 6-DoF predicted trajectory matrices ($T \times K \times 4 \times 4$).
- `v2_kalman_opt/`:
  - `comparison_v1_vs_v2.csv`: Direct delta comparison between v1 baseline and v2.
  - `speed_study.csv`: Tracking rate (Hz) and error across 25, 15, and 8 iterations.
  - `kalman_sweep.csv`: Start-step engagement sweep (0, 2, 5, 10, 20).
