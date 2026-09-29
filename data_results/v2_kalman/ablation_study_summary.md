# POGS v2: Kalman Filter & Ablation Study Benchmark Table

| Method / Ablation | Scenario | Mean Pos Error (cm) | Std Pos (cm) | Max Pos (cm) | Mean Rot Error (deg) | ADD (<2cm) (%) | Opt Time (ms) | FPS | Notes |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **pogs_kalman** | `respected` | **4.25** | 1.90 | 8.49 | **6.83°** | **10.0%** | 8337.9 | 0.1 | Kalman EKF |
| **pogs_kalman** | `threshold` | **4.34** | 1.98 | 8.13 | **7.41°** | **10.0%** | 8308.2 | 0.1 | Kalman EKF |
| **pogs_kalman** | `violated` | **4.15** | 1.92 | 8.30 | **24.36°** | **10.0%** | 8326.1 | 0.1 | Kalman EKF |
| **pogs_base** | `respected` | **1.29** | 0.36 | 2.06 | **2.64°** | **95.0%** | 8374.4 | 0.1 | Full Baseline |
| **pogs_base** | `threshold` | **1.29** | 0.36 | 2.02 | **3.58°** | **95.0%** | 8444.6 | 0.1 | Full Baseline |
| **pogs_base** | `violated` | **1.28** | 0.33 | 1.96 | **10.94°** | **100.0%** | 8406.5 | 0.1 | Full Baseline |
| **ablation_no_depth** | `respected` | **6.22** | 2.87 | 10.50 | **2.21°** | **10.0%** | 5208.4 | 0.2 | Kalman EKF, No-Depth |
| **ablation_no_depth** | `threshold` | **6.07** | 2.82 | 10.07 | **3.72°** | **10.0%** | 5212.2 | 0.2 | Kalman EKF, No-Depth |
| **ablation_no_depth** | `violated` | **5.83** | 2.78 | 9.90 | **16.56°** | **10.0%** | 5175.3 | 0.2 | Kalman EKF, No-Depth |
| **ablation_no_dino** | `respected` | **14.13** | 12.87 | 43.66 | **7.49°** | **10.0%** | 3979.7 | 0.3 | Kalman EKF, No-DINO |
| **ablation_no_dino** | `threshold` | **14.41** | 13.16 | 44.42 | **8.12°** | **10.0%** | 3942.3 | 0.3 | Kalman EKF, No-DINO |
| **ablation_no_dino** | `violated` | **17.25** | 15.98 | 51.36 | **22.17°** | **10.0%** | 3880.3 | 0.3 | Kalman EKF, No-DINO |
| **ablation_fast_10iters** | `respected` | **4.41** | 2.16 | 8.25 | **4.40°** | **10.0%** | 3327.0 | 0.3 | Kalman EKF, 10 iters |
| **ablation_fast_10iters** | `threshold` | **4.41** | 2.16 | 8.30 | **7.83°** | **15.0%** | 3354.8 | 0.3 | Kalman EKF, 10 iters |
| **ablation_fast_10iters** | `violated` | **4.30** | 2.22 | 8.27 | **21.09°** | **15.0%** | 3342.2 | 0.3 | Kalman EKF, 10 iters |
| **ablation_ultra_5iters** | `respected` | **7.11** | 2.59 | 10.70 | **16.33°** | **10.0%** | 1678.6 | 0.6 | Kalman EKF, 5 iters |
| **ablation_ultra_5iters** | `threshold` | **7.18** | 2.62 | 10.70 | **18.93°** | **10.0%** | 1683.5 | 0.6 | Kalman EKF, 5 iters |
| **ablation_ultra_5iters** | `violated` | **7.34** | 2.68 | 10.70 | **31.65°** | **10.0%** | 1669.8 | 0.6 | Kalman EKF, 5 iters |
