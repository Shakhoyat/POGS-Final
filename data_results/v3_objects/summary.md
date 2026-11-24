# Tracking benchmark results

3 seed(s) per level; values are mean ± std over seeds. Target object only, except 'Static drift' (objects that never move).

## rotation

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yaw 30 | Static (no motion) | 17% ± 0% | 20% ± 0% | 21.99 | 75% ± 0% | 21.62 ± 1.59 | 15.00 ± 0.00 | 17.00 | 38.70 | 37.82 | - | - | - |
| yaw 30 | POGS | 9% ± 2% | 100% ± 0% | 0.79 | 0% ± 0% | 0.86 ± 0.06 | 25.86 ± 1.96 | 0.69 | 2.09 | 0.78 | - | 518 | 1.8 |
| yaw 30 | POGS (release: no per-frame depth) | 4% ± 1% | 26% ± 7% | 16.03 | 61% ± 2% | 14.84 ± 2.69 | 35.57 ± 9.07 | 11.44 | 26.70 | 24.42 | - | 426 | 2.2 |
| yaw 30 | POGS + KF | 8% ± 3% | 100% ± 0% | 0.77 | 0% ± 0% | 0.93 ± 0.06 | 30.73 ± 1.31 | 0.73 | 2.04 | 0.71 | - | 525 | 1.8 |

## speed

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x | Static (no motion) | 17% ± 0% | 20% ± 0% | 21.99 | 75% ± 0% | 21.62 ± 1.59 | 15.00 ± 0.00 | 17.00 | 38.70 | 37.82 | - | - | - |
| 1x | POGS | 9% ± 2% | 100% ± 0% | 0.79 | 0% ± 0% | 0.86 ± 0.06 | 25.86 ± 1.96 | 0.69 | 2.09 | 0.78 | - | 518 | 1.8 |
| 1x | POGS (release: no per-frame depth) | 4% ± 1% | 26% ± 7% | 16.03 | 61% ± 2% | 14.84 ± 2.69 | 35.57 ± 9.07 | 11.44 | 26.70 | 24.42 | - | 426 | 2.2 |
| 1x | POGS + KF | 8% ± 3% | 100% ± 0% | 0.77 | 0% ± 0% | 0.93 ± 0.06 | 30.73 ± 1.31 | 0.73 | 2.04 | 0.71 | - | 525 | 1.8 |

## occlusion

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gripper | Static (no motion) | 17% ± 0% | 20% ± 0% | 21.99 | 75% ± 0% | 21.62 ± 1.59 | 15.00 ± 0.00 | 17.00 | 38.70 | 37.82 | - | - | - |
| gripper | POGS | 9% ± 2% | 100% ± 0% | 0.79 | 0% ± 0% | 0.86 ± 0.06 | 25.86 ± 1.96 | 0.69 | 2.09 | 0.78 | - | 518 | 1.8 |
| gripper | POGS (release: no per-frame depth) | 4% ± 1% | 26% ± 7% | 16.03 | 61% ± 2% | 14.84 ± 2.69 | 35.57 ± 9.07 | 11.44 | 26.70 | 24.42 | - | 426 | 2.2 |
| gripper | POGS + KF | 8% ± 3% | 100% ± 0% | 0.77 | 0% ± 0% | 0.93 ± 0.06 | 30.73 ± 1.31 | 0.73 | 2.04 | 0.71 | - | 525 | 1.8 |

## depth

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stereo 3 mm | Static (no motion) | 17% ± 0% | 20% ± 0% | 21.99 | 75% ± 0% | 21.62 ± 1.59 | 15.00 ± 0.00 | 17.00 | 38.70 | 37.82 | - | - | - |
| stereo 3 mm | POGS | 9% ± 2% | 100% ± 0% | 0.79 | 0% ± 0% | 0.86 ± 0.06 | 25.86 ± 1.96 | 0.69 | 2.09 | 0.78 | - | 518 | 1.8 |
| stereo 3 mm | POGS (release: no per-frame depth) | 4% ± 1% | 26% ± 7% | 16.03 | 61% ± 2% | 14.84 ± 2.69 | 35.57 ± 9.07 | 11.44 | 26.70 | 24.42 | - | 426 | 2.2 |
| stereo 3 mm | POGS + KF | 8% ± 3% | 100% ± 0% | 0.77 | 0% ± 0% | 0.93 ± 0.06 | 30.73 ± 1.31 | 0.73 | 2.04 | 0.71 | - | 525 | 1.8 |

## object

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mug | Static (no motion) | 17% ± 0% | 20% ± 0% | 21.99 | 75% ± 0% | 21.62 ± 1.59 | 15.00 ± 0.00 | 17.00 | 38.70 | 37.82 | - | - | - |
| mug | POGS | 9% ± 2% | 100% ± 0% | 0.79 | 0% ± 0% | 0.86 ± 0.06 | 25.86 ± 1.96 | 0.69 | 2.09 | 0.78 | - | 518 | 1.8 |
| mug | POGS (release: no per-frame depth) | 4% ± 1% | 26% ± 7% | 16.03 | 61% ± 2% | 14.84 ± 2.69 | 35.57 ± 9.07 | 11.44 | 26.70 | 24.42 | - | 426 | 2.2 |
| mug | POGS + KF | 8% ± 3% | 100% ± 0% | 0.77 | 0% ± 0% | 0.93 ± 0.06 | 30.73 ± 1.31 | 0.73 | 2.04 | 0.71 | - | 525 | 1.8 |
| mustard bottle | Static (no motion) | 17% ± 0% | 20% ± 0% | 23.25 | 73% ± 5% | 23.03 ± 2.32 | 15.00 ± 0.00 | 19.08 | 41.64 | 40.86 | - | - | - |
| mustard bottle | POGS | 9% ± 13% | 100% ± 0% | 0.61 | 0% ± 0% | 0.60 ± 0.02 | 10.20 ± 2.06 | 0.38 | 0.95 | 0.64 | - | 520 | 1.8 |
| mustard bottle | POGS (release: no per-frame depth) | 1% ± 1% | 56% ± 8% | 2.85 | 3% ± 6% | 3.67 ± 0.69 | 14.09 ± 0.98 | 2.26 | 8.87 | 0.77 | - | 428 | 2.1 |
| mustard bottle | POGS + KF | 11% ± 8% | 100% ± 0% | 0.58 | 0% ± 0% | 0.60 ± 0.00 | 10.52 ± 1.45 | 0.35 | 0.96 | 0.64 | - | 554 | 1.7 |
| cereal box | Static (no motion) | 18% ± 1% | 27% ± 4% | 24.47 | 70% ± 6% | 24.39 ± 1.39 | 15.00 ± 0.00 | 18.70 | 44.59 | 43.92 | - | - | - |
| cereal box | POGS | 8% ± 0% | 46% ± 4% | 25.97 | 52% ± 3% | 32.06 ± 12.40 | 49.07 ± 1.74 | 28.71 | 71.29 | 70.00 | - | 532 | 1.8 |
| cereal box | POGS (release: no per-frame depth) | 8% ± 0% | 43% ± 4% | 7.58 | 37% ± 10% | 16.73 ± 9.91 | 57.99 ± 18.79 | 12.59 | 45.47 | 45.47 | - | 441 | 2.1 |
| cereal box | POGS + KF | 8% ± 0% | 62% ± 26% | 3.97 | 16% ± 21% | 12.47 ± 14.88 | 92.62 ± 15.04 | 10.64 | 43.95 | 43.10 | - | 558 | 1.7 |

## multi-object

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| one moving | Static (no motion) | 17% ± 0% | 20% ± 0% | 21.99 | 75% ± 0% | 21.62 ± 1.59 | 15.00 ± 0.00 | 17.00 | 38.70 | 37.82 | - | - | - |
| one moving | POGS | 9% ± 2% | 100% ± 0% | 0.79 | 0% ± 0% | 0.86 ± 0.06 | 25.86 ± 1.96 | 0.69 | 2.09 | 0.78 | - | 518 | 1.8 |
| one moving | POGS (release: no per-frame depth) | 4% ± 1% | 26% ± 7% | 16.03 | 61% ± 2% | 14.84 ± 2.69 | 35.57 ± 9.07 | 11.44 | 26.70 | 24.42 | - | 426 | 2.2 |
| one moving | POGS + KF | 8% ± 3% | 100% ± 0% | 0.77 | 0% ± 0% | 0.93 ± 0.06 | 30.73 ± 1.31 | 0.73 | 2.04 | 0.71 | - | 525 | 1.8 |
| two moving | Static (no motion) | 17% ± 0% | 20% ± 0% | 23.32 | 69% ± 3% | 23.23 ± 1.17 | 15.00 ± 0.00 | 18.73 | 42.30 | 41.64 | - | - | - |
| two moving | POGS | 12% ± 4% | 100% ± 0% | 0.89 | 0% ± 0% | 0.92 ± 0.06 | 28.31 ± 2.94 | 0.71 | 2.20 | 0.52 | - | 1064 | 0.9 |
| two moving | POGS (release: no per-frame depth) | 4% ± 1% | 27% ± 8% | 16.33 | 59% ± 6% | 13.28 ± 3.24 | 30.32 ± 10.41 | 10.00 | 30.99 | 29.61 | - | 868 | 1.1 |
| two moving | POGS + KF | 8% ± 2% | 96% ± 7% | 1.00 | 0% ± 0% | 1.18 ± 0.61 | 46.44 ± 27.85 | 0.84 | 3.18 | 0.88 | - | 1120 | 0.8 |

## all cases

| Method | Cases | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Static (no motion) | 12 | 17% | 22% | 23.26 | 72% | 23.07 | 15.00 | 18.37 | 41.81 | 41.06 | - | - | - |
| POGS | 12 | 10% | 86% | 7.07 | 13% | 8.61 | 28.36 | 7.62 | 19.13 | 17.98 | - | 658 | 1.6 |
| POGS (release: no per-frame depth) | 12 | 4% | 38% | 10.70 | 40% | 12.13 | 34.49 | 9.07 | 28.01 | 25.07 | - | 541 | 1.9 |
| POGS + KF | 12 | 9% | 89% | 1.58 | 4% | 3.79 | 45.08 | 3.14 | 12.53 | 11.33 | - | 689 | 1.5 |
