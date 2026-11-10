# Tracking benchmark results

3 seed(s) per level; values are mean ± std over seeds. Target object only, except 'Static drift' (objects that never move).

## rotation

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yaw 30 | Static (no motion) | 17% ± 0% | 20% ± 0% | 20.85 | 75% ± 0% | 20.35 ± 1.60 | 15.00 ± 0.00 | 15.30 | 36.08 | 35.14 | - | - | - |
| yaw 30 | POGS | 8% ± 0% | 100% ± 0% | 1.28 | 0% ± 0% | 1.19 ± 0.05 | 33.23 ± 1.18 | 0.77 | 1.82 | 1.34 | - | 514 | 1.8 |
| yaw 30 | POGS (release: no per-frame depth) | 7% ± 0% | 33% ± 1% | 6.60 | 22% ± 19% | 6.29 ± 0.72 | 37.60 ± 1.57 | 3.66 | 11.69 | 5.44 | - | 435 | 2.1 |
| yaw 30 | POGS + KF | 7% ± 1% | 100% ± 0% | 1.26 | 0% ± 0% | 1.17 ± 0.02 | 32.79 ± 1.05 | 0.75 | 1.90 | 0.80 | - | 501 | 1.9 |

## speed

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x | Static (no motion) | 17% ± 0% | 20% ± 0% | 20.85 | 75% ± 0% | 20.35 ± 1.60 | 15.00 ± 0.00 | 15.30 | 36.08 | 35.14 | - | - | - |
| 1x | POGS | 8% ± 0% | 100% ± 0% | 1.28 | 0% ± 0% | 1.19 ± 0.05 | 33.23 ± 1.18 | 0.77 | 1.82 | 1.34 | - | 514 | 1.8 |
| 1x | POGS (release: no per-frame depth) | 7% ± 0% | 33% ± 1% | 6.60 | 22% ± 19% | 6.29 ± 0.72 | 37.60 ± 1.57 | 3.66 | 11.69 | 5.44 | - | 435 | 2.1 |
| 1x | POGS + KF | 7% ± 1% | 100% ± 0% | 1.26 | 0% ± 0% | 1.17 ± 0.02 | 32.79 ± 1.05 | 0.75 | 1.90 | 0.80 | - | 501 | 1.9 |

## occlusion

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gripper | Static (no motion) | 17% ± 0% | 20% ± 0% | 20.85 | 75% ± 0% | 20.35 ± 1.60 | 15.00 ± 0.00 | 15.30 | 36.08 | 35.14 | - | - | - |
| gripper | POGS | 8% ± 0% | 100% ± 0% | 1.28 | 0% ± 0% | 1.19 ± 0.05 | 33.23 ± 1.18 | 0.77 | 1.82 | 1.34 | - | 514 | 1.8 |
| gripper | POGS (release: no per-frame depth) | 7% ± 0% | 33% ± 1% | 6.60 | 22% ± 19% | 6.29 ± 0.72 | 37.60 ± 1.57 | 3.66 | 11.69 | 5.44 | - | 435 | 2.1 |
| gripper | POGS + KF | 7% ± 1% | 100% ± 0% | 1.26 | 0% ± 0% | 1.17 ± 0.02 | 32.79 ± 1.05 | 0.75 | 1.90 | 0.80 | - | 501 | 1.9 |

## depth

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stereo 3 mm | Static (no motion) | 17% ± 0% | 20% ± 0% | 20.85 | 75% ± 0% | 20.35 ± 1.60 | 15.00 ± 0.00 | 15.30 | 36.08 | 35.14 | - | - | - |
| stereo 3 mm | POGS | 8% ± 0% | 100% ± 0% | 1.28 | 0% ± 0% | 1.19 ± 0.05 | 33.23 ± 1.18 | 0.77 | 1.82 | 1.34 | - | 514 | 1.8 |
| stereo 3 mm | POGS (release: no per-frame depth) | 7% ± 0% | 33% ± 1% | 6.60 | 22% ± 19% | 6.29 ± 0.72 | 37.60 ± 1.57 | 3.66 | 11.69 | 5.44 | - | 435 | 2.1 |
| stereo 3 mm | POGS + KF | 7% ± 1% | 100% ± 0% | 1.26 | 0% ± 0% | 1.17 ± 0.02 | 32.79 ± 1.05 | 0.75 | 1.90 | 0.80 | - | 501 | 1.9 |

## object

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mug | Static (no motion) | 17% ± 0% | 20% ± 0% | 20.85 | 75% ± 0% | 20.35 ± 1.60 | 15.00 ± 0.00 | 15.30 | 36.08 | 35.14 | - | - | - |
| mug | POGS | 8% ± 0% | 100% ± 0% | 1.28 | 0% ± 0% | 1.19 ± 0.05 | 33.23 ± 1.18 | 0.77 | 1.82 | 1.34 | - | 514 | 1.8 |
| mug | POGS (release: no per-frame depth) | 7% ± 0% | 33% ± 1% | 6.60 | 22% ± 19% | 6.29 ± 0.72 | 37.60 ± 1.57 | 3.66 | 11.69 | 5.44 | - | 435 | 2.1 |
| mug | POGS + KF | 7% ± 1% | 100% ± 0% | 1.26 | 0% ± 0% | 1.17 ± 0.02 | 32.79 ± 1.05 | 0.75 | 1.90 | 0.80 | - | 501 | 1.9 |
| mustard bottle | Static (no motion) | 17% ± 0% | 21% ± 1% | 22.44 | 73% ± 5% | 22.15 ± 2.33 | 15.00 ± 0.00 | 18.08 | 39.84 | 39.02 | - | - | - |
| mustard bottle | POGS | 4% ± 1% | 100% ± 0% | 0.69 | 0% ± 0% | 0.68 ± 0.03 | 24.35 ± 2.42 | 0.40 | 1.21 | 0.69 | - | 553 | 1.7 |
| mustard bottle | POGS (release: no per-frame depth) | 6% ± 1% | 67% ± 5% | 2.33 | 0% ± 0% | 2.71 ± 0.09 | 25.38 ± 2.81 | 1.52 | 7.80 | 7.80 | - | 430 | 2.1 |
| mustard bottle | POGS + KF | 6% ± 1% | 100% ± 0% | 0.67 | 0% ± 0% | 0.70 ± 0.01 | 22.37 ± 2.94 | 0.41 | 1.23 | 0.70 | - | 515 | 1.8 |
| cereal box | Static (no motion) | 18% ± 1% | 23% ± 1% | 23.84 | 69% ± 6% | 23.71 ± 1.38 | 15.00 ± 0.00 | 17.61 | 43.19 | 42.50 | - | - | - |
| cereal box | POGS | 1% ± 1% | 32% ± 1% | 51.47 | 66% ± 1% | 40.15 ± 0.94 | 31.66 ± 10.63 | 36.88 | 69.54 | 69.34 | - | 497 | 1.9 |
| cereal box | POGS (release: no per-frame depth) | 0% ± 0% | 31% ± 1% | 52.87 | 68% ± 0% | 41.53 ± 3.58 | 46.85 ± 0.62 | 35.94 | 72.48 | 71.95 | - | 489 | 1.9 |
| cereal box | POGS + KF | 0% ± 0% | 42% ± 7% | 31.07 | 54% ± 9% | 43.89 ± 18.95 | 60.78 ± 28.76 | 38.96 | 95.21 | 95.06 | - | 507 | 1.8 |

## multi-object

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| one moving | Static (no motion) | 17% ± 0% | 20% ± 0% | 20.85 | 75% ± 0% | 20.35 ± 1.60 | 15.00 ± 0.00 | 15.30 | 36.08 | 35.14 | - | - | - |
| one moving | POGS | 8% ± 0% | 100% ± 0% | 1.28 | 0% ± 0% | 1.19 ± 0.05 | 33.23 ± 1.18 | 0.77 | 1.82 | 1.34 | - | 514 | 1.8 |
| one moving | POGS (release: no per-frame depth) | 7% ± 0% | 33% ± 1% | 6.60 | 22% ± 19% | 6.29 ± 0.72 | 37.60 ± 1.57 | 3.66 | 11.69 | 5.44 | - | 435 | 2.1 |
| one moving | POGS + KF | 7% ± 1% | 100% ± 0% | 1.26 | 0% ± 0% | 1.17 ± 0.02 | 32.79 ± 1.05 | 0.75 | 1.90 | 0.80 | - | 501 | 1.9 |
| two moving | Static (no motion) | 17% ± 0% | 20% ± 0% | 22.17 | 69% ± 3% | 21.98 ± 1.13 | 15.00 ± 0.00 | 17.08 | 39.75 | 39.04 | - | - | - |
| two moving | POGS | 11% ± 2% | 100% ± 0% | 0.89 | 0% ± 0% | 0.92 ± 0.25 | 26.37 ± 6.57 | 0.67 | 1.69 | 0.73 | - | 989 | 1.0 |
| two moving | POGS (release: no per-frame depth) | 7% ± 0% | 31% ± 1% | 6.82 | 16% ± 27% | 6.26 ± 1.70 | 36.33 ± 9.35 | 3.87 | 10.91 | 7.36 | - | 814 | 1.1 |
| two moving | POGS + KF | 12% ± 3% | 100% ± 0% | 0.61 | 0% ± 0% | 0.69 ± 0.24 | 23.58 ± 6.73 | 0.66 | 1.53 | 0.43 | - | 1004 | 0.9 |

## all cases

| Method | Cases | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Static (no motion) | 12 | 17% | 21% | 22.33 | 72% | 22.05 | 15.00 | 17.02 | 39.71 | 38.92 | - | - | - |
| POGS | 12 | 6% | 83% | 13.58 | 17% | 10.73 | 28.90 | 9.68 | 18.56 | 18.02 | - | 638 | 1.6 |
| POGS (release: no per-frame depth) | 12 | 5% | 40% | 17.16 | 27% | 14.20 | 36.54 | 11.24 | 25.72 | 23.14 | - | 542 | 1.8 |
| POGS + KF | 12 | 6% | 85% | 8.40 | 14% | 11.61 | 34.88 | 10.20 | 24.97 | 24.25 | - | 632 | 1.6 |
