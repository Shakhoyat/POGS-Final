# Tracking benchmark results

3 seed(s) per level; values are mean ± std over seeds. Target object only, except 'Static drift' (objects that never move).

## rotation

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yaw 30 | Static (no motion) | 17% ± 0% | 20% ± 0% | 21.99 | 75% ± 0% | 21.62 ± 1.59 | 15.00 ± 0.00 | 15.84 | 38.70 | 37.82 | - | - | - |
| yaw 30 | POGS | 13% ± 2% | 100% ± 0% | 0.78 | 0% ± 0% | 0.84 ± 0.05 | 23.06 ± 0.93 | 0.71 | 2.33 | 0.79 | - | 538 | 1.8 |
| yaw 30 | POGS (release: no per-frame depth) | 13% ± 2% | 47% ± 21% | 6.14 | 23% ± 25% | 7.22 ± 5.26 | 26.72 ± 2.10 | 4.86 | 15.04 | 11.17 | - | 425 | 2.2 |
| yaw 30 | POGS + KF | 13% ± 1% | 100% ± 0% | 0.74 | 0% ± 0% | 0.84 ± 0.04 | 25.66 ± 0.58 | 0.73 | 2.57 | 0.74 | - | 515 | 1.8 |

## speed

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x | Static (no motion) | 17% ± 0% | 20% ± 0% | 21.99 | 75% ± 0% | 21.62 ± 1.59 | 15.00 ± 0.00 | 15.84 | 38.70 | 37.82 | - | - | - |
| 1x | POGS | 13% ± 2% | 100% ± 0% | 0.78 | 0% ± 0% | 0.84 ± 0.05 | 23.06 ± 0.93 | 0.71 | 2.33 | 0.79 | - | 538 | 1.8 |
| 1x | POGS (release: no per-frame depth) | 13% ± 2% | 47% ± 21% | 6.14 | 23% ± 25% | 7.22 ± 5.26 | 26.72 ± 2.10 | 4.86 | 15.04 | 11.17 | - | 425 | 2.2 |
| 1x | POGS + KF | 13% ± 1% | 100% ± 0% | 0.74 | 0% ± 0% | 0.84 ± 0.04 | 25.66 ± 0.58 | 0.73 | 2.57 | 0.74 | - | 515 | 1.8 |

## occlusion

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gripper | Static (no motion) | 17% ± 0% | 20% ± 0% | 21.99 | 75% ± 0% | 21.62 ± 1.59 | 15.00 ± 0.00 | 15.84 | 38.70 | 37.82 | - | - | - |
| gripper | POGS | 13% ± 2% | 100% ± 0% | 0.78 | 0% ± 0% | 0.84 ± 0.05 | 23.06 ± 0.93 | 0.71 | 2.33 | 0.79 | - | 538 | 1.8 |
| gripper | POGS (release: no per-frame depth) | 13% ± 2% | 47% ± 21% | 6.14 | 23% ± 25% | 7.22 ± 5.26 | 26.72 ± 2.10 | 4.86 | 15.04 | 11.17 | - | 425 | 2.2 |
| gripper | POGS + KF | 13% ± 1% | 100% ± 0% | 0.74 | 0% ± 0% | 0.84 ± 0.04 | 25.66 ± 0.58 | 0.73 | 2.57 | 0.74 | - | 515 | 1.8 |

## depth

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stereo 3 mm | Static (no motion) | 17% ± 0% | 20% ± 0% | 21.99 | 75% ± 0% | 21.62 ± 1.59 | 15.00 ± 0.00 | 15.84 | 38.70 | 37.82 | - | - | - |
| stereo 3 mm | POGS | 13% ± 2% | 100% ± 0% | 0.78 | 0% ± 0% | 0.84 ± 0.05 | 23.06 ± 0.93 | 0.71 | 2.33 | 0.79 | - | 538 | 1.8 |
| stereo 3 mm | POGS (release: no per-frame depth) | 13% ± 2% | 47% ± 21% | 6.14 | 23% ± 25% | 7.22 ± 5.26 | 26.72 ± 2.10 | 4.86 | 15.04 | 11.17 | - | 425 | 2.2 |
| stereo 3 mm | POGS + KF | 13% ± 1% | 100% ± 0% | 0.74 | 0% ± 0% | 0.84 ± 0.04 | 25.66 ± 0.58 | 0.73 | 2.57 | 0.74 | - | 515 | 1.8 |

## object

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mug | Static (no motion) | 17% ± 0% | 20% ± 0% | 21.99 | 75% ± 0% | 21.62 ± 1.59 | 15.00 ± 0.00 | 15.84 | 38.70 | 37.82 | - | - | - |
| mug | POGS | 13% ± 2% | 100% ± 0% | 0.78 | 0% ± 0% | 0.84 ± 0.05 | 23.06 ± 0.93 | 0.71 | 2.33 | 0.79 | - | 538 | 1.8 |
| mug | POGS (release: no per-frame depth) | 13% ± 2% | 47% ± 21% | 6.14 | 23% ± 25% | 7.22 ± 5.26 | 26.72 ± 2.10 | 4.86 | 15.04 | 11.17 | - | 425 | 2.2 |
| mug | POGS + KF | 13% ± 1% | 100% ± 0% | 0.74 | 0% ± 0% | 0.84 ± 0.04 | 25.66 ± 0.58 | 0.73 | 2.57 | 0.74 | - | 515 | 1.8 |
| mustard bottle | Static (no motion) | 17% ± 0% | 20% ± 0% | 23.24 | 73% ± 5% | 23.02 ± 2.32 | 15.00 ± 0.00 | 19.15 | 41.63 | 40.85 | - | - | - |
| mustard bottle | POGS | 26% ± 5% | 100% ± 0% | 0.75 | 0% ± 0% | 0.74 ± 0.02 | 8.09 ± 0.40 | 0.44 | 0.92 | 0.63 | - | 540 | 1.7 |
| mustard bottle | POGS (release: no per-frame depth) | 8% ± 8% | 51% ± 11% | 3.42 | 5% ± 9% | 4.23 ± 0.84 | 8.03 ± 0.44 | 2.55 | 9.36 | 4.50 | - | 468 | 2.0 |
| mustard bottle | POGS + KF | 23% ± 6% | 100% ± 0% | 0.81 | 0% ± 0% | 0.81 ± 0.03 | 7.88 ± 0.38 | 0.46 | 1.05 | 0.61 | - | 543 | 1.7 |
| cereal box | Static (no motion) | 18% ± 1% | 25% ± 3% | 24.47 | 70% ± 6% | 24.39 ± 1.38 | 15.00 ± 0.00 | 18.44 | 44.58 | 43.91 | - | - | - |
| cereal box | POGS | 8% ± 0% | 57% ± 13% | 7.60 | 31% ± 27% | 17.79 ± 15.68 | 71.04 ± 18.37 | 15.13 | 41.30 | 38.71 | - | 541 | 1.7 |
| cereal box | POGS (release: no per-frame depth) | 8% ± 1% | 17% ± 2% | 10.92 | 59% ± 11% | 22.38 ± 6.32 | 53.79 ± 15.71 | 17.98 | 63.27 | 63.27 | - | 459 | 2.0 |
| cereal box | POGS + KF | 9% ± 2% | 72% ± 22% | 2.76 | 14% ± 17% | 9.53 ± 10.24 | 74.22 ± 35.11 | 7.72 | 42.80 | 41.56 | - | 543 | 1.7 |

## multi-object

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| one moving | Static (no motion) | 17% ± 0% | 20% ± 0% | 21.99 | 75% ± 0% | 21.62 ± 1.59 | 15.00 ± 0.00 | 15.84 | 38.70 | 37.82 | - | - | - |
| one moving | POGS | 13% ± 2% | 100% ± 0% | 0.78 | 0% ± 0% | 0.84 ± 0.05 | 23.06 ± 0.93 | 0.71 | 2.33 | 0.79 | - | 538 | 1.8 |
| one moving | POGS (release: no per-frame depth) | 13% ± 2% | 47% ± 21% | 6.14 | 23% ± 25% | 7.22 ± 5.26 | 26.72 ± 2.10 | 4.86 | 15.04 | 11.17 | - | 425 | 2.2 |
| one moving | POGS + KF | 13% ± 1% | 100% ± 0% | 0.74 | 0% ± 0% | 0.84 ± 0.04 | 25.66 ± 0.58 | 0.73 | 2.57 | 0.74 | - | 515 | 1.8 |
| two moving | Static (no motion) | 17% ± 0% | 20% ± 0% | 23.32 | 69% ± 3% | 23.23 ± 1.16 | 15.00 ± 0.00 | 15.80 | 42.31 | 41.65 | - | - | - |
| two moving | POGS | 13% ± 2% | 100% ± 0% | 0.96 | 0% ± 0% | 0.98 ± 0.07 | 24.42 ± 1.24 | 0.72 | 2.60 | 0.42 | - | 994 | 0.9 |
| two moving | POGS (release: no per-frame depth) | 12% ± 1% | 46% ± 13% | 8.41 | 33% ± 29% | 10.06 ± 5.52 | 33.42 ± 15.62 | 7.22 | 24.53 | 22.21 | - | 845 | 1.1 |
| two moving | POGS + KF | 13% ± 2% | 100% ± 0% | 0.60 | 0% ± 0% | 0.74 ± 0.09 | 23.05 ± 1.27 | 0.68 | 2.64 | 0.42 | - | 1021 | 0.9 |

## all cases

| Method | Cases | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Static (no motion) | 12 | 17% | 21% | 23.26 | 72% | 23.07 | 15.00 | 17.31 | 41.80 | 41.06 | - | - | - |
| POGS | 12 | 15% | 89% | 2.52 | 8% | 5.09 | 31.65 | 4.25 | 11.79 | 10.14 | - | 653 | 1.5 |
| POGS (release: no per-frame depth) | 12 | 10% | 40% | 7.22 | 30% | 10.97 | 30.49 | 8.15 | 28.05 | 25.29 | - | 549 | 1.8 |
| POGS + KF | 12 | 14% | 93% | 1.22 | 4% | 2.98 | 32.70 | 2.40 | 12.26 | 10.83 | - | 655 | 1.6 |
