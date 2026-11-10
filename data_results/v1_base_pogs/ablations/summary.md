# Tracking benchmark results

3 seed(s) per level; values are mean ± std over seeds. Target object only, except 'Static drift' (objects that never move).

## rotation

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yaw 30 | Static (no motion) | 17% ± 0% | 20% ± 0% | 20.85 | 75% ± 0% | 20.35 ± 1.60 | 15.00 ± 0.00 | 15.30 | 36.08 | 35.14 | - | - | - |
| yaw 30 | POGS | 8% ± 0% | 100% ± 0% | 1.28 | 0% ± 0% | 1.19 ± 0.05 | 33.23 ± 1.18 | 0.77 | 1.82 | 1.34 | - | 514 | 1.8 |
| yaw 30 | No Depth | 7% ± 1% | 32% ± 3% | 6.62 | 22% ± 20% | 6.44 ± 0.70 | 38.10 ± 1.61 | 3.71 | 11.65 | 5.16 | - | 434 | 2.1 |
| yaw 30 | No DINO (RGB) | 13% ± 1% | 100% ± 0% | 2.41 | 0% ± 0% | 2.06 ± 0.13 | 33.57 ± 1.16 | 0.71 | 3.31 | 0.49 | - | 411 | 2.4 |

## speed

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x | Static (no motion) | 17% ± 0% | 20% ± 0% | 20.85 | 75% ± 0% | 20.35 ± 1.60 | 15.00 ± 0.00 | 15.30 | 36.08 | 35.14 | - | - | - |
| 1x | POGS | 8% ± 0% | 100% ± 0% | 1.28 | 0% ± 0% | 1.19 ± 0.05 | 33.23 ± 1.18 | 0.77 | 1.82 | 1.34 | - | 514 | 1.8 |
| 1x | No Depth | 7% ± 1% | 32% ± 3% | 6.62 | 22% ± 20% | 6.44 ± 0.70 | 38.10 ± 1.61 | 3.71 | 11.65 | 5.16 | - | 434 | 2.1 |
| 1x | No DINO (RGB) | 13% ± 1% | 100% ± 0% | 2.41 | 0% ± 0% | 2.06 ± 0.13 | 33.57 ± 1.16 | 0.71 | 3.31 | 0.49 | - | 411 | 2.4 |

## occlusion

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gripper | Static (no motion) | 17% ± 0% | 20% ± 0% | 20.85 | 75% ± 0% | 20.35 ± 1.60 | 15.00 ± 0.00 | 15.30 | 36.08 | 35.14 | - | - | - |
| gripper | POGS | 8% ± 0% | 100% ± 0% | 1.28 | 0% ± 0% | 1.19 ± 0.05 | 33.23 ± 1.18 | 0.77 | 1.82 | 1.34 | - | 514 | 1.8 |
| gripper | No Depth | 7% ± 1% | 32% ± 3% | 6.62 | 22% ± 20% | 6.44 ± 0.70 | 38.10 ± 1.61 | 3.71 | 11.65 | 5.16 | - | 434 | 2.1 |
| gripper | No DINO (RGB) | 13% ± 1% | 100% ± 0% | 2.41 | 0% ± 0% | 2.06 ± 0.13 | 33.57 ± 1.16 | 0.71 | 3.31 | 0.49 | - | 411 | 2.4 |

## depth

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stereo 3 mm | Static (no motion) | 17% ± 0% | 20% ± 0% | 20.85 | 75% ± 0% | 20.35 ± 1.60 | 15.00 ± 0.00 | 15.30 | 36.08 | 35.14 | - | - | - |
| stereo 3 mm | POGS | 8% ± 0% | 100% ± 0% | 1.28 | 0% ± 0% | 1.19 ± 0.05 | 33.23 ± 1.18 | 0.77 | 1.82 | 1.34 | - | 514 | 1.8 |
| stereo 3 mm | No Depth | 7% ± 1% | 32% ± 3% | 6.62 | 22% ± 20% | 6.44 ± 0.70 | 38.10 ± 1.61 | 3.71 | 11.65 | 5.16 | - | 434 | 2.1 |
| stereo 3 mm | No DINO (RGB) | 13% ± 1% | 100% ± 0% | 2.41 | 0% ± 0% | 2.06 ± 0.13 | 33.57 ± 1.16 | 0.71 | 3.31 | 0.49 | - | 411 | 2.4 |

## object

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mug | Static (no motion) | 17% ± 0% | 20% ± 0% | 20.85 | 75% ± 0% | 20.35 ± 1.60 | 15.00 ± 0.00 | 15.30 | 36.08 | 35.14 | - | - | - |
| mug | POGS | 8% ± 0% | 100% ± 0% | 1.28 | 0% ± 0% | 1.19 ± 0.05 | 33.23 ± 1.18 | 0.77 | 1.82 | 1.34 | - | 514 | 1.8 |
| mug | No Depth | 7% ± 1% | 32% ± 3% | 6.62 | 22% ± 20% | 6.44 ± 0.70 | 38.10 ± 1.61 | 3.71 | 11.65 | 5.16 | - | 434 | 2.1 |
| mug | No DINO (RGB) | 13% ± 1% | 100% ± 0% | 2.41 | 0% ± 0% | 2.06 ± 0.13 | 33.57 ± 1.16 | 0.71 | 3.31 | 0.49 | - | 411 | 2.4 |

## multi-object

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| one moving | Static (no motion) | 17% ± 0% | 20% ± 0% | 20.85 | 75% ± 0% | 20.35 ± 1.60 | 15.00 ± 0.00 | 15.30 | 36.08 | 35.14 | - | - | - |
| one moving | POGS | 8% ± 0% | 100% ± 0% | 1.28 | 0% ± 0% | 1.19 ± 0.05 | 33.23 ± 1.18 | 0.77 | 1.82 | 1.34 | - | 514 | 1.8 |
| one moving | No Depth | 7% ± 1% | 32% ± 3% | 6.62 | 22% ± 20% | 6.44 ± 0.70 | 38.10 ± 1.61 | 3.71 | 11.65 | 5.16 | - | 434 | 2.1 |
| one moving | No DINO (RGB) | 13% ± 1% | 100% ± 0% | 2.41 | 0% ± 0% | 2.06 ± 0.13 | 33.57 ± 1.16 | 0.71 | 3.31 | 0.49 | - | 411 | 2.4 |

## all cases

| Method | Cases | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Static (no motion) | 3 | 17% | 20% | 20.85 | 75% | 20.35 | 15.00 | 15.30 | 36.08 | 35.14 | - | - | - |
| POGS | 3 | 8% | 100% | 1.28 | 0% | 1.19 | 33.23 | 0.77 | 1.82 | 1.34 | - | 514 | 1.8 |
| No Depth | 3 | 7% | 32% | 6.62 | 22% | 6.44 | 38.10 | 3.71 | 11.65 | 5.16 | - | 434 | 2.1 |
| No DINO (RGB) | 3 | 13% | 100% | 2.41 | 0% | 2.06 | 33.57 | 0.71 | 3.31 | 0.49 | - | 411 | 2.4 |
