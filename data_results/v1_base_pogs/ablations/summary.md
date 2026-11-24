# Tracking benchmark results

3 seed(s) per level; values are mean ± std over seeds. Target object only, except 'Static drift' (objects that never move).

## rotation

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yaw 30 | Static (no motion) | 17% ± 0% | 20% ± 0% | 21.99 | 75% ± 0% | 21.62 ± 1.59 | 15.00 ± 0.00 | 17.00 | 38.70 | 37.82 | - | - | - |
| yaw 30 | POGS | 9% ± 2% | 100% ± 0% | 0.79 | 0% ± 0% | 0.86 ± 0.06 | 25.86 ± 1.96 | 0.69 | 2.09 | 0.78 | - | 518 | 1.8 |
| yaw 30 | No Depth | 0% ± 0% | 24% ± 2% | 15.49 | 59% ± 3% | 14.87 ± 2.56 | 34.39 ± 9.13 | 11.38 | 26.59 | 24.25 | - | 427 | 2.2 |
| yaw 30 | No DINO (RGB) | 18% ± 1% | 100% ± 0% | 1.08 | 0% ± 0% | 1.10 ± 0.06 | 19.93 ± 1.78 | 0.61 | 2.19 | 0.62 | - | 410 | 2.4 |

## speed

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x | Static (no motion) | 17% ± 0% | 20% ± 0% | 21.99 | 75% ± 0% | 21.62 ± 1.59 | 15.00 ± 0.00 | 17.00 | 38.70 | 37.82 | - | - | - |
| 1x | POGS | 9% ± 2% | 100% ± 0% | 0.79 | 0% ± 0% | 0.86 ± 0.06 | 25.86 ± 1.96 | 0.69 | 2.09 | 0.78 | - | 518 | 1.8 |
| 1x | No Depth | 0% ± 0% | 24% ± 2% | 15.49 | 59% ± 3% | 14.87 ± 2.56 | 34.39 ± 9.13 | 11.38 | 26.59 | 24.25 | - | 427 | 2.2 |
| 1x | No DINO (RGB) | 18% ± 1% | 100% ± 0% | 1.08 | 0% ± 0% | 1.10 ± 0.06 | 19.93 ± 1.78 | 0.61 | 2.19 | 0.62 | - | 410 | 2.4 |

## occlusion

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gripper | Static (no motion) | 17% ± 0% | 20% ± 0% | 21.99 | 75% ± 0% | 21.62 ± 1.59 | 15.00 ± 0.00 | 17.00 | 38.70 | 37.82 | - | - | - |
| gripper | POGS | 9% ± 2% | 100% ± 0% | 0.79 | 0% ± 0% | 0.86 ± 0.06 | 25.86 ± 1.96 | 0.69 | 2.09 | 0.78 | - | 518 | 1.8 |
| gripper | No Depth | 0% ± 0% | 24% ± 2% | 15.49 | 59% ± 3% | 14.87 ± 2.56 | 34.39 ± 9.13 | 11.38 | 26.59 | 24.25 | - | 427 | 2.2 |
| gripper | No DINO (RGB) | 18% ± 1% | 100% ± 0% | 1.08 | 0% ± 0% | 1.10 ± 0.06 | 19.93 ± 1.78 | 0.61 | 2.19 | 0.62 | - | 410 | 2.4 |

## depth

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stereo 3 mm | Static (no motion) | 17% ± 0% | 20% ± 0% | 21.99 | 75% ± 0% | 21.62 ± 1.59 | 15.00 ± 0.00 | 17.00 | 38.70 | 37.82 | - | - | - |
| stereo 3 mm | POGS | 9% ± 2% | 100% ± 0% | 0.79 | 0% ± 0% | 0.86 ± 0.06 | 25.86 ± 1.96 | 0.69 | 2.09 | 0.78 | - | 518 | 1.8 |
| stereo 3 mm | No Depth | 0% ± 0% | 24% ± 2% | 15.49 | 59% ± 3% | 14.87 ± 2.56 | 34.39 ± 9.13 | 11.38 | 26.59 | 24.25 | - | 427 | 2.2 |
| stereo 3 mm | No DINO (RGB) | 18% ± 1% | 100% ± 0% | 1.08 | 0% ± 0% | 1.10 ± 0.06 | 19.93 ± 1.78 | 0.61 | 2.19 | 0.62 | - | 410 | 2.4 |

## object

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mug | Static (no motion) | 17% ± 0% | 20% ± 0% | 21.99 | 75% ± 0% | 21.62 ± 1.59 | 15.00 ± 0.00 | 17.00 | 38.70 | 37.82 | - | - | - |
| mug | POGS | 9% ± 2% | 100% ± 0% | 0.79 | 0% ± 0% | 0.86 ± 0.06 | 25.86 ± 1.96 | 0.69 | 2.09 | 0.78 | - | 518 | 1.8 |
| mug | No Depth | 0% ± 0% | 24% ± 2% | 15.49 | 59% ± 3% | 14.87 ± 2.56 | 34.39 ± 9.13 | 11.38 | 26.59 | 24.25 | - | 427 | 2.2 |
| mug | No DINO (RGB) | 18% ± 1% | 100% ± 0% | 1.08 | 0% ± 0% | 1.10 ± 0.06 | 19.93 ± 1.78 | 0.61 | 2.19 | 0.62 | - | 410 | 2.4 |

## multi-object

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| one moving | Static (no motion) | 17% ± 0% | 20% ± 0% | 21.99 | 75% ± 0% | 21.62 ± 1.59 | 15.00 ± 0.00 | 17.00 | 38.70 | 37.82 | - | - | - |
| one moving | POGS | 9% ± 2% | 100% ± 0% | 0.79 | 0% ± 0% | 0.86 ± 0.06 | 25.86 ± 1.96 | 0.69 | 2.09 | 0.78 | - | 518 | 1.8 |
| one moving | No Depth | 0% ± 0% | 24% ± 2% | 15.49 | 59% ± 3% | 14.87 ± 2.56 | 34.39 ± 9.13 | 11.38 | 26.59 | 24.25 | - | 427 | 2.2 |
| one moving | No DINO (RGB) | 18% ± 1% | 100% ± 0% | 1.08 | 0% ± 0% | 1.10 ± 0.06 | 19.93 ± 1.78 | 0.61 | 2.19 | 0.62 | - | 410 | 2.4 |

## all cases

| Method | Cases | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Static (no motion) | 3 | 17% | 20% | 21.99 | 75% | 21.62 | 15.00 | 17.00 | 38.70 | 37.82 | - | - | - |
| POGS | 3 | 9% | 100% | 0.79 | 0% | 0.86 | 25.86 | 0.69 | 2.09 | 0.78 | - | 518 | 1.8 |
| No Depth | 3 | 0% | 24% | 15.49 | 59% | 14.87 | 34.39 | 11.38 | 26.59 | 24.25 | - | 427 | 2.2 |
| No DINO (RGB) | 3 | 18% | 100% | 1.08 | 0% | 1.10 | 19.93 | 0.61 | 2.19 | 0.62 | - | 410 | 2.4 |
