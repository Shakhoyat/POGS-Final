# Tracking benchmark results

3 seed(s) per level; values are mean ± std over seeds. Target object only, except 'Static drift' (objects that never move).

## rotation

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yaw 30 | Static (no motion) | 17% ± 0% | 20% ± 0% | 21.99 | 75% ± 0% | 21.62 ± 1.59 | 15.00 ± 0.00 | 15.84 | 38.70 | 37.82 | - | - | - |
| yaw 30 | POGS | 13% ± 2% | 100% ± 0% | 0.78 | 0% ± 0% | 0.84 ± 0.05 | 23.06 ± 0.93 | 0.71 | 2.33 | 0.79 | - | 538 | 1.8 |
| yaw 30 | No Depth | 9% ± 3% | 44% ± 12% | 6.46 | 17% ± 29% | 7.52 ± 5.00 | 25.38 ± 0.28 | 4.97 | 14.83 | 11.17 | - | 434 | 2.2 |
| yaw 30 | No DINO (RGB) | 17% ± 2% | 100% ± 0% | 1.09 | 0% ± 0% | 1.07 ± 0.08 | 22.73 ± 1.93 | 0.66 | 1.85 | 0.96 | - | 417 | 2.3 |

## speed

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x | Static (no motion) | 17% ± 0% | 20% ± 0% | 21.99 | 75% ± 0% | 21.62 ± 1.59 | 15.00 ± 0.00 | 15.84 | 38.70 | 37.82 | - | - | - |
| 1x | POGS | 13% ± 2% | 100% ± 0% | 0.78 | 0% ± 0% | 0.84 ± 0.05 | 23.06 ± 0.93 | 0.71 | 2.33 | 0.79 | - | 538 | 1.8 |
| 1x | No Depth | 9% ± 3% | 44% ± 12% | 6.46 | 17% ± 29% | 7.52 ± 5.00 | 25.38 ± 0.28 | 4.97 | 14.83 | 11.17 | - | 434 | 2.2 |
| 1x | No DINO (RGB) | 17% ± 2% | 100% ± 0% | 1.09 | 0% ± 0% | 1.07 ± 0.08 | 22.73 ± 1.93 | 0.66 | 1.85 | 0.96 | - | 417 | 2.3 |

## occlusion

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gripper | Static (no motion) | 17% ± 0% | 20% ± 0% | 21.99 | 75% ± 0% | 21.62 ± 1.59 | 15.00 ± 0.00 | 15.84 | 38.70 | 37.82 | - | - | - |
| gripper | POGS | 13% ± 2% | 100% ± 0% | 0.78 | 0% ± 0% | 0.84 ± 0.05 | 23.06 ± 0.93 | 0.71 | 2.33 | 0.79 | - | 538 | 1.8 |
| gripper | No Depth | 9% ± 3% | 44% ± 12% | 6.46 | 17% ± 29% | 7.52 ± 5.00 | 25.38 ± 0.28 | 4.97 | 14.83 | 11.17 | - | 434 | 2.2 |
| gripper | No DINO (RGB) | 17% ± 2% | 100% ± 0% | 1.09 | 0% ± 0% | 1.07 ± 0.08 | 22.73 ± 1.93 | 0.66 | 1.85 | 0.96 | - | 417 | 2.3 |

## depth

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stereo 3 mm | Static (no motion) | 17% ± 0% | 20% ± 0% | 21.99 | 75% ± 0% | 21.62 ± 1.59 | 15.00 ± 0.00 | 15.84 | 38.70 | 37.82 | - | - | - |
| stereo 3 mm | POGS | 13% ± 2% | 100% ± 0% | 0.78 | 0% ± 0% | 0.84 ± 0.05 | 23.06 ± 0.93 | 0.71 | 2.33 | 0.79 | - | 538 | 1.8 |
| stereo 3 mm | No Depth | 9% ± 3% | 44% ± 12% | 6.46 | 17% ± 29% | 7.52 ± 5.00 | 25.38 ± 0.28 | 4.97 | 14.83 | 11.17 | - | 434 | 2.2 |
| stereo 3 mm | No DINO (RGB) | 17% ± 2% | 100% ± 0% | 1.09 | 0% ± 0% | 1.07 ± 0.08 | 22.73 ± 1.93 | 0.66 | 1.85 | 0.96 | - | 417 | 2.3 |

## object

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mug | Static (no motion) | 17% ± 0% | 20% ± 0% | 21.99 | 75% ± 0% | 21.62 ± 1.59 | 15.00 ± 0.00 | 15.84 | 38.70 | 37.82 | - | - | - |
| mug | POGS | 13% ± 2% | 100% ± 0% | 0.78 | 0% ± 0% | 0.84 ± 0.05 | 23.06 ± 0.93 | 0.71 | 2.33 | 0.79 | - | 538 | 1.8 |
| mug | No Depth | 9% ± 3% | 44% ± 12% | 6.46 | 17% ± 29% | 7.52 ± 5.00 | 25.38 ± 0.28 | 4.97 | 14.83 | 11.17 | - | 434 | 2.2 |
| mug | No DINO (RGB) | 17% ± 2% | 100% ± 0% | 1.09 | 0% ± 0% | 1.07 ± 0.08 | 22.73 ± 1.93 | 0.66 | 1.85 | 0.96 | - | 417 | 2.3 |

## multi-object

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| one moving | Static (no motion) | 17% ± 0% | 20% ± 0% | 21.99 | 75% ± 0% | 21.62 ± 1.59 | 15.00 ± 0.00 | 15.84 | 38.70 | 37.82 | - | - | - |
| one moving | POGS | 13% ± 2% | 100% ± 0% | 0.78 | 0% ± 0% | 0.84 ± 0.05 | 23.06 ± 0.93 | 0.71 | 2.33 | 0.79 | - | 538 | 1.8 |
| one moving | No Depth | 9% ± 3% | 44% ± 12% | 6.46 | 17% ± 29% | 7.52 ± 5.00 | 25.38 ± 0.28 | 4.97 | 14.83 | 11.17 | - | 434 | 2.2 |
| one moving | No DINO (RGB) | 17% ± 2% | 100% ± 0% | 1.09 | 0% ± 0% | 1.07 ± 0.08 | 22.73 ± 1.93 | 0.66 | 1.85 | 0.96 | - | 417 | 2.3 |

## all cases

| Method | Cases | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Static (no motion) | 3 | 17% | 20% | 21.99 | 75% | 21.62 | 15.00 | 15.84 | 38.70 | 37.82 | - | - | - |
| POGS | 3 | 13% | 100% | 0.78 | 0% | 0.84 | 23.06 | 0.71 | 2.33 | 0.79 | - | 538 | 1.8 |
| No Depth | 3 | 9% | 44% | 6.46 | 17% | 7.52 | 25.38 | 4.97 | 14.83 | 11.17 | - | 434 | 2.2 |
| No DINO (RGB) | 3 | 17% | 100% | 1.09 | 0% | 1.07 | 22.73 | 0.66 | 1.85 | 0.96 | - | 417 | 2.3 |
