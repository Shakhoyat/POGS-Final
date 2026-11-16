# Tracking benchmark results

3 seed(s) per level; values are mean ± std over seeds. Target object only, except 'Static drift' (objects that never move).

## rotation

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yaw 30 | Static (no motion) | 17% ± 0% | 20% ± 0% | 21.99 | 75% ± 0% | 21.62 ± 1.59 | 15.00 ± 0.00 | 17.00 | 38.70 | 37.82 | - | - | - |
| yaw 30 | POGS | 9% ± 1% | 100% ± 0% | 0.80 | 0% ± 0% | 0.87 ± 0.06 | 25.65 ± 1.85 | 0.69 | 2.03 | 0.76 | - | 552 | 1.7 |

## speed

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x | Static (no motion) | 17% ± 0% | 20% ± 0% | 21.99 | 75% ± 0% | 21.62 ± 1.59 | 15.00 ± 0.00 | 17.00 | 38.70 | 37.82 | - | - | - |
| 1x | POGS | 9% ± 1% | 100% ± 0% | 0.80 | 0% ± 0% | 0.87 ± 0.06 | 25.65 ± 1.85 | 0.69 | 2.03 | 0.76 | - | 552 | 1.7 |

## occlusion

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gripper | Static (no motion) | 17% ± 0% | 20% ± 0% | 21.99 | 75% ± 0% | 21.62 ± 1.59 | 15.00 ± 0.00 | 17.00 | 38.70 | 37.82 | - | - | - |
| gripper | POGS | 9% ± 1% | 100% ± 0% | 0.80 | 0% ± 0% | 0.87 ± 0.06 | 25.65 ± 1.85 | 0.69 | 2.03 | 0.76 | - | 552 | 1.7 |

## depth

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stereo 3 mm | Static (no motion) | 17% ± 0% | 20% ± 0% | 21.99 | 75% ± 0% | 21.62 ± 1.59 | 15.00 ± 0.00 | 17.00 | 38.70 | 37.82 | - | - | - |
| stereo 3 mm | POGS | 9% ± 1% | 100% ± 0% | 0.80 | 0% ± 0% | 0.87 ± 0.06 | 25.65 ± 1.85 | 0.69 | 2.03 | 0.76 | - | 552 | 1.7 |

## object

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mug | Static (no motion) | 17% ± 0% | 20% ± 0% | 21.99 | 75% ± 0% | 21.62 ± 1.59 | 15.00 ± 0.00 | 17.00 | 38.70 | 37.82 | - | - | - |
| mug | POGS | 9% ± 1% | 100% ± 0% | 0.80 | 0% ± 0% | 0.87 ± 0.06 | 25.65 ± 1.85 | 0.69 | 2.03 | 0.76 | - | 552 | 1.7 |

## multi-object

| Level | Method | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| one moving | Static (no motion) | 17% ± 0% | 20% ± 0% | 21.99 | 75% ± 0% | 21.62 ± 1.59 | 15.00 ± 0.00 | 17.00 | 38.70 | 37.82 | - | - | - |
| one moving | POGS | 9% ± 1% | 100% ± 0% | 0.80 | 0% ± 0% | 0.87 ± 0.06 | 25.65 ± 1.85 | 0.69 | 2.03 | 0.76 | - | 552 | 1.7 |

## all cases

| Method | Cases | Success (3 cm & 5 deg) | Success (ADD-S < 0.1d) | Median pos (cm) | Lost (>10 cm) | Pos err (cm) | Rot err (deg) | ADD-S (cm) | Max pos (cm) | Final pos (cm) | Static drift (cm) | Opt ms/frame | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Static (no motion) | 3 | 17% | 20% | 21.99 | 75% | 21.62 | 15.00 | 17.00 | 38.70 | 37.82 | - | - | - |
| POGS | 3 | 9% | 100% | 0.80 | 0% | 0.87 | 25.65 | 0.69 | 2.03 | 0.76 | - | 552 | 1.7 |
