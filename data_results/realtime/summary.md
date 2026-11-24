# Real-time tracking results

Camera stream at 30 Hz of scene time; at Nx playback the frames arrive N times faster (2x = 60 frames/s of wall time), so the objects move N times further per tracker step. The tracker always takes the newest frame once it is free; 'now' columns score, at EVERY camera frame, the pose the tracker had published by then. Latency is the measured compute time per tracked frame on this machine. 3 seed(s); mean ± std over seeds.

## base, 1x playback speed

| Method | ADD-S success (now) | Lost (now) | Median pos (cm, now) | Rot err (deg, now) | ADD-S success (tracked frames) | Latency (ms) | Tracker Hz | Frames skipped | Estimate age (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Static (no motion) | 19% ± 0% | 76% ± 1% | 21.99 | 15.0 ± 0.00 | - | - | - | - | - |
| POGS | 54% ± 20% | 17% ± 28% | 5.41 | 28.8 ± 1.42 | 84% ± 27% | 580 ± 51 | 1.7 | 16.4 | 887 |
| POGS, 10 iters | 96% ± 5% | 0% ± 0% | 1.70 | 25.3 ± 1.98 | 100% ± 0% | 242 ± 5 | 4.1 | 6.2 | 379 |
| POGS + KF + check | 52% ± 23% | 20% ± 31% | 6.62 | 25.0 ± 2.52 | 79% ± 36% | 595 ± 46 | 1.7 | 16.9 | 914 |
| POGS + KF + check + latency comp. | 66% ± 32% | 18% ± 32% | 6.05 | 27.1 ± 2.87 | 79% ± 36% | 595 ± 46 | 1.7 | 16.9 | 914 |
| POGS + KF + check, 10 iters | 94% ± 6% | 0% ± 0% | 2.15 | 28.6 ± 2.91 | 100% ± 0% | 271 ± 7 | 3.7 | 7.1 | 422 |
| POGS + KF + check, 10 iters + latency comp. | 100% ± 0% | 0% ± 0% | 1.70 | 29.9 ± 3.25 | 100% ± 0% | 271 ± 7 | 3.7 | 7.1 | 422 |

## base, 2x playback speed

| Method | ADD-S success (now) | Lost (now) | Median pos (cm, now) | Rot err (deg, now) | ADD-S success (tracked frames) | Latency (ms) | Tracker Hz | Frames skipped | Estimate age (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Static (no motion) | 19% ± 0% | 76% ± 1% | 21.99 | 15.0 ± 0.00 | - | - | - | - | - |
| POGS | 20% ± 0% | 58% ± 3% | 16.09 | 27.1 ± 0.69 | 37% ± 4% | 542 ± 39 | 1.9 | 31.5 | 818 |
| POGS, 10 iters | 21% ± 1% | 51% ± 6% | 11.04 | 23.6 ± 0.36 | 38% ± 11% | 250 ± 21 | 4.0 | 14.0 | 384 |
| POGS + KF + check | 20% ± 1% | 59% ± 1% | 18.63 | 35.2 ± 10.66 | 37% ± 3% | 619 ± 42 | 1.6 | 36.2 | 931 |
| POGS + KF + check + latency comp. | 20% ± 1% | 60% ± 2% | 21.13 | 40.3 ± 12.55 | 37% ± 3% | 619 ± 42 | 1.6 | 36.2 | 931 |
| POGS + KF + check, 10 iters | 44% ± 3% | 9% ± 2% | 5.66 | 26.1 ± 0.40 | 94% ± 2% | 295 ± 23 | 3.4 | 16.6 | 451 |
| POGS + KF + check, 10 iters + latency comp. | 51% ± 9% | 0% ± 0% | 4.14 | 28.1 ± 0.17 | 94% ± 2% | 295 ± 23 | 3.4 | 16.6 | 451 |

## all real-time cases

| Method | Cases | ADD-S success (now) | Lost (now) | Median pos (cm, now) | Rot err (deg, now) | ADD-S success (tracked frames) | Latency (ms) | Tracker Hz | Frames skipped | Estimate age (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Static (no motion) | 6 | 19% | 76% | 21.99 | 15.0 | - | - | - | - | - |
| POGS | 6 | 37% | 37% | 10.75 | 27.9 | 61% | 561 | 1.8 | 23.9 | 853 |
| POGS, 10 iters | 6 | 59% | 25% | 6.37 | 24.4 | 69% | 246 | 4.1 | 10.1 | 381 |
| POGS + KF + check | 6 | 36% | 39% | 12.63 | 30.1 | 58% | 607 | 1.7 | 26.5 | 923 |
| POGS + KF + check + latency comp. | 6 | 43% | 39% | 13.59 | 33.7 | 58% | 607 | 1.7 | 26.5 | 923 |
| POGS + KF + check, 10 iters | 6 | 69% | 4% | 3.90 | 27.3 | 97% | 283 | 3.5 | 11.9 | 437 |
| POGS + KF + check, 10 iters + latency comp. | 6 | 76% | 0% | 2.92 | 29.0 | 97% | 283 | 3.5 | 11.9 | 437 |
