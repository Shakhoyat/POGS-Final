# Real-time tracking results

Camera stream at 30 Hz of scene time; at Nx playback the frames arrive N times faster (2x = 60 frames/s of wall time), so the objects move N times further per tracker step. The tracker always takes the newest frame once it is free; 'now' columns score, at EVERY camera frame, the pose the tracker had published by then. Latency is the measured compute time per tracked frame on this machine. 3 seed(s); mean ± std over seeds.

## base, 1x playback speed

| Method | ADD-S success (now) | Lost (now) | Median pos (cm, now) | Rot err (deg, now) | ADD-S success (tracked frames) | Latency (ms) | Tracker Hz | Frames skipped | Estimate age (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Static (no motion) | 19% ± 0% | 76% ± 1% | 21.99 | 15.0 ± 0.00 | - | - | - | - | - |
| POGS | 56% ± 19% | 19% ± 32% | 9.16 | 26.9 ± 2.07 | 80% ± 34% | 526 ± 29 | 1.9 | 14.7 | 802 |
| POGS, 10 iters | 100% ± 1% | 0% ± 0% | 1.69 | 21.8 ± 1.51 | 100% ± 0% | 225 ± 9 | 4.4 | 5.7 | 354 |
| POGS + KF + check | 66% ± 2% | 1% ± 2% | 2.89 | 23.3 ± 3.84 | 100% ± 0% | 540 ± 37 | 1.9 | 15.2 | 824 |
| POGS + KF + check + latency comp. | 91% ± 3% | 0% ± 0% | 2.09 | 24.9 ± 4.14 | 100% ± 0% | 540 ± 37 | 1.9 | 15.2 | 824 |
| POGS + KF + check, 10 iters | 97% ± 3% | 0% ± 0% | 1.86 | 22.7 ± 1.66 | 100% ± 0% | 253 ± 13 | 4.0 | 6.6 | 396 |
| POGS + KF + check, 10 iters + latency comp. | 100% ± 0% | 0% ± 0% | 1.49 | 23.4 ± 1.77 | 100% ± 0% | 253 ± 13 | 4.0 | 6.6 | 396 |

## base, 2x playback speed

| Method | ADD-S success (now) | Lost (now) | Median pos (cm, now) | Rot err (deg, now) | ADD-S success (tracked frames) | Latency (ms) | Tracker Hz | Frames skipped | Estimate age (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Static (no motion) | 19% ± 0% | 76% ± 1% | 21.99 | 15.0 ± 0.00 | - | - | - | - | - |
| POGS | 20% ± 0% | 59% ± 2% | 17.82 | 25.7 ± 1.20 | 33% ± 5% | 551 ± 4 | 1.8 | 31.8 | 825 |
| POGS, 10 iters | 25% ± 6% | 48% ± 3% | 9.39 | 22.9 ± 0.67 | 44% ± 6% | 239 ± 12 | 4.2 | 13.3 | 366 |
| POGS + KF + check | 26% ± 9% | 42% ± 16% | 10.66 | 28.4 ± 15.49 | 74% ± 29% | 585 ± 20 | 1.7 | 33.7 | 879 |
| POGS + KF + check + latency comp. | 34% ± 17% | 29% ± 32% | 9.54 | 32.8 ± 19.35 | 74% ± 29% | 585 ± 20 | 1.7 | 33.7 | 879 |
| POGS + KF + check, 10 iters | 20% ± 0% | 53% ± 6% | 11.77 | 25.9 ± 4.63 | 33% ± 8% | 297 ± 8 | 3.4 | 16.8 | 456 |
| POGS + KF + check, 10 iters + latency comp. | 23% ± 2% | 49% ± 8% | 10.61 | 28.8 ± 5.72 | 33% ± 8% | 297 ± 8 | 3.4 | 16.8 | 456 |

## all real-time cases

| Method | Cases | ADD-S success (now) | Lost (now) | Median pos (cm, now) | Rot err (deg, now) | ADD-S success (tracked frames) | Latency (ms) | Tracker Hz | Frames skipped | Estimate age (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Static (no motion) | 6 | 19% | 76% | 21.99 | 15.0 | - | - | - | - | - |
| POGS | 6 | 38% | 39% | 13.49 | 26.3 | 57% | 538 | 1.9 | 23.2 | 814 |
| POGS, 10 iters | 6 | 62% | 24% | 5.54 | 22.4 | 72% | 232 | 4.3 | 9.5 | 360 |
| POGS + KF + check | 6 | 46% | 22% | 6.78 | 25.9 | 87% | 562 | 1.8 | 24.4 | 852 |
| POGS + KF + check + latency comp. | 6 | 62% | 15% | 5.82 | 28.9 | 87% | 562 | 1.8 | 24.4 | 852 |
| POGS + KF + check, 10 iters | 6 | 58% | 27% | 6.81 | 24.3 | 66% | 275 | 3.7 | 11.7 | 426 |
| POGS + KF + check, 10 iters + latency comp. | 6 | 62% | 24% | 6.05 | 26.1 | 66% | 275 | 3.7 | 11.7 | 426 |
