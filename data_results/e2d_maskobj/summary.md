# E2b/E2c: POGS (DINO in object mask) on O4 (same cases as E0)

100 cases. Single splat.

## Pre-registered rule

| Setting | Static (no motion) | POGS | POGS (DINO in object mask) |
|---|---|---|---|
| -30° | 0/25 [0.00, 0.13] | 3/25 [0.04, 0.30] | 1/25 [0.01, 0.20] |
| -15° | 0/25 [0.00, 0.13] | 7/25 [0.14, 0.48] | 0/25 [0.00, 0.13] |
| +15° | 0/25 [0.00, 0.13] | 1/25 [0.01, 0.20] | 0/25 [0.00, 0.13] |
| +30° | 0/25 [0.00, 0.13] | 6/25 [0.11, 0.43] | 0/25 [0.00, 0.13] |

## Paper-faithful rule (roll free)

| Setting | Static (no motion) | POGS | POGS (DINO in object mask) |
|---|---|---|---|
| -30° | 0/25 [0.00, 0.13] | 9/25 [0.20, 0.55] | 1/25 [0.01, 0.20] |
| -15° | 0/25 [0.00, 0.13] | 12/25 [0.30, 0.67] | 0/25 [0.00, 0.13] |
| +15° | 0/25 [0.00, 0.13] | 16/25 [0.45, 0.80] | 4/25 [0.06, 0.35] |
| +30° | 0/25 [0.00, 0.13] | 23/25 [0.75, 0.98] | 4/25 [0.06, 0.35] |

## Roll before the perturbation and paired difference (paper-faithful)

| Setting | abs roll pre, POGS (deg) | abs roll pre, POGS (DINO in object mask) (deg) | variant − POGS success [95% CI] |
|---|---|---|---|
| -30° | 9.4 | 23.8 | -0.32 [-0.52, -0.16] |
| -15° | 9.3 | 24.1 | -0.48 [-0.68, -0.32] |
| +15° | 9.3 | 24.2 | -0.48 [-0.68, -0.28] |
| +30° | 9.5 | 23.7 | -0.76 [-0.92, -0.60] |
