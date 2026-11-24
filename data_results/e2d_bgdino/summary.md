# E2b/E2c: POGS (DINO vs rendered background) on O4 (same cases as E0)

100 cases. Single splat.

## Pre-registered rule

| Setting | Static (no motion) | POGS | POGS (DINO vs rendered background) |
|---|---|---|---|
| -30° | 0/25 [0.00, 0.13] | 3/25 [0.04, 0.30] | 0/25 [0.00, 0.13] |
| -15° | 0/25 [0.00, 0.13] | 7/25 [0.14, 0.48] | 0/25 [0.00, 0.13] |
| +15° | 0/25 [0.00, 0.13] | 1/25 [0.01, 0.20] | 0/25 [0.00, 0.13] |
| +30° | 0/25 [0.00, 0.13] | 6/25 [0.11, 0.43] | 0/25 [0.00, 0.13] |

## Paper-faithful rule (roll free)

| Setting | Static (no motion) | POGS | POGS (DINO vs rendered background) |
|---|---|---|---|
| -30° | 0/25 [0.00, 0.13] | 9/25 [0.20, 0.55] | 0/25 [0.00, 0.13] |
| -15° | 0/25 [0.00, 0.13] | 12/25 [0.30, 0.67] | 0/25 [0.00, 0.13] |
| +15° | 0/25 [0.00, 0.13] | 16/25 [0.45, 0.80] | 0/25 [0.00, 0.13] |
| +30° | 0/25 [0.00, 0.13] | 23/25 [0.75, 0.98] | 0/25 [0.00, 0.13] |

## Roll before the perturbation and paired difference (paper-faithful)

| Setting | abs roll pre, POGS (deg) | abs roll pre, POGS (DINO vs rendered background) (deg) | variant − POGS success [95% CI] |
|---|---|---|---|
| -30° | 9.4 | 7.4 | -0.36 [-0.56, -0.20] |
| -15° | 9.3 | 7.3 | -0.48 [-0.68, -0.32] |
| +15° | 9.3 | 7.4 | -0.64 [-0.80, -0.44] |
| +30° | 9.5 | 7.5 | -0.92 [-1.00, -0.80] |
