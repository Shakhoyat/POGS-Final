# E2b/E2c: POGS (DINO in frame-start mask) on O4 (same cases as E0)

20 cases. Single splat.

## Pre-registered rule

| Setting | Static (no motion) | POGS | POGS (DINO in frame-start mask) |
|---|---|---|---|
| -30° | 0/5 [0.00, 0.43] | 0/5 [0.00, 0.43] | 0/5 [0.00, 0.43] |
| -15° | 0/5 [0.00, 0.43] | 1/5 [0.04, 0.62] | 0/5 [0.00, 0.43] |
| +15° | 0/5 [0.00, 0.43] | 0/5 [0.00, 0.43] | 0/5 [0.00, 0.43] |
| +30° | 0/5 [0.00, 0.43] | 1/5 [0.04, 0.62] | 0/5 [0.00, 0.43] |

## Paper-faithful rule (roll free)

| Setting | Static (no motion) | POGS | POGS (DINO in frame-start mask) |
|---|---|---|---|
| -30° | 0/5 [0.00, 0.43] | 1/5 [0.04, 0.62] | 1/5 [0.04, 0.62] |
| -15° | 0/5 [0.00, 0.43] | 2/5 [0.12, 0.77] | 0/5 [0.00, 0.43] |
| +15° | 0/5 [0.00, 0.43] | 3/5 [0.23, 0.88] | 2/5 [0.12, 0.77] |
| +30° | 0/5 [0.00, 0.43] | 5/5 [0.57, 1.00] | 2/5 [0.12, 0.77] |

## Roll before the perturbation and paired difference (paper-faithful)

| Setting | abs roll pre, POGS (deg) | abs roll pre, POGS (DINO in frame-start mask) (deg) | variant − POGS success [95% CI] |
|---|---|---|---|
| -30° | 8.9 | 22.4 | +0.00 [-0.60, +0.60] |
| -15° | 8.4 | 22.6 | -0.40 [-0.80, +0.00] |
| +15° | 8.5 | 22.6 | -0.20 [-0.60, +0.00] |
| +30° | 9.0 | 21.5 | -0.60 [-1.00, -0.20] |
