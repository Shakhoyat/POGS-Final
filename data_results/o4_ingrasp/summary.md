| Setting | Method | Recovered (k/n) | Rate | 95% CI (Wilson) | vs 80% | Tip err at N (cm) | Rot err at N (deg) |
|---|---|---|---|---|---|---|---|
| +15° | Static (no motion) | 0/50 | 0% | [0.00, 0.07] | not met | 13.66 ± 3.20 | 23.1 ± 16.7 |
| +15° | POGS | 2/50 | 4% | [0.01, 0.13] | not met | 1.49 ± 0.99 | 10.3 ± 3.0 |
| +15° | POGS (release: no per-frame depth) | 0/50 | 0% | [0.00, 0.07] | not met | 5.80 ± 4.22 | 22.0 ± 7.9 |
| +15° | POGS + KF + check | 3/50 | 6% | [0.02, 0.16] | not met | 1.54 ± 0.88 | 10.4 ± 3.3 |
| -15° | Static (no motion) | 0/50 | 0% | [0.00, 0.07] | not met | 13.75 ± 3.30 | 25.7 ± 16.2 |
| -15° | POGS | 14/50 | 28% | [0.17, 0.42] | not met | 3.94 ± 3.79 | 12.7 ± 16.4 |
| -15° | POGS (release: no per-frame depth) | 1/50 | 2% | [0.00, 0.10] | not met | 6.82 ± 5.53 | 24.4 ± 18.4 |
| -15° | POGS + KF + check | 15/50 | 30% | [0.19, 0.44] | not met | 3.64 ± 2.79 | 11.2 ± 12.7 |
| +30° | Static (no motion) | 0/50 | 0% | [0.00, 0.07] | not met | 15.05 ± 4.53 | 30.6 ± 22.4 |
| +30° | POGS | 13/50 | 26% | [0.16, 0.40] | not met | 1.81 ± 1.68 | 9.4 ± 4.7 |
| +30° | POGS (release: no per-frame depth) | 2/50 | 4% | [0.01, 0.13] | not met | 4.90 ± 4.26 | 16.2 ± 8.8 |
| +30° | POGS + KF + check | 9/50 | 18% | [0.10, 0.31] | not met | 1.63 ± 1.28 | 9.4 ± 4.6 |
| -30° | Static (no motion) | 0/50 | 0% | [0.00, 0.07] | not met | 15.25 ± 4.54 | 33.8 ± 22.7 |
| -30° | POGS | 5/50 | 10% | [0.04, 0.21] | not met | 5.30 ± 4.90 | 21.0 ± 22.3 |
| -30° | POGS (release: no per-frame depth) | 6/50 | 12% | [0.06, 0.24] | not met | 8.54 ± 7.87 | 40.9 ± 31.5 |
| -30° | POGS + KF + check | 7/50 | 14% | [0.07, 0.26] | not met | 4.82 ± 3.92 | 17.6 ± 19.4 |

**Paper-faithful rule** (deviation logged 2026-11-22, taken from POGS Sec. V-B, not from our results): tip < 3 cm and bit-axis direction (swing) < 5 deg at N, never lost; roll about the bit axis free. Pre-registered rule above stays the primary result.

| Setting | Method | Recovered (k/n) | 95% CI (Wilson) | vs 80% | Swing at N (deg) | Roll at N (deg) | Roll pre (deg) |
|---|---|---|---|---|---|---|---|
| +15° | Static (no motion) | 0/50 | [0.00, 0.07] | not met | 23.1 ± 16.7 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| +15° | POGS | 32/50 | [0.50, 0.76] | not met | 4.6 ± 2.5 | 8.6 ± 3.8 | 9.4 ± 2.8 |
| +15° | POGS (release: no per-frame depth) | 14/50 | [0.17, 0.42] | not met | 9.9 ± 7.1 | 18.8 ± 7.0 | 20.6 ± 2.3 |
| +15° | POGS + KF + check | 28/50 | [0.42, 0.69] | not met | 4.7 ± 2.9 | 8.5 ± 4.0 | 8.3 ± 3.8 |
| -15° | Static (no motion) | 0/50 | [0.00, 0.07] | not met | 25.7 ± 16.2 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| -15° | POGS | 25/50 | [0.37, 0.63] | not met | 9.5 ± 16.7 | 6.3 ± 4.8 | 9.6 ± 2.8 |
| -15° | POGS (release: no per-frame depth) | 5/50 | [0.04, 0.21] | not met | 17.5 ± 17.5 | 15.7 ± 9.1 | 20.6 ± 2.5 |
| -15° | POGS + KF + check | 24/50 | [0.35, 0.61] | not met | 7.9 ± 12.8 | 6.2 ± 4.8 | 8.6 ± 3.5 |
| +30° | Static (no motion) | 0/50 | [0.00, 0.07] | not met | 30.6 ± 22.4 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| +30° | POGS | 39/50 | [0.65, 0.87] | inconclusive (CI spans target) | 4.1 ± 2.8 | 7.0 ± 6.1 | 9.6 ± 2.8 |
| +30° | POGS (release: no per-frame depth) | 19/50 | [0.26, 0.52] | not met | 6.2 ± 4.9 | 13.8 ± 9.4 | 20.7 ± 2.5 |
| +30° | POGS + KF + check | 35/50 | [0.56, 0.81] | inconclusive (CI spans target) | 4.2 ± 2.1 | 7.2 ± 6.1 | 8.4 ± 3.7 |
| -30° | Static (no motion) | 0/50 | [0.00, 0.07] | not met | 33.8 ± 22.7 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| -30° | POGS | 17/50 | [0.22, 0.48] | not met | 13.4 ± 20.5 | 13.4 ± 12.9 | 9.5 ± 2.7 |
| -30° | POGS (release: no per-frame depth) | 12/50 | [0.14, 0.37] | not met | 31.0 ± 31.3 | 23.3 ± 15.5 | 20.7 ± 2.4 |
| -30° | POGS + KF + check | 18/50 | [0.24, 0.50] | not met | 11.0 ± 17.4 | 10.9 ± 12.3 | 8.3 ± 3.9 |

Success: tip < 3 cm and rot < 5 deg at 10 frames (2 s) after the perturbation, never lost after it. Paper (physical robot, reference only): 48/50 at 15 deg, 40/50 at 30 deg. Jaws are not collision-checked against the turning drill.
