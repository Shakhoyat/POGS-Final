# E1: anatomy of the O4 rotation floor (evaluation only)

## 1. Pre-perturbation error: roll about the bit axis vs swing of the axis

| Setting | Method | n | |roll| pre (deg) | swing pre (deg) | geodesic pre (deg) | roll share | tip pre (cm) |
|---|---|---|---|---|---|---|---|
| +15° | POGS | 50 | 9.4 ± 2.8 | 4.0 ± 3.4 | 10.5 ± 3.9 | 0.94 | 1.35 ± 1.17 |
| +15° | POGS (release: no per-frame depth) | 50 | 20.6 ± 2.3 | 6.7 ± 4.8 | 22.0 ± 3.6 | 0.96 | 7.50 ± 3.11 |
| +15° | POGS + KF + check | 50 | 8.3 ± 3.8 | 4.0 ± 2.1 | 9.5 ± 3.8 | 0.91 | 1.92 ± 1.11 |
| +15° | Static (no motion) | 50 | 0.0 ± 0.0 | 21.9 ± 12.8 | 21.9 ± 12.8 | 0.00 | 13.21 ± 2.50 |
| +30° | POGS | 50 | 9.6 ± 2.8 | 3.9 ± 3.3 | 10.5 ± 3.8 | 0.94 | 1.34 ± 1.12 |
| +30° | POGS (release: no per-frame depth) | 50 | 20.7 ± 2.5 | 7.0 ± 6.4 | 22.3 ± 4.7 | 0.96 | 7.62 ± 3.35 |
| +30° | POGS + KF + check | 50 | 8.4 ± 3.7 | 3.9 ± 2.0 | 9.4 ± 3.7 | 0.91 | 1.91 ± 1.13 |
| +30° | Static (no motion) | 50 | 0.0 ± 0.0 | 21.9 ± 12.8 | 21.9 ± 12.8 | 0.00 | 13.21 ± 2.50 |
| -15° | POGS | 50 | 9.6 ± 2.8 | 3.9 ± 3.3 | 10.5 ± 3.8 | 0.94 | 1.34 ± 1.11 |
| -15° | POGS (release: no per-frame depth) | 50 | 20.6 ± 2.5 | 7.4 ± 6.4 | 22.4 ± 4.8 | 0.96 | 7.63 ± 3.47 |
| -15° | POGS + KF + check | 50 | 8.6 ± 3.5 | 4.0 ± 2.1 | 9.7 ± 3.5 | 0.92 | 1.85 ± 1.14 |
| -15° | Static (no motion) | 50 | 0.0 ± 0.0 | 21.9 ± 12.8 | 21.9 ± 12.8 | 0.00 | 13.21 ± 2.50 |
| -30° | POGS | 50 | 9.5 ± 2.7 | 4.0 ± 3.6 | 10.6 ± 3.9 | 0.95 | 1.33 ± 1.14 |
| -30° | POGS (release: no per-frame depth) | 50 | 20.7 ± 2.4 | 7.6 ± 6.7 | 22.6 ± 5.0 | 0.96 | 7.81 ± 3.51 |
| -30° | POGS + KF + check | 50 | 8.3 ± 3.9 | 3.9 ± 2.0 | 9.4 ± 3.9 | 0.91 | 1.96 ± 1.12 |
| -30° | Static (no motion) | 50 | 0.0 ± 0.0 | 21.9 ± 12.8 | 21.9 ± 12.8 | 0.00 | 13.21 ± 2.50 |

Roll share = median |roll| / geodesic error. Near 1: the floor is roll about the bit axis, which the paper's rule leaves free (POGS Sec. V-B).

## 2. Signed-bias test of the post-hoc relative rule (POST HOC; artefact check only)

Aligned = the pre-perturbation signed error about the *perturbation* (vertical grasp) axis has the same sign as the perturbation, so the jump moves a frozen estimate's error toward zero in that axis. If success concentrates in 'aligned', the +/- gap is a signed-bias artefact of the relative rule.

| Setting | Method | Aligned: relative-rule success | Opposed: relative-rule success |
|---|---|---|---|
| +15° | POGS | 38/42 [0.78, 0.96] | 6/8 [0.41, 0.93] |
| +15° | POGS (release: no per-frame depth) | 26/43 [0.46, 0.74] | 3/7 [0.16, 0.75] |
| +15° | POGS + KF + check | 27/28 [0.82, 0.99] | 19/22 [0.67, 0.95] |
| +15° | Static (no motion) | 9/29 [0.17, 0.49] | 0/21 [0.00, 0.15] |
| +30° | POGS | 32/41 [0.63, 0.88] | 3/9 [0.12, 0.65] |
| +30° | POGS (release: no per-frame depth) | 28/44 [0.49, 0.76] | 2/6 [0.10, 0.70] |
| +30° | POGS + KF + check | 18/25 [0.52, 0.86] | 14/25 [0.37, 0.73] |
| +30° | Static (no motion) | 7/29 [0.12, 0.42] | 0/21 [0.00, 0.15] |
| -15° | POGS | 6/8 [0.41, 0.93] | 23/42 [0.40, 0.69] |
| -15° | POGS (release: no per-frame depth) | 4/6 [0.30, 0.90] | 30/44 [0.53, 0.80] |
| -15° | POGS + KF + check | 19/21 [0.71, 0.97] | 15/29 [0.34, 0.69] |
| -15° | Static (no motion) | 7/21 [0.17, 0.55] | 0/29 [0.00, 0.12] |
| -30° | POGS | 3/8 [0.14, 0.69] | 13/42 [0.19, 0.46] |
| -30° | POGS (release: no per-frame depth) | 2/7 [0.08, 0.64] | 15/43 [0.22, 0.50] |
| -30° | POGS + KF + check | 16/22 [0.52, 0.87] | 13/28 [0.30, 0.64] |
| -30° | Static (no motion) | 7/21 [0.17, 0.55] | 0/29 [0.00, 0.12] |

## 3. When the floor appears

| Method | |roll| end of approach, drill on table (deg) | end of lift (deg) | just before perturbation (deg) |
|---|---|---|---|
| POGS | 12.5 ± 0.2 | 13.5 ± 2.8 | 9.5 ± 2.8 |
| POGS (release: no per-frame depth) | 16.7 ± 0.2 | 22.9 ± 1.6 | 20.7 ± 2.4 |
| POGS + KF + check | 13.0 ± 0.6 | 7.7 ± 4.0 | 8.4 ± 3.7 |
| Static (no motion) | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |

A floor already present on the table points to initialisation / observability; one that appears during the lift points to gripper occlusion or motion.

Static (no motion) is included in every table as the null model (CLAUDE.md rule 4).
