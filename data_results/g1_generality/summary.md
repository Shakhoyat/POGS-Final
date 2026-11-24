# G1: generality of the background-silhouette mechanism (static sweeps)

Pre-registered in `docs/note/g1_generality_preregistration.md`. Evaluation only; no trajectories. Static (no motion) is not applicable: no tracking is run. Rules first, then exploratory tables.

Split check: max |L - S_in - S_out| = 9.8e-07.

## Pre-registered results

### P1: background share of the restoring curvature

| Splat | Object | median b | n (views x axes, κ_L > 0) | b > 0.5 |
|---|---|---|---|---|
| seed 0 | power_drill | 0.10 | 45 | no |
| seed 0 | ceramic_mug | 0.25 | 42 | no |
| seed 0 | mustard_bottle | 0.54 | 42 | yes |
| seed 0 | cereal_box | 0.51 | 42 | yes |
| seed 0 | soup_can (exploratory) | 0.46 | 42 | no |
| seed 0 | red_apple (exploratory) | 0.37 | 36 | no |
| seed 1 | power_drill | 0.17 | 45 | no |
| seed 1 | ceramic_mug | 0.40 | 42 | no |
| seed 1 | mustard_bottle | 0.60 | 42 | yes |
| seed 1 | cereal_box | 0.61 | 42 | yes |
| seed 1 | soup_can (exploratory) | 0.29 | 41 | no |
| seed 1 | red_apple (exploratory) | 0.30 | 34 | no |
| seed 2 | power_drill | 0.27 | 45 | no |
| seed 2 | ceramic_mug | 0.27 | 42 | no |
| seed 2 | mustard_bottle | 0.55 | 41 | yes |
| seed 2 | cereal_box | 0.50 | 42 | no |
| seed 2 | soup_can (exploratory) | 0.40 | 42 | no |
| seed 2 | red_apple (exploratory) | 0.61 | 36 | yes |

P1: 2/4 primary objects on seed 0 with median b > 0.5; drill on new splats: [False, False]. **FAIL**

### P2: drill roll bias with the frame-0 gripper, by splat

| Splat | argmin L (deg) |
|---|---|
| seed 0 | -10 |
| seed 1 | -10 |
| seed 2 | -5 |

P2: **PASS**

### P3: held-out views, mean |argmin| of release vs rendered background (seed 0)

| Object | axis (largest κ_L) | release | rendered background | lower |
|---|---|---|---|---|
| power_drill | x | 3.0 | 1.2 | yes |
| ceramic_mug | y | 1.7 | 1.5 | yes |
| mustard_bottle | x | 2.9 | 3.2 | no |
| cereal_box | x | 1.1 | 1.2 | no |

P3: 2/4 objects. **FAIL**

### P4/P5: shadow dose-response (drill, tracking camera, frame-0 gripper; roll argmin, deg)

| Splat | s=0 | s=0.25 | s=0.5 | s=0.75 | s=1 | s=1.25 | Spearman ρ | move 0→1 | bg move 0→1 |
|---|---|---|---|---|---|---|---|---|---|
| seed 0 | -2.5 | -7.5 | -8.5 | -8.5 | -10.5 | -10.5 | -0.97 | -8.0 | -1.5 |
| seed 1 | -1.5 | -4.0 | -10.5 | -10.5 | -10.5 | -10.5 | -0.85 | -9.0 | -0.5 |
| seed 2 | +0.5 | -2.0 | -4.5 | -4.5 | -4.5 | -4.5 | -0.85 | -5.0 | -1.0 |

P4: ρ ≤ −0.8 on 3/3, movement ≥ 4° on 3/3. **PASS**
P5: rendered-background movement < half of release on 3/3. **PASS**

### G1b T1: object-feature share of the translational stiffness (seed 0)

| Object | median κ(inside·frac)/κ(L) | median κ(rendered bg)/κ(L) | median argmin L / bg (mm) | < 0.5 |
|---|---|---|---|---|
| power_drill | 0.94 | 1.46 | 0.0 / 1.5 | no |
| ceramic_mug | 0.89 | 1.40 | 1.5 / 1.5 | no |
| mustard_bottle | 0.75 | 1.60 | 1.5 / 1.5 | no |
| cereal_box | 1.00 | 1.43 | 1.5 / 1.5 | no |

T1: 0/4. **FAIL**

## Exploratory: argmins by object and view set (median [min, max], deg)

| Splat | Object | axis | views | release L | inside mean | rendered bg | depth |
|---|---|---|---|---|---|---|---|
| 0 | power_drill | x | tracking cam (3) | -6 [-10, -6] | +2 [+2, +3] | +3 [+2, +4] | -1 [-1, -1] |
| 0 | power_drill | x | held-out ring (12) | +1 [-4, +4] | +2 [-4, +3] | +1 [-1, +2] | +1 [-1, +3] |
| 0 | power_drill | y | tracking cam (3) | -1 [-1, -1] | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] |
| 0 | power_drill | y | held-out ring (12) | +0 [-1, +0] | +0 [-1, +1] | +0 [-3, +1] | +0 [-1, +1] |
| 0 | power_drill | z | tracking cam (3) | -1 [-1, -1] | +0 [-1, +0] | -1 [-1, -1] | +1 [+1, +1] |
| 0 | power_drill | z | held-out ring (12) | +0 [-2, +3] | +0 [-3, +4] | +0 [-3, +3] | +0 [-3, +3] |
| 0 | ceramic_mug | x | tracking cam (2) | -4 [-4, -4] | -2 [-2, -2] | -1 [-1, -1] | -3 [-3, -3] |
| 0 | ceramic_mug | x | held-out ring (12) | +0 [-2, +3] | +0 [-2, +6] | -1 [-3, +2] | -1 [-20, +20] |
| 0 | ceramic_mug | y | tracking cam (2) | -1 [-1, -1] | +0 [+0, +0] | -1 [-1, -1] | -1 [-1, -1] |
| 0 | ceramic_mug | y | held-out ring (12) | +2 [-2, +5] | +1 [-20, +5] | +1 [-2, +4] | +0 [-12, +20] |
| 0 | ceramic_mug | z | tracking cam (2) | +2 [+1, +3] | -1 [-2, +0] | +4 [+4, +5] | +20 [+20, +20] |
| 0 | ceramic_mug | z | held-out ring (12) | +0 [-16, +5] | -2 [-17, +8] | +2 [-10, +6] | -0 [-19, +20] |
| 0 | mustard_bottle | x | tracking cam (2) | -3 [-3, -3] | +5 [+5, +5] | -4 [-5, -4] | +2 [+2, +2] |
| 0 | mustard_bottle | x | held-out ring (12) | -0 [-4, +14] | +2 [-13, +20] | +0 [-13, +6] | +4 [-12, +17] |
| 0 | mustard_bottle | y | tracking cam (2) | +1 [+1, +1] | +3 [+3, +3] | +1 [+1, +1] | +7 [+7, +7] |
| 0 | mustard_bottle | y | held-out ring (12) | +1 [-16, +11] | +0 [-6, +7] | -0 [-18, +1] | +4 [-9, +20] |
| 0 | mustard_bottle | z | tracking cam (2) | +6 [+6, +6] | -6 [-8, -5] | +18 [+17, +20] | -17 [-17, -17] |
| 0 | mustard_bottle | z | held-out ring (12) | +3 [-20, +20] | -1 [-20, +9] | +2 [-17, +18] | +3 [-18, +20] |
| 0 | cereal_box | x | tracking cam (2) | -1 [-1, -1] | +1 [+1, +1] | +0 [+0, +0] | -4 [-4, -4] |
| 0 | cereal_box | x | held-out ring (12) | +0 [-3, +2] | +0 [-3, +2] | -1 [-3, +1] | +0 [-6, +19] |
| 0 | cereal_box | y | tracking cam (2) | -1 [-1, -1] | +0 [+0, +0] | +0 [+0, +0] | +2 [+2, +2] |
| 0 | cereal_box | y | held-out ring (12) | +0 [-6, +3] | +0 [-2, +5] | +0 [-3, +2] | +2 [-9, +10] |
| 0 | cereal_box | z | tracking cam (2) | -2 [-2, -2] | -3 [-3, -3] | -2 [-2, -2] | +1 [+1, +1] |
| 0 | cereal_box | z | held-out ring (12) | +2 [-4, +3] | +0 [-6, +18] | +0 [-5, +7] | +1 [-9, +13] |
| 0 | soup_can | x | tracking cam (2) | +0 [+0, +0] | -10 [-10, -10] | +0 [+0, +0] | -20 [-20, -20] |
| 0 | soup_can | x | held-out ring (12) | +0 [-5, +13] | +4 [-17, +16] | +1 [-6, +20] | +4 [-4, +20] |
| 0 | soup_can | y | tracking cam (2) | -2 [-2, -2] | -4 [-5, -3] | -2 [-4, +0] | -20 [-20, -20] |
| 0 | soup_can | y | held-out ring (12) | +0 [-3, +2] | -0 [-4, +18] | +4 [-4, +15] | +2 [-3, +20] |
| 0 | soup_can | z | tracking cam (2) | -0 [-1, +0] | +5 [+5, +5] | -4 [-4, -4] | -20 [-20, -20] |
| 0 | soup_can | z | held-out ring (12) | +3 [-10, +19] | +4 [-20, +11] | -2 [-20, +9] | +4 [-17, +19] |
| 0 | red_apple | x | tracking cam (2) | +6 [+2, +10] | +1 [+1, +1] | +17 [+17, +17] | -15 [-15, -15] |
| 0 | red_apple | x | held-out ring (12) | +2 [-11, +16] | +1 [-13, +4] | -4 [-20, +20] | -12 [-20, +20] |
| 0 | red_apple | y | tracking cam (2) | +4 [+4, +4] | +3 [+1, +5] | +2 [+2, +2] | -20 [-20, -20] |
| 0 | red_apple | y | held-out ring (12) | -2 [-18, +16] | +2 [-17, +20] | -4 [-20, +14] | +14 [-20, +20] |
| 0 | red_apple | z | tracking cam (2) | +10 [+9, +12] | +15 [+15, +15] | +9 [+8, +10] | +20 [+20, +20] |
| 0 | red_apple | z | held-out ring (12) | -1 [-18, +13] | +0 [-15, +18] | +3 [-15, +20] | +4 [-20, +20] |
| 1 | power_drill | x | tracking cam (3) | -3 [-10, -3] | +0 [+0, +1] | +2 [+2, +2] | -1 [-1, -1] |
| 1 | power_drill | x | held-out ring (12) | +0 [-5, +7] | +0 [-3, +4] | +1 [-2, +2] | +0 [-2, +3] |
| 1 | power_drill | y | tracking cam (3) | -1 [-1, +0] | +1 [+1, +1] | +0 [+0, +0] | +0 [+0, +0] |
| 1 | power_drill | y | held-out ring (12) | +0 [-1, +2] | +0 [-1, +2] | +0 [-3, +1] | +0 [-1, +1] |
| 1 | power_drill | z | tracking cam (3) | -1 [-1, -1] | -1 [-1, -1] | -1 [-1, -1] | +2 [+2, +2] |
| 1 | power_drill | z | held-out ring (12) | +0 [-2, +1] | -0 [-3, +3] | +0 [-3, +2] | +0 [-2, +3] |
| 1 | ceramic_mug | x | tracking cam (2) | -4 [-4, -4] | -1 [-1, -1] | -0 [-1, +0] | -1 [-1, -1] |
| 1 | ceramic_mug | x | held-out ring (12) | +0 [-2, +20] | +0 [-1, +6] | +0 [-2, +2] | +0 [-19, +20] |
| 1 | ceramic_mug | y | tracking cam (2) | +1 [+1, +1] | +2 [+1, +2] | +0 [+0, +0] | -2 [-2, -2] |
| 1 | ceramic_mug | y | held-out ring (12) | +2 [-20, +4] | +1 [-20, +4] | +1 [-1, +5] | +0 [-14, +20] |
| 1 | ceramic_mug | z | tracking cam (2) | +2 [+0, +3] | -4 [-6, -1] | +5 [+4, +6] | +20 [+20, +20] |
| 1 | ceramic_mug | z | held-out ring (12) | -0 [-12, +7] | -1 [-9, +4] | +0 [-8, +6] | -0 [-20, +20] |
| 1 | mustard_bottle | x | tracking cam (2) | -5 [-5, -5] | +6 [+4, +7] | -4 [-4, -4] | +1 [+1, +1] |
| 1 | mustard_bottle | x | held-out ring (12) | -1 [-9, +15] | +2 [-10, +20] | +0 [-19, +6] | +4 [-11, +16] |
| 1 | mustard_bottle | y | tracking cam (2) | +1 [+1, +1] | +2 [+2, +2] | +1 [+1, +1] | +1 [+1, +1] |
| 1 | mustard_bottle | y | held-out ring (12) | +0 [-19, +9] | -0 [-5, +3] | +0 [-11, +8] | +4 [-9, +20] |
| 1 | mustard_bottle | z | tracking cam (2) | +16 [+15, +17] | -2 [-2, -1] | +17 [+14, +20] | -20 [-20, -20] |
| 1 | mustard_bottle | z | held-out ring (12) | +2 [-20, +20] | -3 [-14, +16] | +2 [-20, +20] | +4 [-20, +20] |
| 1 | cereal_box | x | tracking cam (2) | +0 [+0, +0] | +2 [+2, +2] | +1 [+1, +1] | -1 [-1, -1] |
| 1 | cereal_box | x | held-out ring (12) | +0 [-1, +4] | +0 [-3, +5] | -2 [-10, +2] | +2 [-4, +20] |
| 1 | cereal_box | y | tracking cam (2) | -2 [-2, -2] | -1 [-1, -1] | +0 [+0, +0] | -2 [-2, -2] |
| 1 | cereal_box | y | held-out ring (12) | +0 [-14, +5] | -1 [-4, +17] | +0 [-5, +6] | +2 [-6, +9] |
| 1 | cereal_box | z | tracking cam (2) | -4 [-4, -4] | -5 [-5, -5] | -3 [-3, -3] | -3 [-3, -3] |
| 1 | cereal_box | z | held-out ring (12) | +0 [-6, +7] | +3 [-9, +16] | +0 [-9, +9] | +0 [-2, +13] |
| 1 | soup_can | x | tracking cam (2) | +0 [+0, +0] | -7 [-7, -7] | -1 [-1, -1] | -20 [-20, -20] |
| 1 | soup_can | x | held-out ring (12) | -0 [-7, +8] | +2 [-11, +19] | +1 [-2, +15] | +4 [-3, +20] |
| 1 | soup_can | y | tracking cam (2) | -1 [-1, -1] | -1 [-2, +0] | -3 [-3, -3] | -20 [-20, -20] |
| 1 | soup_can | y | held-out ring (12) | +0 [-2, +3] | +1 [-3, +12] | +0 [-4, +3] | +2 [-4, +20] |
| 1 | soup_can | z | tracking cam (2) | +0 [+0, +0] | +2 [+1, +2] | -2 [-2, -2] | -20 [-20, -20] |
| 1 | soup_can | z | held-out ring (12) | +2 [-2, +10] | +0 [-18, +4] | -1 [-7, +7] | -4 [-16, +18] |
| 1 | red_apple | x | tracking cam (2) | +7 [+3, +11] | +2 [+2, +2] | +16 [+16, +17] | -17 [-17, -17] |
| 1 | red_apple | x | held-out ring (12) | +4 [-19, +17] | +2 [-12, +10] | -1 [-20, +20] | -12 [-20, +20] |
| 1 | red_apple | y | tracking cam (2) | +3 [+3, +3] | +1 [+1, +1] | +1 [+1, +1] | -20 [-20, -20] |
| 1 | red_apple | y | held-out ring (12) | -4 [-20, +20] | +0 [-20, +20] | -2 [-20, +15] | +14 [-20, +20] |
| 1 | red_apple | z | tracking cam (2) | +6 [-8, +20] | +6 [-7, +19] | +4 [+4, +4] | +20 [+20, +20] |
| 1 | red_apple | z | held-out ring (12) | +2 [-14, +18] | +6 [-14, +20] | +4 [-10, +20] | +3 [-20, +20] |
| 2 | power_drill | x | tracking cam (3) | -1 [-5, +0] | +2 [+2, +2] | +3 [+3, +3] | -1 [-1, -1] |
| 2 | power_drill | x | held-out ring (12) | +0 [-5, +3] | +0 [-3, +1] | +0 [-2, +3] | +0 [-1, +3] |
| 2 | power_drill | y | tracking cam (3) | -1 [-1, -1] | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] |
| 2 | power_drill | y | held-out ring (12) | +0 [-1, +0] | -0 [-1, +1] | +0 [-4, +1] | +0 [-1, +1] |
| 2 | power_drill | z | tracking cam (3) | +1 [-1, +1] | -1 [-1, -1] | -1 [-1, -1] | +1 [+1, +1] |
| 2 | power_drill | z | held-out ring (12) | +0 [-3, +2] | -0 [-6, +3] | +0 [-4, +3] | +0 [-3, +2] |
| 2 | ceramic_mug | x | tracking cam (2) | -3 [-3, -3] | -2 [-3, -2] | -1 [-1, -1] | -2 [-2, -2] |
| 2 | ceramic_mug | x | held-out ring (12) | +0 [-3, +5] | +0 [-3, +5] | +0 [-1, +3] | -1 [-20, +20] |
| 2 | ceramic_mug | y | tracking cam (2) | -1 [-1, -1] | -0 [-2, +1] | -0 [-1, +0] | +0 [+0, +0] |
| 2 | ceramic_mug | y | held-out ring (12) | +0 [-20, +5] | +0 [-20, +9] | +1 [-1, +5] | +0 [-12, +20] |
| 2 | ceramic_mug | z | tracking cam (2) | +4 [+3, +4] | -3 [-8, +2] | +4 [+3, +5] | +19 [+19, +19] |
| 2 | ceramic_mug | z | held-out ring (12) | +0 [-3, +9] | -1 [-9, +6] | -0 [-11, +7] | -1 [-19, +20] |
| 2 | mustard_bottle | x | tracking cam (2) | -3 [-3, -3] | +2 [+0, +3] | -4 [-5, -4] | +0 [+0, +0] |
| 2 | mustard_bottle | x | held-out ring (12) | +0 [-9, +16] | +0 [-9, +19] | -2 [-20, +5] | +4 [-12, +16] |
| 2 | mustard_bottle | y | tracking cam (2) | +1 [+1, +1] | +0 [+0, +0] | +0 [+0, +1] | -10 [-10, -10] |
| 2 | mustard_bottle | y | held-out ring (12) | +0 [-17, +11] | +0 [-4, +3] | +0 [-20, +10] | +4 [-8, +20] |
| 2 | mustard_bottle | z | tracking cam (2) | +14 [+14, +14] | +0 [+0, +0] | +20 [+20, +20] | -20 [-20, -20] |
| 2 | mustard_bottle | z | held-out ring (12) | -2 [-20, +20] | -2 [-19, +20] | -2 [-20, +20] | -2 [-20, +20] |
| 2 | cereal_box | x | tracking cam (2) | +0 [+0, +0] | +2 [+2, +2] | +1 [+1, +1] | -3 [-3, -3] |
| 2 | cereal_box | x | held-out ring (12) | +0 [-3, +1] | +0 [-3, +3] | -1 [-3, +1] | -0 [-18, +15] |
| 2 | cereal_box | y | tracking cam (2) | -2 [-2, -2] | -1 [-1, -1] | +0 [+0, +0] | +0 [+0, +0] |
| 2 | cereal_box | y | held-out ring (12) | -0 [-10, +4] | -0 [-5, +4] | +0 [-2, +3] | -0 [-9, +12] |
| 2 | cereal_box | z | tracking cam (2) | -3 [-3, -3] | -4 [-4, -4] | -2 [-2, -2] | -1 [-1, -1] |
| 2 | cereal_box | z | held-out ring (12) | +1 [-4, +4] | +2 [-6, +19] | +0 [-4, +5] | +0 [-12, +6] |
| 2 | soup_can | x | tracking cam (2) | +1 [+1, +1] | -1 [-1, -1] | +3 [+2, +4] | -20 [-20, -20] |
| 2 | soup_can | x | held-out ring (12) | -0 [-3, +4] | +0 [-5, +13] | +1 [-20, +12] | +4 [-2, +20] |
| 2 | soup_can | y | tracking cam (2) | -1 [-1, -1] | +0 [+0, +1] | -2 [-2, -1] | -20 [-20, -20] |
| 2 | soup_can | y | held-out ring (12) | +0 [-1, +2] | +1 [-1, +4] | +0 [-20, +7] | +2 [-5, +20] |
| 2 | soup_can | z | tracking cam (2) | -1 [-2, +0] | +0 [+0, +1] | -2 [-2, -2] | -20 [-20, -20] |
| 2 | soup_can | z | held-out ring (12) | +2 [-3, +5] | +2 [-4, +7] | +0 [-20, +20] | -2 [-14, +19] |
| 2 | red_apple | x | tracking cam (2) | +10 [+10, +10] | +2 [+2, +2] | +16 [+16, +16] | -15 [-15, -15] |
| 2 | red_apple | x | held-out ring (12) | +5 [-12, +18] | +2 [-12, +20] | -4 [-20, +20] | -8 [-20, +20] |
| 2 | red_apple | y | tracking cam (2) | +2 [+2, +2] | +0 [+0, +0] | +1 [+1, +1] | -20 [-20, -20] |
| 2 | red_apple | y | held-out ring (12) | -3 [-20, +19] | +0 [-20, +11] | -2 [-20, +16] | +14 [-20, +20] |
| 2 | red_apple | z | tracking cam (2) | -7 [-7, -7] | -1 [-1, -1] | -1 [-2, +0] | +20 [+20, +20] |
| 2 | red_apple | z | held-out ring (12) | -2 [-15, +13] | +2 [-15, +11] | +0 [-16, +20] | +2 [-20, +20] |
