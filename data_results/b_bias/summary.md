# Direction B diagnostics (B0 ground truth, B1 viewpoint / grid / weights)

Evaluation only. Single splat `tool_table`. Argmins in degrees of roll about the bit axis (median [min, max] over views).

## B0: is the ground truth right?

Metadata: 800 approach frames in 200 cases; max GT rotation 0.00e+00 deg, max translation 0.00e+00 m (drill at the scanned pose).

Argmin of each term along each object-frame DOF, tracking camera frame 0 (deg / mm, grid ±6 at 0.5):

| DOF | RGB median [min, max] | depth | DINO |
|---|---|---|---|
| rot_bit(roll) | +0.0 [+0.0, +0.0] | -1.5 [-1.5, -1.5] | -6.0 [-6.0, -6.0] |
| rot_perp | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | -0.5 [-0.5, -0.5] |
| rot_z | +0.0 [+0.0, +0.0] | +1.5 [+1.5, +2.0] | -1.0 [-1.0, -1.0] |
| t_bit | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.5 [+0.5, +0.5] |
| t_perp | +0.5 [+0.5, +0.5] | -4.0 [-4.0, -4.0] | -3.0 [-3.0, -3.0] |
| t_z | +0.5 [+0.5, +0.5] | +0.0 [+0.0, +0.0] | -0.5 [-0.5, -0.5] |

## B1a: roll argmin by viewpoint (grid ±30 at 1)

| View set | n | DINO | depth | RGB |
|---|---|---|---|---|
| training views (distilled on) | 35 | -1 [-6, +8] | +1 [-3, +3] | +0 [+0, +0] |
| tracking camera, frame 0 | 3 | -10 [-10, -10] | -1 [-1, -1] | +0 [+0, +0] |

Per training view (DINO argmin): 00:-3, 01:-6, 02:+8, 03:-1, 04:+1, 05:+2, 06:-3, 07:+7, 08:-1, 09:-4, 10:+1, 11:-2, 12:+2, 13:+4, 14:-4, 15:+4, 16:-5, 17:-1, 18:+4, 19:-4, 20:+8, 21:-2, 22:-5, 23:+3, 24:-4, 25:+2, 26:+2, 27:-1, 28:+4, 29:-1, 30:-1, 31:+3, 32:-2, 33:+4, 34:+1

## B1c: argmin of the combined loss vs term weight

| View set | n | DINO+0·depth | DINO+1·depth | DINO+3.7·depth | DINO+10·depth | DINO+30·depth | DINO+100·depth | DINO+1000·depth |
|---|---|---|---|---|---|---|---|---|
| training views | 35 | -1 [-6, +8] | -1 [-6, +8] | -1 [-6, +8] | -1 [-5, +7] | -1 [-5, +6] | +0 [-4, +6] | +1 [-2, +3] |
| tracking camera | 3 | -10 [-10, -10] | -10 [-10, -10] | -10 [-10, -10] | -8 [-8, -8] | -6 [-6, -6] | -3 [-3, -2] | -2 [-2, -2] |

| View set | n | DINO+0·RGB | DINO+0.1·RGB | DINO+0.3·RGB | DINO+1·RGB | DINO+3·RGB | DINO+10·RGB |
|---|---|---|---|---|---|---|---|
| training views | 35 | -1 [-6, +8] | -1 [-5, +6] | +0 [-5, +6] | +0 [-2, +1] | +0 [-1, +1] | +0 [+0, +0] |
| tracking camera | 3 | -10 [-10, -10] | -10 [-10, -10] | -6 [-6, -6] | -1 [-1, -1] | +0 [+0, +0] | +0 [+0, +0] |

## B1b: patch-grid shift (grasp_+15_s0, frame 0; DINO patch ≈ 10 px at this ROI scale)

| ROI shift (px) | DINO | depth | RGB |
|---|---|---|---|
| +0, +0 | -10 | -1 | +0 |
| +3, +0 | -10 | -1 | +0 |
| +7, +0 | -8 | -1 | +0 |
| +10, +0 | -7 | -1 | +0 |
| +13, +0 | -10 | -1 | +0 |
| +0, +7 | -10 | -1 | +0 |
| +7, +7 | -8 | -1 | +0 |
| -7, +0 | -10 | -1 | +0 |
| +0, -7 | -10 | -1 | +0 |

## B1d: gripper / shadow context and held-out viewpoints (clean re-renders)

| View | DINO | depth | RGB |
|---|---|---|---|
| track, gripper (frame 0) | -10 | -1 | +0 |
| track, no gripper | -6 | -1 | +0 |
| track, gripper (frame 0), no shadows | -2 | -1 | +0 |
| track, no gripper, no shadows | -6 | -1 | +0 |

| Held-out ring | n | DINO median [min, max] | mean abs DINO argmin | depth | RGB |
|---|---|---|---|---|---|
| heldout r0.68 e30 | 12 | +1 [-4, +4] | 2.1 | +1 [-1, +3] | +0 [+0, +0] |
| heldout r0.68 e55 | 12 | +2 [-4, +6] | 3.4 | +1 [-1, +3] | +0 [+0, +0] |
| heldout r0.87 e30 | 12 | +1 [-5, +6] | 3.7 | +1 [-2, +3] | +0 [+0, +1] |

Per held-out view (DINO argmin): r0.87 e30 az000:+1, r0.87 e30 az030:+4, r0.87 e30 az060:+6, r0.87 e30 az090:+5, r0.87 e30 az120:+4, r0.87 e30 az150:+4, r0.87 e30 az180:+1, r0.87 e30 az210:-4, r0.87 e30 az240:-5, r0.87 e30 az270:-5, r0.87 e30 az300:-4, r0.87 e30 az330:-1, r0.68 e30 az000:+1, r0.68 e30 az030:+1, r0.68 e30 az060:+4, r0.68 e30 az090:+3, r0.68 e30 az120:+4, r0.68 e30 az150:+2, r0.68 e30 az180:+1, r0.68 e30 az210:-4, r0.68 e30 az240:-1, r0.68 e30 az270:+2, r0.68 e30 az300:-2, r0.68 e30 az330:+0, r0.68 e55 az000:+1, r0.68 e55 az030:+3, r0.68 e55 az060:+6, r0.68 e55 az090:+5, r0.68 e55 az120:+5, r0.68 e55 az150:+4, r0.68 e55 az180:+3, r0.68 e55 az210:-3, r0.68 e55 az240:-3, r0.68 e55 az270:-1, r0.68 e55 az300:-4, r0.68 e55 az330:-3

