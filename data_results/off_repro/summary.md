# 8.OFF: official POGS code, static roll sweeps

Pre-registered in `docs/note/off_reproduction_preregistration.md`. Static (no motion) is not applicable: no tracking is run. Rules first.

Setup: {"n_clusters": 7, "vote": 2, "n_keep": 12617, "n_gaussians": 388771, "centroid": [0.1936320960521698, 0.11912115663290024, 0.15593497455120087], "dataset_scale": 1.0}

## Pre-registered results

| Rule | Statistic | Threshold | Result |
|---|---|---|---|
| OR1 | roll argmin, gripper + shadow (1° grid) = +2° | ≤ −5° | FAIL |
| OR2 | Spearman(s, argmin) = +0.00; shift s 0→1 = -0.5° | ≤ −0.8 and ≤ −4° | FAIL |
| OR3 | Spearman vs ours = +0.53; median abs diff = 2.0° (n 13) | ≥ 0.5 and ≤ 3° | pass |

## Rows

| View | s | argmin DINO (official) | DINO+depth (official, exploratory) | ours (G1 seed 0, release L) |
|---|---|---|---|---|
| gripper s=0 | 0 | +2.0 | +2.0 |  |
| gripper s=0.25 | 0.25 | -1.5 | -1.5 |  |
| gripper s=0.5 | 0.5 | -1.5 | -1.5 |  |
| gripper s=0.75 | 0.75 | -1.5 | -1.5 |  |
| gripper s=1 | 1 | +1.5 | +1.5 | -10 |
| gripper s=1.25 | 1.25 | +1.5 | +1.5 |  |
| track | 1 | +2.0 | +3.0 | -6 |
| track, no shadows | 1 | +2.0 | +3.0 | -6 |
| heldout az000 | 1 | -1.5 | -1.5 | +1 |
| heldout az030 | 1 | +3.0 | +3.0 | +2 |
| heldout az060 | 1 | +0.5 | +0.5 | +4 |
| heldout az090 | 1 | +1.5 | +1.5 | +4 |
| heldout az120 | 1 | +1.0 | +1.0 | +4 |
| heldout az150 | 1 | +3.5 | +3.5 | +3 |
| heldout az180 | 1 | +0.0 | +0.0 | +1 |
| heldout az210 | 1 | -2.5 | -2.5 | -4 |
| heldout az240 | 1 | -1.0 | -1.0 | -4 |
| heldout az270 | 1 | +0.5 | +0.5 | -4 |
| heldout az300 | 1 | +0.0 | +0.0 | -3 |
| heldout az330 | 1 | -3.0 | -3.0 | -2 |
