# E1a pre-registration: does silhouette over-coverage cause the shadow-driven roll bias? (pruning intervention)

Written 2026-11-25 ≈ 03:20 Dhaka, before any pruned sweep. Motivation (exploratory 8.OFF-S/S2): the official splat
gives no −10° bias; our splats' drill groups render 730–1100 px outside the true silhouette (official 330) and the
over-coverage ranks with the bias over four splats. This is an intervention on our splats. Ground truth (simulator
masks) is used to diagnose only, never by the tracker.

## Intervention
- Splats: ours, seeds 0, 1, 2 (G1). Group: the drill group from our grouping.
- "Outside" Gaussians: project each drill-group Gaussian centre into the 35 capture views; among views where it
  lands in the image, if the simulator id at that pixel is not the drill in ≥ 50% of them, the Gaussian is outside.
- Pruned group = drill group minus outside Gaussians (group rebuilt, centroid recomputed).
- Control group = drill group minus the same number of Gaussians chosen uniformly at random (seed 0), centroid
  recomputed.
- Sweeps: the G1 dose code path (release DINO term, roll about object x, ±20° at 0.5°) at the tracking camera with
  the frame-0 gripper, s ∈ {0, 1}; plus the over-coverage statistics of `silhouette_diag.py` at s = 1.

## Predictions and pass rules (in this order)
- **PR1.** Pruned: |roll argmin| at s = 1 ≤ 3° on ≥ 2 of 3 splats.
- **PR2.** Control: roll argmin at s = 1 ≤ −5° on ≥ 2 of 3 splats (random removal does not remove the bias).
- **PR3.** Pruned: |argmin(s = 1) − argmin(s = 0)| < 2° on ≥ 2 of 3 splats (the shadow dependence goes with it).
PR1–PR3 all pass → over-coverage of the object group is a cause of the static shadow-driven bias. PR1 pass with
PR2 fail → removing any Gaussians changes the minimum (not specific). PR1 fail → over-coverage is not sufficient.

## Budget
One GPU process, ≤ 30 min, outside lab hours. Output `data_results/e1a_prune/` (summary committed).
