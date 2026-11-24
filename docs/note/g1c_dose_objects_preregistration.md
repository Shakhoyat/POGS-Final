# G1c pre-registration: does the shadow dose-response generalise beyond the drill?

Written 2026-11-25 ≈ 00:45 Dhaka, before any G1c sweep. Addresses the stated limitation "the dose-response is
measured on the drill only". Same code path as G1 `dose` (`experiments/g1_generality.py`), static, evaluation only.

## Design
- Splats: G1 seeds 0, 1, 2 of the tool scene.
- Objects: the four G1 primary objects (power_drill, ceramic_mug, mustard_bottle, cereal_box).
- Frame: the tracking camera, scene without the gripper (no new occluder; only the scene's own cast shadows are
  scaled), shadow strength s ∈ {0, 0.25, 0.5, 0.75, 1, 1.25} (shadow factor 1 − s(1 − 0.22), as G1).
- Axes: rotations about object x, y, z through the group centre; grid ±20° at 0.5°.
- Losses: the release DINO term L (background vs. empty render) and the rendered-background variant (as G1 P5).
- For each (splat, object, axis): Spearman(s, argmin L), shift = argmin L(s=1) − argmin L(s=0), and the same for
  the rendered background.

## Predictions and pass rules
- **C1 (generality of the dose-response).** For at least 3 of 4 objects, on at least 2 of 3 splats, at least one
  axis has |Spearman(s, argmin L)| ≥ 0.8 and |shift| ≥ 2°.
- **C2 (the background treatment controls it).** Over all (splat, object, axis) cases that meet C1's criterion, the
  rendered-background argmin moves by less than half the release shift in at least 75% of them.
C1 and C2 are reported in this order. C1 failing narrows the dose-response claim to the drill with a gripper's
shadow (a large occluder close to the object).

## Exploratory
The same per object with the drill's gripper frame (drill only, from G1) for comparison; the axis with the largest
shift per object.

## Budget
One GPU process, ≤ 1 GPU hour, outside lab hours. Output `data_results/g1c_dose_objects/` (summary committed).
