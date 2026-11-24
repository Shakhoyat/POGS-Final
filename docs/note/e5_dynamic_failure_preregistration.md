# E5 pre-registration: do static sweeps along the true lift trajectory explain why the masked tracker drifts?

Written 2026-11-25 ≈ 01:00 Dhaka, before any E5 sweep. Extension E5 of the RSS draft. Background: restricting the
DINO term to the per-pose object mask (`dino_loss_region="object"`, E2d `maskobj`) loses the drill through
translation drift in the lift (1/0/4/4 of 25 vs. 9/12/16/23 for POGS), while static sweeps at frame 0 show that the
object features carry 75–100% of the translational stiffness (G1b T1 fail). G1b used frame 0 only (gripper above the
drill, no contact). E5 sweeps the frames in which the gripper holds and lifts the drill.

## Design
- Splat: tool_table seed 0 (the splat used for E0/E2d). Sequences: O4 `grasp_+15_s{0..9}` and `grasp_-15_s{0..9}`
  (20 sequences; the lift precedes the perturbation, so the sign only changes the start pose).
- Frames: every frame with phase `lift` (primary); `approach` frames as exploratory comparison.
- Pose: the ground-truth pose of the drill group in each frame (`gt_pose_for_group`); evaluation only.
- Sweeps: translations along world x, y, z through the GT pose, ±30 mm at 1.5 mm (G1b grid); rotations not swept.
- Objectives (what the trackers optimise, per-frame depth on as in E0/E2d POGS):
  J_release = DINO over the ROI (release) + 3.7 × depth term;
  J_mask = DINO inside the per-pose object mask (the `maskobj` term) + 3.7 × depth term.
  Depth weight and ignore threshold from `TrackingConfig` defaults.
- Per frame and axis: argmin (mm) and κ (quadratic over ±7.5 mm) of each objective.

## Predictions and pass rules (reported in this order)
- **D1 (translation bias under contact).** Over lift frames, the median of the per-frame maximum-over-axes
  |argmin| of J_mask is ≥ 5 mm and ≥ 2× the same statistic for J_release.
- **D2 (stiffness loss under contact).** Over lift frames × axes with κ(J_release) > 0, the median of
  κ(J_mask) / κ(J_release) is < 0.5.
Reading: D1 or D2 passing → the masked tracker's drift has a static counterpart once the gripper holds the drill
(a biased or flat objective along the true trajectory). Both failing → the failure is dynamic (estimate-dependent
masks and ROI, feedback), not a property of the objective at the true pose; the paper says so.

## Exploratory
Approach frames; the DINO terms alone (without depth); occlusion fraction of the drill per frame vs. argmin.

## Budget
One GPU process, ≤ 1.5 GPU hours, outside lab hours. Output `data_results/e5_dynamic/` (summary committed).
