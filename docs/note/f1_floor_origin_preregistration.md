# F1 pre-registration: where does the shared approach roll floor come from?

Written 2026-11-25 ≈ 03:25 Dhaka, before any F1 sweep. Motivation (8.OFF-C, exploratory): on the 20 OR4 sequences
the official tracker and our release tracker follow the same roll trajectory during the approach (median difference
0.7°, ρ 0.87), both ≈ −7° right after initialisation and ≈ −17 to −19° at the last approach frame, although their
frame-0 ROI minima with the gripper differ (ours −10°, official +2°). The floor's origin must be common to both
splats. Two candidates from the release loop: (a) the first initialisation stage optimises the DINO loss over the
WHOLE image (no ROI, no depth, object-only render), (b) the ROI minimum at the true pose moves as the gripper descends
(frames 1–3), which no sweep so far has covered.

## Design
- Splats: the official splat (8.OFF, used through our loss code, which agrees with the official objective: 8.OFF-S)
  and our seed-0 splat.
- Sequences: O4 `grasp_+15_s{0..9}`, `grasp_-15_s{0..9}`; the drill does not move in the approach, so the true pose is
  the scan pose.
- (a) Frame 0: roll sweep (object x, ±20° at 0.5°) of the whole-image DINO loss (`use_roi=False`, no depth, as the
  release's first init stage).
- (b) Approach frames 1, 2, 3: roll sweep of the release ROI DINO loss (no depth, as the release per-frame step).

## Predictions and pass rules (in this order; each for both splats)
- **F1a.** Median over sequences of the frame-0 whole-image argmin ≤ −5°.
- **F1b.** Median over sequences × frames 1–3 of the ROI argmin ≤ −10°.
Reading: F1a pass → the initialisation stage sets the starting bias; F1b pass → the descending gripper moves the ROI
minimum (the drift has a static counterpart frame by frame). Both fail → the drift is an optimisation-path effect.

## Budget
One GPU process, ≤ 45 min, outside lab hours. Output `data_results/f1_floor/` (summary committed).
