# 8.OFF addendum (OR4): does the OFFICIAL tracker acquire the roll error before contact?

Written 2026-11-25 ≈ 01:35 Dhaka, before any official-tracker trajectory has run and before the OR1–OR3 sweeps have
produced a number. Addendum to `off_reproduction_preregistration.md` (same environment, splat and deviations).

## Why
In our implementation the roll error about the bit axis is already 12.5° at the end of the approach, while the drill
still rests on the table (E1). During the approach the drill does not move, so `Static (no motion)` has zero error
and any tracker error there is a bias, not a failure to follow motion. Running the official tracking loop on the
same frames tests the floor itself with the official code.

## Design
- Sequences: O4 `grasp_+15_s{0..9}`, `grasp_-15_s{0..9}` (20; identical approach phases up to the start pose).
- Frames: as rendered for E0 (889×500, 5 Hz, depth with the stereo noise profile used in E0), exported unchanged.
- Official loop, as `track_main_online_demo.py`: `set_frame` on frame 0 → `initialize_obj_pose()` (release
  defaults: 100 iterations without depth, then 70 with depth and ROI) → for every later frame
  `set_observation(PosedObservation)` + `step(niter=25)` (release: no per-frame depth).
- Error: signed roll about the drill-bit axis and swing of the axis, from the official part pose vs. the ground truth
  (`metrics.pose.roll_swing_error_deg`), at every approach frame; Static is zero by construction in this phase.

## Prediction and rule
- **OR4.** Median over the 20 sequences of |roll| at the last approach frame ≥ 5°.
Reported with Static (0°) and our implementation's value on the same frames where available. OR4 failing with
OR1/OR2 passing would mean the official tracker does not reach the biased minimum within its iteration budget.

## Exploratory
Roll through the lift; swing; the sign of the roll.

## Budget
Inside the 8.OFF cap (≤ 3 GPU hours in total), one process, outside lab hours.
