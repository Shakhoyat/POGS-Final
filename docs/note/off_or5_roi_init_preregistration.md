# OR5 pre-registration: does an ROI-only initialisation remove the official tracker's roll floor?

Written 2026-11-25 ≈ 03:47 Dhaka, before any OR5 run. Motivation: F1a (pass) — the release initialisation's first
stage optimises the DINO loss over the whole image (no ROI) against an object-only render, and its roll minimum at
frame 0 is −7.0° on the official splat (the official tracker starts at −7.4°) and −12° on ours. The official objective
otherwise mostly prefers the truth (8.OFF-P), and the error then grows during the approach.

Caveat recorded with this test: the 20 O4 sequences share identical approach frames (the seed changes only the
later start pose; F1 ranges were zero), so the approach phase is one trajectory; variation across sequences is
optimiser noise. OR5 therefore uses 5 sequences (`grasp_+15_s0..4`), and the lift (which differs per seed only
through the start pose) is exploratory.

## Variant (one change to the official loop; everything else official, same splat and frames as OR4)
`initialize_obj_pose` stage 1 (100 iterations, Adam lr 0.005, no depth) evaluates the loss on the ROI around the
object at the scan pose (`use_roi=True`, ROI from the official `calculate_roi`) instead of the whole image. Stage 2
(70 iterations, ROI + depth) and per-frame steps (25 iterations, ROI, no depth) unchanged.

## Predictions and rules
- **OR5a.** Roll right after initialisation (frame 0): median |roll| ≤ 3° (OR4 release: 7.4°).
- **OR5b.** Roll at the last approach frame: median |roll| ≤ 10° (OR4 release: 19.4°).
Reported against the OR4 release numbers and Static (0°). OR5a pass with OR5b fail → the initialisation is fixed but
the per-frame drift remains.

## Budget
One GPU process, ≤ 20 min, outside lab hours. Output `data_results/off_repro/summary_or5.md`.
