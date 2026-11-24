# OR6 pre-registration: the official tracker on the in-grasp protocol, scored like Table I

Written 2026-11-25 ≈ 04:25 Dhaka, before any OR6 run. Addendum to 8.OFF (same official splat, environment and
deviations). OR4 showed the official tracker's roll error before contact; OR6 scores whole trials.

## Design
- Trials: O4 `grasp_{+15,-15,+30,-30}_s{0..9}` (40 trials; the first 10 seeds of the E0 setting lists).
- Frames: every frame of each trial, as in E0 (stereo depth profile), exported unchanged.
- Official loop as OR4 (release init; 25 iterations per frame; no per-frame depth).
- Scoring: our O4 code (`experiments/o4_ingrasp.judge`): pre-registered full rule (tip < 3 cm, geodesic rotation
  < 5°, never lost) and the paper's roll-free rule (tip < 3 cm, bit-axis swing < 5°, never lost) at N frames after
  the perturbation; tip from the drill tip at scan time relative to the official group centroid.
- Comparison on the same 40 trials: our release-configured tracker (`POGS (release: no per-frame depth)`) and our
  `POGS` from E0, and `Static (no motion)`.

## Predictions and rules (in this order)
- **OR6a (agreement).** |official − ours (release)| roll-free recoveries ≤ 6 of 40.
- **OR6b (paper level).** Official roll-free recovery < 32 of 40 (80%; the paper reports 48/50 at ≤ 15°, 40/50 at 30°).
Static is reported for both rules.

## Budget
One GPU process, ≤ 1.5 GPU hours, outside lab hours. Output `data_results/off_repro/summary_or6.md`.
