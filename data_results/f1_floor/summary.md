# F1: origin of the shared approach roll floor

Pre-registered in `docs/note/f1_floor_origin_preregistration.md`. Static (no motion) has 0° here (the drill does not move in the approach); sweeps at the true pose. Rules first.

| Rule | official splat | our seed-0 splat | Result |
|---|---|---|---|
| F1a median whole-image argmin at frame 0 ≤ −5° | -7.0° | -12.0° | pass |
| F1b median ROI argmin at approach frames 1–3 ≤ −10° | -3.0° | -18.0° | FAIL |

## Exploratory: per-frame medians

- official frame 1: median ROI argmin -3.0° (range -3.0 to -3.0)
- official frame 2: median ROI argmin -20.0° (range -20.0 to -20.0)
- official frame 3: median ROI argmin -3.0° (range -3.0 to -3.0)
- ours_seed0 frame 1: median ROI argmin -18.0° (range -18.0 to -18.0)
- ours_seed0 frame 2: median ROI argmin -20.0° (range -20.0 to -20.0)
- ours_seed0 frame 3: median ROI argmin -15.0° (range -15.0 to -15.0)
