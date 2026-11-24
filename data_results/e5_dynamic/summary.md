# E5: static sweeps along the true lift trajectory (release vs. masked objective)

Pre-registered in `docs/note/e5_dynamic_failure_preregistration.md`. Static (no motion) is not applicable: the tracker is not run; sweeps at the ground-truth pose. Rules first.

Lift frames: 120; depth weight 3.7.

## Pre-registered results

| Rule | Statistic | Threshold | Result |
|---|---|---|---|
| D1 | median max-axis abs(argmin): J_mask 6.0 mm vs J_release 6.0 mm | ≥ 5 mm and ≥ 2× | FAIL |
| D2 | median κ(J_mask)/κ(J_release) = 6.37 (n 360) | < 0.5 | FAIL |

## Exploratory

- Approach frames (80): median max-axis abs(argmin) J_mask 1.5 mm, J_release 3.0 mm; κ ratio 7.76.
- Lift, term L: median abs(argmin) per axis 1.5 mm.
- Lift, term inside: median abs(argmin) per axis 3.0 mm.
- Lift, term depth: median abs(argmin) per axis 9.0 mm.
