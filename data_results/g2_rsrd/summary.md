# G2: RSRD objective, static sweeps (seed-0 splat)

Pre-registered in `docs/note/g2_second_tracker_preregistration.md`. Evaluation only; no trajectories, so Static (no motion) does not apply. Rules first.

| Object | Q1 median b(D) | Q2 median b(L_RSRD) | n |
|---|---|---|---|
| power_drill | 0.09 | 0.08 | 45/45 |
| ceramic_mug | 0.23 | 0.21 | 42/42 |
| mustard_bottle | 0.47 | 0.43 | 42/42 |
| cereal_box | 0.48 | 0.44 | 42/42 |

Q1: 0/4 → **FAIL**
Q2: 0/4 → **FAIL**
Q3: drill roll argmin of L_RSRD with the frame-0 gripper = -10° → **PASS**

