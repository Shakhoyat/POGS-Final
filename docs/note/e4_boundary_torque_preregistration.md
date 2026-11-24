# E4 pre-registration: does the boundary torque of the implicit silhouette term predict the roll bias?

Written 2026-11-25 ≈ 01:10 Dhaka, before any torque value is computed. Extension E4 of the RSS draft
(`docs/paper_rss/main.tex`, Eq. 2). Exploratory additions later are labelled as such.

## Claim under test
The pose gradient of the ROI loss has a boundary term (1/N)∮(e_in − g) v_n ds with g = ‖f‖ the live feature norm
outside the silhouette. If the object features are unbiased (B1e) and the loss is locally quadratic, the roll minimum
is approximately â = −T / (2κ), where T is the derivative of the background term at the true pose and κ the local
curvature of L. A shadow changes g on one side of the silhouette and so changes T.

## Quantities (drill, roll about the bit axis = object x, through the group centre, at the true pose)
- g(p): norm of the live DINO feature (the tracker's own features, PCA-64) in the ROI; invalid pixels skipped.
- α_θ(p): accumulation of the object-only render at roll θ (soft silhouette, the tracker's own renderer).
- Background term B(θ) = (1/N) Σ_p (1 − α_θ(p)) g(p); T = [B(+1°) − B(−1°)] / 2 (per degree).
- κ: quadratic coefficient of L over ±5° (as G1), L measured on a ±20° grid at 0.5°.
- Predictor â_g = −T / (2κ). Null predictor â_c: the same with g replaced by its ROI mean (area change only, no
  spatial asymmetry).
- Measured a: argmin of L on the 0.5° grid (release loss, DINO only, as G1 dose).

## Conditions (60, fixed now)
Three splats (G1 seeds 0, 1, 2) × 20 frames: tracking camera with the frame-0 gripper at the six shadow strengths
s ∈ {0, 0.25, 0.5, 0.75, 1, 1.25}; tracking camera without gripper, shadows on and off; the 12 held-out ring views
(r 0.87 m, elevation 30°, shadows on). Same renders and code path as G1 (`experiments/g1_generality.py`).

## Predictions and pass rules
- **E4a.** Spearman(â_g, a) ≥ 0.5 over the 60 conditions.
- **E4b.** sign(â_g) = sign(a) in ≥ 75% of the conditions with |a| ≥ 2°.
- **E4c.** Spearman(â_g, a) − Spearman(â_c, a) ≥ 0.2 (the spatial asymmetry of g matters, not just mask area).
All three are reported in this order. E4a and E4b passing supports Eq. 2 as the mechanism's first-order account;
E4c failing would mean the shadow acts through total feature mass rather than its location, and the paper says so.

## Exploratory (no pass rule)
Pitch and yaw axes; the same predictor with the exact outside residual e(p) instead of g; the dose rows alone
(Spearman of T with s per splat).

## Budget
One GPU process, ≤ 1 GPU hour, never during lab hours. Output `data_results/e4_torque/` (summary committed).
