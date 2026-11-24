# G1b pre-registration: does the background carry the translational restoring force?

Written 2026-11-24, after the seed-0 G1 rotation sweeps were read and before any translation sweep exists.

## Why (what G1 seed 0 showed; P1 and P3 already fail on seed 0, reported as such)
Exploratory reading of the seed-0 rotation sweeps: the object's own features supply most of the local rotational
curvature at the true pose (κ of the inside-object mean, scaled by the mask fraction, is 0.60–0.89 of κ_L for the four
primary objects), and a rendered background gives more curvature than the release loss (ratio 1.1–1.7). The rotational
basins are ±20° wide for every variant in most views. So the claim "the background supplies most of the restoring force"
does not hold for rotation in static sweeps. What the background clearly does for rotation is shift the minimum (shadow
dose-response, seed 0: ρ = −0.97, −8.0° for s = 0 → 1; rendered background −1.5°). The trajectory failures of the
masked variants (`object`: 1/0/4/4 of 25) were translation drift in the lift (tip error 10 cm). G1b tests whether that
failure has a static counterpart.

## Design
Same splat (seed 0), objects, views and code path as G1, with translations instead of rotations: world x, y, z through
the group centre, grid ±30 mm at 1.5 mm. Quantities: κ (quadratic over ±7.5 mm) of L, of the inside-object mean scaled
by the mask fraction (`inside_mean · mask_frac`, the object-feature part), and of the rendered-background term;
argmins of each.

## Prediction and pass rule
- **T1.** Median over (views × axes) of κ(inside_mean · mask_frac) / κ(L) < 0.5 for at least 3 of 4 primary objects
  (the object features supply less than half of the translational stiffness). Fails otherwise; then the paper states
  that the background term biases the minimum but does not supply the restoring force in either rotation or
  translation, and that the masked variants fail for dynamic reasons (gripper occlusion, mask feedback) not seen in
  static sweeps.
- **T2 (descriptive, no pass rule).** Translation argmins of L vs. rendered background, mm.

Budget: inside the G1 cap; one GPU process; runs after G1/G2. Output `data_results/g1_generality/trans_seed0.json`.
