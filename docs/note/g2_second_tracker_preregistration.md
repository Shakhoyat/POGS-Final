# G2 pre-registration: the same mechanism in a second feature-field tracker (RSRD objective), static sweeps

Written 2026-11-24 before any G2 number exists; additions after the first result are labelled exploratory.
Evaluation-only static sweeps, no trajectories, not gated on the supervisor.

## Which second tracker, and why this one (code survey, 2026-11-24)
- **RSRD** (Kerr et al., CoRL 2024; `github.com/kerrj/rsrd`, `rsrd/motion/motion_optimizer.py`), the published
  ancestor of the POGS tracker, also optimises a rigid pose of a DINO-distilled Gaussian splat (DiG) against live
  DINO features. Its `_get_dino_loss` takes `(frame.dino_feats - dino_feats).norm(-1).mean()` over every pixel of the
  ROI, where `dino_feats` is the object-only render (unblurred inside `accumulation > 0.8`, blurred outside), so
  background pixels are compared with ~0, as in POGS. It adds `0.05 * |rgb_render - rgb_live|.mean()` over the same
  ROI (object-only render, so background RGB is also compared with ~0) and a depth term inside the object mask
  (metric L2 with a 0.1 m ignore threshold when metric depth is given; we use that branch, since our sim provides
  metric depth). Defaults: ROI inflate 0.25, mask threshold 0.8, blur kernel 5.
- POGS (release) blurs the whole render (`rigid_group_optimizer.py:304`); RSRD keeps it sharp inside the mask.
- Not used: methods with explicit segmentation-mask or RGB-only objectives (6DOPE-GS, GSGTrack, GoTrack) are a
  different class (no distilled-feature term); they are related work, not a test of this mechanism.
- Caveat stated up front: RSRD and POGS share authors and code lineage, so G2 tests "the objective family", not an
  independent implementation. The paper will say so.

## Implementation (static, in our framework)
The RSRD objective is evaluated with our splat and renderer (we do not run the RSRD codebase): L_RSRD(θ) =
D(θ) + 0.05 · C(θ) + Z(θ), with D = ROI mean of ||live − r_D||, r_D = sharp render inside the mask and blurred render
outside it; C = ROI mean L1 RGB against the object-only render; Z = metric depth L2 inside the mask (ignore > 0.1 m).
ROI: our POGS ROI construction (same crop as G1), not RSRD's 0.25 inflation; this is a stated deviation.
The pixel-sum split of G1 is applied to D and to C separately (D = D_in + D_out, C = C_in + C_out).

## Material
Same as G1 (seed-0 splat; drill, mug, mustard bottle, cereal box; tracking camera with and without shadows,
12 held-out ring views; drill also with the frame-0 gripper; axes x, y, z; grid ±20° at 1°).

## Predictions and pass rules
- **Q1.** Median background share of the curvature of the RSRD DINO term D (κ_{D,out}/κ_D) > 0.5 for ≥ 3 of 4 objects.
- **Q2.** Median background share of the full objective L_RSRD (κ of D_out + 0.05 C_out over κ of L_RSRD) > 0.5 for ≥ 3
  of 4 objects.
- **Q3.** Drill, tracking camera with the frame-0 gripper: roll argmin of L_RSRD ≤ −5° (same sign as POGS).

Budget: within the G1 cap (4 GPU hours total); one GPU process; runs after the G1 chain.
Outputs: `data_results/g2_rsrd/{sweeps.json, summary.md}`, runner `experiments/g2_rsrd.py`, ledger row 8.G2.
