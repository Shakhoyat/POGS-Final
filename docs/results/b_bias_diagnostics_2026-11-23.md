# Direction B, steps 0–1: where the DINO roll bias comes from (2026-11-23)

Evaluation and diagnostics only (ground truth places the sweeps; the tracker never sees it). Single splat
`tool_table`, POGS paper variant (DINO + 3.7·depth). Script `experiments/b_bias.py`; numbers from
`data_results/b_bias/{gt,views,grid,context}.json` and `summary.md`. Roll = rotation about the drill-bit axis,
a right perturbation in the object frame about the group centroid; argmins come from 1-D sweeps (±30° at 1°
unless stated).

## Step 0 (B0): is the ground truth right? **Yes.**
In simulation, ground truth is the renderer's object transform and does not come from splat registration.
Two checks:
- **Metadata.** All 800 approach frames in 200 O4 cases have an exact identity drill transform (max 0.0°, 0.0 m).
  Capture and sequences use the same scene placement (`build_tool_scene()`).
- **6-DOF sweeps of the RGB-only and depth-only terms** on the tracking camera, frame 0, 8 cases, ±6° / ±6 mm at
  0.5. RGB argmin: 0.0° on all three rotation axes, ≤ 0.5 mm on all three translations (8/8 cases). The splat and
  the ground truth agree to the grid resolution.

The −10° is not a ground-truth artefact. On the same frames the DINO roll argmin is at the −6° grid edge;
the ±30° sweep gives −10°.

## Step 1 (B1): does the DINO minimum stay fixed or move? **It moves, with viewpoint and with the scene's appearance.**

| Condition (drill at the scanned pose) | DINO argmin | depth argmin | RGB argmin |
|---|---|---|---|
| 35 training views (the views the features were distilled on) | median −1°, range [−6°, +8°] | +1° [−3°, +3°] | 0° [0°, 0°] |
| 36 held-out views, 3 rings (r 0.87/0.68 m, elev 30/55°, every 30° az), no gripper | smooth in azimuth: +1…+6° for az 0–180°, −1…−5° for az 210–330° | −2…+3° | 0…+1° |
| Tracking camera, clean re-render, no gripper | **−6°** | −1° | 0° |
| Tracking camera, gripper at frame-0 height, shadows on (= the real frame 0) | **−10°** | −1° | 0° |
| Tracking camera, gripper, shadows off | −2° | −1° | 0° |
| Tracking camera, no gripper, shadows off | −6° | −1° | 0° |
| Tracking camera, ROI origin shifted 0–13 px (< 1 DINO patch) | −7° … −10° | −1° | 0° |

Reading, in order of size:
1. **View-dependent DINO bias (about ±5°, systematic in azimuth).** Even on the views it was distilled on, the
   DINO term does not have its minimum at the true pose. The offset varies smoothly with azimuth: positive on one
   side of the drill, negative on the other. The tracking camera sits at az ≈ 274°, where held-out views give
   −5°, and its clean render gives −6°. This is what view-independent per-Gaussian features should do with
   view-dependent 2-D features. The splat can store only one feature vector per Gaussian, roughly an average
   over the training views, so the residual at the true pose is view-dependent and has a nonzero gradient in
   roll. This is the "distillation / view inconsistency" branch of the plan.
2. **Scene context, here the gripper's shadow on the drill (−6° → −10°).** Removing shadows with the gripper
   present gives −2°, so the cast shadow, not the gripper's silhouette, pushes the minimum further. The two
   effects do not add: shadows off moves −10° to −2° with the gripper but leaves −6° unchanged without it.
   The DINO roll direction is shallow enough that small appearance changes decide where the minimum lands.
3. **Patch-grid dependence, a minor effect (about 3°).** Sub-patch ROI shifts move the minimum between −7° and
   −10° without removing it. Positional-embedding or grid artefacts may contribute but are not the main cause.
4. **Depth is unbiased everywhere (|argmin| ≤ 3°), and RGB is exact on static renders.** Neither term is
   biased in roll on a static scene. Depth is too flat to win at weight 3.7.

### Term weights (single frame; trajectory tests below)
| Combined loss, argmin (median over views) | ×0 | ×3.7 (paper) | ×10 | ×30 | ×100 | ×1000 |
|---|---|---|---|---|---|---|
| DINO + w·depth, tracking camera | −10° | −10° | −8° | −6° | −3° | −2° |
| DINO + w·depth, training views | −1° [−6, +8] | −1° [−6, +8] | −1° [−5, +7] | −1° [−5, +6] | 0° [−4, +6] | +1° [−2, +3] |

DINO + 1·RGB gives −1° on the tracking frame, but RGB fails along the trajectory (next section).

## Why RGB failed in E2b, frame by frame (O4 per-frame roll, 100 cases)
| Frame (phase) | 0 (approach) | 1 | 2 | 3 | 4 (lift) | 6 | 9 | 12 (rest) |
|---|---|---|---|---|---|---|---|---|
| mean abs roll, POGS (deg) | 7.0 | 10.3 | 15.2 | 12.5 | 6.8 | 7.8 | 13.5 | 9.5 |
| mean abs roll, POGS + RGB (deg) | **0.3** | **0.3** | 3.7 | 6.2 | 6.7 | 12.3 | 17.9 | 14.2 |

RGB removes the bias while the scene matches the scan (frames 0–1). Roll error then grows as the gripper's shadow
reaches the drill and as the drill moves under fixed lights, because the shading baked into the splat no longer
matches. This is the photometric failure mode of RSRD (POGS's tracker lineage). POGS's own roll also peaks when
the gripper is closest (frame 2, 15.2°), consistent with the shadow effect in B1.

## Consequences
- The bias has two parts, both appearance changes that a view-independent distilled feature field cannot
  represent: (i) view-dependence of DINO features, about ±5° depending on azimuth; (ii) illumination and
  shadow changes, −4° here. Neither is weak observability, and neither is fixable by a filter (direction A
  stays dropped).
- Cheapest fix to test next (E2c, running): heavier depth weights (×30, ×100) along the trajectory. Depth is
  unbiased and does not depend on lighting, but flat; the single-frame ladder predicts −6° / −3°.
- Step 2 (re-distill features only, on frozen geometry) now has a sharper prediction. A backbone with more
  view-consistent dense features (DINOv2-reg; DINOv3, which is gated on HF, access pending) should shrink the
  ±5° azimuth pattern on held-out views. An upsampler mainly addresses (3), the minor grid term. The B1 sweeps
  (training + held-out + tracking views) are the acceptance test, before any tracking run.

Caveats: one splat; synthetic renderer with one fixed lighting; 1-D sweeps about the group centroid (translation
not re-optimised per roll angle); the tracking-camera rows are near-identical across seeds because frame 0
is the same scene in every case.
