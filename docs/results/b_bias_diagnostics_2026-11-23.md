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

## Step 2 (B2): re-distil features only, geometry frozen (`experiments/b2_redistil.py`, 3000 steps, release LRs)
Acceptance rule (set before the runs): a backbone passes only if its DINO roll argmin shrinks **both** on held-out
views and on the tracking camera relative to the v2 control (same loop, release backbone).

| Features | train views mean abs argmin (sd) | held-out mean abs argmin [range] | tracking f0 (gripper + shadow) | no gripper | gripper, no shadows | Verdict |
|---|---|---|---|---|---|---|
| release splat (joint training) | 3.1° (3.7) | 3.1° [−5, +6] | −10° | −6° | −2° | reference |
| v2 control (DINOv2-S/14, feature-only loop) | 2.5° (2.9) | 2.9° [−5, +6] | −9° | −5° | −2° | reproduces release → loop is sound |
| v2reg (DINOv2-S/14 + 4 registers) | 4.5° (4.9) | 4.6° [−8, +8] | −6° (−6° with 3.7·depth) | −5° | −2° | **fails**: tracking view better, held-out worse |
| v3s (DINOv3 ViT-S/16, timm) | 10.7° (12.7) | 9.1° [−11, +21] | −12° | −10° | −10° | **fails**: much worse everywhere |

Reading: swapping the backbone does not remove the view-dependent bias. Registers trade a smaller offset at this
camera for a larger spread across views. DINOv3-S at this resolution and with 64-D PCA (its features needed a
×8 PCA rescale to match the release feature energy) is far less pose-consistent for this object. The
illumination component (−2° without shadows vs −6/−10° with them) persists for both DINOv2 variants.
Caveats: one setting per backbone (ViT-S, PCA-64, 1260/1248 px long side), and one splat. The DINOv3 result could
depend on the resolution, PCA dimension or layer, which were not tuned, on purpose: tuning them on these sweeps
would overfit the acceptance test.
Upsampler (AnyUp, trained on DINOv2-S/14): implemented but not run. Loading it executes third-party hub code,
which needs the user's approval. Its expected effect is bounded by the grid component (±3°).

## Shadow step (B1e): restrict the DINO loss to the drill (`experiments/b_bias.py masks`, `data_results/b_bias/masks.json`)
The release DINO loss averages ‖live − rendered‖ over the whole ROI. Outside the drill's outline the rendered
feature is zero (alpha < 0.8), so those pixels contribute ‖live background feature‖. **87% of ROI pixels are
outside the drill** (0.85–0.88 across views).

| Condition (clean re-render, drill at the true pose) | release: whole ROI | inside drill rendered at each candidate pose | inside drill at the true pose (fixed) |
|---|---|---|---|
| tracking camera, gripper + shadow (= real frame 0) | **−10°** | +2° | **0°** |
| tracking camera, gripper, no shadows | −2° | +4° | 0° |
| tracking camera, no gripper | −6° | +3° | 0° |
| tracking camera, no gripper, no shadows | −6° | +2° | 0° |
| held-out ring r 0.87 m, elev 30°, 12 azimuths: mean abs argmin | 3.6° | 2.4° | 0.1° |

Reading: **inside the drill's outline the DINO features point at the true pose in every view, with or without the
shadow.** The bias, both the view-dependent ±5° and the shadow's −4°, comes from the background pixels. There the
release loss acts as a silhouette term weighted by the background feature norm, which shadows and viewpoint
change. A mask re-rendered at each candidate pose leaves a smaller bias of the opposite sign (+2…+4°), because the
pixel set changes with pose. Caveat: the fixed mask here sits at the true pose, which leaks ground truth. In
tracking it must come from the current estimate, so the next test places the mask at wrong poses (B1f) and
checks whether the minimum returns to 0° or sticks to the mask.

## B1f: mask placed at a wrong pose (`experiments/b_bias.py maskoff`, `data_results/b_bias/maskoff.json`)
In tracking the mask must come from the current estimate, not from the truth. For the inside-mask DINO loss with
the mask rendered at roll δ, the argmin over ±30° is:

| View | δ = −10° | −5° | 0° | +5° | +10° | re-mask fixed point from −10° / +10° (6 rounds) |
|---|---|---|---|---|---|---|
| tracking cam, gripper + shadow | −10 | −5 | 0 | +4 | +3 | −10 (stuck) / +3 |
| tracking cam, no gripper | −9 | −5 | 0 | +4 | +4 | −6 / +4 |
| held-out az 60° | −9 | −4 | 0 | +5 | +9 | −4 / +9 |
| held-out az 150° | −2 | −4 | 0 | +5 | +9 | −2 / +9 |
| held-out az 240° | −9 | −5 | 0 | +3 | +3 | −8 / 0 |
| held-out az 330° | −9 | −5 | 0 | +5 | +10 | −9 / +10 (stuck) |

**A mask taken from a wrong pose holds the estimate where it is.** The argmin follows the mask in most cases,
and re-masking moves back at most about 1° per round. The inside-drill features are unbiased at the truth but
give almost no restoring force within ±10°. That force came from the background pixels, where the release loss
acts as a silhouette-alignment term (zero render vs live features). The same term carries the shadow and
viewpoint bias.

Consequences:
1. The B1e result (0° with a fixed true-pose mask) is partly a ground-truth leak and is **not a fix by itself**.
   `dino_loss_region="object_frame_start"` is predicted to stick at each frame's start pose; this gets a small
   O4 check only.
2. `dino_loss_region="object"` (mask re-rendered at each evaluated pose, no ground truth) has a real minimum
   at +2…+4° (B1e), against −10° for the release. It runs on O4 next (`maskobj`), judged by the E2c pass rule.
3. Mechanism, as now stated for the note: **distilled-DINO tracking gets its roll restoring force from a
   silhouette term implicit in the zero-rendered background, and that term is biased by shadows and viewpoint.**
   A principled fix keeps a silhouette signal that is robust to shading, for example a rendered-vs-live mask
   term instead of zero-vs-background features. That would be the next experiment if `maskobj` fails.

## B1g: compare background pixels with the rendered static scene, not with zero (`experiments/b_bias.py bgrender`)
Whole-ROI DINO term with the full splat rendered (static background at its scanned pose, drill at the evaluated
pose). No mask, so no ground-truth leak and nothing for the estimate to stick to. Argmin over ±30°:

| Condition | release (background = 0) | background rendered |
|---|---|---|
| tracking cam, gripper + shadow (= real frame 0) | −10° | **+2°** |
| tracking cam, gripper, no shadows | −2° | +4° |
| tracking cam, no gripper | −6° | +3° |
| tracking cam, no gripper, no shadows | −6° | +4° |
| held-out ring (12 az): mean abs argmin / signed mean | 3.7° / +0.5° | **1.3° / +1.0°** |

The shadow no longer pushes the minimum: +2° with it, +4° without. The view-dependent ±5° pattern collapses to
≤ 4°. A small positive offset (+2…+4°) remains at the tracking camera, about the same as the per-pose mask. The
loss is shallower (range over ±30° 0.016 vs 0.027), so the restoring force must be checked on trajectories.
Implemented as `TrackingConfig.dino_background` (`"zero"` = release default, `"rendered"`; one extra render per
iteration); suite 57 passed, 1 skipped. One-case smoke (`grasp_+15_s0`): roll before the perturbation −8.6° → +3.6°,
but post-perturbation recovery failed (tip 9.6 cm, swing 14°; POGS passed this case under the paper's rule). That
is n = 1, so it is not evidence either way. The full O4 run (`bgdino`) is queued after E2c ×100, then the
depth-only control and `maskobj`.

## E2c result (depth weight on trajectories, O4 seeds 0–24, `data_results/e2c_depth{30,100}/summary.md`)
| Rule | Setting | Static | POGS (3.7) | depth ×30 | depth ×100 |
|---|---|---|---|---|---|
| pre-registered | −30 / −15 / +15 / +30° | 0/0/0/0 | 3/7/1/6 | 1/0/0/0 | 0/0/0/0 |
| paper-faithful | −30 / −15 / +15 / +30° | 0/0/0/0 | 9/12/16/23 | 2/1/0/0 | 0/0/0/0 |

Abs roll before the perturbation: POGS 9.3–9.5°, ×30 4.0–4.3°, ×100 5.1–5.2°. Swing before the perturbation
≈ 29° (×30) and 32° (×100), against 3.3° for POGS; it starts at frame 2, when the gripper reaches the drill.
**Both weights fail the pass rule** (paired paper-faithful differences −0.28…−0.92, all CIs < 0). A heavier depth
term does not fix the bias; it trades it for an occlusion failure.

## E2d-1: rendered background on trajectories (`bgdino`, O4 seeds 0–24, `data_results/e2d_bgdino/summary.md`)
**Fails the pass rule.** Pre-registered 0/0/0/0 of 25 (POGS 3/7/1/6); paper-faithful 0/0/0/0 (POGS 9/12/16/23);
Static 0 everywhere; paired paper-faithful differences −0.36…−0.92, all CIs < 0. Abs roll before the perturbation
7.3–7.5° (POGS 9.3–9.5°).

Median per-frame error, bgdino vs POGS:

| Frames (phase) | roll | swing | tip | position |
|---|---|---|---|---|
| 0–2 (approach, drill on the table) | **2.8 / 3.8 / 0.6° vs 7.0 / 10.3 / 15.1°** | 4.2–8.2° vs 1.8–3.2° | 3.0–3.9 vs 0.7–1.5 cm | 1.9–2.5 vs 0.4–1.2 cm |
| 4–6 (lift) | 13.4–17.9° vs 5.3–7.0° | 5.8–7.5° vs 2.5–6.6° | 4.2–5.9 vs 1.5–1.9 cm | 3.5–4.0 vs 1.3–1.6 cm |
| 10–12 (rest, held) | 7.0–7.6° vs 9.4–12.1° | 4.8–5.9° vs 3.0–4.7° | 6.6–7.4 vs 0.9–1.0 cm | 5.9–6.4 vs 1.3–1.5 cm |
| 13–19 (after the perturbation) | 5.7–6.6° vs 7.0–10.1° | 11.7–21.1° vs 3.3–17.1° | 8.5–10.6 vs 2.2–6.0 cm | 6.5–6.6 vs 1.6–2.0 cm |

Reading: **the roll bias is removed while the drill sits where it was scanned** (frame 2: 0.6° vs 15.1°), as B1g
predicted. But translation is worse from frame 0, and roll and position degrade once the drill leaves its
scanned place. A static background model is stale after the object moves: the table under the drill was never
observed, and the drill's own shadow is baked into the table. The rendered background then pulls the estimate.
A ghost check rules out drill fragments left in the background: 537 opaque non-group Gaussians lie inside the
drill's box, but at z ≈ 0.05 m, i.e. the table surface around its base. The 1.9 cm translation error at frame 0,
when nothing is stale yet, is not explained; candidate: the live gripper, absent from the splat, attracts the
drill when the background is rendered.

**Pattern across all four fixes:** each simple substitute for the implicit silhouette term breaks a different
assumption. Heavier depth fails under gripper occlusion. RGB fails under shading change (E2b). An inside-object
mask loses the restoring force (B1f). A rendered background fails once the object leaves its scanned place.

## Depth-only control (`use_dino=False`, O4 seeds 0–24, `data_results/e2c_depthonly/summary.md`)
Pre-registered 0/0/0/0 of 25, paper-faithful 0/0/0/0 (POGS 3/7/1/6 and 9/12/16/23; Static 0). Abs roll before
the perturbation 10.7–10.9°, swing before 25.2°, swing at N 35.6°, tip at N 10.8 cm. This agrees with the POGS
ablation: without DINO, no resets succeed.

What DINO contributes under the paper-faithful rule: 9–23 of 25 recoveries per setting. Depth ×100 (0/25) and
the rendered background (0/25) match the depth-only control; depth ×30 (2/1/0/0) is barely above it. So
**none of the fixes tested so far keeps what the DINO term contributes while removing its bias.**
