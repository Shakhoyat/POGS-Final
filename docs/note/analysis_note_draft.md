# Analysis note (MAIN RESULT since 2026-11-23: direction B found the cause but no fix; draft, direction C): what the POGS in-grasp recovery number measures

Status: living draft, 2026-11-23. Simulation only. One splat (≥ 3 splats before submission, T7). The numbers
come from the result files named in each section; nothing here is new data.

## Framing (user review 2026-11-23)
This is a mechanism finding about feature-field tracking in general, not a failed reproduction. **The DINO
loss of distilled-feature trackers gets much of its pose restoring force from an implicit silhouette term (live
background vs a zero render), and that term is what lighting, shadows and viewpoint bias.** Evidence chain:
ground truth exact (B0) → object features unbiased (B1e) → the background does the work (masking it collapses
recovery: 1/0/4/4 vs 9/12/16/23) → the DINO term matters (depth only: 0 everywhere).

**Final G1/G1b/G2 results 2026-11-24 (all pre-registered; `data_results/g1_generality/summary.md`,
`data_results/g2_rsrd/summary.md`).** P2 pass (drill roll bias −10/−10/−5° on three splats), P4 pass (monotone
shadow dose-response on 3/3 splats, ρ −0.97/−0.85/−0.85, shift −8/−9/−5°), P5 pass (rendered background moves
0.5–1.5°); P1 fail (2/4), P3 fail (2/4), G1b T1 fail (object features carry 0.75–1.00 of the translational
stiffness), G2 Q1/Q2 fail (0/4), G2 Q3 pass (RSRD objective −10° with the gripper shadow). **Thesis now: the implicit
silhouette term is the carrier of a shadow-driven pose bias, replicated across splats and in RSRD's objective; it does
not supply the static restoring force, and why tracking fails without it is open (dynamic).** Paper draft retitled
"Shadows Bias Feature-Field Object Trackers Through an Implicit Silhouette Term".

**Correction 2026-11-24 (G1, pre-registered; seed-0 splat).** The static sweeps do not support "the background
supplies most of the restoring force" for rotation. P1 fails on seed 0 (median background share of the rotational
curvature > 0.5 for 2 of 4 objects; drill 0.10) and P3 fails (2 of 4). Exploratory: the object's own features give
0.60–0.89 of the local rotational curvature, and the rotational basins are ±20° wide whatever the background
treatment. What survives, and is stronger than before, is the **bias**: the roll minimum follows the cast-shadow
strength (seed 0: −2.5° at no shadow → −10.5° at default, Spearman −0.97), while the same loss against a rendered
background moves by 1.5°. The trajectory collapse of the masked variants (1/0/4/4) is a translation drift in the
lift; whether the background carries the translational stiffness is tested by G1b (pre-registered). Until then the
claim is: **the implicit silhouette term biases the minimum with shadows and viewpoint; tracking without it fails in
trajectories; the static mechanism for that failure is open.**

## Thesis in three sentences
1. The POGS drill result (48/50 at ≤ 15°, 40/50 at 30°) uses a rule that leaves roll about the bit axis free
   (Sec. V-B). Under that rule our re-implementation matches the 30° tier in one direction (+30°: 39/50 vs
   40/50, Fisher p = 1.0) and falls short in the other three settings (17–32/50, p ≤ 1e-4). Under the full
   geodesic rule we pre-registered, it recovers 2–14/50.
2. What the rule hides is not noise. It is a **systematic roll bias of 9.5 ± 2.8° about the bit axis**,
   present before the gripper touches the drill. It comes from the DINO feature term, whose minimum at the true
   pose depends on viewpoint (±5°) and on illumination and shadow (−6° → −10° when the gripper's shadow falls on
   the drill).
3. Recovery metrics must be checked against a no-motion baseline. A relative "back to its pre-perturbation
   error" rule passes Static in 7–9 of 21–29 trials when the bias happens to align with the perturbation, and in
   0 when it opposes it. The ± asymmetry is a signed-bias artefact, not a property of the perturbation.

## 1. Rule mismatch (`docs/results/g0_verdict_2026-11-23.md` §§1–3, 6)
| Setting | Pre-registered (tip < 3 cm, geodesic < 5°) | Paper-faithful (roll free) | Paper | Static (both rules) |
|---|---|---|---|---|
| +15° | 2/50 | 32/50 [0.50, 0.76] | 48/50 (Tier 1) | 0/50 |
| −15° | 14/50 | 25/50 [0.37, 0.63] | 48/50 (Tier 1) | 0/50 |
| +30° | 13/50 | 39/50 [0.65, 0.87] | 40/50 (Tier 2) | 0/50 |
| −30° | 5/50 | 17/50 [0.22, 0.48] | 40/50 (Tier 2) | 0/50 |

## 2. Anatomy of the error (`docs/results/e1_floor_summary_2026-11-23.md`)
- Pre-perturbation error is 94% roll about the bit axis (|roll| 9.5 ± 2.8°, swing 4.0 ± 3.4°, tip 1.3 cm).
- It is already 12.5 ± 0.2° at the end of the approach phase (drill on the table, gripper not in contact).
- Signed pre-perturbation yaw error is positive in 167/200 trials. A + perturbation moves the truth toward
  the biased estimate, which explains + > −.
- A filter (POGS + KF + check) does not change it (28 vs 32, 35 vs 39, 24 vs 25, 18 vs 17).

## 3. The loss prefers the wrong pose (`docs/results/e2_loss_bias_2026-11-23.md`)
- loss(estimate) < loss(ground truth) on 87–100% of frames, so this is not an optimisation failure.
- At the true pose the DINO term's minimum is at −10° roll. Depth's is at about 0° but flat (range 0.0012
  over ±30°), and RGB's is at 0°.
- Hessian-based roll information does not predict the error (Spearman +0.22). Weak observability is
  necessary, but the cause is a bias.

## 4. Where the bias comes from (`docs/results/b_bias_diagnostics_2026-11-23.md`)
- Ground truth is exact (B0): the RGB-only argmin is at 0.0° in all rotations, ≤ 0.5 mm in translations.
- View-dependence (B1): DINO roll argmin −1° [−6, +8] over the 35 distillation views. On held-out views it
  varies smoothly with azimuth (±5°), and the tracking camera's azimuth gives −5° / −6°.
- Illumination (B1d): the gripper's cast shadow moves it to −10°; with shadows off and the gripper present it
  is −2°.
- Patch grid: ±3°, a minor effect.
- RGB fixes static frames (0.3° roll on frames 0–1) but fails once shading changes (18° by frame 9; E2b),
  which is the RSRD photometric failure mode.

## 5. Fixes tested (direction B; filled in as results arrive)
| Fix | Single-frame argmin (tracking cam) | O4 trajectories (paper-faithful rule, paired vs POGS) | Status |
|---|---|---|---|
| RGB term, weight 1 (E2b) | −1° | worse: roll 14.3° vs 9.4°; −30° +0.28, +30° −0.20 | negative |
| depth ×30 (E2c) | −6° | fails: 0–2/25 (POGS 9–23/25); roll pre 4.0° but swing 29° once the gripper occludes | negative |
| depth ×100 (E2c) | −3° | fails: 0/25 in all settings; roll pre 5.1°, swing 32° | negative |
| features re-distilled, DINOv2 control (B2) | −9° (release −10°) | not run (control) | reproduces release |
| DINOv2-reg (B2) | −6° | not run: held-out views worse (4.6° vs 2.9°) | rejected |
| DINOv3 ViT-S/16 (B2) | −12° | not run: held-out 9.1°, up to +21° | rejected |
| AnyUp upsampler on DINOv2 (B2, pinned 351807a, 2×) | −15° (control −9°); held-out 4.7° vs 2.9° | not run: fails acceptance | rejected |
| DINO loss inside the drill, mask fixed at the true pose (B1e) | 0° (leaks GT) | not a method | diagnostic |
| same, mask at a wrong pose (B1f) | follows the mask (no restoring force) | see frame-start row | diagnostic |
| DINO loss inside the drill re-rendered per pose (`object`) | +2° (clean frames) | fails: 1/0/4/4 (POGS 9/12/16/23); approach roll unchanged, loses the drill in the lift (swing 20°, tip 10 cm) | negative |
| frame-start mask (`object_frame_start`, 5 seeds) | follows the mask (B1f) | fails: 1/0/2/2 of 5 (POGS 1/2/3/5); loses the drill in the lift | negative |
| DINO vs rendered background (`dino_background="rendered"`, B1g) | +2° (held-out mean abs 1.3°; shadow-insensitive) | fails: 0/25 everywhere; roll fixed on the table (frame 2: 0.6° vs 15.1°) but translation +1.5 cm from frame 0 and roll 13–18° in the lift: the static background is stale once the drill moves | negative |

Depth-only control (DINO off): 0/25 in every setting, both rules (roll pre 10.8°, swing 25°). DINO is worth 9–23
of 25 recoveries under the paper-faithful rule. Depth ×100 and the rendered background are no better than
switching DINO off.

Note on the frame-start mask: the 5-seed check (1/0/2/2 vs 1/2/3/5 of 5) is too small to carry weight. It is
*consistent with* the B1f prediction (the estimate sticks), not a confirmation.

**The ± asymmetry is not explained by the mechanism alone.** Share of positive signed pre-perturbation yaw error
(100 cases each): POGS 83% (median +1.9°), RGB 79%, depth only 72% (+17°), depth ×30 77%, rendered background
**20% (−4.8°)**, DINO inside the outline 100% (+21.7°). The background treatment controls the sign of the yaw
bias, but a positive bias survives with DINO switched off. The asymmetry has more than one contributor and is
left open. Caveat: depth-only and masked runs lose the drill, so their medians partly reflect tracking failure.

## 5b. Mechanism (B1e/B1f)
87% of the ROI is background. There the release loss compares live features with a zero render, which acts as an
implicit silhouette term. (Superseded 2026-11-24 for rotation, see the correction at the top: the term biases the roll
minimum; it does not supply most of the local rotational stiffness.) The cast shadow and viewpoint bias it
(−10° at the tracking camera). Inside the drill the features are unbiased at the truth but nearly flat within
±10°.

## 6. Recommendations for tool-tracking evaluation (to finalise)
1. Report the tool-axis rule and the full-rotation rule side by side, and say which DOF a rule leaves free.
2. Score every recovery metric for Static. Do not use relative-to-pre-perturbation rules, or stratify them by
   the sign of the pre-existing error.
3. Store full SE(3) per frame and decompose the error into roll and swing about the tool axis.
4. Sweep the loss at the true pose, term by term, from several viewpoints and lighting conditions before
   attributing an error to observability.

Threats to validity: one splat and one synthetic lighting setup; real cameras add exposure changes, which the
illumination finding suggests would make the bias worse, not better; the drill model is procedural.
