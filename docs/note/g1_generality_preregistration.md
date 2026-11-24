# G1 pre-registration: does the background-silhouette mechanism generalise? (static loss sweeps only)

Written 2026-11-24, before any G1 number exists. Nothing here may be changed after the first G1 result is read;
anything added later is labelled exploratory. Evaluation-only diagnostics: ground truth places the sweep (object
at its scanned pose) and names the group; the tracker never sees it. No new trajectories. Not gated on the
supervisor (it does not use a gripper model). D-SIL stays gated on Q2.

## Question
B1e/B1g showed on one splat and one object (the drill) that the release DINO term (ROI mean of
`||live − render||`, render = 0 off the object) gets its pose restoring force from the background pixels, and that
shadows bias that force. G1 asks whether this holds (a) on independently trained splats of the same scene and
(b) for other objects.

## Material (fixed)
- Splats: `tool_table` seed 0 (existing), `tool_table` seeds 1 and 2 (new; same data, same config, only
  `TrainConfig.seed` changes; written to `model_checkpoints/tool_table_seed{1,2}`, the seed-0 checkpoint is not touched).
- Objects (groups of each splat): power drill, ceramic mug, mustard bottle, cereal box. Soup can and apple are
  near-symmetric and are reported as exploratory only.
- Views: the benchmark tracking camera (no gripper) with shadows on and off, and 12 held-out views on a ring
  around each object (r = 0.87 m, elevation 30°, azimuth 0..330° step 30°, shadows on). Drill only: tracking
  camera with the frame-0 gripper, shadows on.
- Axes: rotations about world x, y, z through the group centre (the group frame is world-aligned at the scan
  pose); for the drill, additionally roll about the bit axis. Grid ±20° at 1°.

## Quantities (per splat × object × view × axis)
- Pixel-sum decomposition of the release term: with N ROI pixels and the object mask M(θ) rendered at the
  evaluated pose (alpha ≥ `dino_alpha_thresh`), L(θ) = S_in(θ) + S_out(θ), S_in = (1/N) Σ_{M(θ)} e,
  S_out = (1/N) Σ_{¬M(θ)} e. Exact by construction (checked numerically, |L − S_in − S_out| < 1e-5).
- Curvature κ of each of L, S_in, S_out: coefficient of θ² of a least-squares quadratic over |θ| ≤ 5°.
- Background share b = κ_out / κ_L (only where κ_L > 0).
- Argmins over ±20° of: L (release), S_in/|M| (inside-object mean, `dino_loss_region="object"`), the
  rendered-background term (`dino_background="rendered"`), and the depth term.

## Predictions and pass rules (decided now)
- **P1 (mechanism generalises).** Median b over (views × axes) > 0.5 for at least 3 of the 4 objects on the seed-0
  splat, and for the drill on both new splats. Fails if it holds for 2 or fewer objects; then the paper says the
  mechanism is shape-dependent and reports for which objects it holds.
- **P2 (drill bias replicates across splats).** Roll argmin of L at the tracking camera with the frame-0 gripper is
  ≤ −5° on each of the two new splats (seed 0: −10°). Fails otherwise; the paper then reports the bias as
  splat-specific.
- **P3 (rendered background removes most of the bias, generality).** Over the held-out views, the mean absolute
  argmin of the rendered-background term is lower than that of L for at least 3 of the 4 objects (on the axis
  with the largest κ_L for that object; roll for the drill). Seed-0 splat.

All three are reported whatever the outcome, in this order, before any exploratory analysis.

## Dose-response (G1-S, same rules, written now)
Shadow strength s scales the cast-shadow darkening of the key light: shadow factor = 1 − s · (1 − 0.22), so
s = 1 is the renderer default (factor 0.22) and s = 0 means no cast shadow. Levels s ∈ {0, 0.25, 0.5, 0.75, 1.0,
1.25} (1.25 gives factor 0.025). Drill, tracking camera with the frame-0 gripper, roll grid ±20° at 0.5°, all three
drill splats. Only the tracking frame is re-rendered; the splats were trained on s = 1 captures.
- **P4 (dose-response).** Spearman ρ between s and the roll argmin of L is ≤ −0.8 on at least 2 of 3 splats, and
  the argmin moves by at least 4° between s = 0 and s = 1 on at least 2 of 3 splats.
- **P5.** On the same frames, the rendered-background argmin moves by less than half the release movement
  (s = 0 → 1) on at least 2 of 3 splats.

## Budget
Cap: 4 GPU hours in total (2 trainings ≈ 25 min, sweeps estimated < 1 h). One GPU process at a time (well under
the 3-process limit). No GPU job while a latency stage runs. If the cap is reached, stop and report what was done.

## Outputs
`data_results/g1_generality/{sweeps_seed*.json, shadow_dose.json, summary.md}`, runner `experiments/g1_generality.py`,
ledger row 8.G1.
