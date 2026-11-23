# Pre-registration: explicit depth-silhouette term (D-SIL), 2026-11-23

**Gate: runs only if the supervisor answers yes to question 2** (may tracking use a known gripper model to
exclude its pixels?). Written before any code or run. Nothing below may change after the first test-seed run.

## Hypothesis
The roll restoring force of POGS comes from an implicit silhouette term (zero-rendered background,
`docs/results/b_bias_diagnostics_2026-11-23.md` §B1e–B1g). That term is biased by shadows and viewpoint. An
**explicit, shading-free silhouette term** keeps the restoring force without the bias.

## Method (ground-truth-free)
- Live foreground F (per ROI): pixels where live depth < rendered static-background depth − τ, τ = 1 cm (the
  static background is the scene splat without tracked groups).
- Gripper exclusion G: pixels covered by the gripper rendered from its **commanded** pose and its known mesh,
  i.e. robot kinematics. In simulation: render the gripper primitive at the commanded pose. No renderer object
  IDs are used.
- Silhouette term: 1 − soft IoU between the rendered object accumulation A and F, over ROI pixels not in G;
  weight w.
- **Primary arm "D-SIL"** = DINO term inside the rendered object (`dino_loss_region="object"`) + depth term +
  w · silhouette term, all with G excluded. This replaces the implicit silhouette term with the explicit one,
  which is the test of the mechanism claim.
- **Fair reference "POGS + G"** = release POGS with the same gripper exclusion G applied to its DINO and depth
  terms (so any gain is not just from masking the gripper).
- Exploratory arm, labelled as such and not counted for the verdict: release DINO + depth + w · silhouette, with G.

## Calibration (calibration split, never test seeds)
- w ∈ {0.3, 1, 3}, chosen on O4 seeds 40–49 (4 settings, 40 cases per w) by paper-faithful success, ties →
  smaller w.
- τ is fixed at 1 cm, not tuned.

## Test and pass rule (the LEDGER 8.E2c rule, unchanged)
On O4 seeds 0–24 (100 cases), D-SIL vs **POGS + G** on the same cases. Pre-registered rule first, Static in
every table. Pass = paired paper-faithful bootstrap CI > 0 in ≥ 2 of 4 settings and < 0 in none, AND lower mean
abs roll before the perturbation, AND beats the depth-only control in ≥ 2 settings. Also reported, not part of
the verdict: POGS + G vs POGS, i.e. the effect of the gripper exclusion alone.

## Budget
Implementation ≤ 1 working day. GPU cap 6 h wall clock in total (calibration about 1.5 h, two test arms about
2.5 h, exploratory arm about 1.3 h at 2 shards). If the cap is hit, the result is reported as incomplete; no
extension.
