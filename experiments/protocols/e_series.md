# Protocol E-series (plan rev. 2, sec. 5–6): pre-registration for E3, E4, E5, E6, E9

Status: **DRAFT 2026-11-23**. Becomes FROZEN at the git tag `e-series-frozen` (LEDGER 8.T7), which must
precede the first E3 test-seed run. Any change after the tag is a logged deviation.

Written before any E3–E9 data exist. Known inputs at writing time: O4 schema-1 results (pre-registered
rule, post-hoc relative rule), E2 smoke on 1 seed (`docs/results/e2_smoke_1seed_2026-11-22.md`).
E0 / E2 full results are NOT known when this draft is written; if they change a rule below, the change
is made before the tag and recorded in §8.

## 1. Data and splits
- Scene: `tool_table` (drill + 5 objects), O4 in-grasp trials (`tool_perturbation.generate_grasp_sequence`),
  settings ±15°, ±30°.
- **Splat seeds:** ≥ 3 independently trained checkpoints (T7). Until they exist, E3 runs are labelled
  "single splat" and no between-method claim is made from them.
- **Calibration seeds:** trial seeds 40–49 (all settings). Used only to fit (a) the covariance scale c of
  `InfoKalmanPrior`, by mean NEES = 6 over rest_x frames, and (b) the weak-axis temperature τ (E5), by NLL.
- **Test seeds:** 0–39. Never used for fitting anything.

## 2. Methods (same trials, same splats)
| ID | Method | Notes |
|---|---|---|
| S | Static (no motion) | null model, every table |
| P | POGS (paper: DINO + depth) | v1 |
| KF | POGS + KF (`InfoKalmanPrior(mode="fixed")` ≡ v2 "POGS + KF") | reference filter |
| TR | `mode="trace"` | isotropic R from tr Σ |
| FU | `mode="full"` | anisotropic R (the claim) |
| OR | `mode="oracle"` | GT-based upper bound; ablation only |
| MS | equal-compute multi-start | same number of extra renders as FU's Hessian (24 gradients/frame) spent on restarts |
Secondary (reported, not part of P2): FU with fusion (`fuse=True`).

## 3. Primary outcome and rules (reported in this order)
1. **Pre-registered O4 rule** (o4.md): tip < 3 cm and geodesic rotation < 5° at N = 10, never lost.
2. **Paper-faithful rule** (deviation 2026-11-22, POGS Sec. V-B): tip < 3 cm and bit-axis swing < 5°, never lost.
3. RRTrack-style event metric (`metrics.calibration.event_recovered`, window 10) with the paper-faithful
   per-frame criterion; Kaplan–Meier time to recover, censored at 10 frames.
Every rule is reported for S as well; a rule is admissible only if S scores ≤ 2/50 per setting under it.

## 4. Predictions and pass rules
- **P1 (E2).** (a) Spearman(log `roll_info_rel`, |roll error|) ≤ −0.5, as in plan rev. 2 — threshold
  unchanged although the smoke gave +0.47 (no loosening after seeing data); also reported per trial
  (mean rho, bootstrap CI over trials). (b) The "weakest eigenvector within 15° of the bit axis in ≥ 70%
  of frames" criterion is ill-posed when the two smallest rotation eigenvalues nearly tie (seen in the
  smoke). It is kept and reported, and complemented by median `axis_in_weak2` ≥ 0.9, a definitional fix
  for the tie, not a relaxation.
- **P2 (E3).** On test seeds, paper-faithful-rule success: FU − TR > 0 and FU − KF > 0 with 95%
  hierarchical-bootstrap intervals (splat seed → trial) excluding 0, in at least 2 of the 4 settings;
  S ≤ 2/50. If FU ≈ TR (interval includes 0) the anisotropy claim fails, as in the compact pose graph's
  Table 6, and is reported as such.
- **P3 (E4/E5).** Calibration on test seeds: 90% coverage of FU's rotation block in [0.85, 0.95] on
  observable directions; mean NEES inside the χ² acceptance interval; weak-axis belief NLL below the
  single-Gaussian NLL with interval excluding 0.
- **P4 (E6).** Information-gated iterations match fixed-10 success (difference interval includes 0 or is
  positive) at a higher measured rate; timing on an idle GPU only.

## 5. Statistics
- Rates: Wilson 95% CI for description; method differences by paired hierarchical bootstrap
  (`metrics.calibration.paired_difference_bootstrap`, 2000 resamples, levels: splat seed, trial).
- No significance claims from the post-hoc relative rule. Multiple settings: report all four; P2 needs 2 of 4.

## 6. Fixed analysis choices
- Hessian: `loss_hessian_regression`, scale 1°/3 mm, n = 12 antithetic pairs, radius 0.10 m for scaling.
- `information_to_cov` rel_lam = 1e-3; `r_floor` = (1e-6 rad², 1e-8 m²).
- Frames used for calibration fitting: rest_x only (before any perturbation).

## 7. Stop rules (plan rev. 2, sec. 8)
- If E1/E2 show the error is an optimisation failure (loss(est) > loss(GT) on most frames), E3 is not run;
  write the methodology note. (Smoke: the opposite, loss(est) < loss(GT) on 95–100% of frames.)
- If P1 fails on the full E2, E3 still runs (it tests R, not roll), but the paper's roll story is dropped.

## 8. Changes before freezing
| Date | Change | Reason |
|---|---|---|
| 2026-11-23 | P1(b) complemented by `axis_in_weak2` ≥ 0.9; P1(a) threshold kept at −0.5 | E2 smoke: near-tied weak eigenvalues make the single-eigenvector test ill-defined; the correlation threshold is deliberately not loosened after a failing smoke |
