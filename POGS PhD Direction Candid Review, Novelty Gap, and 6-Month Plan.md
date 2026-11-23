# POGS PhD Direction: Revised Research Plan (rev. 2)

Rev. 2, 2026-11-22. Supersedes rev. 1 of the same day (kept verbatim in
`docs/literature/plan_v1_2026-11-22_superseded.md`). This revision follows a full-text reading of the 12 papers in
`ref-paper/` (renamed `YEAR_VENUE_FirstAuthor_topic.pdf`; per-paper notes with section and table references in
`docs/literature/reading_notes.md`). Execution is tracked in `docs/LEDGER.md`, Phase 8.

Tags: **[V]** checked in a paper's full text or in this repository (the source is named). **[I]** my inference.
**[U]** unverified. Numbers quoted from papers are the authors' own, measured under their own protocols and on their own
hardware. They show what reviewers will compare against. They are not a leaderboard.

---

## 0. What changed after reading the papers in full

| # | Rev. 1 claim | What the full text says | Consequence |
|---|---|---|---|
| R1 | Our 5° rule is the paper's rule, so the 10–24° rotation floor means the replication fails | POGS Sec. V-B: *"We leave tool rotation about its target axis (i.e. drill bit axis) unconstrained"*. Its 3 cm / 5° success rule therefore **does not score roll about the bit axis**. Our pre-registered O4 rule uses full geodesic rotation, which does. [V: POGS Sec. V-B] | Our O4 headline compares two different quantities. The fix is a correction to match the paper, not a post-hoc metric: re-score with tip position plus bit-axis direction (roll excluded). This is experiment **E0**. It needs full SE(3) per frame, which O4 does not store yet (`per_frame.json` keeps errors only) [V: repo] |
| R2 | 48/50 is "up to 15°", 40/50 is "30°" | Table II has Tier 1 CW 24/25 and CCW 24/25 (48/50, mean 6.01 s); Tier 2 CW 20/25 and CCW 20/25 (40/50, mean 12.66 s). Tier 1 keeps the target on the table and changes the in-grasp angle "up to 15 degrees". Tier 2 moves the target in 3D. Methods: "15–30° clockwise or counter clockwise about the grasp axis". [V] | Report it as "Tier 1 / Tier 2" and never as "15° / 30°". The 80% is closed-loop servoing on hardware, with no frame budget and no baselines |
| R3 | UGS-Loc: "Fisher information in 3DGS refinement", so "first Fisher in 3DGS" cannot be claimed | UGS-Loc's Fisher information is over the **splat's scene parameters** (FisherRF-style depth uncertainty), and it biases RANSAC sampling. It never computes a pose information matrix or a pose covariance. [V: UGS-Loc Sec. 3.3] | The pose-information claim survives. Cite UGS-Loc and FisherRF. Add a cheap component: a per-Gaussian reliability weight in the tracking residual |
| R4 | Hessian-covariance object tracking (compact pose graph) is close prior art | Confirmed, and sharper than rev. 1 said. It already ran an **Identity / Trace / Covariance / Oracle** ladder. On 2 of 4 benchmarks the full anisotropic covariance brings **no consistent benefit over trace**, because "it only helps when its estimated directions are correct". It reports a 37.66° error with 1.22° predicted (Fig. 5), and says its local covariance "cannot capture a well-separated alternative solution". [V: Table 6, Fig. 5, Sec. 6] | P2 must be pre-registered as *anisotropic beats trace on the near-symmetric subset*, not in general. The weak-axis multimodal belief is the part that is new |
| R5 | FoundationPose tracks at about 32 Hz | The paper gives about 32 Hz on an RTX 3090 (Sec. 4.5). Other papers measured 42.9 fps (RTX 6000 Ada, RoPT Table 2) and 41.3 fps (RTX 4090, Opra Table II), and 10–33 fps depending on the dataset (H100, compact pose graph Table 9). Its tracking refiner is trained on perturbations of at most 20° (Supp. 5.2). [V] | Quote speed per GPU and per dataset, measured on our own GPU. The 20° training basin explains why it collapses under 8–16× frame downsampling (RoPT Table 3) |
| R6 | Real data is non-negotiable | RRTrack has **synthetic-only quantitative results** (Isaac Sim) plus qualitative real drone footage. [V] | This makes our own synthetic results publishable as *support*. It does not lower the bar for ICRA, CoRL or CVPR, where the closest competitors all report YCBInEOAT, HO3D or YCB-Video. We still plan at least one public real benchmark [I] |
| R7 | — | Every 2026 tracker is weakest on rotationally symmetric objects: Opra's mug (66.09 ADD, Table I), ICG+ on the soup can (79.83 ADD), compact pose graph's failures (Fig. 5), BundleSDF's symmetric re-acquisition failure (App. F). RRTrack and the compact pose graph both list "symmetry-aware / multimodal pose distributions" as future work. [V] | The gap is real and documented, so the scoop risk is real too. Move fast, and claim only the near-symmetric subset |
| R8 | — | X-ICP: eigenvalue thresholds do not transfer across sensors or scenes (threshold 120 vs 200, Fig. 17). It analyses the rotation and translation blocks separately because their scales differ, and keeps the prior along degenerate directions. Its baselines are Zhang et al. 2016 (min-eigenvalue "solution remapping") and Hinduja et al. 2019. [V] | Our observability measure must be scale-free (rotation normalised by object radius; ratios, not raw λ). Solution remapping and a fixed-threshold variant become mandatory baselines |
| R9 | — | The DQ-UKF paper separates **local weak observability** (the rank of the information) from **global multiplicity** (the four P3P solutions). The prior estimate plays the role of the fourth point. [V: Sec. V] | This is the theory backbone. H explains local weak directions. Symmetric modes are global, and only a multimodal belief can represent them |
| R10 | — | Implicit-PDF's evaluation uses the log-likelihood of the ground truth (uniform on SO(3) = −2.29), spread, and top-k recall. SYMSOL II (a cylinder with one off-centre dot) is exactly the near-symmetric case. [V] | Adopt the LL metric with its uniform reference for P3. The "decal" intervention is our analogue of SYMSOL II |

**Net effect.** The direction from rev. 1 stands: calibrated, observability-aware tracking of near-symmetric in-hand
objects. Two things change. First, the first experiment now asks whether the replication "failure" is mostly a metric
mismatch (R1). Second, the claim narrows to what the full texts leave open (R3, R4, R7).

---

## 1. Verdict

1. **The supervisor's three targets (30°, 80%, 5 Hz) are POGS's own headline numbers, measured on a different task.**
   Replicating them is a report, not a contribution. [I]
2. **The O4 "not met" verdict is not yet interpretable.** Our rule scores bit-axis roll and POGS's rule does not (R1). The
   tip error is already 1.5–5.6 cm, but this cannot be judged until roll is removed. E0 settles it in about one GPU-hour plus
   a re-run that stores poses. Until then, the sentence for the supervisor is: "not met under a stricter rule than the
   paper's; the paper-faithful re-score is pending". [V for the rule mismatch, I for the reading]
3. **The PhD thread is the calibrated pose belief for persistent-splat tracking.** It has three parts: per-frame pose
   information from the foundation-feature + depth render loss, a multimodal correction along the weak axis for global
   ambiguity, and compute gated by information. Its value is judged by a recovery metric with null models. [I]
4. **What is not a contribution:** a Kalman filter on POGS (RoPT, RGBTrack, EnDKF already filter); gripper masking or an
   end-effector prior (POGS lists both as future work, Sec. VI); reaching 5 Hz by cutting iterations; the index-matched KL
   (a function of the pose error). [V for the first two, I for the rest]
5. **Venues.** No credible sim-only submission is possible by Dec 4 (RSS stage 1). The realistic sequence is IROS 2027
   (Mar 1) only if gate G3 passes, otherwise CoRL 2027 (≈ late May, call not out) or ICRA 2028 (Aug 2027). Re-verify every
   date on the official page before acting on it. [U for 2027 dates]
6. **Confidence:** high on R1–R10 (full text); medium on the direction; low to medium on novelty. "Not found" covers 12
   papers read in full plus about 40 earlier snippet searches, and it is not proof of absence. Two targeted searches stay on
   the task list (§9).

---

## 2. Problem statement and claims

**Setting.** A rigid object is scanned once into a persistent Gaussian splat (POGS training: RGB + DINO features + depth
regularisation). It is then tracked frame by frame from a single RGB-D (stereo) stream by render-and-compare. The focus
is in-hand objects that are **near-symmetric**: drill, mug, soup can, bottle. Their symmetry-breaking features are small
and often occluded by the gripper or a hand.

**Core claim (falsifiable).** On near-symmetric in-hand objects, a splat tracker that (a) estimates per-frame pose
information from its own render loss, (b) represents the weak-axis ambiguity as a 1-D multimodal belief and (c) spends
iterations according to that information gives **calibrated** uncertainty: coverage close to nominal, and lower NLL of the
true pose than a single Gaussian. It also **recovers** from in-grasp perturbations more often than POGS, a fixed-noise
filter and an equal-compute multi-start baseline, with the Static null model near zero under the same rule. Accuracy on
well-observed objects stays within noise of POGS.

**Explicit non-claims.** Not best ADD on YCBInEOAT; FoundationPose, RoPT and Opra et al. all sit at 92–93 ADD there
(FoundationPose Table 4, RoPT Table 1, Opra Table II). Not a new filter. Not real-time against 40+ fps learned trackers.

---

## 3. Prior-art map (from full texts; details in `docs/literature/reading_notes.md`)

| Line of work | Representative (read) | What it has | What it lacks, relative to our claim |
|---|---|---|---|
| Splat / feature-field tracker | POGS (ICRA 2025) | Persistent splat, DINO + depth loss, multi-object | No uncertainty, no symmetry handling, no failure detection; 5 Hz on a 4090 |
| Learned render-and-compare | FoundationPose (CVPR 2024) | Strongest general tracker; 1-iteration refiner | No uncertainty; basin ≈ 20°; collapses under fast motion |
| Reconstruction trackers | BundleSDF (CVPR 2023) | Model-free; evaluation protocol | 10 Hz; symmetric re-acquisition failure |
| Filter + recovery systems | RoPT, Opra et al., RRTrack (2026) | Factor graph / reliability check / retrieval recovery | Fixed observation noise (RoPT), inlier counts (Opra), mask agreement (RRTrack); none is a calibrated pose belief |
| Hessian covariance in tracking | Compact pose graph (Oct 2026) | Σ = σ²H⁻¹ per edge; I/T/C/Oracle ladder; Spearman 0.65–0.72 | Correspondence space; local only; overconfident on symmetric objects (self-reported) |
| Localizability / degeneracy | X-ICP (T-RO 2024) | Scale-free localizability; constrained updates | LiDAR point-to-plane; no render loss, no multimodality |
| Uncertainty in 3DGS | UGS-Loc (2026) | Particles + Fisher over splat parameters | Camera relocalisation; no pose information; 1.1 s per iteration |
| Filters on SO(3) | EnDKF (2026), DQ-UKF (2026) | Directional / Lie-group filters; observability theory | Synthetic only; R is fixed |
| Distributions on SO(3) | Implicit-PDF (ICML 2021) | Non-parametric densities; LL metric; SYMSOL | Single image, learned; no tracking |

**The gap, stated narrowly.** No paper we read derives the measurement uncertainty of a *dense foundation-feature render
loss on a persistent splat*. None uses that uncertainty to (i) set an anisotropic filter covariance, (ii) trigger and
locate a multimodal search along the weak axis, and (iii) gate compute. None evaluates calibration with proper scores on
near-symmetric in-hand objects. [I, bounded by the 12 papers plus the snippet searches]

---

## 4. Method (to be built, tested and ablated)

Notation: the pose of object k is T ∈ SE(3), perturbed on the right, T·Exp(δ) with δ = (δθ, δt) ∈ ℝ⁶. The residual r(T)
stacks per-pixel DINO-feature residuals (64-D, from the PCA basis stored in the checkpoint) and depth residuals inside the
object ROI, using the release weights (3.7 depth, ignore beyond 0.26 m).

**M1. Information matrix.** H = JᵀWJ with J = ∂r/∂δ at the converged pose. J comes from 6 forward-mode JVPs, or from
central finite differences (12 renders) as a check. σ̂² = rᵀWr / (m − 6), Σ = σ̂²(H + λI)⁻¹. We store the eigenpairs in
the object frame. Scale-free measures (R8): rotation is scaled by the object radius ρ, so δθ·ρ is in metres. The reported
quantity is κ = λ_max / λ_min, together with the alignment of the weakest eigenvector with the object's principal axes.
Cost: about 6 renders per frame, out of 25 iterations of 1 render plus 1 backward each. E6 measures it.

**M2. Observability-aware update (ablation ladder).** The variants are: plain Adam (POGS); Tikhonov damping per block
(Opra/ICG+ style, isotropic); solution remapping (Zhang 2016: project the update onto eigenvectors with
λ > τ·λ_max); X-ICP-style partial constraints (keep the prior along weak directions); and the proposed information-form
prior, where the previous belief enters the loss as ½‖δ − δ̂⁻‖²_{P⁻¹}. That prior is the iterated measurement update of
an error-state filter.

**M3. Filter.** An iterated, right-invariant error-state Kalman filter on SE(3) with a constant-velocity prior. The
measurement covariance is R_t = Σ_t, and updates are gated by NIS against a χ²₆ bound. The ladder follows the compact pose
graph (R4): fixed R, trace R, full Σ, and an oracle R built from the true error.

**M4. Weak-axis multimodal belief (the novel part).** It is triggered when κ exceeds a threshold calibrated on held-out
sequences, or when the NIS gate fails. We render K poses on the 1-D orbit T·Exp(φ·v_min), φ ∈ (−π, π], read the loss
profile L(φ) and form p(φ) ∝ exp(−L(φ)/τ). The temperature τ is fitted once on calibration sequences, by NLL. The modes
of p seed hypotheses for M3. A flat profile means a continuous symmetry, so roll is reported as unobservable (the
axis-only error applies). A multimodal profile means discrete or near-symmetry, and every mode is kept until evidence
separates them. Cost is K = 16–32 renders, only when triggered. This addresses the documented failure in R4 and R7.

**M5. Information-gated compute.** The iteration budget is set from tr(Σ) and the NIS. Few iterations when the pose is
well constrained, more when it is not. E6 tests this against fixed budgets of 5, 10, 15 and 25 iterations on the same
frames.

**M6 (cheap add-on, optional). Splat reliability weights.** Each Gaussian gets a weight w_i ∝ 1/(1 + tr G_i), from Fisher
information over splat parameters computed once after training (UGS-Loc / FisherRF). The weights are rendered into the
residual weights W.

**Theory section (C).** (i) If the object has a continuous symmetry axis a, then L(T·Exp(t·a)) = L(T) for every image,
so a ∈ null(H), and roll about a is unobservable for any estimator. That is global multiplicity in the DQ-UKF sense (R9).
(ii) For a near-symmetric object, λ_min is bounded by the image footprint of the symmetry-breaking features, which
predicts that λ_min grows with decal area. Simulation can test this. (iii) A Cramér–Rao-style efficiency ratio
(empirical error covariance against Σ in the observed subspace) holds only if the residual-noise model is validated. It
is reported as heuristic until E10 checks it. [I]

---

## 5. Predictions and pre-registered pass rules

These are frozen in `experiments/protocols/e_series.md` before any E3+ run (task T7). Every rule is reported together with
Static and equal-compute baselines.

- **P0 (replication sanity).** Under the paper-faithful rule (tip < 3 cm and bit-axis direction < 5°, roll free), POGS
  in-grasp recovery is ≥ 46/50 in at least one setting. That is the smallest count whose Wilson lower bound reaches 80%.
  If P0 fails as well, the replication shortfall is real, and E1 explains it.
- **P1 (observability explains the error).** Spearman(log λ along the bit axis, |roll error|) ≤ −0.5 on the drill, and
  the weakest eigenvector lies within 15° of the bit axis in at least 70% of frames. Both are computed at the estimate and
  at the ground truth.
- **P2 (anisotropic information helps where directions are right).** On the near-symmetric subset, full Σ beats trace
  Σ and fixed R in recovery, with non-overlapping 95% hierarchical-bootstrap intervals. We pre-state that it may not beat
  trace on well-observed objects (R4).
- **P3 (multimodal belief is calibrated).** The mean log-likelihood of the true roll under M4 exceeds that of a single
  Gaussian, with a bootstrap interval that excludes 0. 90% credible sets cover 85–95% of true poses on observable
  directions.
- **P4 (compute).** Gated M5 matches fixed-10 success at a higher rate. A 5 Hz claim counts only at ≥ 95% success, on the
  same GPU, measured alone (O6 / realtime rules).

---

## 6. Experiments

Common rules: ≥ 3 independently trained splats per scene, 50 trials per condition, hierarchical bootstrap over (scene,
splat seed, trial), full SE(3) stored for every frame, the pre-registered rule printed beside any new rule, and Static,
no-op reset and equal-compute baselines everywhere. GPU hours are for one RTX 4070 (12 GB) and are rough until measured.

| ID | Question | Protocol | Pass rule | GPU-h | Depends on |
|---|---|---|---|---|---|
| **E0** | Is the O4 shortfall mostly roll about the bit axis? | Re-run O4 with full SE(3) stored (T1). Re-score with (a) the pre-registered rule, (b) the paper-faithful rule (R1), (c) axis-angle decomposition: roll about the bit axis vs swing | P0 | 1.5 (3 shards) | T1 |
| **E1** | Which cause makes the floor and the ± gap? | Signed roll stratified by the sign of the pre-perturbation error; loss at the estimate vs at the ground truth; 1-D roll sweep of the loss | Names one cause: roll observability / occlusion / optimisation / signed bias. Stop rule in §8 | 3 | T1 |
| **E2** | Does H see the roll? | 6×6 spectrum at the estimate and at the ground truth, every O4 frame | P1 | 4 | T2 |
| **E3** | Does an anisotropic R beat fixed and trace R? | M2/M3 ladder on O4: Static, POGS, fixed-R KF (existing), trace, full Σ, oracle, + M4 | P2; Static ≤ 2/50 | 25 | T4 |
| **E4** | Is the belief calibrated? | NEES, NIS, reliability of predicted σ vs error, 90% coverage, NLL with the uniform reference | P3 coverage part | 5 | T6 |
| **E5** | Is the ambiguity real and handled? | 2×2: asymmetric decal on/off × gripper rendered/hidden, on drill, mug and soup can (SYMSOL-II analogue, R10) | P3 LL part; decal raises λ_min as theory (ii) predicts | 30 | T5 |
| **E6** | Can information gate compute? | Success vs Hz for 5/10/15/25 iterations vs M5; reuse the O6 setup | P4 | 6 | T2 |
| **E7** | Does it hold on real data? | YCBInEOAT (9 videos, 5 objects; the soup can is the symmetric case) and HO3D. Splats from posed frames or CAD renders, disclosed. Same masks for all methods; first-frame alignment; ADD/ADD-S AUC 0–0.1 m (BundleSDF / compact pose graph protocol) | Beats POGS on every object; within 1 ADD of FoundationPose on the soup can or better; E4 calibration holds | 80 | T8 |
| **E8** | Graceful under fast motion? | Frame downsampling 1/2/4/8/16× (RoPT protocol) | Degrades more slowly than POGS and the fixed-R KF | 30 | T8 |
| **E9** | Does it recover from loss? | RRTrack event metric (recovered if any of the 10 frames after reappearance has ADD-S AR > 0.9 at 0.1 d), time-to-recover survival curve, with Static and a frozen estimate scored by the same rule | Excess recovery over null with an interval that excludes 0 | 15 | T6 |
| **E10** | How efficient is the estimator? | Empirical error covariance vs Σ in the observed subspace; residual-noise model check | Report only | 5 | E4 |
| E11 | (if E2 and E5 point to roll) Can an adapter raise λ_min? | Small adapter on frozen DINO, trained on splat-rendered perturbations to maximise λ_min | λ_roll × 2, other λ within −10%, roll error down | 30 | E2, E5 |
| E12 | (if speed still blocks) Amortised update | Self-supervised from the object's own splat; heteroscedastic head | Same success at 2× the rate of M5 | 40 | E6 |

---

## 7. Baselines reviewers will require

Static (no update) · no-op reset · POGS (paper: DINO + depth; release: no per-frame depth) · POGS + fixed-R KF (existing
v2) · equal-compute multi-start (UGS-Loc-style particles, same render budget) · Tikhonov-damped (Opra / ICG+) · solution
remapping (Zhang 2016, X-ICP baseline) · FoundationPose, model-free mode, with reference views rendered from our splat
(the same object knowledge) and fps measured on our GPU · BundleSDF on real data · a published-number row for RoPT, Opra
et al. and the compact pose graph, labelled "published, different hardware" unless their code is released. [I; code
availability for RoPT and Opra is U]

---

## 8. Timeline and gates (Nov 23, 2026 → May 2027)

| When | Work | Gate |
|---|---|---|
| Nov 23–26 | T1 full-pose storage; E0 re-run (3 GPU shards after the pipeline's `realtime` stage finishes, so latency timings stay clean); E1 | **G0 (Nov 26):** P0 result. If the paper-faithful rule meets 80%, the supervisor's O4 target is met, and the PhD story becomes "the headline survives, but roll is unobservable and the 5° claim hides it". Continue with E2 either way |
| Nov 27 – Dec 10 | T2 information matrix + tests; E2; T3–T4 filter ladder; E3 starts | **G1 (Dec 10):** P1 and P2 on simulation. Neither passes → pivot to an analysis + benchmark paper (RA-L or a workshop) |
| Dec 7–22 (parallel) | T8 YCBInEOAT loader; splats from posed frames; POGS end to end on ≥ 3 real sequences | **G2 (Dec 22):** metrics computed on real data |
| Dec 11 – Jan 3 | T5 weak-axis belief; E5 2×2; E6; E9 | — |
| Jan 4 – Feb 7 | E7, E8, E10; optional E11 | **G3 (Feb 7):** go for IROS if P1–P3 hold in simulation and E7 shows a gain on the symmetric subset with calibration intact. Otherwise target CoRL 2027 |
| Feb 8 – Mar 1 | Write; supervisor reads a full draft by Feb 15; arXiv at submission | — |
| Mar – May | More real sequences, a robot demo if an arm is available (RoPT-style robot-TCP ground truth is accepted, R6 / RoPT App. C), E11–E12 if justified | — |

**Stop and re-scope rules.** (1) If at G0/E1 the cause is optimisation failure or signed bias, method work stops. We write
the methodology note (the metric mismatch, the null-model recovery metric, the pseudo-replication fix) and decide again.
(2) If by the end of April there is no real-data gain over POGS on the symmetric subset, the paper becomes an analysis and
benchmark paper. (3) Never cut: the pre-registered result, Static and equal-compute baselines, per-frame storage,
calibration (E4) and one real benchmark.

**Direction decision (2026-11-23, after G0 / E2 / E2b; LEDGER 8.B).** G0 failed, P1 failed, and the floor is a
*bias* of the DINO feature term (minimum ≈ −10° roll at the true pose), not weak observability. A filter that
models zero-mean scatter cannot remove a bias, so the method line M1–M4 / E3 is **on hold**, and its freeze tag
is held. The programme follows **direction B**, in a fixed order with a deadline, and the analysis note (C) is
written alongside it:

| Step | Test | Outcome that decides |
|---|---|---|
| B0 | Is the ground truth right? (sim: renderer transform, 6-DOF RGB/depth sweeps; real: ArUco) | done: GT exact. Bias is real |
| B1 | Roll sweeps from many viewpoints, DINO / depth / RGB separated; grid shift; scene context; weight ladder | done: the DINO minimum moves with viewpoint (±5°) and with shadows (−6° → −10°); depth unbiased but flat. E2c (depth ×30/×100 on trajectories) running |
| B2 | Re-distil features only on frozen geometry: DINOv2-reg, DINOv3, one upsampler; accept on the B1 sweeps first | by **2026-11-30** |
| B3 | Only if B0–B2 fail: 3-D-aware feature fine-tuning (FiT3D-style) | new research question, expensive |

If a step removes the bias, the paper is: *a systematic, view- and illumination-dependent roll bias in
distilled-DINO tracking, its cause, the fix, and a recovery metric checked against Static.* If none does, the
analysis note becomes the main result (stop rule 1).

**Resource rules (unchanged from PLAN §3.4).** Accuracy runs may share the GPU (O4 as 3 shards). Latency stages (o5, o6,
realtime, E6 timing) run alone. Report the hardware with every timing.

---

## 9. Tasks for Claude Code, in order (mirrored in `docs/LEDGER.md` Phase 8)

1. **T1.** Store full SE(3) estimate and ground truth per frame in O3/O4 runs (`per_frame.json` → add `pred_T`, `gt_T`).
   Add `metrics.pose.axis_angle_decomposition` (roll about a given object-frame axis vs swing) and the paper-faithful
   tool rule. Unit tests. Re-run O4 (E0).
2. **T2.** `model/pogs/information.py::info_matrix(...)`: the 6×6 Gauss-Newton matrix of the DINO + depth residual for a
   right-multiplied se(3) perturbation, scale-normalised eigenpairs in the object frame. Tests against finite differences
   and against a synthetic object with a known null direction (a textureless cylinder gives roll in the null space).
3. **T3.** E1 and E2 analysis scripts on the T1 runs.
4. **T4.** Iterated error-state filter with the information-form prior in the loss, NIS gate, and ladder switches: fixed,
   trace, full, oracle. Plus the Tikhonov and solution-remapping update variants. All through config or the motion-prior
   hook of `track_sequence` (CLAUDE.md rule 2).
5. **T5.** Weak-axis probe: orbit renders, L(φ), temperature calibration, mode extraction, NLL scoring.
6. **T6.** Metrics: NEES, NIS, reliability curve, coverage, NLL (with the uniform reference), hierarchical bootstrap,
   survival curve, RRTrack event metric.
7. **T7.** Train ≥ 3 splats per scene. Freeze `experiments/protocols/e_series.md`, log the deviation, and tag the commit
   before any E3 run.
8. **T8.** Real-data loader for YCBInEOAT and HO3D, ADD/ADD-S AUC, first-frame alignment.
9. **T9.** Check FoundationPose in 12 GB (model-free mode) and record whether code for RoPT and Opra is available, with the
   date checked.
10. **T10.** Two novelty searches before writing related work: "localizability render-and-compare object pose" and
    "observability Gaussian splatting object tracking uncertainty". Add them to the reading notes.

**Open questions for the supervisor** (send with the E0 result): (a) is the 80% target the paper's closed-loop servoing
success, or a tracking-recovery rate? (b) is an arm or an RGB-D camera available for 30 minutes of real capture? (c) does
the supervisor accept the paper-faithful rule (roll free) as the replication rule, with the stricter rule kept as our
own?

---

## 10. Likely reviewer attacks and defences

| Attack | Defence | Evidence |
|---|---|---|
| "A KF on a known tracker is incremental" | The claim is the information source, the calibration and the weak-axis belief. The ladder shows fixed R gives little | E3, E4 |
| "Hessian covariance for tracking exists" | Cite the compact pose graph. Ours is a dense render loss on a persistent splat, and it fixes that paper's self-reported overconfidence on symmetric objects | E4, E5 |
| "Not better than FoundationPose / RoPT" | Narrow claim: calibrated belief and recovery on near-symmetric in-hand objects; report every loss | E7, E8 |
| "You changed the metric after seeing results" | The paper-faithful rule is quoted from POGS Sec. V-B (R1). The pre-registered rule stays first; deviation log; null models under every rule | E0 |
| "Simulation only" | YCBInEOAT + HO3D; simulation used only for the interventions real data cannot make (decal, hidden gripper) | E5, E7 |
| "The mixture is ad hoc" | The probe axis is the Hessian's weakest eigenvector; τ is fitted on held-out data; scored by a proper scoring rule; oracle-axis ablation | E5 |
| "Dependent trials" | ≥ 3 splats per scene; hierarchical bootstrap; Wilson intervals only for descriptive counts | all |

---

## 11. Sources

All 12 papers are in `ref-paper/`, read in full; section, table and figure references are in
`docs/literature/reading_notes.md`. Venue dates come from rev. 1's search snippets and are not re-checked here [U].
Re-verify before use: CVPR 2027 dates page, RSS 2027 call, IROS 2027 listing (IEEE RAS), CoRL 2026 call as a proxy for
2027, ICCV 2027 (CVF).
