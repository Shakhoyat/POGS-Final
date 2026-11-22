# POGS PhD Direction: Candid Review, Novelty Gap, and 6-Month Plan

Nov 22, 2026

## Verdict (read this first)

Yes, there is a PhD-worthy thread, but it is not "reproduce POGS's three numbers". It is ambiguity- and observability-aware tracking of in-hand tools, judged with a recovery metric that has a null model. Tags: \[V\] verified source, \[I\] inference, \[U\] unsure.

1. Your supervisor's targets (30 degrees, 80% recovery, 5 Hz) are POGS's own headline numbers. A replication checklist is not a contribution, so his "no depth yet" worry is fair. \[I\]
2. POGS's 80% is 40/50 on a real UR5 servoing a drill to a target, with no baselines. It is not a 10-frame tracking-recovery test like your O4. \[V: arXiv 2503.05189, fetched summary; re-read Sec. V.B and Table II\]
3. Your pre-registered rule (under 3 cm and 5 degrees) cannot pass when rotation error is already 10 to 24 degrees before the perturbation. It measures steady-state accuracy, not recovery. \[I\]
4. Your post-hoc rule has a leak: Static scores up to 7/50 and the plus/minus gap is large. A signed pre-perturbation bias can explain both, with or without roll symmetry. \[I\]
5. Best direction: observability-aware (Hessian or Fisher information) tracking on Gaussian splats, an SO(3) ambiguity prior, and a null-model recovery metric. It fits an estimation and signal-processing profile. \[I\]
6. Weakest direction: a learned update to reach 5 Hz. FoundationPose already tracks at about 32 Hz, so you would have to beat it. \[V\]
7. Timing: CVPR 2027 closed on Nov 16 and ICRA 2027 on Sep 15. RSS 2027 stage 1 is Dec 4, twelve days away. Do not rush a sim-only paper into it. First realistic target is IROS 2027 (Mar 1). \[V dates, I advice\]
8. Fall 2028 PhD applications are due around Dec 2027, so a CoRL or ICRA 2028 submission plus arXiv by autumn 2027 serves you better than a rushed December paper. \[I\]
9. Real evidence is non-negotiable. Public datasets (YCBInEOAT, HO3D, HOT3D with BOP-H3 onboarding) give it without owning a robot. \[I\]
10. Confidence: medium on the direction, high on the metric critique, low on exact novelty (my search was about 25 queries, mostly snippet-level).

## What your numbers actually say

The 80% target is borrowed from a different experiment, and your own results are better explained by a leaky metric than by a failed tracker. Wilson 95% intervals below are my calculation from your counts.

| In-grasp setting | Pre-registered rule (k/50, 95% CI) | Post-hoc rule (k/50, 95% CI) |
| --- | --- | --- |
| +15 degrees | 4/50 = 8% \[3.2, 18.8\] | 49/50 = 98% \[89.5, 99.6\] |
| -15 degrees | 14/50 = 28% \[17.5, 41.7\] | 29/50 = 58% \[44.2, 70.6\] |
| +30 degrees | 12/50 = 24% \[14.3, 37.4\] | 46/50 = 92% \[81.2, 96.8\] |
| -30 degrees | 6/50 = 12% \[5.6, 23.8\] | 17/50 = 34% \[22.4, 47.8\] |
| Static baseline (best setting) | 0/50 | 7/50 = 14% \[7.0, 26.2\] |

- **What POGS actually measured.** A human rotated a drill 15 to 30 degrees about the grasp axis, and success meant the tip ended within 3 cm and 5 degrees of the servo target. Five trials per tier, five target poses, both directions: 48/50 (tier 1) and 40/50 (tier 2, 3D target motion) on a real UR5 with a ZED 2. No frame budget and no baselines are reported for this test. The text ties 48/50 to "up to 15 degrees" and 40/50 to 30, but the methods say every trial is 15 to 30, so those labels probably track the tier, not the angle. \[V: fetched summary of arXiv 2503.05189; check the PDF yourself\]
- **So your O4 is a different quantity.** Closed-loop servoing success on hardware is not open-loop tracking recovery at 2 s in simulation. A perfect replication of your tracker would still not reproduce their number. \[I\]
- **Clearing 80% needs 46 of 50.** That is the smallest count whose Wilson lower bound reaches 80%. Only your post-hoc +15 and +30 settings clear it, and only under a rule that fails its own sanity check (Static scores above zero). \[I, my arithmetic\]
- **Pseudo-replication.** The 50 trials share one trained splat per scene, so Wilson assumes independence you do not have, and splat-training variance is uncounted. Retrain several splats per scene and bootstrap hierarchically (scene, seed, trial), in the spirit of the stratified-bootstrap advice in Agarwal et al. (NeurIPS 2021). \[V for the paper's advice, I for applying it\]
- **Why tip error can stay small while rotation error is large.** A tip on the symmetry axis does not move under roll. With a 20 cm lever arm, a 10 degree swing gives about 3.5 cm and 24 degrees gives about 8.3 cm, so tip errors of 1.5 to 5.6 cm fit mostly-roll error plus some off-axis error. \[I, my arithmetic\]
- **Why plus and minus can differ.** If rotation error before the perturbation has a consistent sign, a perturbation of one sign moves even a frozen estimate closer to truth, so it passes "within +5 degrees of pre-perturbation error", and the opposite sign fails. That alone can produce Static's 7/50 and the sign gap. Check signed errors before crediting symmetry. \[I\]
- **Axis mismatch to rule out.** POGS perturbs about the grasp axis, which for a drill need not be the bit axis. If yours does too, the roll hypothesis may explain the pre-existing rotation floor but not directly the recovery gap. Log the perturbation axis in the object frame. \[V for the POGS protocol, I for the rest\]

## Candidate contributions at a glance

Combine candidates 1 and 3 into the core story, use 2 as a component, keep 4 as an ablation, and treat 5 as a stretch. Novelty and risk are my judgement from a non-exhaustive search. \[I\]

| # | Candidate | Novelty | Feasibility on 12 GB | Risk it is already done | Closest prior work |
| --- | --- | --- | --- | --- | --- |
| 1 | Observability-aware tracking: Hessian or Fisher information of the render loss, damp or hold weak directions, SO(3) prior along them | Medium to high | High (6-DoF Jacobian, cheap) | Medium | X-ICP localizability-aware registration (LiDAR); Implicit-PDF, Bingham and matrix-Fisher SO(3) distributions |
| 2 | Symmetry- or roll-aware hypothesis sets and SO(3) mixture priors | Low to medium; medium if symmetry is discovered from the splat for near-symmetric tools | High | High to medium | Manhardt et al. ICCV 2019; Implicit-PDF ICML 2021; Diff-DOPE; multi-hypothesis object SLAM |
| 3 | Recovery metric with null model, plus small benchmark | Medium | Very high | Low to medium, but weak alone | VOT reset-based accuracy and robustness; BOP and ADD/ADD-S AUC |
| 4 | Gripper-aware occlusion in the splat render loss | Low | High (gripper pose is known from kinematics) | High | GHOST hand-aware loss; ComPose; Real2Sem2Real mask-gated loss |
| 5 | Amortised init or learned update replacing 25 iterations | Low | Medium (training data is free from the splat itself) | High | DeepIM, RePOSE, MegaPose, FoundationPose (about 32 Hz), iGaussian |

## Q1. Prior art map

POGS-style tracking is a rigid per-object render-and-compare refiner. It sits below FoundationPose and BundleSDF on published real-benchmark evidence and has no uncertainty, symmetry handling or failure detection. Confidence: medium. \[I unless tagged\]

- **Splat and neural-field trackers.**
  - POGS (Yu et al., ICRA 2025): DINO plus depth loss, per-object SE(3) by gradient descent. \[V\]
  - RSRD (Kerr et al., CoRL 2024): DINO-embedded Gaussians, part poses tracked by differentiable rendering. \[V\]
  - Dynamic 3D Gaussians (Luiten et al., 3DV 2024): dense 6-DoF tracking from persistent dynamic view synthesis, multi-view. \[V\]
  - BundleSDF (Wen et al., CVPR 2023): about 10 Hz, neural object field plus pose graph, first-frame mask only, evaluated on HO3D, YCBInEOAT and BEHAVE. \[V\]
  - 6DOPE-GS (3.5 Hz, HO3D and YCBInEOAT), GSGTrack (HO3D and OnePose), GTR (TRI, 2025). \[V\]
  - GS-Pose and GS2Pose use splats for pose refinement. \[V\]
- **Foundation-model and render-and-compare pose.**
  - FoundationPose (CVPR 2024): model-based and model-free, tracking at about 32 Hz after one estimation pass. \[V\]
  - MegaPose (CoRL 2022), se(3)-TrackNet (RSS 2020, introduced YCBInEOAT), BundleTrack, RePOSE (ICCV 2021), Diff-DOPE (restarts with random learning rates to escape symmetry minima). \[V\]
- **SAM-based and filter-based recovery.** SuperPose combines SAM2 with FoundationPose; RGBTrack and FoundationPose++ add a 2D tracker, Kalman filter and recovery logic. These are the practical competitors for any "recovery" claim. \[V\]
- **In-hand and hand-aware tracking.**
  - Neural Feels (visuo-tactile neural fields) reports that axis-symmetric objects are hard for its depth-based estimator, which supports your roll hypothesis. \[V\]
  - TEG-Track (tactile-enhanced in-hand tracking).
  - ComPose (arXiv, May 2026) uses hand cues when the object is occluded.
  - Real2Sem2Real (arXiv, Oct 2026) tracks a DINOv3-distilled Gaussian splat by render-and-compare with a mask-gated loss. This is the closest recent neighbour to your setup and a scoop risk for candidate 4. \[V from snippets\]
  - GHOST and interaction-aware 4DGS are reconstruction methods, not trackers. \[V\]
- **Framing fix.** Your drill is rigid, so "non-rigid" is the wrong label. In-hand rigid tracking under grasp slip is accurate, and non-rigid belongs to Dynamic 3D Gaussians and 4DGS. \[I\]
- **Not verified.** I did not find POGS numbers on YCBInEOAT or HO3D, so I cannot say how it compares to FoundationPose or BundleSDF on them. \[U\]

## Q2. What is genuinely open

The open gap is explaining and handling weak observability in render-and-compare object tracking, and measuring recovery honestly. Matching POGS numbers or hitting 5 Hz is not a gap. Confidence: low to medium, because absence of prior work is the hardest thing to verify. \[U\]

1. **Observability-aware splat tracking plus ambiguity prior plus null-model metric (core).** LiDAR registration already analyses Hessian eigenvalues to detect degenerate directions and constrain updates (X-ICP, arXiv 2211.16335). I found no such analysis for foundation-feature, Gaussian-splat object tracking. The novelty is carrying that idea to a DINO-plus-depth render loss, adding a calibrated SO(3) prior (Bingham or matrix-Fisher) along weak axes, and showing it improves recovery against Static and equal-compute baselines. To confirm before committing, run two more searches: "localizability render-and-compare object pose" and "recovery after in-hand slip pose tracking benchmark". \[V for X-ICP, I for the gap\]
2. **Hessian-guided iteration budget (bonus link to your O6).** Your data show 10 iterations at 250 to 500 px keep 100% success at about 3.2 Hz. Spending iterations only on well-conditioned directions ties speed to observability and gives a systems result that is not just "fewer iterations". \[I\]
3. **Roll-aware hypothesis sets.** Multi-hypothesis and mixture-on-SO(3) ideas are well established in single-image pose and SLAM, so as a standalone claim this is likely seen as known. It becomes publishable only if symmetry is discovered from the splat for near-symmetric tools without annotation, and the hypotheses are driven by the Hessian. \[I\]
4. **Gripper-aware occlusion.** POGS itself says partial occlusion hurts tracking, and several 2025 to 2026 papers use hand-aware or mask-gated losses. In a robot setting you can render the known gripper and z-buffer it, which is neat but incremental. Use it as an ablation. \[V that these papers exist, I for the judgement\]
5. **Learned update or amortised init.** The idea is old (DeepIM, RePOSE, MegaPose) and FoundationPose tracks at about 32 Hz. Pursue only if 1 to 3 succeed and speed is still the blocker. It is also the only candidate that fits CoRL's "learning" scope. \[V for the facts, I for advice\]

**Not contributions:** the KL between matched Gaussians (you noted it is a function of pose error), the foundation-model latency table (an off-the-shelf benchmark), and the 5 Hz target as such. Keep them as supporting evidence only. \[I\]

## Q3. Methodology critique

Saying "fails the target under the pre-registered rule" is sound if you also say the rule measures steady-state accuracy and cannot detect recovery. Saying "POGS fails to recover" is not supported. Confidence: high on the logic, medium on the stats advice. \[I\]

**Standard ways to define recovery**

- **Threshold and hold:** error below a threshold for k consecutive frames after the event, reported as success within a horizon and as time-to-recover (a survival curve censored at 10 frames).
- **Return to baseline:** error within a margin of the pre-event level. This is your post-hoc rule. It needs signed or axis-wise errors and a baseline that is itself trustworthy.
- **Reset-based robustness:** count failures and re-initialise, as in the VOT accuracy, robustness and expected-average-overlap protocol. \[V\]
- **AUC of ADD and ADD-S** over a whole sequence, as used on YCBInEOAT by se(3)-TrackNet and BundleSDF. \[V\]

**Null models.** Report Static (no update), a no-op that resets to the pre-perturbation pose, and an equal-compute alternative (depth-only ICP, or multi-start restarts with the same iteration budget). Report excess success over the null with a hierarchical-bootstrap interval. A rule is acceptable only if Static scores about zero on it. Yours scored 7/50, so it fails that check. \[I\]

**Reporting a post-hoc rule without p-hacking**

1. Keep the pre-registered result first and unchanged, with its interval.
2. Label the new rule as a deviation, with a dated log and git commit showing it was chosen after seeing results.
3. Fix its parameters on separate calibration scenes or seeds, then evaluate on untouched ones.
4. Show a small family of reasonable rules (a specification curve) rather than one tuned rule, each with its Static score.
5. Make no significance claims from the post-hoc rule alone. \[I: general statistical practice\]

**Is a 10 to 24 degree floor consistent with symmetry ambiguity?** Plausible, not established. It fits if the error is mostly about the bit axis, tip error stays small, and the floor appears right after initialisation. It does not fit if tip error is large, or if the floor is absent when the splat is initialised at ground truth. Also note a drill is only approximately symmetric: the handle, chuck and battery break symmetry, but ViT-S/14 patches are coarse, so DINO may not see them. Score rotation with a symmetry-aware error (axis only for a continuous symmetry) as well as geodesic error. \[I\]

**Diagnostics that separate the three causes**

| Hypothesis | What you should see | Test |
| --- | --- | --- |
| Roll is weakly observable | Error concentrated about the bit axis; tip error small; loss flat along roll; smallest Hessian eigenvector aligned with the bit axis | Axis-angle decomposition of the error; 1-D roll sweep of the loss; 6x6 Gauss-Newton eigenspectrum; add an asymmetric decal in simulation |
| Gripper occlusion | Error grows with the occluded fraction; direction-dependent; vanishes when the gripper is hidden | Hide the gripper in simulation; plot error against visible-pixel fraction per sign |
| Optimisation failure | More iterations, restarts or learning-rate changes reduce error | Loss at the estimate versus loss at ground truth: if the estimate's loss is higher, it is optimisation; if lower, the loss or features are at fault |
| Signed bias artefact | Plus and minus differ by the sign of the initial error | Stratify by the sign of pre-perturbation signed roll error |

The loss-at-estimate versus loss-at-ground-truth comparison is the cleanest single discriminator and costs one extra render per frame. \[I\]

## Q4. Venue and evidence bar

Nothing top-tier is realistic by Dec 31 with simulation-only evidence. The first sensible shot is IROS 2027, then CoRL 2027 or ICRA 2028. Confidence: high on dates, medium on the evidence bar, which is my inference from what comparable papers did. \[V dates, I bar\]

| Venue | Deadline | Status |
| --- | --- | --- |
| CVPR 2027 | Nov 16, 2026 (supplementary Nov 23) | Closed |
| ICRA 2027 | Sep 15, 2026 | Closed; ICRA 2028 is listed for Aug 16, 2027 |
| RSS 2027 | Dec 4, 2026 extended abstract; final paper Apr 16, 2027 | New two-stage format; not recommended for sim-only |
| IROS 2027 | Mar 1, 2027 | First realistic target |
| CoRL 2027 | Not announced (CoRL 2026 was May 28) | Plan for late May 2027, unconfirmed |
| ICCV 2027, NeurIPS 2027 | Not checked | \[U\] |

**Minimum evidence by venue** \[I, anchored on peers\]

- **ICRA and IROS:** at least one public real benchmark plus a real capture or robot demo. Peers: 6DOPE-GS used HO3D and YCBInEOAT plus a live real-world demo; GSGTrack used HO3D and OnePose. \[V\]
- **CoRL:** real robot results and a learning component. CoRL's reviewer guidance lists "works only in a simulator" as a grey area and "no learning" as a scope concern. \[V from CoRL 2026 reviewer instructions\]
- **CVPR, ICCV, NeurIPS:** state-of-the-art comparisons on about three public real datasets against FoundationPose and BundleSDF. BundleSDF used HO3D, YCBInEOAT and BEHAVE; a recent tracker used HOI4D, YCBInEOAT, HO3D and YCB-Video. NeurIPS additionally wants a theory or insight angle. \[V for peers\]
- **How many real sequences:** there is no official number. A defensible target is 30 to 50 real sequences covering 8 to 10 objects, at least two near-symmetric. YCBInEOAT has 9 videos of 5 objects, HO3D has 13 scenes, DexYCB has about 1,000 sequences. \[I; dataset sizes V from snippets\]

**Datasets.** YCBInEOAT (robot in-hand, ADD and ADD-S AUC, direct comparison with published numbers), HO3D, DexYCB, HOT3D with the BOP-H3 onboarding videos (needed because your method requires a multi-view scan; BOP 2024 supplies onboarding videos for HOT3D, HOPEv2 and HANDAL), and BEHAVE. If a dataset has no scan views, render views from its CAD to build the splat and say so. \[V that BOP-H3 has onboarding videos, I for the rest\]

**Baselines reviewers will demand:** FoundationPose (model-based and model-free), BundleSDF, BundleTrack, se(3)-TrackNet, 6DOPE-GS, GSGTrack, the original POGS, a depth-only ICP baseline, Static, and an equal-compute baseline. For recovery claims, also FoundationPose with a 2D tracker and Kalman filter (the FoundationPose++ and RGBTrack style). Whether FoundationPose fits your 12 GB card alongside the rest is untested. \[U\]

## Q5. Six-month plan (Nov 23, 2026 to May 2027)

Run one causal experiment first, because simulation lets you intervene in ways real data cannot. Then build the smallest method that exploits what it shows, and test it on real public data before any submission. All steps fit a single 12 GB GPU: the 6x6 Hessian needs a handful of backward passes per frame. \[I\]

**The single highest-information experiment: a 2x2 intervention in simulation**

Factors: drill as is versus drill with an asymmetric decal that breaks roll symmetry, crossed with gripper rendered versus gripper hidden. Use 50 trials per angle (plus and minus 15 and 30), at least 3 independently trained splats, full SE(3) stored every frame, signed and axis-decomposed errors, and loss at estimate versus loss at ground truth. Two free precursors on existing runs: re-score with signed and axis-wise errors, and a 1-D roll sweep of the loss.

| Outcome | Meaning | Direction |
| --- | --- | --- |
| Decal removes the floor and the sign gap | Roll observability is the cause | Go: candidates 1 and 3, a real paper |
| Hiding the gripper removes it | Occlusion is the cause | Shrinks to a hand-aware loss; pivot to metric and benchmark |
| Neither helps and loss at estimate is higher than at ground truth | Optimisation or initialisation failure | Fix with restarts; check for a bug; low novelty, re-scope |
| Neither helps and loss at estimate is lower | Features or loss mismatch (patch size, depth bias) | Test DINOv3 and higher resolution before anything else |
| Both help partly | Additive causes | Report the decomposition; method handles both |

**Schedule**

1. **Nov 23 to Dec 6:** freeze the protocol and write a deviation log; store full poses; re-score old runs; add hierarchical intervals and symmetry-aware rotation error. Decision D0 on Dec 6: is the floor explained by roll, occlusion, optimisation or sign bias?
2. **Dec 7 to Jan 3:** run the 2x2; log the per-frame Hessian eigenspectrum at ground truth and at the estimate; write a four-page internal note.
3. **Jan 4 to Feb 7:** implement the method (Hessian-conditioned update that damps weak directions, SO(3) prior along them, multi-start baseline with equal compute, Hessian-guided iteration budget). In parallel build splats for YCBInEOAT and HOT3D or HO3D and run POGS, FoundationPose and BundleSDF on the same sequences.
4. **Feb 10, decision D1 (go or no-go for IROS Mar 1):** go only if the method beats plain POGS on a real benchmark with non-overlapping intervals on near-symmetric objects and is within reach of FoundationPose. Otherwise skip IROS and continue.
5. **Feb 11 to Mar 1:** write the paper, or if no-go, extend data collection.
6. **Mar to May:** reach 30 to 50 real sequences, release the recovery benchmark with null models, ask whether any lab at KUET or a collaborator has an arm for a closed-loop demo. Target CoRL 2027 (about late May, unconfirmed) or ICRA 2028 (Aug 16, 2027).

**Stop and re-scope rules**

- If at D0 the cause is optimisation or sign bias, stop the method work and write a short methodology note.
- If by the end of April the method shows no significant real-data gain over plain POGS and trails FoundationPose, re-scope to an analysis and benchmark paper (workshop or RA-L) and stop chasing state of the art.
- Fold this into your SOP in autumn 2027: the story is that you found why a published headline did not replicate, explained the mechanism, and fixed it. \[I\]

## Q6. Three likeliest rejection reasons

These are the objections a reviewer at ICRA, IROS or CoRL is most likely to raise, with what defuses each. \[I\]

1. **"Simulation only, no real evidence."** Defuse with at least one public real benchmark (YCBInEOAT or HO3D) and 30 to 50 real sequences, and a closed-loop demo if any arm is available. Keep simulation only for causal interventions, and say that is why: you cannot swap a decal or hide a gripper on real footage.
2. **"Incremental over POGS, and FoundationPose or BundleSDF are stronger."** Compare to them directly and report where you lose. Claim a narrow, testable win: near-symmetric, in-hand objects, with intervals, plus the diagnosis of why existing trackers fail there. Do not claim general state of the art.
3. **"The metric and analysis look post-hoc, and statistics ignore dependence."** Freeze the pre-registered rule, show its result first, log deviations, calibrate on held-out scenes, use a hierarchical bootstrap across scenes, seeds and trials, and show Static and equal-compute scores under every rule.

A fourth risk to watch: scooping by Real2Sem2Real or ComPose on the occlusion angle. It is another reason to lead with observability, not occlusion. \[I\]

## Read first, and what to send your Claude Code

**Five papers to read first**

1. [POGS (Yu et al., ICRA 2025)](https://arxiv.org/abs/2503.05189): Sec. V.B and Table II, to see exactly what the 80% means.
2. [FoundationPose (Wen et al., CVPR 2024)](https://arxiv.org/abs/2312.08344): tracking mode, speed and the YCBInEOAT numbers you must meet.
3. [BundleSDF (Wen et al., CVPR 2023)](https://arxiv.org/abs/2303.14158): the evaluation protocol on HO3D, YCBInEOAT and BEHAVE, and its supplement.
4. [X-ICP (Tuna et al., arXiv 2211.16335)](https://arxiv.org/abs/2211.16335): the Hessian-based localizability idea to port to render-and-compare.
5. [Implicit-PDF (Murphy et al., ICML 2021)](https://arxiv.org/abs/2106.05965): representing symmetric and ambiguous rotations as distributions on SO(3).

Skim next: [6DOPE-GS](https://arxiv.org/abs/2412.01543), [Real2Sem2Real](https://arxiv.org/abs/2610.28880), [ComPose](https://arxiv.org/abs/2605.23523), [Neural Feels](https://arxiv.org/abs/2312.13469), [Agarwal et al. on reliable evaluation](https://arxiv.org/abs/2108.13264).

**Questions and tasks for your Claude Code**

- What initialises the pose before the perturbation, and why is rotation error already 10 to 24 degrees there?
- In which object-frame axis is the drill perturbed: bit axis or grasp axis?
- Dump per-frame full SE(3), signed rotation error, error about the bit axis versus the swing, and loss at estimate versus loss at ground truth.
- Does the O4 setup include an equal-compute baseline, or only Static, POGS, Kalman and release?
- How many independently trained splats per scene exist, and can you train at least three?
- What are the Adam learning rate, DINO feature resolution and render resolution?
- Why do the O3 outlier seeds at +90 degrees fail?
- Does your supervisor or anyone at KUET have access to a robot arm or a depth camera?

**Sources.** Only the POGS paper was opened in full (via a fetch summary); everything else comes from search-result snippets and should be checked at the links above and in the venue pages: [CVPR 2027 dates](https://cvpr.thecvf.com/Conferences/2027/Dates), [ICRA 2027 call](https://2027.ieee-icra.org/announcements/call-for-technical-papers/), [RSS 2027 call](https://roboticsconference.org/information/cfp/), [IROS 2027 listing](https://www.ieee-ras.org/events/category/important-dates/), [CoRL 2026 call](https://2026.corl.org/contributions), [BOP Challenge 2024](https://arxiv.org/abs/2504.02812), [VOT protocol](https://link.springer.com/10.1007/978-3-030-11009-3_1).

## Addendum D. Research-grade plan toward an A\* submission (added 2026-11-22)

A Kalman filter on top of POGS will not carry an A\* paper, because 2026 trackers already combine filters, mask tracking and recovery. What can carry one is a calibrated pose belief for splat tracking: use the Hessian of the render loss to say which pose directions the image constrains, where the object is ambiguous, and when to spend compute. \[I\]

Three findings from the October to November 2026 literature drive this plan.

- **Filters and recovery are crowded.** RoPT puts a constant-velocity factor graph, multi-hypothesis scoring and self-recovery on top of FoundationPose; two other trackers add failure detection and re-localization. A filter alone is table stakes. \[V: opened the PDFs\]
- **Your diagnosed gap is documented as unsolved.** A compact pose-graph tracker derives per-edge covariance from the inverse Hessian and reports one failure case with 37.66° rotation error but only 1.22° estimated uncertainty (30.9 times overconfident), on a symmetric object under occlusion. Its authors say a local covariance cannot see a separated alternative solution. That is the gap your roll ambiguity sits in. \[V\]
- **No general SOTA claim is available.** On YCBInEOAT, FoundationPose, RoPT and Opra et al. all sit around 92 to 93 ADD and 95 to 96.6 ADD-S. A POGS-based tracker should claim calibrated uncertainty, ambiguity handling and compute efficiency on near-symmetric in-hand objects, not best accuracy. \[V for the numbers, I for the advice\]

I cannot promise A\* acceptance. What I can state are the conditions a reviewer will check: one falsifiable claim, a win over the strongest baseline on a named subset of real data, calibration evidence, and honest negative results.

## D1. What the newest papers change

Every row below was read in the PDF text, not only the abstract, unless marked. Numbers are the papers' own, on their own protocols, so do not compare across rows.

| Paper | What it does | Result worth knowing | What it means for you |
| --- | --- | --- | --- |
| [RoPT](https://arxiv.org/abs/2610.28962) (Oct 2026) | FoundationPose refiner + EdgeTAM masks + GTSAM iSAM2 constant-velocity factor graph + multi-hypothesis + failure-triggered re-localization | YCBInEOAT all-object ADD 92.02 at original rate; 90.85 at 16× frame-rate downsampling, where FoundationPose falls to 38.87. 18.6 FPS | The strongest filter-plus-recovery competitor. Its factor graph uses fixed observation sigmas (0.02 for three terms, 0.10 for three), so per-frame measurement quality is not modelled. Opening for a Hessian-driven R |
| [Opra et al.](https://arxiv.org/abs/2607.23468) (Jul 2026) | Optimisation tracker (ICG+) with cycle-consistency reliability check and TEASER-based global recovery | YCBInEOAT ADD 93.14, ADD-S 96.57 at 57.6 FPS; recovery about 210 ms | Shows reviewers expect a reliability signal plus recovery. Its signal is match inliers, not pose uncertainty |
| [RRTrack](https://arxiv.org/abs/2607.23669) (Jul 2026) | Memory-based mask tracker + FoundationPose refiner + DINOv2 dual-bank recovery | 55.2 FPS; event metric: recovered if any of 10 frames after reappearance has ADD-S AR above 0.9 | Gives an event-level recovery metric you can adopt for comparability, then extend with null models |
| [Compact pose graph](https://arxiv.org/abs/2610.11815) (Oct 2026) | Per-edge covariance from inverse Hessian, weighted pose graph, iSAM2 | Covariance weighting adds +1.3 to +4.3 ADD with LoFTR and +0.0 to +2.3 with RoMa v2. Rotation uncertainty vs error Spearman 0.65 to 0.72 | Closest prior art for Hessian covariance in object tracking, but in correspondence space, not a splat render loss. Its overconfidence failure on symmetric objects is your opening |
| [UGS-Loc](https://arxiv.org/abs/2603.16538) (Mar 2026) | Monte Carlo pose particles + Fisher-information-guided PnP for 3DGS camera relocalization | About 30% lower median translation error than GS-CPR on Cambridge Landmarks | Different task (camera, scene splat), but shows particles plus Fisher information are accepted in 3DGS refinement. Cite it; do not claim "first Fisher in 3DGS" |
| [EnDKF](https://arxiv.org/abs/2605.03105) (May 2026) | Ensemble directional Kalman filter on FoundationPose measurements | Synthetic and head digital-twin only; real data left as future work | A filter on top of a pose detector is publishable at a small venue, not at A\*. Use as a baseline idea only |
| [DQ-UKF observability](https://arxiv.org/abs/2605.02054) (May 2026) | Lie-algebraic observability conditions for PnP tracking with dual quaternions | Recovers the collinear-marker degeneracy from control theory | A template for the theory section: observability of a rendered-feature tracker as rank of an information matrix |

Not opened, snippet only: KineFuse (in-hand tracking with haptics), Point2Pose, Real2Sem2Real, ComPose. Treat their claims as unverified.

## D2. Core thesis and the four-part method

Thesis: a splat tracker should be as accurate as POGS where the image constrains the pose, and honest where it does not. Per frame, the method does five things in order. \[I: design proposal, not yet tested\]

1. **Update.** Run the existing POGS Adam optimiser on the DINO plus depth render loss, as now.
2. **Information matrix.** At the converged pose, compute the 6×6 Gauss-Newton matrix H = JᵀWJ of the residuals with respect to a right-multiplied se(3) perturbation. This costs six extra backward passes or finite-difference renders. Set Σ = σ̂² (H + λI)⁻¹, with σ̂² from the residual variance, and store λ\_min and its eigenvector in the object frame.
3. **Ambiguity probe.** If the condition number of H exceeds a threshold, render K poses along the weakest eigenvector and read the loss profile L(φ). Fit a small mixture on that axis (von Mises components on the orbit). A flat profile gives one broad component; a multimodal one gives separate modes. This is what a local covariance cannot do.
4. **Filter.** Run an iterated, error-state (multiplicative) Kalman filter on SE(3) with a constant-velocity prior. Use R\_t = Σ\_t as measurement noise and gate each update with the normalised innovation squared (NIS) against a chi-square bound. The optimiser in step 1 is the iterated measurement update, so the previous belief enters as a prior term in the loss.
5. **Gated compute and recovery.** Choose the iteration budget from the information (few iterations when well constrained, more when not). If NIS stays above bound for m frames, or the mixture entropy is high, launch restarts along the weak axis. Global re-detection à la Opra et al. or RRTrack stays an optional last resort.

What is new relative to neighbours: the information source (Hessian of a foundation-feature splat render loss), the multimodal correction on the weak axis, consistency evidence (NEES, NIS, proper scoring rules), and uncertainty-driven compute. \[I: gap judged from the papers in D1, not an exhaustive search\]

Three falsifiable predictions, each tested in D5:

- **P1.** log λ\_min of the roll direction correlates with signed roll error (Spearman at least 0.5) on the drill, mug and soup can.
- **P2.** A Hessian-adaptive R beats your current fixed-noise KF on recovery, with non-overlapping hierarchical-bootstrap intervals, and Static stays near 0 under the same rule.
- **P3.** The mixture belief gives lower negative log-likelihood of the true rotation than a single Gaussian on near-symmetric objects, and its 90% regions cover about 90% of true poses.

## D3. The Kalman filter: why it did nothing, and what would make it a contribution

Your filter changed almost nothing because it cannot tell when the measurement is weak. A fixed-noise KF trusts the render-and-compare output equally when roll is constrained and when it is not. \[I\]

Your own O4 numbers (Addendum A4) already hint at something useful. Under the post-hoc rule, POGS scores 49, 29, 46, 17 of 50 at +15, -15, +30, -30 degrees, while POGS+KF scores 40, 41, 33, 34. The KF shrinks the sign gap (49 vs 29 becomes 40 vs 41; 46 vs 17 becomes 33 vs 34) and lowers the best settings. That fits the signed-bias explanation, because a smoother damps the part of the response that depends on the sign of the jump. It does not prove it. E1 below is the test. \[M for the counts, I for the reading\]

The gain you want is anisotropic: high where the image constrains the pose, low along weak directions. In covariance form and information form:

```latex
K_t = P_t^{-}\,(P_t^{-} + \Sigma_t)^{-1},\qquad \Sigma_t = \hat\sigma_t^{2}\,(H_t + \lambda I)^{-1}
\qquad\Longleftrightarrow\qquad (P_t^{+})^{-1} = (P_t^{-})^{-1} + H_t/\hat\sigma_t^{2}
```

The information form says the previous belief becomes a quadratic prior term in the render loss. That is a one-line change to the POGS optimiser, so it is cheap to build and easy to ablate.

What turns this from engineering into a contribution:

- **Where R comes from.** The Hessian of a foundation-feature splat render loss. The nearest prior art takes it from point correspondences, and RoPT uses fixed sigmas.
- **Evidence that it is calibrated.** NEES and NIS consistency, reliability curves, and 90% region coverage, not only lower error.
- **An ablation ladder with an oracle.** Static, POGS, fixed-R KF, trace-only R (isotropic), full anisotropic Σ, oracle R built from ground-truth error, then the mixture. The compact pose-graph paper ran the same trace, covariance, oracle ladder and found that direction only helps when the estimated directions are right. Expect that here, and report it. \[V for their finding\]
- **The weak-axis mixture** (D2 step 3), because the local Hessian alone fails exactly on symmetric objects.

Filter choice: an iterated error-state filter on SE(3) with right-invariant error, not an ensemble or a quaternion-sphere filter like EnDKF. Ensembles would multiply the render cost, and with a base rate of 1.5 Hz you cannot afford it. Allow particles only on the 1-D weak axis. Extra cost of the Jacobian is six renders per frame; if each costs about one iteration, that is about a quarter of the 25-iteration budget, which E6 measures. \[I: cost estimate to be measured\]

## D4. Deeper contribution options, ranked

Depth for an A\* reviewer means mechanism, theory and evidence, not a longer component list. The core claim is A plus B. C is the theory section, D is the architectural contribution if E2 says roll observability is the cause, and E is a conditional extra. Novelty and risk are my judgement from a non-exhaustive search. \[I\]

| Option | Idea | Novelty vs D1 | A\* value | Main risk | Fits 12 GB |
| --- | --- | --- | --- | --- | --- |
| A. Hessian-adaptive filter | Steps 2 and 4 of D2 | Medium | High if calibrated | Compact pose-graph paper is close (different space) | Yes, 6 extra renders |
| B. Weak-axis mixture belief | Step 3 of D2 | Medium to high | High: fixes a documented overconfidence | Mixture fit unstable on flat profiles | Yes, 8 to 16 renders only when triggered |
| C. Estimation-theory section | Cramér-Rao style bound from the information matrix; efficiency = error / bound in the observed subspace; roll eigenvalue vs asymmetry of the object | Medium; I found no splat-tracker version | High for a signal-processing profile | A render loss is not a true likelihood, so the bound is heuristic unless the noise model is validated \[U\] | Yes |
| D. Observability-driven feature adapter | Tiny adapter on frozen DINO features, trained on splat-rendered perturbations to raise the smallest eigenvalue of H (E-optimal design) | Medium to high, unsearched \[U\] | High: an architectural fix aimed at the diagnosed cause | May raise roll sensitivity and hurt other directions | Yes, DINO stays frozen |
| E. Splat-conditioned amortised update | Small network predicts the SE(3) update and a matrix-Fisher or Bingham head, trained on self-generated perturbations from the object's own splat | Low to medium (DeepIM, RePOSE, MegaPose, FoundationPose precedent) | Medium; fits CoRL's learning scope | FoundationPose and RoPT already run at 19 to 43 FPS | Yes, small net |
| F. Recovery metric with null models | Event-level metric like RRTrack's, plus Static, no-op reset, equal-compute baselines | Medium | Medium alone; strong as support | Weak if it is the only claim | Yes |
| G. Gripper-aware occlusion | Render and z-buffer the known gripper | Low, scoop risk | Low | Real2Sem2Real and ComPose overlap \[snippet only\] | Yes; ablation only |

Decision rule: build A and B first. Build D only if E2 and E5 show the weak roll eigenvalue explains the error floor. Build E only if A to D work and speed is still the blocker. Drop G to an ablation. Treat the global re-detection modules of Opra et al. and RRTrack as baselines, not as things to reinvent.

## D5. Experiments E1 to E12

Every experiment uses at least 3 independently trained splats, 50 trials per condition, hierarchical bootstrap over scene, seed and trial, Static and equal-compute baselines, and the pre-registered rule reported unchanged beside any new rule. Thresholds below are my pre-registration guesses and should be frozen in the deviation log before running. GPU hours are rough guesses on one RTX 4070. \[I\]

| ID | Question | Protocol | Metric and pass rule | GPU h |
| --- | --- | --- | --- | --- |
| E1 | Which cause makes the 10 to 24° floor and the sign gap? | Addendum C D0: store full SE(3), signed roll about the bit axis, loss at estimate vs ground truth; stratify by sign | Names one of four causes. If sign bias or optimisation, stop and write the methodology note | 3 |
| E2 | Does the Hessian see the roll? | 6×6 spectrum at ground truth and at estimate for every O4 frame | Spearman of log λ\_roll vs absolute roll error at least 0.5; weakest eigenvector within 15° of the bit axis for most frames | 4 |
| E3 | Does an anisotropic R beat a fixed R? | Ladder: Static, POGS, fixed-R KF, trace-R, full Σ, oracle R, full Σ + mixture, on O4 | Full Σ beats fixed-R on the post-hoc rule with non-overlapping 95% intervals, and Static at most 2/50 | 25 |
| E4 | Is the belief calibrated? | NEES, NIS, reliability curve of predicted σ vs error, coverage of 90% regions | Coverage 85 to 95% in well-observed directions; report failures on weak ones | 5 |
| E5 | Is the ambiguity real, and does the mixture handle it? | The 2×2 from Q5 (decal, gripper) plus mug and soup can | Mixture NLL of true rotation below a single Gaussian's, interval excluding zero, on near-symmetric objects | 30 |
| E6 | Can information gate compute? | Success vs Hz for 5, 10, 15, 25 iterations vs gated; reuse O6 setup | Gated matches fixed-10 success at a higher rate. A 5 Hz claim counts only with at least 95% success | 6 |
| E7 | Does it hold on real data? | YCBInEOAT (9 videos, 5 objects) and HO3D; build splats from CAD renders if no scan views | Beats POGS on every object; within 1 ADD point of FoundationPose on the symmetric can, or better; calibration (E4) still holds | 80 |
| E8 | Does it degrade gracefully with motion? | Frame-rate downsampling 1×, 2×, 4×, 8×, 16×, the RoPT protocol | Degrades more slowly than POGS | 30 |
| E9 | Does it recover from loss? | Lost-reappear events (RRTrack-style, success within 10 frames), time-to-recover survival curve, null models; occlusion by the gripper | Excess recovery over null with interval excluding zero | 15 |
| E10 | How efficient is the tracker? | Compare empirical error in the observed subspace with the bound from the information matrix | Report the ratio; no pass rule until the noise model is validated | 5 |
| E11 (if E2, E5 say roll) | Can an adapter raise roll observability? | Train a small adapter on frozen DINO features to maximise λ\_min over splat-rendered perturbations | λ\_roll up at least 2× with other eigenvalues not down by more than 10%, and roll error down | 30 |
| E12 (if speed still blocks) | Can a network replace the iterations? | Self-supervised from the object's own splat; heteroscedastic head | Same success at 2× the rate of gated POGS | 40 |

Two reuse notes. First, E6 can reuse the O6 numbers you already have: base POGS 1.5 Hz at 100% success, 10 iterations 3.2 Hz at 100%, 5 iterations 5.6 Hz but only 68 to 100% success. Second, for E7 the published YCBInEOAT numbers (FoundationPose 93.1 ADD, 96.4 ADD-S; RoPT about 92.0 and 95.1) come from other protocols; rerun baselines yourself where code exists and say so where you cannot. \[V for the published numbers, as read in the papers\]

## D6. Timeline and go/no-go gates (Nov 23, 2026 to May 2027)

This extends Addendum C rather than replacing it: days 1 to 4 stay as written, and the 40-day plan becomes the first phase of a longer one. The venue that fits the evidence is decided at G3, not now. \[I\]

1. **Nov 23 to 26: E1.** Gate G0 on Nov 26: which cause explains the floor and the sign gap? Sign bias or optimisation means stop the method work and write the methodology note first.
2. **Nov 27 to Dec 10: E2, E3, start E4.** Implement the information matrix and the iterated filter ladder. Gate G1 on Dec 10: P1 (Spearman at least 0.5) and P2 (full Σ beats fixed-R with non-overlapping intervals). Both pass: continue. Neither passes: pivot to a metric, benchmark and analysis paper (arXiv plus a workshop or RA-L).
3. **Dec 7 to 22, in parallel: real-data splats.** Build splats for YCBInEOAT from CAD renders and run POGS end to end. Gate G2 on Dec 22: POGS runs on at least 3 real sequences with metrics computed.
4. **Dec 11 to Jan 3: E5, E6, E9.** Weak-axis mixture, the 2×2 intervention, compute gating, recovery events.
5. **Jan 4 to Feb 7: E7, E8, E10, optional E11.** Real benchmark runs against POGS and FoundationPose, motion stress, efficiency analysis. Gate G3 on Feb 7: go if P1 to P3 hold in simulation, and E7 shows a gain on the symmetric subset with calibration intact on real data.
6. **Feb 8 to Mar 1: write.** Supervisor reads a full draft by Feb 15. Post to arXiv at submission.
7. **Mar to May: strengthen.** Conditional E11 and E12, more sequences, a robot demo if an arm exists. Target CoRL 2027 or ICRA 2028 if G3 was no-go.

| Venue | Deadline | Status and fit |
| --- | --- | --- |
| IROS 2027 | Mar 1, 2027 | From the earlier review; first realistic target |
| ICCV 2027 (Hong Kong, Oct 2 to 8) | Mar 8, 2027 projected | Dates and city confirmed by the CVF site; the March 8 deadline comes from a third-party tracker that says the call was not yet released \[U\] |
| CoRL 2027 | Not announced | CoRL 2026 closed May 28, 2026, so expect late May. Its call lists "probabilistic learning and representation of uncertainty in robotics" and "learning for robot vision", which fits this thesis \[V for the 2026 call\] |
| CVPR 2027 | Nov 16, 2026 | Closed |

If your thesis or exams take January, E7 slips first, and the first realistic submission becomes CoRL 2027.

## D7. Reviewer attack map

The likeliest A\* objections and the experiment that answers each. \[I\]

| Likely objection | Defence | Evidence |
| --- | --- | --- |
| "A Kalman filter on a known tracker is incremental." RGBTrack, ROFT and RoPT already filter | The claim is the information source, the calibration and the weak-axis mixture. The ladder shows a fixed-R filter alone gives about nothing | E3, E4 |
| "Hessian covariance for object tracking exists." The compact pose-graph paper | Cite it. Differences: foundation-feature splat render loss, and a fix for its documented overconfidence on symmetric objects | E4, E5 |
| "Not better than FoundationPose or RoPT." | Claim a narrow, testable win: calibrated belief, symmetric in-hand objects, compute. Report every object where you lose | E7, E8 |
| "Simulation only." | Real benchmark plus lost-reappear events; robot-kinematics ground truth if an arm exists | E7, E9 |
| "Post-hoc metric, dependent trials." | Pre-registered rule first and unchanged, hierarchical bootstrap, null models, deviation log | E1, E3 |
| "The mixture is ad hoc." | The probe axis is the Hessian's weakest eigenvector, scored by a proper scoring rule (NLL), with an oracle-mode ablation | E5 |
| "The 5 Hz claim is cherry-picked." | Report the whole success-vs-rate curve and count 5 Hz only with at least 95% success | E6 |

Cut order if time slips: E12, E11, the gripper ablation, extra datasets, then the ICCV target in favour of CoRL. Never cut: the pre-registered result, Static and equal-compute baselines, per-frame storage, calibration (E4), and one real benchmark.

## D8. Tasks for Claude Code, in order

Hand these over one at a time and ask for tests plus a short result note after each.

1. Extend the per-frame record: full SE(3) estimate and ground truth, signed axis-angle error in the object frame, loss at estimate, loss at ground truth (Addendum B).
2. Write `info_matrix(pose, frame)`: the 6×6 Gauss-Newton matrix of the DINO plus depth residual for a right-multiplied se(3) perturbation, with regulariser λ and eigenpairs in the object frame. Unit-test it against finite differences and a synthetic case with a known null direction.
3. Run E1 and E2 on the stored O4 runs. Report the signed-error stratification, the spectrum, and Spearman of log λ\_roll vs roll error.
4. Implement the iterated error-state filter with the previous belief as a prior term in the loss, NIS gating, and ladder switches: fixed R, trace R, full Σ, oracle R.
5. Implement the weak-axis probe: K renders along the weakest eigenvector, loss profile, von Mises mixture fit, and NLL scoring of the true rotation.
6. Add metrics: NEES, NIS, reliability curve, 90% coverage, NLL, hierarchical bootstrap, time-to-recover survival curve, and the RRTrack-style event metric.
7. Train at least 3 splats per scene; freeze the protocol and write the deviation log; tag the commit before any E3 run.
8. Add a real-data loader for YCBInEOAT and HO3D, build splats from CAD renders when no scan views exist, and compute ADD and ADD-S AUC.
9. Check what fits in 12 GB: FoundationPose first. Check whether RoPT or Opra et al. released code; if not, report their published numbers as such. \[U: I did not find released code for either\]
10. Keep the daily status check and the independent recomputation of every table from `row.json` (Addendum C).

Questions for you or your supervisor, still open: is an arm or depth camera available for 30 minutes of real capture, and is the 80% target the paper's servoing success or a tracking-recovery rate?

## D9. What this pass revises, sources and caveats

**Revisions to the earlier review.**

- Q2 item 1 said I found no Hessian-based analysis for object tracking. The compact pose-graph paper does derive Hessian covariance for object tracking, in correspondence space. The gap is narrower than Q2 stated: foundation-feature splat render loss, symmetry-aware belief, and calibration evidence.
- Q1 gave FoundationPose tracking as about 32 Hz. Two 2026 papers measured 42.9 and 41.3 Hz on their own GPUs, so quote it per GPU, not as one figure.

**Read in PDF text this pass.** [RoPT](https://arxiv.org/abs/2610.28962), [Opra et al.](https://arxiv.org/abs/2607.23468), [RRTrack](https://arxiv.org/abs/2607.23669), [compact pose graph](https://arxiv.org/abs/2610.11815), [UGS-Loc](https://arxiv.org/abs/2603.16538), [EnDKF](https://arxiv.org/abs/2605.03105), and the first pages of the [DQ-UKF observability paper](https://arxiv.org/abs/2605.02054).

**Snippet only, not verified.** KineFuse (2607.14842), Point2Pose (2604.10415), ComPose (2605.23523), Real2Sem2Real, NovaSplat (2610.13309), GRC-Pose (2609.39116).

**Venues.** [CVPR 2027 call](https://cvpr.thecvf.com/Conferences/2027/CallForPapers) and [ICCV 2027 dates](https://www.thecvf.com/?p=137) are from search snippets of the official sites. The ICCV deadline is from a [third-party tracker](https://www.opencurious.com/ai-conference-deadlines/iccv-2027) that says it is projected. CoRL 2027 has no call yet; the [CoRL 2026 call](https://sites.google.com/corl.org/corl2026/contributions/call-for-papers) is the only reference. Check each official page before planning around a date.

**Caveats.**

- The D2 method, the thresholds in D5 and the GPU-hour figures are my proposals, untested on your code. P1 to P3 may fail, and G1 is there to find out early.
- Published numbers in D1 use different protocols and hardware. They show what reviewers will compare against, not a leaderboard.
- Novelty judgements rest on about 14 searches this pass on top of the earlier 25. Absence of prior work is the hardest thing to verify, so run the two extra searches proposed in Q2 and read the compact pose-graph and RoPT papers yourself before writing related work.
- I have not seen your repo, so file names and costs in D8 follow what Addendum B reported.
