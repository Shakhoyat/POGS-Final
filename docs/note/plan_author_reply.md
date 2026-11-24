# Plan addendum: branches on the POGS author's reply

Written 2026-11-24, the day the data request was sent to the POGS first author (ledger 8.AR). There was no plan
file in `docs/note/`; this file is an addendum to `docs/PLAN.md` and the rev. 2 research plan, which it does not
change. No branch below may delay the RSS submission.

## Fixed (not changed by any reply)
- Paper claim, title ("Shadows Bias Feature-Field Object Trackers Through an Implicit Silhouette Term"), results,
  and the pre-registered order of the generality results (G1 P1–P5, G1b T1, G2 Q1–Q3).
- **RSS 2027 Stage 1 extended abstract:** due 2026-12-04 23:59 AoE (2026-12-05 ≈ 18:00 Dhaka). **Submit by
  2026-12-03.** A reply, or no reply, never moves this date.
- Extension priority: (1) official-code reproduction, (2) D-SIL as pre-registered (`dsil_preregistration.md`,
  still gated on supervisor Q2), (3) real data, (4) explaining the dynamic failure without the background term.
  Real data that also runs the official code counts toward (1).
- Pre-registration: every new test is written in `docs/note/` and committed before it runs, with a fixed pass rule.
  A reply from the author never changes a pre-registered criterion.
- Fallback venue: IROS 2027 (deadline 2027-03-01) if RSS rejects at Stage 1 (2027-02-05) or Stage 2 (2027-02-26).
  No dual submission.
- GPU rules (≤ 3 GPU processes, no GPU work during latency timing, cap stated in each pre-registration).
  Large artefacts stay local and out of git.

## Timeline and triggers
| Date | Event | Branch |
|---|---|---|
| 2026-11-24 | Request sent (items 1–4) | – |
| on reply | Data shared: items 1 + 2 (± 3) | A |
| on reply | Some items shared | B (per item) |
| on reply | Questions, objections or an offer to collaborate | D (can run together with A/B) |
| 2026-12-01 23:59 Dhaka | Cut-off for real-data results in the RSS abstract. Complete R1–R3 results → "preliminary real-data validation"; otherwise "planned validation" | A/B |
| 2026-12-03 | RSS Stage 1 abstract submitted | all |
| 2026-12-15 | Decline, or no reply → one nudge (draft below, sent only by the user) and start own captures | C |
| 2027-02-05 / 02-26 | RSS Stage 1 / Stage 2 decisions; rejection → IROS by 2027-03-01 | all |

## Shared predictions R1–R3 (form copied from G1 P2, P5, P4)
Each branch copies these into its own pre-registration (`g3_real_data_preregistration.md` for A/B,
`g3c_own_capture_preregistration.md` for C) and fixes the details listed there **before** any loss is computed on
real frames.

| | Prediction | Simulation analogue | Proposed threshold (fixed in the pre-registration) |
|---|---|---|---|
| R1 | Full-ROI DINO loss: drill roll argmin biased in the same sign as in simulation | P2 (−10/−10/−5°) | median signed argmin over selected frames ≤ −5° (simulation sign convention: right-hand roll about the drill-bit axis, object +x) |
| R2 | Loss restricted to the object mask removes most of that bias | P5; B1e | median abs masked argmin < 0.5 × median abs full-ROI argmin |
| R3 | Frames with the gripper's shadow in the ROI show a larger roll argmin shift than frames without | P4 (shift ≥ 4°) | difference of medians (shadow − no shadow) ≥ 4° toward the bias, with a bootstrap 95% CI that excludes 0; needs ≥ 10 frames per group or R3 is reported "not testable" |

Fixed alongside: grid ±20° at 1° and κ over ±5° (as G1); a frame-selection rule written before any loss is
computed; shadow labels from RGB only (annotated, committed, then the sweeps run); views and objects as available.
A sign mismatch caused by a different light geometry is reported as an R1 fail plus a description, not
re-interpreted after the fact. The loss code is the G1 code path (`experiments/g1_generality.py::_curves`) unless
the official tracker is used, which the pre-registration then says.

## Task 0 (done 2026-11-24): the email's numbers
See ledger 8.AR0. **Result: incorrect as worded.** The POGS paper reports neither splat PSNR nor grouping IoU.
"34.0 vs. 34.1" compares our rev2 retrain (34.02 dB, 2026-11-17) with our own 2026-11-10 run (34.1 dB). Both are of
the simulated pick-and-place scene, which has no drill. The drill-scene splat is 33.54 dB (33.54–33.64 over three
seeds). "Similar grouping IoU" is also our run vs our run. Our results are not "close to the paper" in general:
under the paper's roll-free rule we match its 30° result (39 vs 40/50) but fall short at 15° (32 and 25 vs 48/50).
The correction draft is under Branch C/D.

## Branch A: substantial data shared (items 1 and 2, with or without 3)
1. Storage: `data/external/pogs_real/` (local, git-ignored; `.gitignore` entry added 2026-11-24). Follow any data
   agreement to the letter (no redistribution, no frames in the paper beyond what it allows).
2. Write `docs/note/g3_real_data_preregistration.md` (R1–R3 above, frame-selection rule, thresholds, GPU cap) and
   commit it **before opening the streams**. A file listing to check formats is allowed beforehand. Looking at
   frames is not.
3. Train the scene with the **official POGS code** (`third_party/POGS-author-github`) from the item 1 scans; run the
   official tracker on the item 2 streams. This also covers extension (1) on real data. Static loss sweeps for R1–R3
   use the official splat; a re-implementation splat on the same scans is an exploratory second column.
4. If item 3 is present (GT or end-effector logs): evaluate under both rules (full rotation, roll-free) plus
   `Static (no motion)`, exactly as in the simulation tables, with `metrics/recovery_report.py`.
5. Paper: add a "Real-data validation" section. Results complete by the 2026-12-01 cut-off → "preliminary
   real-data validation" in the RSS abstract; otherwise listed under planned validation.

## Branch B: partial data shared (do only what each item supports)
| Shared | Enables | Not claimed |
|---|---|---|
| Item 4 only (checkpoints) | Official tracker on the released `test_data`; compare its loss sweeps with the re-implementation's on the same frames. **Reproduction check, extension (1)** | Not real-data validation of R1–R3 |
| Item 1 only (scans + poses) | Official-code splat; static sweeps at the scan poses (truth known there); test **R1, R2** | R3 (no gripper, no shadow frames) |
| Item 2 without item 3 | Depth-only ICP as a pseudo-reference for frame alignment; test **R3 only** (relative shift with vs without shadow) | No absolute pose error; the paper states this limitation, and that ICP uses the same depth sensor as the tracker's depth term |
| Items 1 + 3 or 2 + 3 | As above plus both-rule scoring where GT covers the frames | – |

Each case still needs its own pre-registration (or the Branch A file with the untestable predictions marked "not
testable with the shared data") before the data is looked at.

## Branch C: decline, or no reply by 2026-12-15
1. At most one follow-up: the nudge below, sent by the user on 2026-12-15, not by Claude. Nothing after that.
2. Own capture (one afternoon, protocol below). Pre-register R1–R3 for it in
   `docs/note/g3c_own_capture_preregistration.md` before the capture.
3. Paper: the real-data check uses our own captures, not the original robot setup.

### Capture protocol (≈ 4 h)
Equipment: one power drill; table; printed AprilTag board (tag36h11, e.g. 4 × 6 tags on A3, tag size measured
with a ruler); phone or RGB-D camera on a tripod and by hand; one desk lamp with a known position (and a dimmer
or 3 distances); a matte card or box on a stand as the occluder (gripper proxy) that casts a shadow into the ROI.
The drill **never moves** after placement: the scan pose is the truth, as in Branch B item 1.
1. **Setup (30 min).** Tape the AprilTag board flat; place the drill on it, bit axis horizontal and pointing along a
   board axis; mark lamp positions L1–L3 on the table (left, back, right at ≈ 45° elevation, ≈ 60 cm away). Fix
   camera exposure, white balance and focus (locked; a phone pro mode or the camera's manual mode).
2. **Scan (45 min), lamp OFF, room light diffuse.** 50 views: two rings of 20 (elevation ≈ 30° and ≈ 55°, radius
   ≈ 50 cm, every 18°) plus 10 higher views. The board is visible in every view. Poses from COLMAP
   (`ns-process-data`), scaled and aligned to the board frame with the tag corners (Umeyama). The tag pose gives an
   independent check: report the reprojection error. Reject the scan and redo it if the median is > 2 px.
3. **Test frames (90 min), tripod at 3 fixed viewpoints V1–V3** (elevation 30°, azimuths 0°, 90°, 270° from
   the bit side). At each viewpoint, 5 frames per condition: lamp off; lamp at L1/L2/L3; at the lamp position that
   casts the occluder's shadow onto the ROI, occluder in / out × 3 lamp strengths (dimmer levels or distances
   30/60/90 cm). Camera pose per frame from the board. ≈ 3 × (4 + 6) × 5 = 150 frames.
4. **Dose measure (computed from images only, before any loss):** shadow strength
   s = 1 − I_shadow / I_lit in a fixed board patch next to the drill, inside the ROI.
5. **Write-down (15 min):** lamp positions (tape measure), tag size, camera settings, any frame discarded and why.
Analysis: official POGS splat on the scan; static roll sweeps on each test frame at the known pose (R1, R2);
shadow-in vs shadow-out and a dose-response of argmin vs s (R3, plus the P4-form Spearman, exploratory if not
pre-registered as a rule).

### Nudge (draft, for the user to send on 2026-12-15 if there is no reply; no em dashes)
> Subject: Re: POGS real captures request
>
> Hi Justin,
>
> A short follow-up on my earlier email, with one correction. I wrote that our splat PSNR is 34.0 vs. 34.1 with
> similar grouping IoU, and called that close to the paper. That was wrong: the POGS paper reports neither number.
> Both values are from our own simulator runs (a retrain vs. our earlier run, on a pick-and-place scene). Our
> drill-scene splat is 33.5 dB and the drill's grouping IoU against simulator labels is 0.96. Under your roll-free
> rule our tool-perturbation result matches your 30 degree result (39 vs. 40 of 50) but is lower at 15 degrees.
>
> If sharing even one drill sequence is possible, it would help us check the roll behaviour on real data. If not,
> no problem at all; we will validate with our own captures.
>
> Best regards,
> Shujon

(If the user sends the correction earlier, see Task 0 / Branch D. That message is then the one follow-up and the
2026-12-15 nudge is dropped.)

## Branch D: questions, objections or an offer to collaborate
1. Claude drafts a reply in `docs/note/author_reply_draft_<date>.md`. Nothing is sent by Claude.
2. If the finding is questioned: share privately the evaluation-rule results (both rules + Static, E0/G0) and
   the mechanism results (B1e–B1g, G1 P2/P4/P5, G2 Q3), and ask what the recovery rule computes in their code
   (which rotation is compared, whether roll about the bit axis is free, the success window). Never change a
   pre-registered criterion because of the reply. Record any disagreement in the paper's Limitations, as theirs
   vs ours, with the evidence.
3. If the author contributes data or analysis: note here an offer of co-authorship (decided by the user and the
   supervisor). The RSS submission stays anonymous: no names, no acknowledgments, no links that identify either
   group. Credit goes in the camera-ready only.
4. A reply that is a question only (no data) does not change the 2026-12-15 trigger for Branch C unless it
   promises data with a date.

## Log of changes to this file
- 2026-11-24: created (branches A–D, triggers, R1–R3 form, capture protocol, nudge draft, Task 0 result).
