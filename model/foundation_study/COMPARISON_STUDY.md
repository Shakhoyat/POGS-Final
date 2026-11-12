# POGS model-substitution study — comparison analysis

*Independent analysis of every artefact under `artefacts/`, written 2026-11-08. Numbers are recomputed from the raw per-frame JSON, the training logs and the scene ground truth, not copied from `RESULTS.md`.*

---

## 1. Verdict in one paragraph

**No model combination in this study is measurably better than the stock POGS baseline, and the study as it stands cannot tell them apart.** Three independent checks each show that the tracking metric carries no usable signal on 9 of the 10 scenes in the primary matrix: errors are 1.4–54× worse than simply assuming the object never moved, error is uncorrelated with the difficulty axis the ladder was built to vary (ρ = −0.13 to +0.13 against ρ = +1.00 for the null), and re-running the *identical* pipeline at three seeds moves the number by 605 mm — 10–36× the largest between-row difference. Separately, only **one** of the four substitution rows actually substituted anything in the train/track path: rows 01 and 02 ran the baseline's own DETIC + DINOv2 + OpenCLIP, and row 04 never trained at all. The results that *are* trustworthy are the deterministic ones — latency, VRAM, detector IoU — and those say the baseline's choices are the right ones: every alternative is slower in its role, or cannot satisfy POGS's mask contract, or both.

---

## 2. The central correction: what each row actually ran

This is the single most important thing to understand before reading any table. The study's title matrix describes five *intended* combinations. The configs in the training logs show what was really loaded.

| row | intended combination | **what actually ran in train + track** | genuine substitution? |
|---|---|---|---|
| **00** | DETIC + DINOv2 ViT-S/14 + CLIP ViT-B/16 | DETIC + DINOv2 ViT-S/14 + `OpenCLIPNetwork ViT-B-16/laion2b` | reference (stock) |
| **01** | LocateAnything3D + dino.txt (both roles) | **identical to row 00** — DETIC + DINOv2 + OpenCLIP ViT-B-16 | **no** |
| **02** | RF-DETR + DINOv3 + MobileCLIPv2 | **identical to row 00** — `Detic Detector` ran over all 33 frames; `OpenCLIPNetwork ViT-B-16` | **no** |
| **03** | VFM-DETR + SigLIP2 | DETIC + DINOv2 + **`SigLIP2Network` (`siglip2-base-patch16-224`, 768-d)** | **yes — alignment only** |
| **04** | YOLO + DINOv2 + CLIP | never trained, never tracked — `bench` only | n/a |

Evidence, per row:

- Row 01's and 02's configs both print `network=OpenCLIPNetworkConfig`, `clip_model_type='ViT-B-16'`, `clip_model_pretrained='laion2b_s34b_b88k'` — byte-identical to row 00. Row 02's log contains 35 `Detic Detector` progress records and the `Detic_LCOCOI21k...` checkpoint name; RF-DETR never ran in the pipeline.
- Row 03's config prints `network=SigLIP2NetworkConfig` / `SigLIP2Network` and `siglip2-base-patch16-224`. This one is real.
- Median training iteration time confirms it from the other direction: 65.4 / 64.8 / 64.5 / 65.2 ms per iteration for rows 00 / 01 / 02 / 03 on the v2 pendulums, and 110.0 / 108.3 / 107.8 ms on pickplace. Four "different" pipelines cannot land within 1.5% of each other unless they are the same pipeline.

**To the study's credit, this is already disclosed** — each row's `RESULTS.md` carries an explicit "Which models ran in train/track" banner saying the numbers are a baseline rerun. But the merged `RESULTS.md`, `RESULTS_v2.md` and `RESULTS_handle.md` tables do **not** carry that caveat, and they present rows 01 and 03 side by side as if they were different pipelines. Anyone reading only the merged table will draw a false conclusion.

**Consequence that works in the study's favour:** rows 01 and 02's train/track runs are unintentionally a *replication experiment*. Row 01 gives nine extra samples of the baseline on the same nine scenes at the same seed. That is what makes section 6's variance analysis possible, and it is the most valuable measurement in the whole dataset.

---

## 3. Evidence inventory

451 files under `artefacts/`. What exists, by measurement type:

| measurement | row 00 | row 01 | row 02 | row 03 | row 04 |
|---|---|---|---|---|---|
| weight probe | yes | — | yes | yes | yes |
| standalone latency (`bench`) | pickplace only | dino.txt only | 12 scenes | — | 10 scenes |
| LocateAnything3D detection | — | **17 scenes** | — | — | — |
| DETIC capture detect | pickplace | pendulum_v2_30 | — | — | — |
| detector IoU on authors' real frames | **yes (14 frames)** | — | — | — | — |
| training runs (4000 it) | 23 | 13 | 2 (1 truncated at 88%) | 12 (1 truncated) | 0 |
| tracking runs | 24 | 13 | 1 | 11 | 0 |
| determinism check | attempted, `match: null` | — | — | — | — |

Scene families, which the merged reports blur together:

| family | scenes | what it is |
|---|---|---|
| `pendulum_05…60` (v1) | 9 | cylinder spans **0.95%** of frame on 85% black background. The registry itself records these as degenerate: the cylinder was "narrower than one DINOv2 patch on a black background, which left the DINO-only tracker nothing to match." |
| `pendulum_v2_05…60` | 9 | same geometry and ground truth, cylinder thickened to **4.2%** of frame, textured object and background, camera noise. **This is the valid ladder.** |
| `pendulum_v2_01` | 1 | **null control** — 1° swing, 2.6 mm peak motion |
| `pickplace`, `pickplace_v2` | 2 | mug carried 465 mm by a gripper that occludes it; mug is 2.4% of frame |
| `pickplace_handle` | 1 | same trajectory, gripper holds only the handle — occlusion-free control |
| `pickplace_seed7/13` | 2 | baseline pickplace re-run at seeds 7 and 13 — the variance probe |
| `real_14` | 1 | authors' real ZED capture, detector IoU only, never trained |

---

## 4. Deterministic results — these are trustworthy

Latency carries no seed variance, and the measurements prove it: across 9–10 repeats of the same model on different scenes at the same resolution, the median moves by less than 1 ms (SD 0.09–0.79 ms for the feature and alignment models, 2.9 ms for RF-DETR). Protocol is sound — 20 warmup iterations, 100 timed, `cudnn.benchmark`, `cuda.synchronize()` both sides, fp32 and fp16 measured separately. RTX 4070, 12,282 MB, torch 2.6.0+cu118.

### 4a. Per-role comparison at matched resolution (fp16 median, 1280×720 scene)

This is the honest way to read the latency data, because the pipeline totals in `RESULTS.md` are not comparable — the baseline's total **excludes DETIC** (it only loads inside the POGS environment), while rows 02 and 04 include their detector.

| role | baseline | alternative | result |
|---|---|---|---|
| **detection** | DETIC ≈ **148 ms** *(see 4b)* | RF-DETR 51.2 ms · YOLO11x 39.8 ms | alternatives **3–4× faster** — but neither can serve POGS |
| **features** | DINOv2 ViT-S/14 **19.6 ms**, 876 MB | DINOv3 ViT-B/16 34.1 ms, 1,627 MB | baseline **1.74× faster, 1.9× lighter** |
| **alignment** | CLIP ViT-B/16 **24.0 ms**, 246 MB | MobileCLIPv2 20.0 ms, 258 MB · SigLIP2 *(see 4c)* | MobileCLIPv2 **1.20× faster** — the one free win, never wired in |
| features+alignment fused | 43.7 ms combined | dino.txt **170.8 ms**, 7,062 MB | **3.9× slower, 8× the VRAM** |

### 4b. DETIC's missing latency, recovered

The study states DETIC has no latency figure because it only loads inside POGS. It can be recovered from the training logs, which print the detector's own progress bar over the 33 capture frames:

| row | scene | rate | per frame |
|---|---|---|---|
| 02 | pendulum_v2_05 | 7.80 it/s | 128 ms |
| 01 | pendulum_v2_30 | 6.73 it/s | 149 ms |
| 03 | pendulum_v2_30 | 6.51 it/s | 154 ms |
| 01 | pickplace_v2 | 6.63 it/s | 151 ms |

**DETIC ≈ 128–154 ms per frame in-pipeline (fp32, including post-processing).** This closes the study's one acknowledged measurement gap and reverses the direction of the latency story: the baseline's detector is the slowest component in the entire study, 3–4× slower than RF-DETR or YOLO. The baseline is not fast — it is *capable*, and it pays for that in detection latency.

### 4c. The real cost of the one genuine substitution

SigLIP2's standalone latency was never benchmarked (`bench.py` did not run for row 03, and the Windows-side prefetch failed on a `huggingface-hub` version conflict). But its in-pipeline cost is in every row-03 training log, in the image-embedding pass:

| scene | row 00 (CLIP ViT-B/16) | row 03 (SigLIP2-base) | cost |
|---|---|---|---|
| `pendulum_v2_30` (640×480) | 31.88 it/s | 11.00 it/s | **2.90× slower** |
| `pickplace_v2` (1280×720) | 17.28 it/s | 7.22 it/s | **2.39× slower** |

The substitution also forced a second patch: POGS hardcodes the GaussianField CLIP head to 512 dimensions, and SigLIP2-base is 768-d, so both the encoder choice and the head width had to be env-gated. Training iteration time is unaffected (65.2 vs 65.4 ms) — the cost is confined to the one-time embedding pass.

**So the only real substitution in the study costs 2.4–2.9× on its stage and, per section 6, buys no measurable accuracy.**

### 4d. VRAM, and a hard limit

| model | fp32 peak | fp16 peak | note |
|---|---|---|---|
| dino.txt | **11,353 MB** | 7,062 MB | **92% of the 4070's 12,282 MB** at fp32 |
| DINOv3 ViT-B/16 | 3,011 MB | 2,116 MB | |
| DINOv2 ViT-S/14 | 1,448 MB | 876 MB | baseline |
| LocateAnything3D | — | 7,938–10,909 MB | measured via its own harness |

dino.txt at fp32 takes 5,866 ms per forward pass — 299× the baseline's feature model. It is only viable at fp16, and even then needs 7 GB before POGS has allocated a single gaussian. On this GPU, row 01's intended combination is not merely slow, it is close to infeasible.

---

## 5. Detector capability and accuracy

### 5a. The mask contract decides most of the matrix before any run

POGS passes a `custom_vocab` to its detector and consumes per-instance 2D masks. This is not a preference; `full_images_datamanager.py` feeds those masks into the CLIP/lerf loss and asserts `"No masks found"` without them.

| detector | open-vocab | instance masks | can serve POGS | measured |
|---|---|---|---|---|
| **DETIC** | yes | yes | **yes** | IoU 0.617 on real frames; ~148 ms |
| RF-DETR | **no** (COCO-closed) | yes | no | 51.2 ms |
| YOLO11x | **no** | **no** (boxes) | no | 39.8 ms |
| LocateAnything3D | yes | **no** (3D/2D boxes) | no | IoU 0.46–0.97; 0.43–1.64 s |
| VFM-DETR | unknown | unknown | **cannot be evaluated** | no weights exist |

Two of the five intended detection swaps are ruled out by capability regardless of speed, one is ruled out by having no weights, and the fourth (LocateAnything3D) emits boxes. **The detection column of this study has exactly one viable entrant, and it is the baseline's.** That conclusion is robust — it does not depend on any noisy measurement.

### 5b. DETIC on the authors' real capture — the best-measured accuracy number in the study

Row 00 only. 14 real ZED frames, stock LVIS vocabulary, 120 ground-truth labels.

| metric | value |
|---|---|
| mean IoU | **0.617** |
| median IoU | **0.779** |
| labels with IoU > 0.75 | 70 / 120 (58%) |
| labels with IoU > 0.50 | 79 / 120 (66%) |
| labels with IoU < 0.05 (misses) | 21 / 120 (18%) |
| coverage | 0.975 |
| best / worst | 0.969 / 0.000 |

The mean/median gap (0.617 vs 0.779) is the signature of a bimodal result: DETIC either segments an object well or misses it outright, with little in between. **This number is absent from the merged `RESULTS.md`, which reports "not run" for every row** — because `compare.py` reads `validate/detector_real_iou.json` (still a stub saying "not run — DETIC inference pending") while the real result sits in `real_14/detector_real_iou.json`. A reporting-path bug, not a missing measurement.

### 5c. LocateAnything3D — well measured, and the clearest difficulty signal in the study

Row 01's `LOCATEANYTHING_DETECT.md` covers 17 scenes. This is the only place where a measurement responds cleanly to the intended difficulty axis:

| scene family | first-box IoU | precision | s / detection | peak GPU |
|---|---|---|---|---|
| `pendulum_v2_01…30` (thick cylinder) | **0.961–0.972** | 1.00 | 0.43–0.52 | 7.9 GB |
| `pendulum_05…10` (thin, v1) | 0.906–0.911 | 1.00 | 0.44–0.45 | 7.9 GB |
| `pendulum_15…60` (thin, v1) | **0.459–0.701** | 0.50–0.75 | 0.44–0.57 | 7.9 GB |
| `pickplace` (mug, occluded) | 0.818 | 0.875 | 1.59 | 10.9 GB |
| `pickplace_v2` | 0.693 | 0.75 | 1.64 | 10.9 GB |

Two real findings here: the v1 thin-cylinder scenes degrade sharply with swing angle (0.91 → 0.46) while the v2 thick-cylinder scenes do not degrade at all (0.96–0.97 flat), which independently confirms the registry's own diagnosis that v1 is too small to be a valid test; and occlusion costs ~0.13–0.28 IoU on pickplace. But at 0.43–1.64 s per detection and ~8–11 GB, LocateAnything3D is 3–11× slower than *all three* other detectors combined, and it still emits no masks.

---

## 6. Tracking — why the headline table cannot be used

The merged reports rank rows by `centroid_err_mm_mean`. Four independent checks show that ranking is noise.

### 6a. Check 1 — the null predictor beats almost every run

The metric is displacement error, `|(est_i − est_0) − (gt_i − gt_0)|`. A trivial tracker that outputs "the object never moved" scores exactly the mean ground-truth displacement. That is the floor any real tracker must beat. Computed per scene from the ground truth in each `tracking.json`:

| row | v2 pendulum scenes worse than the null |
|---|---|
| 00 baseline | **8 / 9** |
| 01 | **9 / 9** |
| 03 | **9 / 9** |

Worked example: on `pendulum_v2_05` the cylinder moves 13.1 mm at peak, mean 9.0 mm. Row 00 reports 446.5 mm, row 01 488.2 mm, row 02 208.8 mm, row 03 37.5 mm. The *best* of those is 4× the motion being tracked; the worst is 54×. Only `pendulum_v2_60` (ratio 0.96) reaches parity with the null, and only for the baseline.

The null control proves the mechanism. `pendulum_v2_01` swings 2.6 mm; the baseline reports **42.5 mm** of error. **The tracker has a resting noise floor around 40 mm, and random excursions to 450 mm.** Every pendulum motion in the ladder (13–150 mm) sits inside that noise band.

`pickplace` is the one family where tracking genuinely works, because the mug moves 465 mm — well clear of the floor:

| run | error | vs null (267 mm) |
|---|---|---|
| 03 `pickplace_v2` | **47.4 mm** | 0.18 ✓ |
| 00 `pickplace` (opt_iters 30) | 50.8 mm | 0.19 ✓ |
| 00 `pickplace` (opt_iters 15) | 63.4 mm | 0.24 ✓ |
| 01 `pickplace` | 85.1 mm | 0.32 ✓ |
| 01 `pickplace_v2` | 146.7 mm | 0.55 ✓ |
| 00 `pickplace_seed13` | 159.8 mm | 0.60 ✓ |
| 00 `pickplace_v2` | 280.9 mm | 1.05 ✗ |
| 00 `pickplace_seed7` | 668.7 mm | 2.50 ✗ |

### 6b. Check 2 — seed variance dwarfs every between-row difference

The baseline was run on `pickplace` at three seeds. Same pipeline, same scene, same `opt_iters`, same data:

| seed | error |
|---|---|
| 42 | 63.4 mm |
| 13 | 159.8 mm |
| 7 | **668.7 mm** |

Mean 297.3 mm, **SD 325.2 mm, range 605.3 mm, CV 1.09.** The standard deviation exceeds the mean.

Set that against the between-row differences on the v2 ladder (paired by scene, n = 9):

| comparison | mean difference | paired t(8) | significant? |
|---|---|---|---|
| row 01 − row 00 | +59.8 mm | +0.92 | no (needs \|t\| > 2.31) |
| row 03 − row 00 | +16.8 mm | +0.22 | no |
| row 03 − row 01 | −43.0 mm | −0.49 | no |

Rows 00 and 01 are the *same pipeline*, so their +59.8 mm difference is a direct measurement of pure run-to-run noise. Row 03's difference from the baseline (+16.8 mm) is **smaller than that noise floor**. The genuine SigLIP2 substitution produced a change indistinguishable from re-running the same code twice.

Row means on the v2 ladder, for completeness: row 00 **202.5 ± 143.8**, row 01 **262.3 ± 153.0**, row 03 **219.3 ± 147.6**, null **50.4 ± 32.4**. The three rows' confidence intervals overlap almost entirely, and all three sit far above the null.

### 6c. Check 3 — error does not track the difficulty axis

The ladder exists so that peak displacement grows 13 → 150 mm and a row that "holds at 60°" can be distinguished from one that only holds at 5°. Spearman correlation between swing angle and error:

| row | ρ(angle, error) |
|---|---|
| 00 | **−0.13** |
| 01 | **+0.07** |
| 03 | **+0.13** |
| null predictor | **+1.00** (by construction) |

All three rows are flat. If the tracker were following the cylinder at all, error would respond to how far the cylinder travels. It does not. The ladder's independent variable has no effect on the dependent variable — which is the definition of a measurement that is not measuring.

### 6d. Check 4 — failed runs are reported as results

`artefacts/00_.../sweep.txt` records `pendulum_15:FAILED`, `pendulum_25:FAILED`, `pendulum_40:FAILED`. Those three scenes have `lost_at_frame` set in their tracking JSON (10, 17, 9). The merged `RESULTS.md` prints their errors — 280.7, 312.5, 158.6 mm — as ordinary measurements, with no marker.

Worse, the inverse case produces the study's apparent best result. `01/pendulum_05` reports **9.0 mm**, the lowest error anywhere in the dataset — and `lost_at_frame: 0`. The track was lost at the very first frame, so the estimate never moved, so the error collapsed onto the null predictor's score exactly (9.0 mm null, 9.0 mm reported, ratio 1.00). **The best number in the study is a total tracking failure.** Any ranking that sorts on this column will put failures at the top.

Runs with `lost_at_frame` set: `00/pendulum_15` (10), `00/pendulum_25` (17), `00/pendulum_40` (9), `01/pendulum_05` (0), `01/pickplace_handle` (85), `03/pickplace_handle` (48). Six of 48 tracking runs, none flagged in the merged tables.

### 6e. What the occlusion control shows

`pickplace_handle` removes the occlusion (gripper holds only the handle) and should therefore be *easier* than `pickplace`. Both runs on it lost track — row 01 at frame 85 (199.5 mm), row 03 at frame 48 (547.1 mm), and row 03's run split the mug across 2 HDBSCAN clusters. The baseline never ran it. So the occlusion hypothesis is untested: the control is harder than the condition it controls for, which points at the clustering/seeding step rather than at occlusion.

---

## 7. Scene-set comparison

| family | baseline mean | verdict |
|---|---|---|
| `pendulum_v1` (9) | 261.0 ± 130.9 mm | invalid by the registry's own account — object thinner than one DINOv2 patch |
| `pendulum_v2` (9) | 202.5 ± 143.8 mm | 22% better than v1, but still 4× the null. The thickening helped the *detector* (LocateAnything IoU 0.46→0.97) and did not fix the *tracker* |
| `pendulum_v2_01` (control) | 42.5 mm | establishes the ~40 mm noise floor |
| `pickplace` / `pickplace_v2` | 50.8–280.9 mm | **the only family with real tracking signal** |
| `pickplace_handle` | lost in both runs | control fails |

The v1 → v2 regeneration was the right call and it is well documented, but it only moved the baseline from 5.2× the null to 4.0× the null. The problem is not the scenes' texture. It is that pendulum displacements are smaller than the tracker's noise floor, and no amount of texturing changes that.

---

## 8. Per-combination verdicts

**Row 00 — DETIC + DINOv2 ViT-S/14 + CLIP ViT-B/16 — the one to keep.**
Only combination that satisfies the mask contract end to end and the only one with a real-imagery accuracy number (IoU 0.617 / median 0.779). Cheapest feature model (19.6 ms, 876 MB). Its detector is the study's slowest component at ~148 ms, which is the honest cost of open-vocabulary instance masks. Tracking numbers are not usable, for the reasons above.

**Row 01 — LocateAnything3D + dino.txt — rejected on capability and cost, measured well.**
LocateAnything3D localises excellently on the v2 scenes (IoU 0.961–0.972, precision 1.00) but emits boxes, not masks, and takes 0.43–1.64 s per detection at 8–11 GB. dino.txt is 170.8 ms fp16 (3.9× the two baseline models combined) at 7 GB, and 5,866 ms / 11.3 GB at fp32 — 92% of this GPU. Its train/track numbers are a baseline rerun; read them only as the variance estimate they accidentally are.

**Row 02 — RF-DETR + DINOv3 + MobileCLIPv2 — one free win, never claimed.**
RF-DETR is COCO-closed: it cannot take POGS's `custom_vocab`, so its 3× speed advantage is unusable. DINOv3 is a straight regression in its role — 1.74× slower and 1.9× more VRAM than DINOv2 ViT-S/14 — and needed a real patch (stride 16 vs the hardcoded 14, plus `transformers > 4.44`). **MobileCLIPv2 is the one genuine improvement anywhere in the study: 20.0 ms vs CLIP's 24.0 ms at equal VRAM, loading through the existing `open_clip` call with no adapter.** It was never wired into train/track, so its effect on accuracy is unmeasured. This is the cheapest open experiment in the folder.

**Row 03 — (VFM-DETR blocked) + SigLIP2 — the only real substitution, and it cost 2.4–2.9× for nothing measurable.**
VFM-DETR has no weights in existence, so detection and features fell back to stock and only the alignment encoder changed. The change is real and required patching two hardcoded assumptions (encoder choice and the 512-d CLIP head). Measured cost: 2.90× slower embedding at 640×480, 2.39× at 1280×720. Measured benefit: +16.8 mm mean difference from the baseline across nine scenes, t(8) = 0.22 — smaller than the noise between two runs of the same code. Its two eye-catching wins (`pendulum_v2_05` 37.5 mm, `pickplace_v2` 47.4 mm) are single samples from a distribution with SD > 300 mm.

**Row 04 — YOLO + DINOv2 + CLIP — correctly excluded.**
Boxes only, closed vocabulary; cannot serve POGS. Fastest detector measured (39.8 ms fp16, 82 MB). Latency-only row, as designed, and the fastest-detector result is the right thing to record about it.

---

## 9. What would make this comparison decisive

Ordered by information gained per GPU-hour.

1. **Fix the two reporting bugs first — zero GPU cost.** Point `compare.py` at `real_14/detector_real_iou.json` so the measured IoU 0.617 stops showing as "not run", and propagate `lost_at_frame` into the merged tables so failures are not ranked as successes (`01/pendulum_05`'s 9.0 mm is currently the study's best number and is a frame-0 failure).
2. **Carry the per-row "which models actually ran" banner into the merged tables.** It already exists in each row's own `RESULTS.md`. Without it the merged table reads as a four-way pipeline comparison when three of the four columns are the same pipeline.
3. **Drop the pendulum ladder from the accuracy comparison, or fix the amplitude.** Nine scenes whose motion is below the tracker's noise floor cost ~2 hours per row and return noise. Either raise displacement well above ~450 mm (where `pickplace` demonstrably works) or report the pendulums as a determinism regression track only — which is what the registry says they were for.
4. **Replicate on `pickplace` instead.** With SD ≈ 325 mm and a target of resolving ~50 mm differences, n = 1 cannot work. Five seeds per row on the one scene that beats the null is ~6 GPU-hours and would, for the first time, let a between-row claim be made.
5. **Then run the two experiments the study set up but never finished:** MobileCLIPv2 in the alignment slot (the only component measured faster than the baseline's, and it needs no adapter), and row 04's train/track to confirm the box-only failure mode empirically rather than by argument.
6. **Measure reconstruction quality.** No PSNR was recorded for any row — the logs track only `Train Rays/Sec`, and `determinism.json` has `match: null`, so the determinism check never completed either. A 768-d alignment encoder's effect would surface in the CLIP/lerf loss and relevancy quality long before it surfaced in a centroid. The study is currently blind to the axis where SigLIP2 is most likely to differ.

---

## 10. Summary table

| | row 00 | row 01 | row 02 | row 03 | row 04 |
|---|---|---|---|---|---|
| serves POGS mask contract | **yes** | no | no | **yes** | no |
| really substituted in train/track | reference | **no** | **no** | **yes** (alignment) | not run |
| detection latency | ~148 ms | 0.43–1.64 s | 51.2 ms | ~148 ms | **39.8 ms** |
| feature latency / VRAM | **19.6 ms / 876 MB** | 170.8 ms / 7,062 MB | 34.1 ms / 1,627 MB | 19.6 ms / 876 MB | 19.6 ms / 876 MB |
| alignment latency | 24.0 ms | (dino.txt, fused) | **20.0 ms** | 2.4–2.9× CLIP | 24.0 ms |
| detector IoU, real frames | **0.617** | — | — | — | — |
| tracking vs baseline | reference | +59.8 mm, t = 0.92 | n/a | +16.8 mm, t = 0.22 | n/a |
| **overall** | **keep** | reject (capability + cost) | reject, but **test MobileCLIPv2** | no measurable gain, 2.4–2.9× cost | correctly excluded |

**Bottom line: the baseline's model choices survive the study. DETIC is the only detector that can do the job, DINOv2 ViT-S/14 is the cheapest feature model measured, and the only alternative that beats CLIP on speed (MobileCLIPv2) was never actually tested on accuracy. The study's deterministic measurements are sound and well protocolled; its accuracy comparison needs the pendulum scenes replaced and n > 1 before it can rank anything.**
