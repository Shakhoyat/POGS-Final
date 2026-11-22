# Execution plan: supervisor requirements → final report → paper

Owner: Md.Shakhoyat Rahman Shujon (first author). Working copy: `E:\POGS-Final\POGS-Final`
(WSL: `/mnt/e/POGS-Final/POGS-Final`). Revised 2026-11-13 after re-reading
`Robotic-Manipulation-v1-requirement-supervisor.pdf` against the repository. Tick boxes as steps finish.

## 1. Requirement → current state (verified 2026-11-13)

| ID | Supervisor requirement | Target | State in repo | Gap |
|---|---|---|---|---|
| O1 | Splats with / without one object; check that the group really holds that object's Gaussians | qualitative + quantitative | HDBSCAN grouping, per-object IoU 0.67–0.98; object-only renders in dashboard videos | No scene-minus-object render, no metric, soup can weak (0.67) |
| O2 | Structure: **Model** (detection, feature extraction, alignment, POGS), **Data** (scene generation), **Metrics** (Gaussian, Pose); Python 3.12, CUDA 13.0, PyTorch 2.14 + torchmetrics | layout + env | `model/pogs`, `model/foundation_study`, `data/scene_generator` exist | **No `metrics/` package** (pose metrics live in `model/pogs/evaluate.py`, grouping metrics in `grouping.py`); no Gaussian metrics; no torchmetrics; `.venv` absent; system Python is 3.14, no 3.12; workstream B on torch 2.6/cu118 |
| O3 | Tool perturbation, recovery angle 30°: x → +5,+10,… (p) → −5,−10,… back (y); compare splats y vs x, same for pose; KL divergence | 30° | Drill object + `tool_tip_frame` exist in `data/scene_generator/objects.py`; only retracted `legacy/servoing.py` | Trajectory, KL metric, experiment all missing |
| O4 | In-grasp tool perturbation recovery rate | 80 % | — | Grasp-perturbation scene, trials, rate with CI |
| O5 | Latency of every 2D model in the table (Detic, LocateAnything3D, RF-DETR, VFM-DETR, YOLO / DINOv2, dino.txt, DINOv3 / CLIP, MobileCLIPv2, SigLIP2) | per model | 7 models benchmarked standalone (fp32/fp16/VRAM) on torch 2.6/cu118 | Detic and SigLIP2 only from logs; LocateAnything3D own harness; VFM-DETR no weights; not on target stack; no per-stage time inside the tracker |
| O6 | Tracking frequency | 5 Hz | 1.9 Hz (25 it), 4.4 Hz (10 it), 7.1 Hz (5 it) | No setting reaches 5 Hz at useful accuracy |

## 2. Research-integrity issues to settle before running anything

These are the points a strict A*/Q1 reviewer would raise. Each one has a decision recorded here.

1. **Index-matched KL is the pose error in another form.** POGS moves each group rigidly, so
   KL(N(μᵢˣ,Σᵢˣ) ‖ N(μᵢʸ,Σᵢʸ)) over the same index i is a deterministic function of the residual
   rigid transform. It adds no evidence beyond pose error. *Decision:* report it (it is what the
   supervisor asked for, and it is covariance-aware), state this equivalence explicitly, and add
   one KL that is **independent of ground truth**: the KL between the rendered and observed
   per-pixel feature/colour distributions inside the object ROI. That second KL can serve as a
   recovery-confidence signal without GT, which is a possible contribution.
2. **Step-wise ±5° is easier than the paper's test.** The paper perturbs by a sudden 15°/30° jump
   (Table II). *Decision:* run both the step-wise protocol (supervisor) and the single-jump
   protocol (paper), and label each.
3. **Targets come from a physical robot.** 30°, 80 % and 48/50 / 40/50 are real-robot numbers.
   Our results are synthetic. *Decision:* show the paper's values as reference only, never as a
   head-to-head comparison.
4. **Pre-register the protocols.** Success thresholds, number of trials, seeds, frames allowed for
   recovery and the analysis (Wilson CI) go into `experiments/protocols/*.md`, which is committed
   **before** the first run. Any later change is logged with its reason.
5. **Ground truth for evaluation only** (`Claude.md` rule 3): grouping, selection and initialisation
   never read GT.
6. **Workstream B accuracy numbers are not usable**: n = 1, unseeded, `lost@N` cells, and rows 01/02
   ran the baseline models (report Table "Study matrix"). Use B for latency only, unless rows are re-run.
7. **Retracted results stay retracted.** Nothing from `legacy/` is cited.
8. **Novelty.** A re-implementation plus a synthetic benchmark is unlikely to be enough for A*/Q1.
   Candidate contributions, to agree with the supervisor (see §5):
   (a) latency-aware real-time evaluation of POGS with Kalman prior + latency compensation that
   reaches 5 Hz; (b) GT-free splat-consistency (KL) signal for detecting and recovering lost tracks;
   (c) controlled benchmark of POGS failure modes (rotation, speed, occlusion, depth, shape);
   (d) foundation-model substitution cost/capability analysis. Most robotics venues will also
   expect **at least one real-world sequence**.
9. **Credit.** Md Nayeem (workstream A) and Dadhichi (workstream B) wrote most of the existing
   code (`CONTRIBUTIONS.md`); authorship of the report and paper must reflect that.

## 3. Ground rules (every step, every table)

1. Report what happened. Negative and null results get the same prominence as positive ones.
2. Every number is regenerable from a script reading cached results; no hand-typed numbers in
   `main.tex` except paper reference values, marked as such.
3. Every table has the `Static (no motion)` baseline; every mean has a seed std or 95 % CI;
   ≥ 3 seeds per condition, ≥ 5 where a difference between methods is claimed; rates get Wilson CIs.
4. Like with like: equal iteration budgets for speed claims; no sim vs real comparisons.
5. Deviations from paper, release or brief are listed (`Claude.md` "Known deviations") and repeated in the report.
6. `env.json` (Python, CUDA, PyTorch, GPU, driver, git SHA) is written into every result directory.
7. `.venv/bin/python -m pytest -q tests` after every change to `model/pogs/` or `metrics/`.

## 4. Phases

### Phase 0: Environment and structure (O2), ~2 days
- [ ] 0.1 Install Python 3.12 (`uv python install 3.12` or deadsnakes), create `.venv`, install
      PyTorch 2.14 + CUDA 13.0 wheels, `torchmetrics`, then `requirements.txt`; build gsplat
      (`tools/setup_gsplat.sh`). Update the `requirements.txt` header (it currently says Python 3.14).
- [ ] 0.2 `pytest -q tests`: all 17 pass.
- [ ] 0.3 Create `metrics/` with `metrics/pose.py` (ADD, ADD-S, translation/rotation error, success
      rates, moved from `model/pogs/evaluate.py` and re-exported there so nothing breaks) and
      `metrics/gaussian.py` (Phase 1–2). Use `torchmetrics` for PSNR/SSIM/LPIPS. Tests in `tests/test_metrics.py`.
- [ ] 0.4 Regenerate the benchmark (`python -m experiments.v1_base_pogs.run_all`); confirm PSNR
      ≈ 34.1 dB and grouping IoU within ±0.03 of `docs/RESULTS_reproduction.md`. Record any difference.
- [ ] 0.5 `env.json` writer in `experiments/pipeline.py`.

**Exit:** Python 3.12 / CUDA 13.0 / PyTorch 2.14 env, tests green, base case reproduces 100 % ADD-S success.

### Phase 1: O1 splats with / without object, ~2 days
- [ ] 1.1 `metrics/gaussian.py`: render (a) full scene, (b) scene minus group k, (c) group k only,
      from the 35 capture views.
- [ ] 1.2 Per object: alpha IoU of (c) vs GT mask; leakage = opacity of (c) outside the GT mask;
      completeness = GT-mask pixels not covered by (c); PSNR/SSIM of (b) vs a GT render of the scene
      without the object (generator renders it); count of object Gaussians assigned to other groups
      (via `assign_gt_ids`, evaluation only).
- [ ] 1.3 Figure 5 objects × 3 renders; table of the metrics with 3 training seeds.
- [ ] 1.4 Explain the residual red patch at the mug's scan position and the soup can's 0.67 IoU.

**Exit:** figure + table for all 5 objects, failures explained.

### Phase 2: O3 perturbation recovery to 30° (splats + pose + KL), ~4 days
- [ ] 2.1 Send the supervisor the questions in §5 and write `experiments/protocols/o3.md`; commit before running.
- [ ] 2.2 Generator: extend `object_frame` beyond yaw (it is yaw-only today) so the drill can rotate
      about any axis; trajectories (i) step-wise 0 → +5 → … → +30 (p) → … → 0 (y), hold h frames per
      step; (ii) single jump 0 → ±15 / ±30 → hold. 5 Hz, static camera, axes: tool vertical, plus one horizontal.
- [ ] 2.3 `metrics/gaussian.py::kl_gaussians(x, y)`: opacity-weighted mean KL over matched Gaussians
      (formula in `report/supervisor/main.tex` §Remaining work); unit tests: identity → 0, pure
      translation → closed form, symmetry check of the symmetrised variant.
- [ ] 2.4 `metrics/gaussian.py::kl_render(obs, render)`: GT-free KL between observed and rendered
      feature/colour histograms in the ROI; test that it rises when the object is lost.
- [ ] 2.5 Per step: position error, rotation error, ADD-S, KL(x_gt-posed ‖ tracked), KL_render;
      final KL(x ‖ y). Sweep step size 5/10/15° and find the largest angle that still recovers.
- [ ] 2.6 Methods: Static, POGS (paper, release), POGS + KF + check; 5 seeds.

**Exit:** error/KL vs commanded angle (out and back), recovery verdict per method with CI.

### Phase 3: O4 in-grasp recovery rate (target 80 %), ~3 days
- [ ] 3.1 `experiments/protocols/o4.md` (pre-registered): ±15°, ±30°, CW/CCW, 50 trials per setting,
      random start pose + seed, success = tool tip within 3 cm and 5° of GT within N frames (N fixed now).
- [ ] 3.2 Gripper holds the drill (benchmark gripper, scale for the drill); perturb in grasp.
- [ ] 3.3 Rate + Wilson 95 % CI per method; verdict "met" / "not met" / "inconclusive (CI spans 80 %)".
      Paper's 48/50 and 40/50 as reference only.

- [x] 3.4 Resource use (decided 2026-11-22): accuracy experiments (O3/O4, v1–v3 accuracy) may share the GPU, so
      `tools/o4_parallel.sh N` runs O4 as N shards (`--shard k --nshards N`, case index % N, disjoint `row.json`
      files, then `--aggregate`). N=3 fits 12 GB VRAM (4 shards filled it). Latency stages (o5, o6, realtime)
      still run alone: a shared GPU would corrupt their timings. WSL RAM 28 GB (`C:\Users\User\.wslconfig`).
      Reboot cause: Windows power-off (Start menu 16:25) and one unclean reboot (09:13), not WSL; keep the PC awake.

**Exit:** 4 settings × methods table.

### Phase 4: O6 5 Hz tracking, ~3 days
- [ ] 4.1 Per-stage profile of one step (DINO forward, render, backward, optimiser, ROI crop).
- [ ] 4.2 Candidates on all 54 offline cases + 30 Hz protocol: fp16/bf16 DINO, `torch.compile`,
      smaller MATCH_RESOLUTION, 10–15 iterations + KF latency compensation, early stopping on loss
      plateau. Baselines with equal iteration budgets.
- [ ] 4.3 Choose the cheapest configuration ≥ 5 Hz; report its accuracy cost vs 25 iterations
      and the hardware.

**Exit:** a ≥ 5 Hz configuration with success rate and CI, or a clear statement that none exists.

### Phase 5: O5 foundation-model latency table, ~2 days
- [ ] 5.1 Port `model/foundation_study/shared/bench.py` to the target env; re-measure every model
      (20 warm-up, 100 timed, fp32 + fp16, one input size 1280×720, same GPU).
- [ ] 5.2 Standalone Detic and SigLIP2 benches; LocateAnything3D with the same protocol if it fits in 12 GB.
- [ ] 5.3 VFM-DETR: state "no public weights", with the date checked and the source URL.
- [ ] 5.4 Separate training-time models (detection, alignment) from the per-frame tracking path
      (DINO only), so the table connects to O6.
- [ ] 5.5 Optional: DINOv3 / MobileCLIPv2 in train+track, 5 seeds, ADD-S on pick-place.

**Exit:** one table, one stack, every model measured or explained.

### Phase 6: Report, ~3 days
- [ ] 6.1 Extend `report/supervisor/make_figures.py` with Phase 1–5 outputs.
- [ ] 6.2 Update the status table; move an item to "done" only when its exit criterion is met;
      "Limitations" lists every unmet target.
- [ ] 6.3 Second person checks every number against its source file.
- [ ] 6.4 Build PDF + 10-slide summary; `git tag report-v1`.

### Phase 7 (paper, after supervisor sign-off)
- [ ] 7.1 Choose the headline contribution (§2.8) and venue (ICRA/IROS → RA-L, or T-RO/RAS for Q1 journals).
- [ ] 7.2 Real-world sequence(s): tool scan + perturbation with an RGB-D camera.
- [ ] 7.3 Related work since 2025 (dynamic/persistent GS tracking, FoundationPose-style trackers) and
      a comparison against at least one external tracker on the same data.
- [ ] 7.4 Release code + data; reproducibility checklist.

## 5. Questions for the supervisor (send before Phase 2)

1. KL: per-Gaussian matched KL (pose-equivalent under rigid motion), image-space KL (GT-free), or both?
2. Recovery: step-wise ±5° only, or also the paper's single 15°/30° jump? Rotation axis?
3. Recovery success criterion and time window (we propose 3 cm / 5° within N frames).
4. Is synthetic-only acceptable for the report, and is a real capture expected for the paper?
5. Python 3.12 is requested; is Python 3.14 (current, working) acceptable if 3.12 wheels for
   PyTorch 2.14 / CUDA 13.0 cause trouble?

## 6. Timeline (indicative)

| Week | Phases |
|---|---|
| 1 | 0, 1; send §5 questions |
| 2 | 2, start 3 |
| 3 | 3, 4 |
| 4 | 5, 6 (report to supervisor) |
| 5+ | 7 (paper) |

## 7. Critique of this plan (2026-11-17) and the changes it forces

| # | Weakness | Change |
|---|---|---|
| C1 | **Contribution is decided last (Phase 7).** Which experiments matter depends on the paper's claim; without it Phases 2–5 can produce a report but not a paper. | Decide the headline after Phase 2/4 pilot numbers, not after Phase 6. Candidates stay as in §2.8; default = latency-aware tracking (O6) + GT-free loss-of-track signal. |
| C2 | **Blocked on answers nobody was asked for.** §5 questions were never sent; Phase 2 waited on them. | Run with the stated defaults (both KLs, both protocols, 3 cm / 5°), mark protocols "provisional", send the questions with the first results. |
| C3 | **O3 as specified cannot fail informatively.** ±5° steps, no occluder, exact poses: likely 100 % everywhere, and "recovery angle 30°" is then a ceiling we set, not a limit we found. | Extend the jump sweep past 30° (45, 60, 90) until recovery breaks, so the reported angle is a measured boundary. |
| C4 | **One trained model.** Every tracking number is conditional on a single checkpoint; seeds vary trajectories only. | State it as a threat; O1 and the O3 headline rows get 3 training seeds if GPU time allows, otherwise the limitation is written in the report. |
| C5 | **GPU budget never estimated.** The old run cache is invalid (checkpoint lost), so "all 54 cases × 13 methods" ≈ 700 runs ≈ 8–10 h plus ~1 min rendering per sequence. | Re-run only what a report table needs: 54 cases × {Static, POGS} first; KF / iteration sweeps on the factors where they are claimed. |
| C6 | **`kl_render` is RGB; POGS tracks DINO features.** A colour histogram is a weak, viewpoint-blind proxy. | Keep as baseline signal; compare against the tracker's own DINO loss as a loss-of-track detector (AUROC on lost vs. tracked frames) before claiming anything. |
| C7 | **Index-matched KL is redundant** (§2.1) and heavy-tailed. | Reported because requested; never used as evidence separate from pose error. |
| C8 | **O5 port may not be feasible.** Detic needs detectron2, unlikely to build on Python 3.12 / torch 2.14. | Time-box to 1 day; models that cannot run on the target stack keep their old-stack number, labelled with the stack. |
| C9 | **Timing on WSL + NTFS, GPU shared with the desktop.** Hz numbers carry I/O and contention noise. | Preload frames to RAM for timing runs; report hardware, repeat timing 3×, give spread. |
| C10 | **External validity.** Primitive-based renderer, exact camera poses, synthetic depth noise. No real sequence until Phase 7. | Unchanged scope for the report; stated in Limitations. A real RGB-D capture is the first Phase 7 item, not optional for a robotics venue. |
| C11 | **Process overhead.** Ledger rows per sub-step and per-session logs cost more than they return. | Ledger keeps: resume block, one line per step, deviations. No narrative log. |
