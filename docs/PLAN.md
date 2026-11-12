# Execution plan: supervisor requirements → final report

Owner: Md.Shakhoyat Rahman Shujon (first author). Working copy: `E:\POGS-Final\POGS-Final`
(WSL: `/mnt/e/POGS-Final/POGS-Final`). Status as of 2026-11-13; tick boxes as steps finish.

## Ground rules (applied to every step and to the final report)

1. **Report what happened, not what we hoped.** Negative and null results go in the report,
   with the same prominence as positive ones.
2. **Every number is regenerable.** Each table and figure comes from a script in this repo,
   reading cached results under `data_results/` or `model/foundation_study/artefacts/`. No
   hand-typed numbers in `main.tex` except paper reference values (marked as such).
3. **Baselines and spread.** Every table has the `Static (no motion)` baseline; every mean has
   a seed std or a 95% interval. At least 3 seeds per stochastic condition, 5 where we claim a
   difference between methods.
4. **Like with like.** No comparison of our simulation numbers with the paper's physical-robot
   numbers. Speed claims compare equal iteration budgets.
5. **Disclose deviations.** Anything that differs from the paper, the release or the brief is
   listed (see `Claude.md` "Known deviations") and repeated in the report.
6. **Credit.** Per-path authorship in `CONTRIBUTIONS.md`; new work is committed under the first author.
7. **Environment recorded.** Python / CUDA / PyTorch / GPU / driver printed into every result file.

## Phase 0: Foundation (environment and sanity), ~2 days

- [ ] 0.1 Create `.venv` in WSL with **Python 3.12**, PyTorch 2.14, CUDA 13.0
      (`requirements.txt`; build gsplat with `tools/setup_gsplat.sh`).
- [ ] 0.2 `pytest -q tests`: all 17 must pass after the Model/Data restructure.
- [ ] 0.3 Regenerate the benchmark data (`python -m experiments.v1_base_pogs.run_all`); it retrains
      the checkpoint (not in git). Confirm PSNR ≈ 34.1 dB and grouping IoU within ±0.03 of
      `docs/RESULTS_reproduction.md`. If they differ, record by how much and why.
- [ ] 0.4 Add `env.json` (versions, GPU, driver) to every run directory.

**Exit:** tests green; base case reproduces 100% ADD-S success at 25 iterations.

## Phase 1: O1 object-grouping splats (with / without object), ~2 days

- [ ] 1.1 `metrics/gaussian.py`: render full / scene-minus-object / object-only for each cluster
      from the 35 capture views.
- [ ] 1.2 Metrics per object: alpha IoU vs GT mask; **leakage** (object-only opacity outside the
      GT mask); PSNR of scene-minus-object vs a GT render without the object (the generator can
      produce that render).
- [ ] 1.3 Figure: 5 objects × 3 renders. Table: IoU / leakage / PSNR per object.
- [ ] 1.4 Explain the residual red patch seen at the mug's scan position (dashboard figure).

**Exit:** table + figure for all 5 objects; soup can (IoU 0.67) discussed honestly.

## Phase 2: O3 tool perturbation recovery to 30° (splats + pose + KL), ~4 days

- [ ] 2.1 Confirm the protocol with the supervisor before running (one message): step size 5°,
      axis (tool's own vertical?), hold time per step, and which KL (we propose the per-Gaussian
      Gaussian-to-Gaussian KL below).
- [ ] 2.2 Scene: add a tool object (drill-like, asymmetric) to `data/scene_generator/`; trajectory
      0 → +5 → … → +30 (state *p*) → … → 0 (state *y*), static camera, 5 Hz.
- [ ] 2.3 `metrics/gaussian.py::kl_divergence(x, y)`: opacity-weighted mean of
      KL(N(μ_x, Σ_x) ‖ N(μ_y, Σ_y)) over corresponding Gaussians (same indices; rigid groups).
      Unit tests: identity → 0; known translation → closed form.
- [ ] 2.4 Report, per step: position error, rotation error, ADD-S, and KL(x‖current); final KL(x‖y).
      Also the largest step size that still recovers (sweep 5 / 10 / 15°).
- [ ] 2.5 Methods: POGS and POGS + KF + check, 3 seeds each.

**Exit:** curve of error and KL vs commanded angle, there and back; numeric recovery verdict.

## Phase 3: O4 in-grasp recovery rate (target 80%), ~3 days

- [ ] 3.1 Gripper holds the tool; perturb in-grasp by ±15° and ±30° (CW/CCW), as in paper Table II.
- [ ] 3.2 50 trials per setting (randomised start pose and seed). Success = tool tip within
      3 cm and 5° of GT within N frames after the perturbation (N fixed in advance and stated).
- [ ] 3.3 Recovery rate with Wilson 95% interval, per method. Compare with 80% honestly:
      "met", "not met", or "inconclusive (CI spans 80%)".

**Exit:** table of 4 settings × methods; paper's 48/50 and 40/50 shown as reference only.

## Phase 4: O6 5 Hz tracking, ~2 days

- [ ] 4.1 Per-stage profile of one tracking step (DINO forward, render, backward, optimiser).
- [ ] 4.2 Candidates, each measured on all 54 offline cases plus the 30 Hz protocol:
      fp16 DINO; 10–15 iterations + KF latency compensation; MobileCLIPv2 only if alignment is on
      the per-frame path (it is not during tracking; state this).
- [ ] 4.3 Pick the cheapest configuration with ≥ 5 Hz and report its accuracy loss vs 25 iterations.

**Exit:** a configuration at ≥ 5 Hz with its success rate, or a clear statement that none exists.

## Phase 5: O5 foundation-model latency, completing the table, ~2 days

- [ ] 5.1 Re-benchmark all models on the target stack (PyTorch 2.14 / CUDA 13) with the existing
      `model/foundation_study/shared/bench.py` protocol (20 warm-up, 100 timed, fp32 + fp16).
- [ ] 5.2 Add SigLIP2 standalone; Detic standalone (not only from logs).
- [ ] 5.3 VFM-DETR: record "no public weights" with the date checked and the source.
- [ ] 5.4 Optional: wire MobileCLIPv2 into train/track and run 5 seeds on pick-place.

**Exit:** one latency table, one input size, one stack, every model either measured or explained.

## Phase 6: Final report and presentation, ~3 days

- [ ] 6.1 Extend `report/supervisor/make_figures.py` with phase 1–5 outputs; no manual numbers.
- [ ] 6.2 Rewrite results sections; move "planned" items to "results" only when the exit
      criterion is met. Keep a "Limitations" section listing every unmet target.
- [ ] 6.3 Internal review: a second person checks every number against its source file.
- [ ] 6.4 Build the PDF (`tectonic` or Overleaf), plus a 10-slide summary for the meeting.
- [ ] 6.5 Tag the commit used for the report (`git tag report-v1`) so the PDF is reproducible.

## Timeline (indicative)

| Week | Phases |
|---|---|
| 1 | 0, 1 |
| 2 | 2 (after supervisor confirms the protocol), start 3 |
| 3 | 3, 4 |
| 4 | 5, 6 |

## Open questions for the supervisor

1. KL divergence: per-Gaussian (our proposal) or between rendered feature/colour distributions?
2. Recovery: step-wise (±5°) only, or also a single 30° jump as in the paper?
3. Is synthetic-only evaluation acceptable, or is a real capture expected for the final report?
