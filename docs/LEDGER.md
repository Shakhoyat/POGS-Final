# Progress ledger

Tracks execution of `docs/PLAN.md`. One row per plan step; one dated log entry per working
session. Update both in the same commit as the work they describe.

**Status keys:** `todo` · `doing` · `done` (exit criterion met, evidence linked) · `blocked`
(reason in Notes) · `dropped` (reason in Notes, decision logged) · `changed` (step altered from
the plan; the reason is logged).

**Evidence** is a file path, commit SHA or result directory, never a claim alone. A step is
`done` only when its evidence is in the repository.

## ▶ Resume here
- **2026-11-22 (research plan rev. 2):** all 12 papers in `ref-paper/` read in full and renamed `YEAR_VENUE_Author_topic.pdf` (index `ref-paper/README.md`; notes `docs/literature/reading_notes.md`). Plan rewritten: `POGS PhD Direction Candid Review, Novelty Gap, and 6-Month Plan.md` (rev. 1 kept in `docs/literature/plan_v1_2026-11-22_superseded.md`). **Key finding R1:** POGS Sec. V-B leaves roll about the drill-bit axis unconstrained, so its 3 cm / 5° rule ignores bit-axis roll; our pre-registered O4 rule uses full geodesic rotation. The O4 "not met" is therefore not comparable until E0 (paper-faithful re-score) runs; it needs full SE(3) per frame (T1), which O4 does not store yet. Work is tracked in **Phase 8** below. 8.T1 done; **8.E0 queued** (`tools/e0_after_realtime.sh`, starts when `realtime.done` appears); 8.T2 done (`model/pogs/information.py`); next **8.T3/E1–E2 analysis scripts** on the E0 runs once `e0.done` exists. GPU rule: E0 runs only after the pipeline's `realtime` stage (latency timings must run alone).
- **Unattended pipeline:** `tools/run_pipeline.sh --detach | --status | --stop | --only <stage> | --redo <stage>`. Order: tests → repro (0.4) → o1 → o3_smoke → o6 (4.x) → o3 → o4_smoke → o4 (3.x) → o5 (5.x, own venv `~/venvs/pogs-o5`) → v1 → v2 → realtime → v3. State + logs: `data_results/_pipeline/` (`<stage>.done|failed|log`, `status.tsv`). A failed stage is skipped and retried on the next start. Started 2026-11-17 05:07; estimated ≥ 12 h for everything (O4 alone ≈ 3.5 h: 200 sequences + 600 tracking runs).
- **2026-11-22:** WSL RAM raised to 28 GB. O4 now runs as 3 parallel GPU shards (`tools/o4_parallel.sh 3`, plan 3.4); the pipeline was relaunched at 18:1x and resumes from cached rows. Reboots were Windows power-offs, not WSL (plan 3.4).
- **O4 result (2026-11-22, 200 cases x 4 methods, provisional protocol):** POGS recovers 4/50 (+15°), 14/50 (-15°), 12/50 (+30°), 6/50 (-30°); POGS+KF 4/13/10/8; release 0-4/50; Static 0/50. Verdict vs 80%: **not met** in every setting (Wilson upper bounds <= 0.42). Failures are mostly rotation: tip error is 1.5-5.6 cm but rotation error at 2 s is 9-20°, so the 5° rule decides; the protocol is not frozen, so check rotation-vs-tip before reporting (an honest negative result, not a bug fix target). o5 done, v1 running.
- **O4 diagnosis (2026-11-22, POST-HOC, labelled as such; `experiments/o4_posthoc_relative.py` -> `data_results/o4_ingrasp/posthoc_relative.md`):** rotation error is already 10-24° (median) BEFORE the perturbation, so the pre-registered 5° rule measures steady-state accuracy, not recovery. Relative rule (error back within +5°/+2 cm of its pre-perturbation level, never lost): POGS 49/50 (+15), 29/50 (-15), 46/50 (+30), 17/50 (-30); POGS+KF 40/41/33/34 of 50. Caveats: Static still scores up to 7/50 under it, and the asymmetry between +/- directions is unexplained. Open question for the paper: why is pre-perturbation rotation error 10-24° (tip error only ~1 cm)? Candidates: roll about the drill axis is unobservable (near-symmetric tool), or the gripper occludes it. Test: score axis-direction error and roll separately (needs poses stored per frame). O6 finding: base POGS runs at 1.5 Hz; 5 Hz only at 5 iterations, with unstable accuracy (68-100% success), so O6 is NOT met robustly.
- **Never edit `tools/run_pipeline.sh` while it runs** (bash reads it incrementally; the stage list is fixed at start). To add a stage: `kill <wrapper pid>` (not `--stop`, which also kills the experiment), edit, `--detach`; `.done` stages are skipped.
- **On resume:** `--status`; read the `.log` of any FAILED stage, fix, `--detach`. Then fill rows below from `data_results/{repro_check/report.json, o1_grouping/metrics.json, o3_perturbation/summary.md, o4_ingrasp/summary.md, o6_speed/summary.md}` and `model/foundation_study/artefacts/*/*/bench`.
- **Status 2026-11-17 10:55:** tests, repro, o1, o3_smoke, o6, o3, o4_smoke finished without failures (results in `data_results/*/summary.md`, not yet analysed into rows). The pipeline wrapper died during `o4` (about 3 h in, cause unknown, probably WSL/session end); relaunched with `--detach`, o4 resumes from its run cache. o5 (not yet run) uses the HF token from `~/.cache/huggingface/token`.
- **Early O3 reading (POGS, 5 seeds):** recovery 4-5/5 for jumps up to 90 deg, but jump +90 has outlier seeds (tip err 5.3 +- 10 cm), so the limit lies near 60-90 deg. Not final: protocol not frozen.
- **Not written:** report figures/tables from the new results (6.x), 54-case + 30 Hz re-check of the chosen O6 configuration, 3-seed training repeats (C4), real capture (7.2).
- **Rules:** run cache keyed by `ckpt_id`; the 2026-11-10 results are git tag `results-2026-11-10` (v1/v2/realtime/v3 overwrite `data_results/` with the new checkpoint's numbers).
- **Env:** `.venv` → `~/venvs/pogs-py312` (py 3.12.16, torch 2.14.1+cu130, gsplat 1.5.3; pins `requirements-lock.txt`). Tests: 30.

## Step status

### Phase 0: Environment and structure (O2)
| Step | Description | Status | Started | Finished | Evidence | Notes |
|---|---|---|---|---|---|---|
| 0.1 | Python 3.12 venv, PyTorch 2.14 + CUDA 13.0, torchmetrics, gsplat | done | 2026-11-13 | 2026-11-13 | `requirements-lock.txt`, `tools/setup_gsplat.sh`, `tools/cuda_env.sh` | torch 2.14.1+cu130; gsplat CUDA OK |
| 0.2 | 17 regression tests pass | done | 2026-11-13 | 2026-11-13 | pytest 25 passed (`74bdef2`) | 17 existing + 8 new |
| 0.3 | `metrics/` package (pose, gaussian) + tests | done | 2026-11-13 | 2026-11-13 | `metrics/pose.py`, `metrics/gaussian.py`, `metrics/stats.py`, `tests/test_metrics.py` | Tests pass. Pose metrics moved from `model/pogs/evaluate.py`, re-exported there |
| 0.4 | Regenerate benchmark, check PSNR ≈ 34.1 dB, IoU ± 0.03 | doing | 2026-11-13 | | `experiments/reproduce_check.py` → `data_results/repro_check/` | Base case 3 seeds, fresh runs dir; compares with old cached rows. Capture regenerated: 35 views, seed pts {0:4000, 1–5,9: 2500} |
| 0.5 | `env.json` in every result directory | doing | 2026-11-13 | | `experiments/env_info.py`, hooks in `experiments/pipeline.py` | Per run dir + suite out_dir; verify on first run. Also: rows stamped with `ckpt_id` (sha256[:12] of checkpoint), stale cache rejected |

### Phase 1: O1 splats with / without object
| Step | Description | Status | Started | Finished | Evidence | Notes |
|---|---|---|---|---|---|---|
| 1.1 | Render full / minus-object / object-only | doing | 2026-11-13 | | `experiments/o1_grouping_splats.py` | Written, compiles; waits for checkpoint |
| 1.2 | IoU, leakage, completeness, PSNR/SSIM, misassigned Gaussians | doing | 2026-11-13 | | same script | Amodal GT mask (object rendered alone); purity/recall per Gaussian; PSNR of minus vs GT-without outside/inside mask; residual alpha |
| 1.3 | Figure 5 × 3 + table, 3 training seeds | todo | | | | |
| 1.4 | Explain mug residual patch and soup-can IoU | todo | | | | |

### Phase 2: O3 perturbation recovery to 30°
| Step | Description | Status | Started | Finished | Evidence | Notes |
|---|---|---|---|---|---|---|
| 2.1 | Supervisor questions sent; `experiments/protocols/o3.md` committed before runs | doing | 2026-11-13 | | `experiments/protocols/o3.md` (draft) | Needs a new `tool_table` scene + checkpoint (pick-place has no tool). Freeze after Q1–3 answered |
| 2.2 | Full-rotation object frames; step-wise and single-jump trajectories | doing | 2026-11-13 | | `data/scene_generator/tool_perturbation.py`, `experiments/o3_perturbation.py` (`df21131`) | Written, schedules unit-checked, never rendered/tracked. Trials start at scanned pose + tracked settle (≤5 cm, ≤45°) |
| 2.3 | `kl_gaussians` + unit tests | done | 2026-11-13 | 2026-11-13 | `metrics/gaussian.py`, `tests/test_metrics.py` | Done early with 0.3; tests: identity, translation closed form, isotropic closed form, monotone in rotation, ±5° round trip |
| 2.4 | `kl_render` (GT-free) + test | done | 2026-11-13 | 2026-11-13 | `metrics/gaussian.py::kl_render`, test in `tests/test_metrics.py` | RGB 8³ histogram KL in rendered mask; feature-space variant may follow |
| 2.5 | Per-step errors, KL; step-size sweep | todo | | | | |
| 2.6 | Static / POGS (paper, release) / POGS + KF, 5 seeds | todo | | | | |

### Phase 3: O4 in-grasp recovery rate (80 %)
| Step | Description | Status | Started | Finished | Evidence | Notes |
|---|---|---|---|---|---|---|
| 3.1 | `experiments/protocols/o4.md` pre-registered | doing | 2026-11-13 | | `experiments/protocols/o4.md` (provisional) | Success: tip<3 cm, rot<5° at N=10 frames after perturbation, never lost after it |
| 3.2 | Drill-in-gripper perturbation scene | doing | 2026-11-17 | | `tool_perturbation.generate_grasp_sequence`, `tests/test_tool.py` | Gripper scale 1.0, fingers straddle housing; jaws not collision-checked; occlusion fraction not computed (flagged in metadata). Not rendered yet |
| 3.3 | Rates + Wilson CI, verdict per setting | doing | 2026-11-17 | | `experiments/o4_ingrasp.py` → `data_results/o4_ingrasp/summary.md` | Pipeline stages o4_smoke, o4; 4 settings × 50 trials × 4 methods |

### Phase 4: O6 5 Hz tracking
| Step | Description | Status | Started | Finished | Evidence | Notes |
|---|---|---|---|---|---|---|
| 4.1 | Per-stage profile | doing | 2026-11-17 | | `experiments/o6_speed.py` → `data_results/o6_speed/profile.txt` | In pipeline stage `o6`; untested |
| 4.2 | Speed candidates on 54 cases + 30 Hz protocol | doing | 2026-11-17 | | same | Narrowed: base condition, 3 seeds, iters {25,15,10,5} × match_res {500,350,250} × {POGS, KF+check} + Static; 54-case / 30 Hz re-check only for the chosen config |
| 4.3 | Cheapest ≥ 5 Hz configuration + accuracy cost | todo | | | | |

### Phase 5: O5 foundation-model latency
| Step | Description | Status | Started | Finished | Evidence | Notes |
|---|---|---|---|---|---|---|
| 5.1 | Re-benchmark on target stack | doing | 2026-11-17 | | `tools/o5_latency.sh` (venv `~/venvs/pogs-o5` built) | dino_vits14, clip_vitb16, mobileclipv2, siglip2, yolo, rfdetr, dinov3 (gated: needs HF_TOKEN). Stage o5 |
| 5.2 | Detic, SigLIP2 (and LocateAnything3D) standalone | todo | | | | |
| 5.3 | VFM-DETR availability recorded with date + source | done | 2026-11-17 | 2026-11-17 | `tools/o5_latency.sh` note → `data_results/o5_latency_note.json` | Two web searches found no repo/weights (hits: VFMTok, RT-DETRv4); paper not located. Worded "not found", not "does not exist" |
| 5.4 | Training-time vs per-frame split | todo | | | | |
| 5.5 | Optional: DINOv3 / MobileCLIPv2 in train + track | todo | | | | |

### Phase 6: Report
| Step | Description | Status | Started | Finished | Evidence | Notes |
|---|---|---|---|---|---|---|
| 6.1 | Figures/tables from scripts | todo | | | | |
| 6.2 | Status table, Limitations | todo | | | | |
| 6.3 | Independent number check | todo | | | | |
| 6.4 | PDF, slides, `git tag report-v1` | todo | | | | |

### Phase 7: Paper
| Step | Description | Status | Started | Finished | Evidence | Notes |
|---|---|---|---|---|---|---|
| 7.1 | Headline contribution + venue agreed | todo | | | | |
| 7.2 | Real-world sequence(s) | todo | | | | |
| 7.3 | Related work + external tracker comparison | todo | | | | |
| 7.4 | Code + data release | todo | | | | |

### Phase 8: Research programme (plan rev. 2, `POGS PhD Direction ... Plan.md` §6, §9)
| Step | Description | Status | Started | Finished | Evidence | Notes |
|---|---|---|---|---|---|---|
| 8.L | Read 12 reference papers in full; rename; reading notes; plan rev. 2 | done | 2026-11-22 | 2026-11-22 | `docs/literature/reading_notes.md`, `ref-paper/README.md`, plan rev. 2 | Findings R1–R10 in plan §0 |
| 8.T1 | Full SE(3) per frame in O3/O4 runs; roll/swing decomposition; paper-faithful tool rule; tests | done | 2026-11-22 | 2026-11-22 | `metrics/pose.py` (`swing_twist_deg`, `roll_swing_error_deg`, `tool_success_roll_free`), `experiments/o3_perturbation.py::score_trial(axis_local=)`, `experiments/o4_ingrasp.py` (`ROW_SCHEMA = 2`, `success_rollfree`), 3 new tests in `tests/test_metrics.py` (32 passed, 1 skipped) | Bit axis = drill object +x (`DRILL_TIP_LOCAL` lies on it); group frame has identity rotation at scan. Schema-1 rows are re-run, not reused |
| 8.E0 | O4 re-run with poses; pre-registered vs paper-faithful (roll-free) rule | doing | 2026-11-22 | | queued: `tools/e0_after_realtime.sh` (waits for `realtime.done`, then `tools/o4_parallel.sh 3`; log `data_results/_pipeline/e0.log`, marker `e0.done`/`e0.failed`) | Old schema-1 results archived in `data_results/_archive/o4_ingrasp_schema1_2026-11-22/` (800 rows, not in git). Gate G0 (Nov 26) |
| 8.E1 | Cause of the 10–24° floor and the ± gap (signed roll, loss at est vs GT, roll sweep) | todo | | | | Stop rule plan §8 |
| 8.T2 | `model/pogs/information.py` (`gauss_newton_info`, `loss_hessian`, `observability_summary`, `tracker_loss_hessian`) + tests | done | 2026-11-22 | 2026-11-22 | `tests/test_information.py` (6 passed; full suite 38 passed, 1 skipped) | Right perturbation in the object frame, exact vs matrix product; textureless cylinder → single null direction = roll about its axis; a decal restores roll information and more decal gives more. `tracker_loss_hessian` (real render loss) untested on GPU until E2 |
| 8.E2 | Hessian spectrum vs roll error on O4 (P1) | todo | | | | Gate G1 (Dec 10) |
| 8.T4 | Error-state filter ladder (fixed / trace / full / oracle R; Tikhonov; solution remapping) via config/hook | todo | | | | CLAUDE.md rule 2 |
| 8.E3 | Ladder on O4 (P2) | todo | | | | Needs T7 freeze first |
| 8.T5 | Weak-axis orbit belief (M4) | todo | | | | |
| 8.T6 | Calibration + recovery metrics (NEES, NIS, coverage, NLL, bootstrap, survival, RRTrack event) | todo | | | | |
| 8.T7 | ≥ 3 splats per scene; freeze `experiments/protocols/e_series.md`; tag | todo | | | | Before any E3 run |
| 8.E4–E6, E9 | Calibration, 2×2 decal × gripper, compute gating, recovery events | todo | | | | |
| 8.T8 | YCBInEOAT / HO3D loader, ADD/ADD-S AUC | todo | | | | Gate G2 (Dec 22) |
| 8.E7–E8, E10 | Real benchmark, frame downsampling, efficiency | todo | | | | Gate G3 (Feb 7) |
| 8.T9 | FoundationPose in 12 GB; RoPT / Opra code availability (dated) | todo | | | | |
| 8.T10 | Two targeted novelty searches | todo | | | | Before related work |

## Supervisor questions (PLAN §5)
| # | Question | Sent | Answer | Date |
|---|---|---|---|---|
| 1 | Which KL | no | | |
| 2 | Step-wise vs single jump; axis | no | | |
| 3 | Success criterion + window | no | | |
| 4 | Synthetic-only acceptable? | no | | |
| 5 | Python 3.14 acceptable if 3.12 fails? | no | | |

## Deviations and decisions
Anything that departs from the plan, the paper, the release or the brief. Never delete a row.

| Date | Step | Decision / deviation | Reason |
|---|---|---|---|
| 2026-11-13 | 0.3 | Pose metrics moved to `metrics/pose.py`; `model.pogs.evaluate` re-exports them | Brief asks for a Metrics package; re-export keeps every caller and cached result unchanged |
| 2026-11-13 | 2.3 | Covariance floor `COV_EPS = 1e-8 m²` in KL | Trained splats contain flat Gaussians with numerically singular covariances; must be stated next to KL values |
| 2026-11-13 | 2.3 | `weighted_kl` also reports median and p95 | Per-Gaussian KL is heavy-tailed; the weighted mean alone can be driven by a few flat Gaussians |
| 2026-11-13 | 0.4 | Run cache keyed by checkpoint (`ckpt_id`); old rows not reused | Old checkpoint not in git; reusing its rows with a retrained model would mix two models in one table |
| 2026-11-13 | 0.4 | Reproduction check in a fresh dir (`data_results/repro_check/runs`), not the shared cache | Keeps the 2026-11-10 rows intact as the reference |
| 2026-11-13 | 1.2 | O1 uses amodal GT masks and black background | Occlusion by other objects must not count as leakage; black makes alpha and holes unambiguous |
| 2026-11-13 | 2.4 | `kl_render` smoothing 0.01/bin (not 1) | Add-1 over 512 bins swamps a few-hundred-pixel object |
| 2026-11-17 | plan | Critique C1–C11 added to PLAN §7; protocols run provisional without waiting for supervisor (C2); jump sweep to 90° (C3); selective re-runs (C5) | See PLAN §7 |
| 2026-11-13 | 2.2 | O3 trials start at the scanned pose and settle to a random start pose | Random initial pose would mix initialisation failures into recovery |
| 2026-11-17 | 5.2 | LocateAnything3D and Detic skipped in the new-stack benchmark; old-stack numbers kept and labelled | 9.5 GB / 8–11 GB VRAM; detectron2 not buildable on py3.12 / torch 2.14. Time-box (PLAN §7 C8) |
| 2026-11-17 | 3.2 | In-grasp rotation about the vertical grasp axis with a static gripper, no jaw collision | Simulator limitation; stated in o4 summary |
| 2026-11-22 | 3.1 / 8.E0 | O4 will additionally be scored with the paper-faithful rule (tip < 3 cm and bit-axis direction < 5°, roll about the bit axis free); the pre-registered rule stays first and unchanged | POGS Sec. V-B: "We leave tool rotation about its target axis (i.e. drill bit axis) unconstrained". Rule taken from the paper, not chosen from our results; recorded before E0 runs |
| 2026-11-22 | plan | Research plan rev. 2 adds Phase 8 (E0–E12, T1–T10, gates G0–G3) | Full-text reading of 12 papers (plan §0) |
| 2026-11-22 | refs | `ref-paper/*.pdf` renamed from arXiv ids to `YEAR_VENUE_Author_topic.pdf` | Readability; mapping in `ref-paper/README.md` |

## Session log
Dropped (C11): history is in git log and the tables above.
