# Progress ledger

Tracks execution of `docs/PLAN.md`. One row per plan step; one dated log entry per working
session. Update both in the same commit as the work they describe.

**Status keys:** `todo` · `doing` · `done` (exit criterion met, evidence linked) · `blocked`
(reason in Notes) · `dropped` (reason in Notes, decision logged) · `changed` (step altered from
the plan; the reason is logged).

**Evidence** is a file path, commit SHA or result directory, never a claim alone. A step is
`done` only when its evidence is in the repository.

## ▶ Resume here
- **Unattended pipeline:** `tools/run_pipeline.sh --detach | --status | --stop | --only <stage> | --redo <stage>`. Stages: tests → repro (0.4) → o1 → o3_smoke → o3 → v1 → v2 → realtime → v3. State + logs in `data_results/_pipeline/` (`<stage>.done|failed|log`, `status.tsv`). Failed stages are skipped over and retried on the next start. Started 2026-11-17 05:07.
- **On resume:** run `--status`; read `.log` of any FAILED stage, fix, `--detach` again. Then fill the step rows below from `data_results/{repro_check/report.json, o1_grouping/metrics.json, o3_perturbation/summary.md}`.
- **Not automated yet (no code):** O4 in-grasp trials (3.2–3.3), O6 profile + speed candidates (4.x), O5 re-benchmark (5.x), report figures (6.x). Add each as a line in `STAGES` when written.
- **Untested code in the pipeline:** `o3_perturbation` (first real run is stage `o3_smoke`).
- **Rules:** run cache keyed by `ckpt_id`; the 2026-11-10 results are git tag `results-2026-11-10` (v1/v2/realtime/v3 stages overwrite `data_results/` with the new checkpoint's numbers).
- **Env:** `.venv` → `~/venvs/pogs-py312`. Py 3.12.16, torch 2.14.1+cu130, gsplat 1.5.3; pins `requirements-lock.txt`.

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
| 3.1 | `experiments/protocols/o4.md` pre-registered | doing | 2026-11-13 | | `experiments/protocols/o4.md` (draft) | Freeze before first O4 run |
| 3.2 | Drill-in-gripper perturbation scene | todo | | | | |
| 3.3 | Rates + Wilson CI, verdict per setting | todo | | | | |

### Phase 4: O6 5 Hz tracking
| Step | Description | Status | Started | Finished | Evidence | Notes |
|---|---|---|---|---|---|---|
| 4.1 | Per-stage profile | todo | | | | |
| 4.2 | Speed candidates on 54 cases + 30 Hz protocol | todo | | | | |
| 4.3 | Cheapest ≥ 5 Hz configuration + accuracy cost | todo | | | | |

### Phase 5: O5 foundation-model latency
| Step | Description | Status | Started | Finished | Evidence | Notes |
|---|---|---|---|---|---|---|
| 5.1 | Re-benchmark on target stack | todo | | | | |
| 5.2 | Detic, SigLIP2 (and LocateAnything3D) standalone | todo | | | | |
| 5.3 | VFM-DETR availability recorded with date + source | todo | | | | |
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

## Session log
Dropped (C11): history is in git log and the tables above.
