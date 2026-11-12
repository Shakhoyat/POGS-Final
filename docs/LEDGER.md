# Progress ledger

Tracks execution of `docs/PLAN.md`. One row per plan step; one dated log entry per working
session. Update both in the same commit as the work they describe.

**Status keys:** `todo` · `doing` · `done` (exit criterion met, evidence linked) · `blocked`
(reason in Notes) · `dropped` (reason in Notes, decision logged) · `changed` (step altered from
the plan; the reason is logged).

**Evidence** is a file path, commit SHA or result directory, never a claim alone. A step is
`done` only when its evidence is in the repository.

## ▶ Resume here
- **Next:** 0.4 `source tools/cuda_env.sh && .venv/bin/python -m experiments.v1_base_pogs.run_all` (retrains checkpoint; GPU, long → run in background); check PSNR ≈ 34.1 dB, grouping IoU ±0.03 vs `docs/RESULTS_reproduction.md`, and that `env.json` appears in run dirs. Then Phase 1.
- **Env:** `.venv` → `~/venvs/pogs-py312` (symlink; venv on ext4 for speed). Python 3.12.16, torch 2.14.1+cu130, gsplat 1.5.3 (CUDA built), torchmetrics 1.9.0. Exact pins: `requirements-lock.txt`. Rebuild: `uv venv --python 3.12 --seed ~/venvs/pogs-py312`, torch from `download.pytorch.org/whl/cu130`, `uv pip install -r requirements.txt`, `bash tools/setup_gsplat.sh`.
- **Tests:** `source tools/cuda_env.sh && .venv/bin/python -m pytest -q tests` → 25 passed (2026-11-13).

## Step status

### Phase 0: Environment and structure (O2)
| Step | Description | Status | Started | Finished | Evidence | Notes |
|---|---|---|---|---|---|---|
| 0.1 | Python 3.12 venv, PyTorch 2.14 + CUDA 13.0, torchmetrics, gsplat | done | 2026-11-13 | 2026-11-13 | `requirements-lock.txt`, `tools/setup_gsplat.sh`, `tools/cuda_env.sh` | torch 2.14.1+cu130; gsplat CUDA OK |
| 0.2 | 17 regression tests pass | done | 2026-11-13 | 2026-11-13 | pytest 25 passed | 17 existing + 8 new |
| 0.3 | `metrics/` package (pose, gaussian) + tests | done | 2026-11-13 | 2026-11-13 | `metrics/pose.py`, `metrics/gaussian.py`, `metrics/stats.py`, `tests/test_metrics.py` | Tests pass. Pose metrics moved from `model/pogs/evaluate.py`, re-exported there |
| 0.4 | Regenerate benchmark, check PSNR ≈ 34.1 dB, IoU ± 0.03 | todo | | | | |
| 0.5 | `env.json` in every result directory | doing | 2026-11-13 | | `experiments/env_info.py`, hooks in `experiments/pipeline.py` | Written per run dir and per suite out_dir; to verify on the first real run |

### Phase 1: O1 splats with / without object
| Step | Description | Status | Started | Finished | Evidence | Notes |
|---|---|---|---|---|---|---|
| 1.1 | Render full / minus-object / object-only | todo | | | | |
| 1.2 | IoU, leakage, completeness, PSNR/SSIM, misassigned Gaussians | todo | | | | |
| 1.3 | Figure 5 × 3 + table, 3 training seeds | todo | | | | |
| 1.4 | Explain mug residual patch and soup-can IoU | todo | | | | |

### Phase 2: O3 perturbation recovery to 30°
| Step | Description | Status | Started | Finished | Evidence | Notes |
|---|---|---|---|---|---|---|
| 2.1 | Supervisor questions sent; `experiments/protocols/o3.md` committed before runs | todo | | | | |
| 2.2 | Full-rotation object frames; step-wise and single-jump trajectories | todo | | | | |
| 2.3 | `kl_gaussians` + unit tests | done | 2026-11-13 | 2026-11-13 | `metrics/gaussian.py`, `tests/test_metrics.py` | Done early with 0.3; tests: identity, translation closed form, isotropic closed form, monotone in rotation, ±5° round trip |
| 2.4 | `kl_render` (GT-free) + test | todo | | | | |
| 2.5 | Per-step errors, KL; step-size sweep | todo | | | | |
| 2.6 | Static / POGS (paper, release) / POGS + KF, 5 seeds | todo | | | | |

### Phase 3: O4 in-grasp recovery rate (80 %)
| Step | Description | Status | Started | Finished | Evidence | Notes |
|---|---|---|---|---|---|---|
| 3.1 | `experiments/protocols/o4.md` pre-registered | todo | | | | |
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

## Session log

### 2026-11-13
- Re-read supervisor requirements against the repo; revised `docs/PLAN.md`; created this ledger.
- Environment check: RTX 4070 (12 GB, driver 610.88) visible via `/usr/lib/wsl/lib`; system
  Python 3.14 only, no pip, no torch, no `.venv`. Started 0.1.
- 0.1: made `tools/setup_gsplat.sh` / `tools/cuda_env.sh` work for any venv Python; updated
  `requirements.txt` (target stack, torchmetrics). Installing uv was refused by the permission
  check; the user needs to approve or run it.
- 0.3 / 2.3: wrote `metrics/` (pose, gaussian incl. KL, stats incl. Wilson) and
  `tests/test_metrics.py`; byte-compiles; **not yet executed**.
- 0.5: `experiments/env_info.py`, wired into `run_suite` and `run_realtime_suite`.
- 0.1/0.2 done: uv installed (user granted full permission); Python 3.12 venv, torch 2.14.1+cu130, gsplat built; 25/25 tests pass; lock file written. Usage limit reached here; next = 0.4.
