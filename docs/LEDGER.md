# Progress ledger

Tracks execution of `docs/PLAN.md`. One row per plan step; one dated log entry per working
session. Update both in the same commit as the work they describe.

**Status keys:** `todo` · `doing` · `done` (exit criterion met, evidence linked) · `blocked`
(reason in Notes) · `dropped` (reason in Notes, decision logged) · `changed` (step altered from
the plan; the reason is logged).

**Evidence** is a file path, commit SHA or result directory, never a claim alone. A step is
`done` only when its evidence is in the repository.

## ▶ Resume here
- **Running:** 0.4 `experiments/reproduce_check.py` (bg; log `data_results/repro_check/log.txt`, result `report.json`). If not running and no `report.json`: relaunch `source tools/cuda_env.sh && nohup .venv/bin/python -u -m experiments.reproduce_check > data_results/repro_check/log.txt 2>&1 &` (data + checkpoint are reused if present).
- **Then:** fill 0.4 row from `report.json`; run `python -m experiments.o1_grouping_splats` (1.1–1.3); commit.
- **Do not** run `experiments.*.run_all` against the old cache expecting reuse: rows now carry `ckpt_id` and rows of the old (lost) checkpoint are re-run. Old rows stay on disk as the 2026-11-10 reference.
- **Env:** `.venv` → `~/venvs/pogs-py312` (ext4). Py 3.12.16, torch 2.14.1+cu130, gsplat 1.5.3, torchmetrics 1.9.0; pins `requirements-lock.txt`. Rebuild: `uv venv --python 3.12 --seed ~/venvs/pogs-py312`; torch from `download.pytorch.org/whl/cu130`; `uv pip install -r requirements.txt`; `bash tools/setup_gsplat.sh`.
- **Tests:** `source tools/cuda_env.sh && .venv/bin/python -m pytest -q -p no:cacheprovider tests` (26 expected; `.pytest_cache` can't be written on /mnt/e).

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
| 2.2 | Full-rotation object frames; step-wise and single-jump trajectories | todo | | | | |
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

## Session log

### 2026-11-13
- Plan revised; ledger created. Env (0.1) + tests (0.2) + metrics (0.3, 2.3) done → `74bdef2`.
- 0.4 launched (`reproduce_check`); cache keyed by `ckpt_id`; `env.json` writer fixed to collect before opening.
- Written while GPU busy: O1 script (1.1/1.2), `kl_render` (2.4), draft protocols O3/O4 (2.1/3.1).
