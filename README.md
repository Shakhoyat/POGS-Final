# Persistent Object Gaussian Splatting (POGS) -- re-implementation

Re-implementation of **Persistent Object Gaussian Splat (POGS) for Tracking Human
and Robot Manipulation of Irregularly Shaped Objects** (Yu, Hari, El-Refai et al.,
ICRA 2025), validated on a synthetic robot pick-and-place benchmark with exact
ground truth.

`v1` follows the paper and the authors' release (`POGS-author-github/`) as closely
as possible; `v2` and `v3` are built strictly on top of it. Grasp planning and tool
servoing are out of scope.

## Layout

```
common/            POGS library: config (release defaults), model, gsplat/torch
                   rasteriser, feature field, encoders, losses, trainer,
                   HDBSCAN grouping, rigid-group tracker, SE(3) Kalman filter,
                   checkpoints, the shared tracking loop + metrics
scene_generator/   synthetic scenes; pick_place_benchmark.py builds the benchmark
                   (12 conditions x 3 seeds, + speed / depth-noise at load time)
experiments/
  pipeline.py      train -> group -> track -> report (shared by all versions)
  visuals.py       dashboard videos
  v1_base_pogs/    base POGS (paper and release settings) + the paper's ablations
  v2_kalman_pogs/  v1 + constant-velocity SE(3) Kalman motion prior (+ iteration sweep)
  v3_scene_mustard/ object and multi-object factors (folder name kept for history)
tests/             regression tests (pose maths, PCA persistence, metrics, ...)
tools/             gsplat CUDA setup without sudo
legacy/            unmaintained notebook-era modules (grasping, servoing, old generator)
```

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
tools/setup_gsplat.sh        # builds gsplat's CUDA kernels inside .venv (no sudo)
.venv/bin/python -m pytest -q tests
```

Without gsplat everything still runs on a pure-PyTorch rasteriser, but about two
orders of magnitude slower, and densification is only approximate.

## Run

```bash
.venv/bin/python -m experiments.v1_base_pogs.run_all      # data (if missing), train, group, track, videos
.venv/bin/python -m experiments.v2_kalman_pogs.run_all    # reuses the v1 checkpoint and runs
.venv/bin/python -m experiments.v3_scene_mustard.run_all  # reuses the v1 checkpoint and runs
# any of them: --factors rotation speed occlusion depth object multi-object
```

Outputs: `model_checkpoints/pick_place_floor_table/` (one self-contained
checkpoint, including the DINO PCA basis), `data_results/benchmark_runs/` (every
run, shared and resumable) and `data_results/<version>/` (`summary.md` with one
table per benchmark factor, `summary.csv`, `videos/`).

See [RESULTS.md](RESULTS.md) for numbers, [JOURNEY.md](JOURNEY.md) for the audit
history, and [Claude.md](Claude.md) for implementation notes and deviations from
the release.
