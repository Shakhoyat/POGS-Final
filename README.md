# POGS-Final

Consolidated research repository for **Persistent Object Gaussian Splatting (POGS)**
(Yu, Hari, El-Refai *et al.*, ICRA 2025): a faithful re-implementation with a synthetic
pick-and-place benchmark, a real-time tracking study, and a 2D foundation-model
substitution study.

Start with the supervisor report: [`report/supervisor/main.tex`](report/supervisor/main.tex).

## Layout

```
model/
  pogs/               POGS library: model, gsplat/torch rasteriser, feature field, encoders,
                      losses, trainer, HDBSCAN grouping, rigid-group tracker, SE(3) Kalman
                      filter, checkpoints, shared tracking loop + metrics (evaluate.py)
  foundation_study/   detection / visual-feature / feature-alignment model study
                      (Detic, LocateAnything3D, RF-DETR, VFM-DETR, YOLO; DINOv2, DINOv3,
                      dino.txt; CLIP, MobileCLIPv2, SigLIP2): code, patches, results archive
data/
  scene_generator/    synthetic scenes and the 54-case pick-and-place benchmark
experiments/          v1 base POGS, v2 Kalman prior, v3 objects, real-time protocol
tests/                regression tests
report/
  supervisor/         consolidated progress report (LaTeX) + figure script
  reproduction/       detailed report of the re-implementation
docs/                 paper, results (RESULTS_reproduction.md), audit history (JOURNEY.md)
third_party/          authors' release (POGS-author-github), exploratory notebooks
legacy/               unmaintained pre-audit modules (results there were retracted)
tools/                gsplat CUDA build without sudo
datasets/ model_checkpoints/ data_results/   generated data, checkpoints and cached runs
```

## Setup (Linux / WSL2, NVIDIA GPU)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
tools/setup_gsplat.sh                 # builds gsplat's CUDA kernels inside .venv
.venv/bin/python -m pytest -q tests
```

## Run

```bash
.venv/bin/python -m data.scene_generator.preview_scene     # review the scene (CPU)
.venv/bin/python -m experiments.v1_base_pogs.run_all       # data, train, group, track, videos
.venv/bin/python -m experiments.v2_kalman_pogs.run_all
.venv/bin/python -m experiments.v3_scene_mustard.run_all
.venv/bin/python -m experiments.realtime.run_all           # needs an idle GPU
.venv/bin/python -m experiments.results_tables > tables.md
```

The foundation-model study runs per row: `cd model/foundation_study/<row> && bash run.sh`
(see `model/foundation_study/README.md` and `MERGE_NOTES.md`).

## Status

| Objective | Status |
|---|---|
| POGS re-implementation, scene generation, pose metrics | done |
| Object-grouping visuals | partial (object-only splats; with/without comparison missing) |
| 2D foundation-model latency | partial (VFM-DETR has no public weights) |
| 5 Hz tracking | partial (1.9 Hz at 25 iterations, 4.4 Hz at 10) |
| Tool perturbation recovery (30 deg, KL divergence) | not started |
| In-grasp recovery rate (80 %) | not started |
| Python 3.12 / CUDA 13.0 / PyTorch 2.14 | partial |

Details and the work plan: `report/supervisor/main.tex`.
