# Contributions

Who did what, so that authorship can be attributed precisely when writing the paper.
Git history carries the same information: Md Nayeem's original commits are preserved
unchanged (`git log --author=Nayeem`).

## Md Nayeem: POGS re-implementation and benchmark (workstream A)

Imported with full history from `github.com/codernayeem/pogs-dev` @ `1a559cf`.
Files were moved (Model / Data layout) and imports rewritten; the content is his.

| Path in this repo | Original path | Work |
|---|---|---|
| `model/pogs/` | `common/` | POGS library: model, rasteriser, feature field, encoders, losses, trainer, HDBSCAN grouping, rigid-group tracker, SE(3) Kalman filter, checkpoints, tracking loop and metrics |
| `data/scene_generator/` | `scene_generator/` | synthetic scenes, procedural materials, 54-case pick-and-place benchmark, 30 Hz renders |
| `experiments/` | `experiments/` | v1 base POGS + paper ablations, v2 Kalman prior, v3 objects / multi-object, real-time protocol, dashboard videos |
| `tests/` | `tests/` | 17 regression tests |
| `tools/` | `tools/` | gsplat CUDA build |
| `report/reproduction/` | `report/` | detailed LaTeX report and figures of the re-implementation |
| `data_results/`, `datasets/`, `model_checkpoints/` | same | cached benchmark runs, summaries, videos |
| `docs/RESULTS_reproduction.md`, `docs/JOURNEY.md`, `Claude.md` | `RESULTS.md`, `JOURNEY.md`, `Claude.md` | results, audit history, implementation notes |
| `legacy/`, `third_party/` | `legacy/`, `POGS-author-github/`, `base_notebooks/` | pre-audit modules (retracted), authors' release, exploratory notebook |

## Dadhichi: 2D foundation-model substitution study (workstream B)

Imported from `pogs_v2/model_study` (not under version control; imported as one commit).
Every file under `model/foundation_study/` is his work; the results archive is
byte-identical to his `model_study_results/` (md5-verified).

| Path in this repo | Work |
|---|---|
| `model/foundation_study/shared/` | pipeline: weight probing/download, latency benchmark, offline tracking, detector validation, comparison/report generation, tests |
| `model/foundation_study/0*_*/` | five model rows (Detic/DINOv2/CLIP baseline; LocateAnything3D + dino.txt; RF-DETR + DINOv3 + MobileCLIPv2; VFM-DETR + SigLIP2; YOLO): `run.sh`, POGS patches, per-row reports |
| `model/foundation_study/artefacts/`, `incomplete/`, `inputs/` | latency, VRAM, detector IoU, tracking measurements and ground truth |
| `model/foundation_study/*.md` | study design (`README.md`), merged results, `COMPARISON_STUDY.md` analysis, `MANIFEST.md` |

## First author (research coordination)

| Path | Work |
|---|---|
| repository structure, `README.md`, `.gitattributes`, `CONTRIBUTIONS.md`, `model/foundation_study/MERGE_NOTES.md` | consolidation of both workstreams |
| `report/supervisor/` | consolidated progress report, cross-workstream figures and tables, protocols for the remaining objectives |
| remaining objectives (with/without-object splats, perturbation recovery + KL divergence, in-grasp recovery rate, 5 Hz tracker, environment port) | planned; see `report/supervisor/main.tex` |
