# Merge notes (POGS-Final)

This folder is the 2D foundation-model substitution study (detection / visual features /
feature alignment), imported from the `pogs_v2/model_study` workstream on 2026-11-13.

Imported: the pipeline code (`shared/`), every row's `run.sh`, `patch/` and reports, and the
results archive (`artefacts/`, `incomplete/`, `inputs/` ground truth, `weights/manifest.json`).
Every JSON is byte-identical to `model_study_results/` (see `MANIFEST.md`).

Not imported (multi-GB, regenerable), all git-ignored:

| Missing | How to restore |
|---|---|
| `<row>/POGS/` | a checkout of the authors' POGS release per row (`run.sh setup` applies `patch/`) |
| `deps/` (nerfstudio, detectron2, Detic) | as in the row `run.sh setup` stage |
| `weights/` blobs (~16 GB) | `python shared/download_weights.py` (URLs + sha256 in `weights/manifest.json`) |
| `inputs/<scene>/` frames | regenerate the scenes; only ground-truth JSON is kept |
| checkpoints, `clusters.npy`, `train.log` | re-run `bash run.sh train` / `track` |

Read `COMPARISON_STUDY.md` first: rows 01 and 02 ran the baseline models in train/track,
so only latency, VRAM and detector IoU are valid per-model results.
