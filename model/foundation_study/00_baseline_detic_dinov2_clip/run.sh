#!/usr/bin/env bash
# POGS model-substitution study - row 00 baseline (DETIC + DINOv2 + CLIP).
#
# This is the reference row. Everything else is measured against it, so it runs
# stock POGS with nothing patched: `PATCHED=0` and no patch/ directory.
#
# The install sequence follows the authors' README verbatim. Deviating from it
# is how you end up debugging a torch/CUDA mismatch instead of the experiment.
#
# Usage, on the GPU host:
#     cd 00_baseline_detic_dinov2_clip && bash run.sh
#
# Stages can be re-run individually. Each acts on one scene, `pickplace` by
# default; SCENE=... picks another from the matrix:
#     bash run.sh setup     # environment + imports + patches
#     bash run.sh weights   # download this row's weights
#     bash run.sh prefetch  # warm caches so timing is not a download
#     bash run.sh smoke     # 50 iterations, twice, to verify determinism
#     bash run.sh detect    # does the detector find the object? ~1 min, run early
#     bash run.sh train     # full 4000
#     bash run.sh bench     # latency + memory
#     bash run.sh validate  # detector IoU on the authors' real frames
#     bash run.sh track     # headless tracking, n=1 at the fixed seed
#     bash run.sh report    # RESULTS.md
#     bash run.sh sweep     # train + track on all 10 scenes
#     bash run.sh all       # one scene, everything (default)
#
#   SCENE=pendulum_60 bash run.sh train      # one scene, one stage
#   bash run.sh sweep bench                  # a different stage per scene
#   SKIP_TRAIN=1 bash run.sh all             # everything but training/tracking
#   bash run.sh track --max-frames 20        # flags pass through to the tracker

# Resolved here, before the source below, because BASH_SOURCE inside a sourced
# file names the sourced file rather than this one -- shared/row_common.sh
# needs this row's directory and cannot work it out for itself.
ROW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROW=00_baseline_detic_dinov2_clip
ROW_DESC="DETIC + DINOv2 ViT-S/14 + CLIP ViT-B/16 -- stock POGS, unmodified"
PATCHED=0
BLOCKED_MODELS=""
GATED_MODELS=""

# shellcheck source=../shared/row_common.sh
source "$ROW_DIR/../shared/row_common.sh"
row_main "$@"