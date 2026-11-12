#!/usr/bin/env bash
# POGS model-substitution study - row 04 (YOLO detection only).
#
# The source table leaves this row's feature and alignment cells empty, so it
# reuses the baseline's DINOv2 and CLIP: only the detector changes.
#
# This row is measured as LATENCY PLUS A DOCUMENTED CAPABILITY GAP, and that is
# the finding rather than a shortfall:
#
#   * YOLO emits boxes, not instance masks. POGS consumes instance_masks, so it
#     cannot serve that contract at all.
#   * YOLO is COCO-closed-set with no text interface, so it cannot accept the
#     custom_vocab POGS passes to its detector.
#
# No adapter engineering was attempted. The gap is the result.
#
# NOT PATCHED: bench.py times YOLO through ultralytics on the baseline's tree.
# detic_dataloader.py still loads DETIC, so `detect` and `validate` measure the
# *baseline* detector for this row. That is stated in RESULTS.md rather than
# passed off as a YOLO measurement.
#
# Run one scene (pickplace by default):
#     bash run.sh setup     # environment + imports
#     bash run.sh weights   # download this row's weights
#     bash run.sh detect    # does the detector find the object? ~1 min
#     bash run.sh train     # full 4000
#     bash run.sh track     # headless tracking, n=1 at the fixed seed
#     bash run.sh bench     # latency and memory -- the row's actual measurement
#     bash run.sh validate  # detector IoU on the authors' real frames
#     bash run.sh report    # RESULTS.md
#     bash run.sh all       # one scene, everything
#
# Run every scene:
#     bash run.sh sweep                    # train + track on all 10
#     bash run.sh sweep bench              # a different stage per scene
#     SCENE=pendulum_60 bash run.sh train   # one scene, one stage
#     bash run.sh track --max-frames 20    # flags pass through to the tracker

# Resolved here, before the source below, because BASH_SOURCE inside a sourced
# file names the sourced file rather than this one.
ROW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROW=04_yolo
ROW_DESC="YOLO11x detection only -- closed-set, capability gap recorded"
PATCHED=0
BLOCKED_MODELS=""
GATED_MODELS=""

# shellcheck source=../shared/row_common.sh
source "$ROW_DIR/../shared/row_common.sh"
row_main "$@"