#!/usr/bin/env bash
# POGS model-substitution study - row 02 (RF-DETR + DINOv3 + MobileCLIPv2).
#
# PATCHED, for two reasons:
#
#   * DINOv3 is stride 16, but dino_dataloader.py:24 hardcodes dino_stride = 14
#     and dino_dataloader.py:12 hardcodes MAX_DINO_SIZE against that stride.
#     Feeding DINOv3 a stride-14 grid would silently mis-shape its token map.
#   * transformers==4.44.0 in pyproject.toml predates DINOv3's loading path.
#
# MobileCLIP2 needs no patch at all: it loads through the same
# open_clip.create_model_and_transforms call POGS already makes for CLIP, with a
# different clip_model_type. That is the row's positive finding and it is
# recorded as such.
#
# GATED: DINOv3 requires an HF licence acceptance and token. Without it this row
# runs its available models and records DINOv3 as gated.
#
# Capability note: RF-DETR is COCO-closed-set. POGS passes a custom_vocab to its
# detector, so this row cannot serve that contract however fast it is. That is
# the finding; no adapter was attempted.

# Run one scene (pickplace by default):
#     bash run.sh setup     # environment + imports
#     bash run.sh weights   # download this row's weights
#     bash run.sh detect    # does the detector find the object? ~1 min
#     bash run.sh train     # full 4000
#     bash run.sh track     # headless tracking, n=1 at the fixed seed
#     bash run.sh bench     # latency + memory
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

ROW=02_rfdetr_dinov3_mobileclip
ROW_DESC="RF-DETR + DINOv3 ViT-B/16 + MobileCLIP2-B"
PATCHED=1
BLOCKED_MODELS=""
GATED_MODELS="dinov3"

ROW_ENV=pogs_env_r02
ROW_PIP="transformers==4.49.0 rfdetr==1.5.2"

# shellcheck source=../shared/row_common.sh
source "$ROW_DIR/../shared/row_common.sh"
row_main "$@"