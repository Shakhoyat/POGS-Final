#!/usr/bin/env bash
# POGS model-substitution study - row 01 (LocateAnything3D + DINOv2 Meets Text).
#
# dino.txt spans BOTH the feature and alignment columns in the source table, so
# this row replaces two of POGS's three components with one model.
#
# PATCHED, because stock POGS hardcodes the models it uses:
#
#   * dino_dataloader.py:23  dino_model_type = "dinov2_vits14" is a class
#     attribute, not a config field, and cfg is only read for image_shape. There
#     is no supported way to express dino.txt without editing this line.
#   * detic_dataloader.py:76  the DETIC config path and weights URL are literals.
#     LocateAnything3D is not a drop-in for either.
#
# The patches live in patch/ and are applied by run.sh, which also records that
# it did so. See RESULTS.md section 6 for the diff.
#
# Capability note: LocateAnything3D emits 3D boxes, not per-instance 2D masks.
# POGS consumes instance_masks. This row therefore measures latency and states
# the gap rather than bridging it -- bridging would mean writing the adapter the
# study was meant to evaluate the *need* for.
#
# GATED: dino.txt needs an HF token. export HF_TOKEN=... before running.
#
# Run one scene (pickplace by default):
#     bash run.sh setup     # environment + imports + patches
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

ROW=01_locateanything_dinotxt
ROW_DESC="LocateAnything3D-3B + DINOv2 Meets Text (features AND alignment)"
PATCHED=1
BLOCKED_MODELS=""
GATED_MODELS=""

# dino.txt is a ViT-L/14 open_clip-shaped checkpoint; stock POGS pins
# transformers==4.44.0, which predates it.
ROW_ENV=pogs_env_r01
ROW_PIP="transformers==4.51.3 timm==1.0.15 peft==0.15.2 decord==0.6.0 lmdb==1.7.5"

# shellcheck source=../shared/row_common.sh
source "$ROW_DIR/../shared/row_common.sh"
row_main "$@"