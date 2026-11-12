#!/usr/bin/env bash
# POGS model-substitution study - row 03 (DETIC + DINOv2 + SigLIPv2).
#
# VFM-DETR spans the detection AND feature columns in the source table. It is
# BLOCKED with evidence: the repo exists (github.com/lch216/VFMDETR, NeurIPS
# 2025) but ships no released weights anywhere, and its protocol is closed-set
# few-shot on six CD-FSOD benchmarks -- incompatible with POGS's zero-shot
# open-vocab instance_masks contract even if weights appeared. Recorded, not
# faked.
#
# So this row keeps POGS's stock detector (DETiC) and stock DINOv2 features,
# and substitutes ONLY the relevancy/alignment encoder: OpenCLIP ViT-B-16
# 512-d -> SigLIP2-base-patch16-224 768-d.
#
# PATCHED, because stock POGS hardcodes that encoder twice over:
#
#   * pogs_config.py builds OpenCLIPNetworkConfig(ViT-B-16, 512) as a literal.
#   * pogs.py GaussianField() defaults clip_n_dims=512 for its CLIP head, so
#     swapping only the encoder would silently mismatch head vs encoder.
#
# Both are env-gated behind POGS_ALIGNMENT_ENCODER=siglip2 (exported below);
# unset, the tree behaves exactly as upstream. See patch/ for the diff.
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

ROW=03_vfmdetr_siglip2
ROW_DESC="VFM-DETR (BLOCKED) -- DETIC + SigLIPv2 alignment substitution"
PATCHED=1
BLOCKED_MODELS="vfmdetr"
GATED_MODELS=""

# SigLIP2 needs transformers >= 4.50 (released 2025-02-21); stock POGS pins
# 4.44.0. Own env, so rows 00/01/02/04 are unaffected.
ROW_ENV=pogs_env_r03
ROW_PIP="transformers==4.50.0 timm==1.0.15"

# Read by pogs_config.py (encoder choice) and pogs/encoders/siglip2_encoder.py
# (offline weights lookup under $POGS_WEIGHTS/siglip2).
export POGS_ALIGNMENT_ENCODER=siglip2

# shellcheck source=../shared/row_common.sh
source "$ROW_DIR/../shared/row_common.sh"
row_main "$@"
