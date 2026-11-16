#!/usr/bin/env bash
# PLAN Phase 5 (O5): latency of every 2D foundation model on the target stack, GPU otherwise idle.
# Runs in its own venv (~/venvs/pogs-o5) because rfdetr / ultralytics may pin other torch builds.
# Gated or unavailable models are recorded as blocked/gated, never fatal. Skipped by default:
# LocateAnything3D (~9.5 GB, 8-11 GB VRAM; old-stack numbers are kept and labelled), Detic
# (needs detectron2; not buildable on Python 3.12 / torch 2.14 -> old-stack number, labelled).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
# Gated models (DINOv3): take the token from huggingface's own token file; never from the repo.
if [ -z "${HF_TOKEN:-}" ] && [ -f ~/.cache/huggingface/token ]; then export HF_TOKEN="$(cat ~/.cache/huggingface/token)"; fi
V=~/venvs/pogs-o5
UV=~/.local/bin/uv
if [ ! -x "$V/bin/python" ]; then
  $UV venv --python 3.12 --seed "$V" || exit 1
  $UV pip install --python "$V/bin/python" torch torchvision --index-url https://download.pytorch.org/whl/cu130 || exit 1
  $UV pip install --python "$V/bin/python" numpy open_clip_torch timm transformers ultralytics rfdetr \
      huggingface_hub safetensors pillow || exit 1
fi
cd model/foundation_study
MODELS="dino_vits14 clip_vitb16 mobileclipv2 siglip2 yolo rfdetr dinov3"
"$V/bin/python" shared/download_weights.py --models $MODELS || echo "download reported gaps (gated / unavailable models are expected)"
for m in $MODELS; do   # one model per call: a crash in one cannot lose the others
  "$V/bin/python" shared/bench.py --models "$m" --dtypes fp32 fp16 || echo "bench $m failed"
done
"$V/bin/python" - <<'PY'
import datetime, json, os, subprocess, sys
info = {"venv": sys.prefix, "written": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "skipped": {"locateanything3d": "9.5 GB / 8-11 GB VRAM; old-stack number retained, labelled",
                    "detic": "detectron2 not buildable on py3.12 / torch 2.14; old-stack number retained, labelled",
                    "vfmdetr": "no public code or weights found: two web searches on 2026-11-17 returned only unrelated papers (VFMTok, RT-DETRv4); the paper itself was not located, so this is not exhaustive"}}
json.dump(info, open("../../data_results/o5_latency_note.json", "w"), indent=1)
PY
