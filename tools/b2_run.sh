#!/usr/bin/env bash
# B2 (direction B, step 2): feature-only re-distillation + B1 acceptance sweeps per backbone, in order.
# Logs data_results/_pipeline/b2_<kind>.log, markers b2_<kind>.done/failed. Usage: tools/b2_run.sh v2 v2reg v3s
cd "$(dirname "$0")/.."
source tools/cuda_env.sh >/dev/null 2>&1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=data_results/_pipeline
for k in "$@"; do
  if .venv/bin/python -u -m experiments.b2_redistil $k > $S/b2_$k.log 2>&1; then touch $S/b2_$k.done; else touch $S/b2_$k.failed; fi
done
