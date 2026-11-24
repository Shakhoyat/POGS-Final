#!/usr/bin/env bash
# 8.OFF-S (exploratory): export the official splat, then run our tracking code on it (and on our splat as control).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
CFG=$(ls -t $HOME/pogs_external/off/runs/tool_table_ns/pogs/*/config.yml | head -1)
( source tools/official_cuda_env.sh; $HOME/venvs/pogs-official/bin/python -u -m experiments.off_repro.export_official_splat --config "$CFG" )
( source tools/cuda_env.sh; .venv/bin/python -u -m experiments.off_repro.swap_ours_on_official )
