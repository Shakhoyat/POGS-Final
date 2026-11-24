#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source tools/official_cuda_env.sh
CFG=$(ls -t $HOME/pogs_external/off/runs/tool_table_ns/pogs/*/config.yml | head -1)
OFF_PROBE=1 $HOME/venvs/pogs-official/bin/python -u -m experiments.off_repro.official_track run --config "$CFG"
