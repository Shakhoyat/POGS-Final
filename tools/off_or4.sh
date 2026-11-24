#!/usr/bin/env bash
# 8.OFF OR4 (docs/note/off_or4_tracking_preregistration.md): official tracking loop on O4 approach+lift frames.
# Waits for the OR1-OR3 chain (off.done/failed) and the sequence export. One GPU process.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
P=data_results/_pipeline
until [ -e $P/off.done ] || [ -e $P/off.failed ]; do sleep 30; done
until grep -q "grasp_-15_s9" $P/off_seq_export.log 2>/dev/null; do sleep 30; done
source tools/official_cuda_env.sh
CFG=$(ls -t $HOME/pogs_external/off/runs/tool_table_ns/pogs/*/config.yml | head -1)
$HOME/venvs/pogs-official/bin/python -u -m experiments.off_repro.official_track run --config "$CFG"
$HOME/venvs/pogs-official/bin/python -u -m experiments.off_repro.official_track summarise
