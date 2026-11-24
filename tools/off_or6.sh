#!/usr/bin/env bash
# OR6: official tracker on 40 O4 trials (pre-registered), then scoring with our judge.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
until grep -q "grasp_-30_s9" data_results/_pipeline/off_seq_full_export.log 2>/dev/null; do sleep 20; done
source tools/official_cuda_env.sh
CFG=$(ls -t $HOME/pogs_external/off/runs/tool_table_ns/pogs/*/config.yml | head -1)
OFF_FULL=1 $HOME/venvs/pogs-official/bin/python -u -m experiments.off_repro.official_track run --config "$CFG"
.venv/bin/python -u -m experiments.off_repro.or6_score
