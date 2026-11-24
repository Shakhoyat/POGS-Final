#!/usr/bin/env bash
# 8.OFF chain (docs/note/off_reproduction_preregistration.md): official training -> our frames -> official-loss sweeps
# -> summary. One GPU process at a time. Run via: tools/supervisor.sh run off bash tools/off_run.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OPY=$HOME/venvs/pogs-official/bin/python
source tools/official_cuda_env.sh
W=$HOME/pogs_external/off
( cd "$ROOT" && $OPY -c "import experiments.off_repro.shims, pogs.pogs" ) >/dev/null 2>&1 || { echo "official env not ready"; exit 1; }
CFG=$(ls -t $W/runs/tool_table_ns/pogs/*/config.yml 2>/dev/null | head -1 || true)
if [ -z "$CFG" ] || [ ! -f "$W/train.done" ]; then
  echo "== official training"
  ( cd "$ROOT" && $OPY -u -m experiments.off_repro.official_train )
  touch "$W/train.done"
  CFG=$(ls -t $W/runs/tool_table_ns/pogs/*/config.yml | head -1)
fi
echo "config: $CFG"
echo "== frames (main venv)"
( source tools/cuda_env.sh; .venv/bin/python -u -m experiments.off_repro.frames )
source tools/official_cuda_env.sh
echo "== official-loss sweeps"
$OPY -u -m experiments.off_repro.official_sweeps run --config "$CFG"
$OPY -u -m experiments.off_repro.official_sweeps summarise
