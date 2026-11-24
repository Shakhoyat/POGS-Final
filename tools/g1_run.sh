#!/usr/bin/env bash
# G1 chain, one GPU process at a time (docs/note/g1_generality_preregistration.md).
set -u
cd "$(dirname "$0")/.."
source tools/cuda_env.sh
LOG=data_results/_pipeline
mkdir -p "$LOG"
run() { echo "[g1_run] $(date +%H:%M:%S) $*"; .venv/bin/python -m experiments.g1_generality "$@" >> "$LOG/g1_$1_${3:-x}.log" 2>&1 || { echo "[g1_run] FAILED $*"; touch "$LOG/g1.failed"; exit 1; }; }
run dose --seed 0
run sweeps --seed 0
for s in 1 2; do run train --seed $s; run dose --seed $s; run sweeps --seed $s; done
run summarise
touch "$LOG/g1.done"; echo "[g1_run] $(date +%H:%M:%S) done"
