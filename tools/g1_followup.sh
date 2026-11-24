#!/usr/bin/env bash
# After the G1 chain: G1b translation sweeps, G2 RSRD sweeps, summaries. One GPU process at a time.
set -u
cd "$(dirname "$0")/.."
source tools/cuda_env.sh
LOG=data_results/_pipeline
while [ ! -f "$LOG/g1.done" ] && [ ! -f "$LOG/g1.failed" ]; do sleep 30; done
[ -f "$LOG/g1.failed" ] && { echo "[g1_followup] G1 failed, not starting"; exit 1; }
echo "[g1_followup] $(date +%H:%M:%S) trans"; .venv/bin/python -m experiments.g1_generality trans --seed 0 > "$LOG/g1b_trans.log" 2>&1 || { touch "$LOG/g1f.failed"; exit 1; }
echo "[g1_followup] $(date +%H:%M:%S) g2"; .venv/bin/python -m experiments.g2_rsrd sweeps > "$LOG/g2_sweeps.log" 2>&1 || { touch "$LOG/g1f.failed"; exit 1; }
.venv/bin/python -m experiments.g2_rsrd summarise > "$LOG/g2_summary.log" 2>&1
.venv/bin/python -m experiments.g1_generality summarise > "$LOG/g1_summary.log" 2>&1
touch "$LOG/g1f.done"; echo "[g1_followup] $(date +%H:%M:%S) done"
