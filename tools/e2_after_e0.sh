#!/usr/bin/env bash
# Plan rev. 2, E2: run after E0 (which itself waits for the `realtime` latency stage), so E2 never
# overlaps a latency measurement. 10 seeds x 4 settings, 2 shards. Log: data_results/_pipeline/e2.log
cd "$(dirname "$0")/.."
S=data_results/_pipeline
while [ ! -f $S/e0.done ] && [ ! -f $S/e0.failed ]; do sleep 300; done
source tools/cuda_env.sh >/dev/null 2>&1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[e2] start $(date -Is)" >> $S/e2.log
pids=()
N=${E2_SHARDS:-1}   # 2026-11-23: parallel shards stall on this GPU (see LEDGER); default 1
for ((k=0; k<N; k++)); do .venv/bin/python -u -m experiments.e2_information --trials 10 --shard $k --nshards $N >> $S/e2_shard$k.log 2>&1 & pids+=($!); sleep 20; done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
if [ $rc -eq 0 ] && .venv/bin/python -m experiments.e2_information --summarise >> $S/e2.log 2>&1; then
  echo "[e2] done $(date -Is)" >> $S/e2.log; touch $S/e2.done
else echo "[e2] FAILED $(date -Is)" >> $S/e2.log; touch $S/e2.failed; fi
