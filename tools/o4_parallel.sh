#!/usr/bin/env bash
# Run O4 as N shards sharing the GPU (O4 measures accuracy, not latency, so sharing is safe), then aggregate.
# Resumable: finished cases are skipped via their row.json. Usage: tools/o4_parallel.sh [N_SHARDS]
cd "$(dirname "$0")/.."
# gsplat needs the CUDA toolkit env; without it the shards fall back to a CPU-bound path (2026-11-23 stalls)
source tools/cuda_env.sh >/dev/null 2>&1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
N=${1:-3}; PY=".venv/bin/python -u"
pids=()
for ((k=0; k<N; k++)); do
  $PY -m experiments.o4_ingrasp --shard $k --nshards $N > data_results/_pipeline/o4_shard$k.log 2>&1 &
  pids+=($!)
  sleep 20   # stagger start-up so model loading does not spike RAM/VRAM together
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
[ $rc -eq 0 ] || { echo "a shard failed; see data_results/_pipeline/o4_shard*.log"; exit 1; }
$PY -m experiments.o4_ingrasp --aggregate
