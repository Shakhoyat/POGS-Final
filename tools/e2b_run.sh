#!/usr/bin/env bash
# E2b: POGS + RGB on O4 seeds 0-24, 2 shards. Log data_results/_pipeline/e2b*.log, markers e2b.done/failed.
cd "$(dirname "$0")/.."
source tools/cuda_env.sh >/dev/null 2>&1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=data_results/_pipeline; pids=()
for k in 0 1; do .venv/bin/python -u -m experiments.e2b_rgb --seeds 25 --shard $k --nshards 2 > $S/e2b_shard$k.log 2>&1 & pids+=($!); sleep 20; done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
if [ $rc -eq 0 ] && .venv/bin/python -m experiments.e2b_rgb --summarise > $S/e2b.log 2>&1; then touch $S/e2b.done; else touch $S/e2b.failed; fi
