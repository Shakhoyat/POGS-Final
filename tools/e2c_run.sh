#!/usr/bin/env bash
# E2c (direction B, step 1): POGS with a heavier depth term on O4 seeds 0-24, 2 shards per variant, variants
# in order. Logs data_results/_pipeline/e2c_<v>*.log, markers e2c_<v>.done/failed. Usage: tools/e2c_run.sh depth30 depth100
cd "$(dirname "$0")/.."
source tools/cuda_env.sh >/dev/null 2>&1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=data_results/_pipeline
for v in "$@"; do
  pids=()
  for k in 0 1; do .venv/bin/python -u -m experiments.e2b_rgb --variant $v --seeds 25 --shard $k --nshards 2 > $S/e2c_${v}_shard$k.log 2>&1 & pids+=($!); sleep 20; done
  rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
  if [ $rc -eq 0 ] && .venv/bin/python -m experiments.e2b_rgb --variant $v --summarise > $S/e2c_$v.log 2>&1; then touch $S/e2c_$v.done; else touch $S/e2c_$v.failed; fi
done
