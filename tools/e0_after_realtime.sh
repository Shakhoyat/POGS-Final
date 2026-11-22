#!/usr/bin/env bash
# Plan rev. 2, E0: re-run O4 with full SE(3) storage (schema 2) once the latency-sensitive `realtime`
# pipeline stage has finished (latency stages must run alone; PLAN 3.4). Accuracy-only, so it may share
# the GPU with the pipeline's later v3 stage. Log: data_results/_pipeline/e0.log
# 2 shards (2026-11-23): 3 shards filled 12 GB VRAM (11.95 GB) and stalled at 4% GPU utilisation.
cd "$(dirname "$0")/.."
S=data_results/_pipeline
while [ ! -f $S/realtime.done ] && [ ! -f $S/realtime.failed ]; do sleep 300; done
echo "[e0] start $(date -Is)" >> $S/e0.log
if tools/o4_parallel.sh ${E0_SHARDS:-2} >> $S/e0.log 2>&1; then echo "[e0] done $(date -Is)" >> $S/e0.log; touch $S/e0.done
else echo "[e0] FAILED $(date -Is)" >> $S/e0.log; touch $S/e0.failed; fi
