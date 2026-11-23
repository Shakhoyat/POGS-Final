#!/usr/bin/env bash
# After e2c_maskobj: (1) 5-seed check of dino_loss_region=object_frame_start (B1f predicts it sticks), 2 shards;
# (2) AnyUp (pinned) acceptance sweeps alone on the GPU. Markers e2c_maskstart5.done/failed, b2_v2up.done/failed.
cd "$(dirname "$0")/.."
source tools/cuda_env.sh >/dev/null 2>&1
S=data_results/_pipeline
while [ ! -e $S/e2c_maskobj.done ] && [ ! -e $S/e2c_maskobj.failed ]; do sleep 60; done
pids=()
for k in 0 1; do .venv/bin/python -u -m experiments.e2b_rgb --variant maskstart --seeds 5 --shard $k --nshards 2 > $S/e2c_maskstart5_shard$k.log 2>&1 & pids+=($!); sleep 20; done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
if [ $rc -eq 0 ] && .venv/bin/python -m experiments.e2b_rgb --variant maskstart --summarise > $S/e2c_maskstart5.log 2>&1; then touch $S/e2c_maskstart5.done; else touch $S/e2c_maskstart5.failed; fi
rm -f $S/b2_v2up.deferred $S/b2_v2up.failed
bash tools/b2_run.sh v2up
