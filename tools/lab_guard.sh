#!/usr/bin/env bash
# Stop official GPU jobs at the start of a lab window; the resumable OR6 chain restarts after it via supervisor.sh.
cd "$(dirname "$0")/.."
P=data_results/_pipeline
while true; do
  if bash tools/supervisor.sh labhours; then
    pids=$(pgrep -f "experiments.off_repro.official_track run" || true)
    if [ -n "$pids" ]; then
      kill $pids; sleep 5
      echo "$(date '+%F %T') lab_guard: stopped OR6 for lab hours; resuming after" >> $P/supervisor.log
      sleep 30
      rm -f $P/off_or6.running
      sed -i 's/^until grep -q "grasp_-30_s9".*$/true/' tools/off_or6.sh
      nohup bash tools/supervisor.sh run off_or6 bash tools/off_or6.sh >/dev/null 2>&1 &
    fi
  fi
  [ -e $P/off_or6.done ] && exit 0
  sleep 60
done
