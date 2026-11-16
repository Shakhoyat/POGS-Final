#!/usr/bin/env bash
# Unattended, resumable experiment pipeline (docs/PLAN.md). No supervision needed.
#
#   tools/run_pipeline.sh --detach     start in the background (survives logout / session end)
#   tools/run_pipeline.sh --status     what is done / running / failed
#   tools/run_pipeline.sh --stop       stop the running pipeline
#   tools/run_pipeline.sh              run in the foreground
#   tools/run_pipeline.sh --only o3    run one stage (also: --redo o3 to force it)
#
# Each stage runs once: success writes data_results/_pipeline/<stage>.done and is skipped on
# the next start; a failed stage is logged and the pipeline moves on, so one bug does not
# stop the night. Re-running after a fix repeats only failed / new stages. Every experiment
# is itself resumable (run cache keyed by checkpoint), so an interrupted stage loses little.
# Logs: data_results/_pipeline/<stage>.log     Status: data_results/_pipeline/status.tsv
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
P=data_results/_pipeline
mkdir -p "$P"
PY=".venv/bin/python -u"

# name | command                      (order = execution order; add new stages here)
STAGES=(
  "tests|$PY -m pytest -q -p no:cacheprovider tests"
  "repro|$PY -m experiments.reproduce_check"
  "o1|$PY -m experiments.o1_grouping_splats"
  "o3_smoke|$PY -m experiments.o3_perturbation --seeds 0"
  "o3|$PY -m experiments.o3_perturbation"
  "o6|$PY -m experiments.o6_speed"
  "v1|$PY -m experiments.v1_base_pogs.run_all --no-videos"
  "v2|$PY -m experiments.v2_kalman_pogs.run_all --no-videos"
  "realtime|$PY -m experiments.realtime.run_all --no-videos"
  "v3|$PY -m experiments.v3_scene_mustard.run_all --no-videos"
)

status() {
  [ -f "$P/pid" ] && kill -0 "$(cat "$P/pid")" 2>/dev/null && echo "pipeline RUNNING (pid $(cat "$P/pid"))" || echo "pipeline not running"
  for s in "${STAGES[@]}"; do
    n="${s%%|*}"
    if [ -f "$P/$n.done" ]; then st="done    $(cat "$P/$n.done")"
    elif [ -f "$P/$n.running" ]; then st="RUNNING since $(cat "$P/$n.running")"
    elif [ -f "$P/$n.failed" ]; then st="FAILED  $(cat "$P/$n.failed")  -> $P/$n.log"
    else st="pending"; fi
    printf "  %-10s %s\n" "$n" "$st"
  done
}

case "${1:-}" in
  --status) status; exit 0 ;;
  --stop)   [ -f "$P/pid" ] && pkill -TERM -g "$(ps -o pgid= -p "$(cat "$P/pid")" | tr -d ' ')" && echo stopped; rm -f "$P"/*.running; exit 0 ;;
  --detach) shift; setsid nohup "$0" "$@" > "$P/pipeline.log" 2>&1 < /dev/null &
            echo "started; check with: tools/run_pipeline.sh --status"; exit 0 ;;
esac
ONLY=""; REDO=""
[ "${1:-}" = "--only" ] && ONLY="$2"
[ "${1:-}" = "--redo" ] && { ONLY="$2"; REDO=1; }

if [ -f "$P/pid" ] && kill -0 "$(cat "$P/pid")" 2>/dev/null; then echo "already running (pid $(cat "$P/pid"))"; exit 1; fi
echo $$ > "$P/pid"; trap 'rm -f "$P/pid" "$P"/*.running' EXIT
rm -f "$P"/*.running
source tools/cuda_env.sh

# One GPU: wait for any experiment started outside the pipeline to finish first.
while pgrep -f "python.* -m experiments\." > /dev/null; do echo "$(date '+%F %T') waiting for a running experiment"; sleep 60; done

for s in "${STAGES[@]}"; do
  n="${s%%|*}"; cmd="${s#*|}"
  [ -n "$ONLY" ] && [ "$n" != "$ONLY" ] && continue
  [ -n "$REDO" ] && rm -f "$P/$n.done"
  [ -f "$P/$n.done" ] && continue
  rm -f "$P/$n.failed"; date '+%F %T' > "$P/$n.running"
  echo "$(date '+%F %T') START $n"
  t0=$(date +%s)
  if $cmd > "$P/$n.log" 2>&1; then
    echo "$(date '+%F %T') ($(( ($(date +%s) - t0) / 60 )) min, commit $(git rev-parse --short HEAD))" > "$P/$n.done"; res=ok
  else
    echo "$(date '+%F %T') exit $? after $(( ($(date +%s) - t0) / 60 )) min" > "$P/$n.failed"; res=FAILED
  fi
  rm -f "$P/$n.running"
  printf "%s\t%s\t%s\n" "$(date '+%F %T')" "$n" "$res" >> "$P/status.tsv"
  echo "$(date '+%F %T') END   $n $res"
done
echo "$(date '+%F %T') pipeline finished"; status
