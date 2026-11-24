#!/usr/bin/env bash
# Central supervisor for unattended research work (user request 2026-11-25).
#
#   tools/supervisor.sh status          GPU, lab-hours window, running jobs, git sync state, open findings
#   tools/supervisor.sh labhours        exit 0 if NOW is inside a lab class window (no heavy GPU work then)
#   tools/supervisor.sh run NAME CMD..  run a job with markers/logs in data_results/_pipeline; waits outside
#                                       lab hours before starting; NAME.done / NAME.failed on exit
#   tools/supervisor.sh harvest         copy new "FINDING:" lines from job logs into docs/LEDGER.md (Findings inbox)
#   tools/supervisor.sh sync [MSG]      harvest, then commit docs/code/small summaries and push research-plan-rev2
#   tools/supervisor.sh loop [SECS]     harvest + sync every SECS (default 1800) until stopped (detached use: nohup)
#
# Findings protocol: any experiment may print a line
#     FINDING: <id> | <one-sentence result with numbers> | <evidence path>
# `harvest` appends each new <id> once to the "Findings inbox" table in docs/LEDGER.md with the source log and
# Paper = "pending". Claude (or the user) later sets Paper to "in: <section>" or "out: <reason>" when deciding what
# reaches docs/paper_draft/main.tex. The ledger is the single source; the paper only takes from it.
#
# Lab routine (lab-routine.jpg, KUET CSE, updated 31.08.2026; Dhaka time):
#   Sun, Mon, Wed 08:00-10:30 and 14:30-17:00; Tue 14:30-17:00; Thu/Fri/Sat free.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
P=data_results/_pipeline
LEDGER=docs/LEDGER.md
mkdir -p "$P"
export TZ=Asia/Dhaka
BRANCH=research-plan-rev2

labhours() {  # exit 0 = inside a class window
  local dow hm
  dow=$(date +%u)            # 1=Mon .. 7=Sun
  hm=$((10#$(date +%H%M)))
  within() { [ "$hm" -ge "$1" ] && [ "$hm" -lt "$2" ]; }
  case "$dow" in
    7|1|3) within 800 1030 || within 1430 1700 ;;
    2)     within 1430 1700 ;;
    *)     return 1 ;;
  esac
}

wait_outside_lab() {
  while labhours; do
    echo "$(date '+%F %T') lab hours: waiting" >> "$P/supervisor.log"
    sleep 300
  done
}

gpu() { (/usr/lib/wsl/lib/nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "nvidia-smi n/a"); }

status() {
  echo "time      $(date '+%a %F %T %Z')"
  if labhours; then echo "lab hours YES (no heavy GPU work)"; else echo "lab hours no"; fi
  echo "gpu       $(gpu)"
  echo "running:"; for f in "$P"/*.running; do [ -e "$f" ] || continue
    pid=$(cat "$f"); if kill -0 "$pid" 2>/dev/null; then echo "  $(basename "$f" .running) pid $pid"; else echo "  $(basename "$f" .running) (stale marker)"; fi; done
  echo "recent:"; ls -t "$P"/*.done "$P"/*.failed 2>/dev/null | head -5 | sed 's/^/  /'
  git fetch -q origin 2>/dev/null
  echo "git       local $(git rev-parse --short HEAD)  remote $(git rev-parse --short origin/$BRANCH 2>/dev/null)  dirty(non-data) $(git status --porcelain -- . ':!data_results' | wc -l)"
  echo "findings pending: $(grep -c '| pending |' "$LEDGER" 2>/dev/null)"
}

run() {
  local name="$1"; shift
  wait_outside_lab
  rm -f "$P/$name.done" "$P/$name.failed"
  echo $$ > "$P/$name.running"
  echo "$(date '+%F %T') start $name: $*" >> "$P/supervisor.log"
  if "$@" > "$P/$name.log" 2>&1; then
    touch "$P/$name.done"; printf '%s\t%s\tok\n' "$(date '+%F %T')" "$name" >> "$P/status.tsv"
  else
    touch "$P/$name.failed"; printf '%s\t%s\tFAILED\n' "$(date '+%F %T')" "$name" >> "$P/status.tsv"
  fi
  rm -f "$P/$name.running"
  echo "$(date '+%F %T') end $name" >> "$P/supervisor.log"
  harvest
}

harvest() {
  grep -q '^## Findings inbox' "$LEDGER" || printf '\n## Findings inbox\nWritten by `tools/supervisor.sh harvest` from `FINDING:` lines in job logs. Paper: pending / in: <section> / out: <reason>.\n\n| Date | ID | Finding | Evidence | Source log | Paper |\n|---|---|---|---|---|---|\n' >> "$LEDGER"
  local n=0
  while IFS= read -r hit; do
    local src="${hit%%:FINDING:*}" body="${hit#*FINDING:}"
    local id ev txt
    id=$(echo "$body" | cut -d'|' -f1 | xargs)
    txt=$(echo "$body" | cut -d'|' -f2 | xargs)
    ev=$(echo "$body" | cut -d'|' -f3 | xargs)
    [ -z "$id" ] && continue
    grep -q "| $id |" "$LEDGER" && continue
    # insert at the end of the Findings inbox table
    python3 - "$LEDGER" "| $(date +%F) | $id | ${txt//|//} | $ev | \`$src\` | pending |" <<'PY'
import sys
p, row = sys.argv[1], sys.argv[2]
s = open(p).read().split('\n')
i = s.index('## Findings inbox')
j = i + 1
while j < len(s) and not s[j].startswith('|'): j += 1
while j < len(s) and s[j].startswith('|'): j += 1
s.insert(j, row)
open(p, 'w').write('\n'.join(s))
PY
    n=$((n+1))
  done < <(grep -H 'FINDING:' "$P"/*.log 2>/dev/null)
  echo "harvest: $n new finding(s)"
}

sync() {
  harvest
  local msg="${1:-supervisor: ledger, docs and summaries}"
  # small, reviewable paths only; bulk data_results stays local (.gitignore + this list)
  git add -- docs tools experiments model metrics tests data/scene_generator .gitignore .github CLAUDE.md 2>/dev/null
  git add -- data_results/*/summary.md data_results/*/summary.csv 2>/dev/null
  git add -- data_results/_pipeline/status.tsv 2>/dev/null
  if git diff --cached --quiet; then echo "sync: nothing to commit"; else
    git commit -q -F - <<EOF
$msg

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016TFSbkPUR4K1RftUUD3u9R
EOF
  fi
  git fetch -q origin
  if ! git merge-base --is-ancestor "origin/$BRANCH" HEAD; then git merge -q --no-edit "origin/$BRANCH" || { echo "sync: merge conflict, not pushed"; return 1; }; fi
  git push -q origin "$BRANCH" && echo "sync: pushed $(git rev-parse --short HEAD)"
}

loop() {
  local secs="${1:-1800}"
  while true; do sync "supervisor: periodic sync" >> "$P/supervisor.log" 2>&1; sleep "$secs"; done
}

cmd="${1:-status}"; shift || true
case "$cmd" in
  status) status ;;
  labhours) labhours ;;
  run) run "$@" ;;
  harvest) harvest ;;
  sync) sync "$@" ;;
  loop) loop "$@" ;;
  *) sed -n 2,20p "$0"; exit 2 ;;
esac
