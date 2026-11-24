#!/usr/bin/env bash
# Sequential GPU chain: OFF-S swap (exploratory) then OR4 (pre-registered). One GPU process at a time.
set -uo pipefail
cd "$(dirname "$0")/.."
bash tools/supervisor.sh run off_swap bash tools/off_swap.sh
sed -i 's/^until \[ -e \$P\/off.done \]; do sleep 30; done.*$/true/' tools/off_or4.sh
bash tools/supervisor.sh run off_or4 bash tools/off_or4.sh
