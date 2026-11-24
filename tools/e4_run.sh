#!/usr/bin/env bash
# E4 boundary-torque sweeps on the three G1 splats, then the summary (pre-registered rules first). One GPU process.
set -euo pipefail
cd "$(dirname "$0")/.."
source tools/cuda_env.sh
for s in 0 1 2; do .venv/bin/python -u -m experiments.e4_torque run --seed $s; done
.venv/bin/python -u -m experiments.e4_torque summarise
