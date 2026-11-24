#!/usr/bin/env bash
# G1c dose-response on 4 objects (pre-registered C1, C2). One GPU process.
set -euo pipefail
cd "$(dirname "$0")/.."
source tools/cuda_env.sh
for s in 0 1 2; do .venv/bin/python -u -m experiments.g1c_dose_objects run --seed $s; done
.venv/bin/python -u -m experiments.g1c_dose_objects summarise
