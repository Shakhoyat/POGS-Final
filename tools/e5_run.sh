#!/usr/bin/env bash
# E5 (pre-registered D1, D2): waits for G1c to finish (one heavy GPU job at a time), then runs. Lab hours gate via supervisor.
set -euo pipefail
cd "$(dirname "$0")/.."
P=data_results/_pipeline
until [ -e $P/g1c.done ] || [ -e $P/g1c.failed ]; do sleep 30; done
source tools/cuda_env.sh
.venv/bin/python -u -m experiments.e5_dynamic run
.venv/bin/python -u -m experiments.e5_dynamic summarise
