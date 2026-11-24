#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
P=data_results/_pipeline
until [ -e $P/off_or4.done ] || [ -e $P/off_or4.failed ]; do sleep 20; done
source tools/cuda_env.sh
.venv/bin/python -u -m experiments.off_repro.silhouette_diag
