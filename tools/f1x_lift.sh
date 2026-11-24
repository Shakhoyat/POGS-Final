#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source tools/cuda_env.sh
.venv/bin/python -u -m experiments.f1x_wide lift
