#!/usr/bin/env bash
# One-time setup of a working gsplat CUDA backend inside .venv, without sudo.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
.venv/bin/pip install "nvidia-cuda-nvcc==13.0.*" "nvidia-cuda-cccl==13.0.*" \
    "nvidia-cuda-crt==13.0.*" "nvidia-nvvm==13.0.*" ninja
C=.venv/lib/python3.14/site-packages/nvidia/cu13
ln -sfn lib "$C/lib64"
ln -sf libcudart.so.13 "$C/lib/libcudart.so"
PYVER=$(dpkg-query -W -f='${Version}' python3.14)
TMP=$(mktemp -d)
( cd "$TMP" && apt-get download "libpython3.14-dev=$PYVER" "python3.14-dev=$PYVER" )
mkdir -p .venv/sysinclude
for f in "$TMP"/*.deb; do dpkg-deb -x "$f" .venv/sysinclude; done
rm -rf "$TMP"
source tools/cuda_env.sh
.venv/bin/python -c "from gsplat.cuda._backend import _C; assert _C is not None; print('gsplat CUDA backend OK')"
