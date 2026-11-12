#!/usr/bin/env bash
# One-time setup of a working gsplat CUDA backend inside .venv, without sudo.
# Works for any Python in .venv (3.12 target; 3.14 previously).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PIP=(.venv/bin/python -m pip)
"${PIP[@]}" install "nvidia-cuda-nvcc==13.0.*" "nvidia-cuda-cccl==13.0.*" \
    "nvidia-cuda-crt==13.0.*" "nvidia-nvvm==13.0.*" ninja
PYV=$(.venv/bin/python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
C=.venv/lib/python$PYV/site-packages/nvidia/cu13
ln -sfn lib "$C/lib64"
ln -sf libcudart.so.13 "$C/lib/libcudart.so"
PYINC=$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
if [ ! -f "$PYINC/Python.h" ]; then
    # System interpreter without headers: unpack Ubuntu's -dev packages (no sudo).
    PKGVER=$(dpkg-query -W -f='${Version}' "python$PYV")
    TMP=$(mktemp -d)
    ( cd "$TMP" && apt-get download "libpython$PYV-dev=$PKGVER" "python$PYV-dev=$PKGVER" )
    mkdir -p .venv/sysinclude
    for f in "$TMP"/*.deb; do dpkg-deb -x "$f" .venv/sysinclude; done
    rm -rf "$TMP"
fi
source tools/cuda_env.sh
.venv/bin/python -c "from gsplat.cuda._backend import _C; assert _C is not None; print('gsplat CUDA backend OK')"
