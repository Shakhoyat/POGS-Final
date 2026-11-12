# Source this before building/running anything that uses gsplat:
#   source tools/cuda_env.sh
# It points torch's JIT extension builder at the CUDA 13.0 toolkit installed
# from pip into .venv (nvidia-cuda-nvcc / -cccl / -crt / -nvvm) and at the
# Python headers: the interpreter's own include dir when it ships one (uv /
# python-build-standalone), else headers unpacked from Ubuntu's pythonX.Y-dev .deb.
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_PYV="$("$_ROOT/.venv/bin/python" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
_SP="$_ROOT/.venv/lib/python$_PYV/site-packages"
_PYINC="$("$_ROOT/.venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
export CUDA_HOME="$_SP/nvidia/cu13"
export PATH="$_ROOT/.venv/bin:$CUDA_HOME/bin:$PATH"
if [ -f "$_PYINC/Python.h" ]; then
    export CPATH="$_PYINC${CPATH:+:$CPATH}"
else
    export CPATH="$_ROOT/.venv/sysinclude/usr/include/python$_PYV:$_ROOT/.venv/sysinclude/usr/include${CPATH:+:$CPATH}"
fi
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export MAX_JOBS="${MAX_JOBS:-8}"
