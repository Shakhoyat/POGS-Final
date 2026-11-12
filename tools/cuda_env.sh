# Source this before building/running anything that uses gsplat:
#   source tools/cuda_env.sh
# It points torch's JIT extension builder at the CUDA 13.0 toolkit installed
# from pip into .venv (nvidia-cuda-nvcc / -cccl / -crt / -nvvm) and at the
# Python 3.14 headers unpacked from Ubuntu's python3.14-dev .deb (no sudo).
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_SP="$_ROOT/.venv/lib/python3.14/site-packages"
export CUDA_HOME="$_SP/nvidia/cu13"
export PATH="$_ROOT/.venv/bin:$CUDA_HOME/bin:$PATH"
export CPATH="$_ROOT/.venv/sysinclude/usr/include/python3.14:$_ROOT/.venv/sysinclude/usr/include${CPATH:+:$CPATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export MAX_JOBS="${MAX_JOBS:-8}"
