# Source before running anything in the official venv (8.OFF): gsplat's JIT extension needs the same toolchain env
# it was built with (tools/official_env.sh).
V=$HOME/venvs/pogs-official
_SP=$($V/bin/python -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')
export CUDA_HOME=$_SP/nvidia/cu13
export PATH=$V/bin:$CUDA_HOME/bin:$PATH
export CPATH=$($V/bin/python -c 'import sysconfig;print(sysconfig.get_paths()["include"])')
export TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=16
