#!/usr/bin/env bash
# Build an environment for the OFFICIAL POGS code (8.OFF, extension 1) without conda or sudo.
# Release: Python 3.10, CUDA 11.8, torch 2.6 (README). Here: system gcc 15 cannot host CUDA 11.8's nvcc, so we use
# the project's working toolchain (Python 3.12, torch 2.14.1+cu130, CUDA 13.0 nvcc from pip, gsplat 1.5.3).
# Deviations are listed in docs/note/off_reproduction_preregistration.md. Official code itself is not modified.
#   bash tools/official_env.sh      (idempotent; log via tools/supervisor.sh run off_env bash tools/official_env.sh)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OFF=$HOME/third_party_pinned/POGS-official
V=$HOME/venvs/pogs-official
PY=$V/bin/python
cd "$ROOT"
[ -x "$PY" ] || uv venv -q --python 3.10 "$V"   # official: Python 3.10 (3.12 rejects a dataclass default in pogs)
export VIRTUAL_ENV=$V
uvp() { uv pip install --python "$PY" "$@"; }
echo "== torch"; uvp -q "torch==2.14.1+cu130" "torchvision==0.29.1+cu130" --index-url https://download.pytorch.org/whl/cu130
echo "== toolchain"; uvp -q "nvidia-cuda-nvcc==13.0.*" "nvidia-cuda-cccl==13.0.*" "nvidia-cuda-crt==13.0.*" "nvidia-nvvm==13.0.*" ninja setuptools wheel
SP=$($PY -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')
export CUDA_HOME=$SP/nvidia/cu13
ln -sfn lib "$CUDA_HOME/lib64"; ln -sf libcudart.so.13 "$CUDA_HOME/lib/libcudart.so"
export PATH=$V/bin:$CUDA_HOME/bin:$PATH
PYINC=$($PY -c 'import sysconfig;print(sysconfig.get_paths()["include"])')
[ -f "$PYINC/Python.h" ] && export CPATH=$PYINC
export TORCH_CUDA_ARCH_LIST=8.9 TCNN_CUDA_ARCHITECTURES=89 MAX_JOBS=16
# glibc >= 2.41 declares rsqrt/rsqrtf noexcept; CUDA 13.0's crt/math_functions.h does not -> nvcc error in tcnn.
H=$CUDA_HOME/include/crt/math_functions.h
grep -q "rsqrt(double x) noexcept" "$H" || sed -i.orig -e 's/\(double *rsqrt(double x)\);/\1 noexcept (true);/' -e 's/\(float *rsqrtf(float x)\);/\1 noexcept (true);/' "$H"
echo "== nerfstudio (official submodule 5003d0e)"; uvp -q -e "$OFF/pogs/dependencies/nerfstudio"
echo "== gsplat, warp, extras"; uvp -q "gsplat==1.5.3" warp-lang jaxtyping rich "fast_simplification==0.1.9" scikit-learn hdbscan
echo "== pogs (official, editable)"; uvp -q -e "$OFF"
echo "== gsplat CUDA JIT"; $PY -c "from gsplat.cuda._backend import _C; assert _C is not None; print('gsplat OK')"
echo "== tiny-cuda-nn (bindings/torch, build from source)"
if [ "${SKIP_TCNN:-1}" != "1" ] && ! $PY -c "import tinycudann" 2>/dev/null; then   # default: skip (nvcc 13 ICE; stand-in in shims)
  [ -d "$HOME/third_party_pinned/tiny-cuda-nn" ] || git clone -q --recursive https://github.com/NVlabs/tiny-cuda-nn "$HOME/third_party_pinned/tiny-cuda-nn"
  # torch 2.14 headers require C++20; tcnn's setup.py pins C++17 for CUDA >= 11 (build flag only)
  sed -i.orig 's/^cpp_standard = 17$/cpp_standard = 20/; s/\t\t\tcpp_standard = 17$/\t\t\tcpp_standard = 20/' "$HOME/third_party_pinned/tiny-cuda-nn/bindings/torch/setup.py"
  # fmt v9 (bundled) consteval format strings do not compile under nvcc + C++20: define FMT_CONSTEVAL empty
  grep -q "DFMT_CONSTEVAL" "$HOME/third_party_pinned/tiny-cuda-nn/bindings/torch/setup.py" || python3 - "$HOME/third_party_pinned/tiny-cuda-nn/bindings/torch/setup.py" <<'PYEOF'
import sys;p=sys.argv[1];s=open(p).read()
s=s.replace('\t"-U__CUDA_NO_HALF2_OPERATORS__",\n]','\t"-U__CUDA_NO_HALF2_OPERATORS__",\n\t"-DFMT_CONSTEVAL=",\n]')
s=s.replace('\tbase_cflags = [f"-std=c++{cpp_standard}"]','\tbase_cflags = [f"-std=c++{cpp_standard}", "-DFMT_CONSTEVAL="]')
open(p,'w').write(s)
PYEOF
  ( cd "$HOME/third_party_pinned/tiny-cuda-nn/bindings/torch" && uvp --no-build-isolation . ) || true
fi
$PY -c "import tinycudann" 2>/dev/null && echo "tcnn OK" || echo "tcnn NOT built: experiments/off_repro/shims.py uses the pure-torch stand-in (deviation 5)"
echo "== import check (with shims)"; ( cd "$ROOT" && $PY -c "import experiments.off_repro.shims, nerfstudio, pogs.pogs, pogs.tracking.rigid_group_optimizer as r; print('official POGS imports OK')" )
echo "ENV_DONE"
