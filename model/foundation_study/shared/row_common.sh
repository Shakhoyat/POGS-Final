#!/usr/bin/env bash
# Shared body for every row's run.sh.
#
# Sourced, not executed: each row file sets ROW, its blocked models and any
# patch step, then calls `row_main "$@"`. One implementation means a fix to the
# training or tracking path lands in all five rows at once, instead of the five
# copies drifting -- which they would, silently, because only row 00 ever gets
# run and debugged.
#
# Kept in shared/ rather than copied per row for the same reason: four copies of
# a 300-line script is four places to forget the answer was fixed.

set -uo pipefail

# --------------------------------------------------------------------------- #
# what a row must define before calling row_main
# --------------------------------------------------------------------------- #
#: ROW            row name, matching a key in study_config.ROWS
#: PATCHED        1 if POGS/ in this folder has been modified
#: BLOCKED_MODELS models with no weights, space separated
#: GATED_MODELS   models needing HF_TOKEN, space separated
#: ROW_DESC       one line for the report header

ROW_DESC=""
PATCHED=0
BLOCKED_MODELS=""
GATED_MODELS=""

row_main() {
  # ROW_DIR, set by the row's own run.sh before sourcing this file.
  #
  # It cannot be derived from BASH_SOURCE here: inside a sourced file
  # ${BASH_SOURCE[0]} names the *sourced* file, so this function would resolve
  # HERE to shared/ and POGS to shared/POGS -- a directory that does not exist.
  # Caught by driving the sweep with fakes on PATH, where ns-train was silently
  # never reached.
  [ -n "${ROW_DIR:-}" ] || die "ROW_DIR unset - call this from a row's run.sh, which sets it before sourcing"
  HERE="$(cd "$ROW_DIR" && pwd)"
  STUDY="$(dirname "$HERE")"
  SHARED="$STUDY/shared"
  POGS="$HERE/POGS"
  # The active scene, and every path derived from it.
  #
  # Derived by a function rather than assigned once, because the sweep changes
  # SCENE between scenes: computing INPUT and ART here and leaving them alone
  # means all ten scenes quietly train on the first one. That is exactly what
  # happened, and it produced ten "successful" runs of a single scene.
  SCENE="${SCENE:-${SCENE_OVERRIDE:-pickplace}}"
  # SEED=7 bash run.sh train  -- a repeat for run-to-run spread. Non-default
  # seeds get their own artefact folder so a repeat never overwrites, or is
  # mistaken for, the seed-42 result the report reads.
  export SEED="${SEED:-42}"
  set_scene() {
    INPUT="$STUDY/inputs/$SCENE"
    ART="$STUDY/artefacts/$ROW/$SCENE"
    [ "$SEED" = "42" ] || ART="${ART}_seed$SEED"
    mkdir -p "$ART"
  }
  WEIGHTS="$STUDY/weights"
  set_scene
  SMOKE_ITERS=50

  # Pin the RNG that POGS does not pin. See shared/study_config.py:POGS_SEED.
  export PYTHONHASHSEED=0
  export CUBLAS_WORKSPACE_CONFIG=:4096:8

  # The downloader prefers weights/ over the HF cache; make sure torch.hub and
  # open_clip also stop reaching out mid-benchmark.
  export TORCH_HOME="${TORCH_HOME:-$WEIGHTS/torch}"
  export HF_HOME="${HF_HOME:-$WEIGHTS/hf}"
  # Row-03's SigLIP2 encoder reads this to prefer weights/<name>/ over the
  # hub when the config.json is present (offline-first, like every loader).
  export POGS_WEIGHTS="$WEIGHTS"

  step() { printf '\n\033[1m==== %s ====\033[0m\n' "$1"; }
  warn() { printf '\033[33m%s\033[0m\n' "$1"; }
  die()  { printf '\033[31m%s\033[0m\n' "$1" >&2; exit 1; }
  note() { printf '\033[36m%s\033[0m\n' "$1"; }

  need() { command -v "$1" >/dev/null 2>&1 || die "missing required tool: $1"; }

  ENVNAME=pogs_env
  if [ -n "${ROW_ENV:-}" ]; then ENVNAME="$ROW_ENV"; fi

  # The row env, for every stage. Only setup ever activated it, so
  # `bash run.sh detect` from a plain shell silently used the base interpreter
  # and every import in it failed. setup creates the env, so it calls this a
  # second time once the env exists.
  activate_row_env() {
    command -v conda >/dev/null 2>&1 || return 0
    [ -d "$(conda info --base 2>/dev/null)/envs/$ENVNAME" ] || return 0
    # conda's activate.d scripts for binutils and gcc read a dozen variables a
    # fresh shell has not set -- ADDR2LINE was the one that aborted a run -- so
    # nounset is off for the activation only.
    set +u
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENVNAME" || die "cannot activate conda env $ENVNAME"
    set -u
    # conda.sh redefines `conda` as a shell *function* whose post-install
    # reactivation evals deactivate.d scripts into this shell, and gcc 11's
    # reads the CONDA_BACKUP_CXX only its activate script sets -- fatal under
    # set -u, and the reason the first compiler install killed a setup run. The
    # binary does the same work without eval'ing scripts into us.
    unset -f conda
  }
  activate_row_env

  # gsplat JIT-compiles its rasteriser on first use when the wheel has no kernel
  # for this GPU. Unset, torch targets every arch it knows, the compile eats
  # 10+ GB, and WSL's 50%-of-RAM cgroup OOM-kills the process (exit "Killed").
  # RTX 4070 = Ada = sm_89; overridable if this ever runs elsewhere.
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
  # cicc (CUDA compiler) peaks at ~2 GB per job; ninja defaults to one job per
  # core, which OOM-kills the box before the first training iteration.
  export MAX_JOBS="${MAX_JOBS:-2}"
  # Ceiling on the gaussian count, read by POGSModel.step_post_backward. Stock POGS
  # has none: pendulum_05 reached 1.5M gaussians by step 3300 and the CLIP hash
  # grid then ran the 12 GB 4070 out of memory (tiny-cuda-nn
  # CUDA_ERROR_OUT_OF_MEMORY). Applied identically to every row and scene so the
  # comparison stays fair; POGS_MAX_GAUSSIANS=0 restores uncapped behaviour.
  export POGS_MAX_GAUSSIANS="${POGS_MAX_GAUSSIANS:-300000}"

  header() {
    step "$ROW"
    echo "  $ROW_DESC"
    [ "$PATCHED" = "1" ] && note "  POGS/ in this folder is PATCHED (see patch/)"
    if [ -n "$BLOCKED_MODELS" ]; then
      warn "  BLOCKED models: $BLOCKED_MODELS"
    fi
    if [ -n "$GATED_MODELS" ]; then
      warn "  GATED models (need HF_TOKEN): $GATED_MODELS"
    fi
  }

  # --------------------------------------------------------------------------- #
  stage_weights() {
    step "weights for this row"
    # Downloaded before anything is timed, so a latency figure is never a
    # measurement of the network.
    python "$SHARED/download_weights.py" --row "$ROW" --dest "$WEIGHTS" \
      || warn "some weights unavailable -- continuing; blocked models are recorded"
    if [ -n "$GATED_MODELS" ] && [ -z "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]; then
      warn "no HF_TOKEN set, so gated models cannot download:"
      warn "  $GATED_MODELS"
      warn "  export HF_TOKEN=... then 'bash run.sh weights' again"
    fi
  }

  # --------------------------------------------------------------------------- #
  stage_setup() {
    need python3
    need conda
    # tcnn, detectron2, pyliblzfse and fpsample all compile. Without a toolchain
    # pip fails four times over with errors that look like four different bugs,
    # and the nerfstudio install aborts silently with it.
    need gcc
    need g++
    need cmake
    need git

    step "environment (authors' README sequence)"
    echo "conda env: $ENVNAME   python 3.10   torch 2.6.0   cuda 11.8"
    # This file only ever reaches the GPU host by hand, and a stale copy has
    # already wasted a run; the mtime says which version a pasted log ran.
    echo "row_common.sh: $(stat -c %y "${SHARED}/row_common.sh" 2>/dev/null || echo unknown)"
    if ! conda env list | grep -qE "^$ENVNAME[[:space:]]"; then
      conda create --name "$ENVNAME" -y python=3.10 || die "conda create failed"
    fi
    # row_main's attempt at this ran before the env existed; now that it does.
    activate_row_env

    conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit \
      || die "cuda-toolkit install failed"

    pip install --quiet torch==2.6.0 torchvision==0.21.0 \
      --index-url https://download.pytorch.org/whl/cu118 \
      || die "torch install failed"
    pip install --quiet ninja || die "ninja install failed"
    # Placed immediately before its one consumer, and after conda, so nothing
    # later can quietly restore a newer one: pkg_resources (which tcnn's
    # setup.py:5 parses versions with) was removed in setuptools 82, and pip's
    # build isolation always fetches the newest setuptools. --no-build-isolation
    # then makes this env's setuptools the one that runs -- tcnn has no
    # pyproject.toml, so an isolated build env would never see torch either
    # (its setup.py:8 imports it).
    pip install --quiet "setuptools<81" wheel || die "setuptools downgrade failed"
    # CUDA 11.8's nvcc hard-refuses any host gcc newer than 11 (crt/host_config.h
    # errors with "gcc versions later than 11 are not supported"), and ML20's
    # Ubuntu 26.04 ships gcc 15: every .cu file dies before compiling, which
    # reads like a broken tcnn and would break detectron2's CUDA ops below too.
    # nvcc looks its host compiler up as gcc/g++ on PATH and ninja uses CC/CXX,
    # so conda-forge's gcc 11 -- the compiler nvcc 11.8 was built against -- is
    # exposed under all four names.
    host_gcc_major="$(gcc -dumpfullversion 2>/dev/null | cut -d. -f1)"
    if [ -z "$host_gcc_major" ] || [ "$host_gcc_major" -gt 11 ]; then
      step "host compiler: gcc ${host_gcc_major:-unknown} cannot drive CUDA 11.8 nvcc"
      conda install -y -c conda-forge "gxx_linux-64=11" "gcc_linux-64=11" \
        || warn "conda gcc 11 install failed -- falling back to -allow-unsupported-compiler"
      gcc11_cxx="$(command -v x86_64-conda-linux-gnu-g++ \
        || command -v x86_64-conda-linux-gnu-g++-11 || true)"
      gcc11_cc="$(command -v x86_64-conda-linux-gnu-gcc \
        || command -v x86_64-conda-linux-gnu-gcc-11 || true)"
      if [ -n "$gcc11_cxx" ] && [ -n "$gcc11_cc" ]; then
        NVCC_HOST="$CONDA_PREFIX/nvcc-host-gcc11"
        mkdir -p "$NVCC_HOST"
        ln -sf "$gcc11_cc" "$NVCC_HOST/gcc"
        ln -sf "$gcc11_cc" "$NVCC_HOST/cc"
        ln -sf "$gcc11_cxx" "$NVCC_HOST/g++"
        ln -sf "$gcc11_cxx" "$NVCC_HOST/c++"
        export PATH="$NVCC_HOST:$PATH" CC="$NVCC_HOST/gcc" CXX="$NVCC_HOST/g++"
        # Prove the nvcc/g++ pair in two seconds rather than ten minutes into
        # tcnn's own build.
        echo "int main(){return 0;}" > "$NVCC_HOST/probe.cu"
        nvcc -c "$NVCC_HOST/probe.cu" -o "$NVCC_HOST/probe.o" \
          || die "nvcc cannot compile through the gcc 11 shim (g++ is $(g++ -dumpfullversion 2>/dev/null || echo unknown))"
        echo "  host compiler: $(g++ -dumpfullversion) at $(command -v g++)"
      else
        warn "gcc 11 is not on conda-forge here; building against nvcc's"
        warn "-allow-unsupported-compiler and the system gcc ${host_gcc_major}"
        warn "instead -- revisit this if tcnn builds but training misbehaves"
        export NVCC_PREPEND_FLAGS="-allow-unsupported-compiler"
      fi
    fi
    # libcuda.so is the *driver* library: the CUDA toolkit does not ship it, and
    # WSL keeps it in /usr/lib/wsl/lib, which is not on the linker's search
    # path -- so every extension that links -lcuda (tcnn now, detectron2's
    # CUDA ops next) fails at the link step with "cannot find -lcuda" after
    # compiling perfectly. On bare-metal Ubuntu the driver puts it in
    # /usr/lib/x86_64-linux-gnu, which is already on the path.
    if [ -d /usr/lib/wsl/lib ]; then
      export LIBRARY_PATH="/usr/lib/wsl/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
      echo "  libcuda.so: /usr/lib/wsl/lib via LIBRARY_PATH"
    fi
    pip install --quiet --no-build-isolation \
      "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch" \
      || die "tiny-cuda-nn build failed"
    pip install --quiet jaxtyping rich || die "jaxtyping/rich install failed"
    # Pinned to nerfstudio's own gsplat pin (deps/nerfstudio/pyproject.toml:67).
    # Unpinned, this index serves 1.5.3, and nerfstudio's ==1.4.0 then makes pip
    # rebuild gsplat from the PyPI sdist with nvcc in the middle of its install.
    pip install --quiet "gsplat==1.4.0" \
      --index-url https://docs.gsplat.studio/whl/pt20cu118 \
      || die "gsplat install failed"
    pip install --quiet warp-lang || die "warp-lang install failed"
    # cuml is a hard requirement: pogs.py imports cuml.cluster.hdbscan for the
    # clustering step that defines object clusters after training.
    pip install --quiet --extra-index-url=https://pypi.nvidia.com \
      cudf-cu11==25.4.* cuml-cu11==25.4.* \
      || die "cudf/cuml install failed"

    step "POGS package"
    mkdir -p "$POGS/pogs/dependencies"
    for dep in nerfstudio Detic detectron2; do
      src="$STUDY/deps/$dep"
      [ -d "$src" ] || die "missing $src - the deps folder did not transfer intact"
      rm -rf "${POGS:?}/pogs/dependencies/$dep"
      ln -s "$src" "$POGS/pogs/dependencies/$dep"
      echo "  linked dependencies/$dep"
    done

    ( cd "$POGS" && python -m pip install --quiet -e . ) || die "pip install -e . failed"
    ( cd "$POGS" && python -m pip install --quiet pogs/dependencies/nerfstudio/ ) \
      || die "nerfstudio install failed"
    # detectron2 is symlinked above but never *installed* -- DeticDataloader
    # imports it at module level (detic_dataloader.py:29) and its _C extension
    # needs a compiler, hence --no-build-isolation alongside the torch already
    # in this env (its setup.py imports torch too).
    python -m pip install --quiet --no-build-isolation \
      -e "$STUDY/deps/detectron2" || die "detectron2 install failed"
    # sklearn: detic_dataloader.py:48 (DBSCAN), pulled by neither POGS nor
    # nerfstudio. openai-clip: detic/modeling/text/text_encoder.py does
    # `from clip.simple_tokenizer import ...`, and POGS only ships open_clip --
    # a different module -- so nothing installs `clip` for us.
    pip install --quiet scikit-learn openai-clip || die "sklearn/clip install failed"
    # README lines 47-48: fast_simplification==0.1.9 first, then numpy==1.26.4,
    # as two commands -- 0.1.9 declares numpy>=2, so pip cannot resolve the pair
    # at once. trimesh imports it lazily inside simplify_quadric_decimation,
    # which ToadObject._points_to_mesh calls for every tracked part; a grep for
    # top-level importers misses that, and tracking died on it. numpy is
    # re-asserted after nerfstudio's install has had its say.
    pip install --quiet fast_simplification==0.1.9 || die "fast_simplification install failed"
    pip install --quiet numpy==1.26.4 || die "numpy pin failed"
    ns-install-cli || warn "ns-install-cli failed; ns-train may not be on PATH"

    # Last, on purpose: this row's version overrides must land after the two
    # installs above, which unconditionally pin transformers==4.44.0 (POGS) and
    # timm==0.6.7 (nerfstudio). Installed before them, as it was, ROW_PIP was
    # silently reverted for every row that carries one.
    step "row-specific pip packages"
    if [ -n "${ROW_PIP:-}" ]; then
      echo "  $ROW_PIP"
      # shellcheck disable=SC2086
      pip install --quiet $ROW_PIP || die "row pip install failed"
    else
      echo "  none -- this row needs no version changes"
    fi

    step "patched POGS tree"
    # Recorded before anything runs, so a report always states whether the tree
    # it measured was stock POGS.
    #
    # --binary is required, not optional: every .py in the upstream tree uses CRLF
    # line endings (46/46 checked), and a patch without it fails with "different
    # line endings" while looking otherwise correct. Patch files are therefore
    # stored with CRLF too.
    if [ -d "$HERE/patch" ] && [ -n "$(ls -A "$HERE/patch" 2>/dev/null)" ]; then
      note "  this row carries patches in patch/ -- the diff is in RESULTS.md"
      if [ -f "$HERE/patch/APPLIED" ]; then
        echo "  already applied"
      else
        for p in "$HERE"/patch/*.patch; do
          [ -e "$p" ] || continue
          ( cd "$POGS" && patch -p1 --binary --forward < "$p" ) \
            && echo "  applied $(basename "$p")" \
            || die "patch failed: $(basename "$p") (a partly-applied tree is" \
                    "  not runnable -- restore POGS/ from row 00 and retry)"
        done
        touch "$HERE/patch/APPLIED"
      fi
      # Record what the patch actually did, so the report can print the real
      # diff instead of a description of it.
      {
        echo "row: $ROW"
        echo "applied: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        for p in "$HERE"/patch/*.patch; do
          [ -e "$p" ] || continue
          echo "--- $(basename "$p") ---"
          cat "$p"
        done
      } > "$ART/patches_applied.txt" 2>/dev/null || true
    else
      echo "  none -- running stock POGS"
    fi

    step "import chain"
    python - <<'PY'
import importlib
import sys

# "Detic" used to be in this list. It can never import: there is no top-level
# module by that name, so the check printed FAIL forever and taught nothing.
# The detic_dataloader entry is the real gate -- importing it runs detectron2,
# the CenterNet2 sys.path insert (detic_dataloader.py:34), the whole detic/
# package chain (clip, timm, fvcore), plus open3d and sklearn. Everything
# stage_detect and stage_train need arrives or fails in one place.
mods = [
    ("nerfstudio", "dataparser + splatfacto"),
    ("gsplat", "rasteriser used by pogs.py"),
    ("cuml", "HDBSCAN clustering, required"),
    ("detectron2", "DeticDataloader import"),
    ("open_clip", "CLIP relevancy encoder"),
    ("open3d", "seed point cloud"),
    ("sklearn", "DBSCAN, detic_dataloader.py:48"),
    ("clip", "Detic text encoder, text_encoder.py"),
    ("pogs", "the installed POGS package"),
    ("pogs.data.utils.detic_dataloader", "detic + centernet + detectron2 gate"),
]
bad = []
for name, why in mods:
    try:
        importlib.import_module(name)
        print(f"  OK    {name:<34} {why}")
    except Exception as exc:
        print(f"  FAIL  {name:<34} {type(exc).__name__}: {exc}")
        bad.append(name)
if bad:
    sys.exit("missing imports: " + ", ".join(bad))
print("\n  import chain OK")
PY
    [ $? -eq 0 ] || die "import chain incomplete - fix the environment before training"
  }

  stage_prefetch() {
    step "pre-caching model weights through their own loaders"
    # Weights are already downloaded into weights/; this warms the in-process
    # caches (torch.hub, open_clip) so the first timed iteration is not also a
    # model build.
    python "$SHARED/prefetch.py" --row "$ROW" --scene "$SCENE" \
      --weights "$WEIGHTS" \
      || warn "prefetch incomplete -- the row's blocked models are expected to fail"
  }

  stage_smoke() {
    step "smoke train (${SMOKE_ITERS} iterations) - gate before the long run"
    echo "A broken environment should fail here in a minute, not three hours in."
    for pass_no in 1 2; do
      ( cd "$POGS" && ns-train pogs \
          --data "$INPUT" \
          --machine.seed "$SEED" \
          --viewer.quit-on-train-completion True \
          --max-num-iterations "$SMOKE_ITERS" \
          --output-dir "$ART/smoke_$pass_no" \
        ) 2>&1 | tee "$ART/smoke_$pass_no.log" | tail -20
      [ "${PIPESTATUS[0]}" -eq 0 ] || {
        echo "smoke train (pass $pass_no) failed on $SCENE" >&2
        return 1
      }
    done

    step "determinism check"
    python - "$ART" <<'PY'
import glob, json, os, sys
art = sys.argv[1]
out = {"seed": 42, "iterations": 50, "runs": []}
for run in ("smoke_1", "smoke_2"):
    f = os.path.join(art, run, "experiment.yml")
    out["runs"].append(run)
losses = []
for run in out["runs"]:
    found = glob.glob(os.path.join(art, run, "**", "*.json"), recursive=True)
    for p in found:
        try:
            d = json.load(open(p))
        except Exception:
            continue
        for e in (d if isinstance(d, list) else d.get("steps", [])):
            if isinstance(e, dict) and "loss" in e:
                losses.append(float(e["loss"]))
if len(losses) >= 2 * out["iterations"]:
    a, b = losses[:out["iterations"]], losses[out["iterations"]:2 * out["iterations"]]
    out["max_abs_delta"] = round(max(abs(x - y) for x, y in zip(a, b)), 8)
    out["match"] = out["max_abs_delta"] < 1e-6
    print(f"  max |loss difference| across runs: {out['max_abs_delta']}")
    print(f"  deterministic: {out['match']}")
else:
    out["match"] = None
    print("  could not read both loss traces; treat tracking as noisy")
os.makedirs(os.path.join(art, "train"), exist_ok=True)
json.dump(out, open(os.path.join(art, "train", "determinism.json"), "w"), indent=2)
PY
  }

  stage_train() {
    step "full training (4000 iterations, seed $SEED)"
    ( cd "$POGS" && ns-train pogs \
        --data "$INPUT" \
        --machine.seed "$SEED" \
        --viewer.quit-on-train-completion True \
        --output-dir "$ART/train" \
      ) 2>&1 | tee "$ART/train.log" | tail -30
    # return, not die: inside a sweep, `die` would exit the whole shell and the
    # remaining nine scenes would never run. The sweep records the failure and
    # moves on; a single-scene invocation still stops, because the case handler
    # propagates this non-zero status.
    [ "${PIPESTATUS[0]}" -eq 0 ] || {
      echo "training failed on $SCENE" >&2
      return 1
    }

    python - "$ART" <<'PY'
import glob, json, os, sys
art = sys.argv[1]
res = {"iterations": 4000}
# nerfstudio 1.1.1 writes step-000003999.ckpt under nerfstudio_models/;
# older builds wrote step-*-train.ckpt beside config.yml. Match both.
ckpt = glob.glob(os.path.join(art, "train", "**", "step-*.ckpt"), recursive=True)
if ckpt:
    res["checkpoint"] = max(ckpt, key=os.path.getmtime)
for p in glob.glob(os.path.join(art, "train", "**", "*.json"), recursive=True):
    try:
        d = json.load(open(p))
    except Exception:
        continue
    for e in (d if isinstance(d, list) else d.get("steps", [])):
        if not isinstance(e, dict):
            continue
        for k, out in (("loss", "psnr"), ("Test/loss", "psnr")):
            if k in e and out not in res:
                res[out] = float(e[k])
        for k, out in (("depth_loss", "depth_loss"), ("dino_loss", "dino_loss")):
            if k in e and out not in res:
                res[out] = float(e[k])
det = glob.glob(os.path.join(art, "train", "**", "outputs", "*", "detic.npy"),
                recursive=True)
if det:
    res["detic_masks"] = int(os.path.getsize(det[0]))
os.makedirs(os.path.join(art, "train"), exist_ok=True)
json.dump(res, open(os.path.join(art, "train", "training.json"), "w"), indent=2)
print("  wrote training.json:", {k: v for k, v in res.items() if k != "checkpoint"})
PY
  }

  stage_bench() {
    step "latency and memory"
    echo "  scene: $SCENE"
    # bench.py is standalone, but the row env pins transformers/timm for
    # Detic and the bench-only models (DINOv3, MobileCLIP2) need newer
    # versions than those pins allow. bench_env holds just the benchmarking
    # stack; fall back to the row env where the baseline models live.
    local py=python
    if [ -x "$HOME/miniconda3/envs/bench_env/bin/python" ]; then
      py="$HOME/miniconda3/envs/bench_env/bin/python"
    fi
    "$py" "$SHARED/bench.py" --row "$ROW" --scene "$SCENE" \
      --warmup 20 --iters 100 --weights "$WEIGHTS"
    # LocateAnything3D needs a prompt and generate(); the generic bench cannot time it.
    if [ "$ROW" = "01_locateanything_dinotxt" ]; then
      step "LocateAnything3D detection (prompt + generate)"
      python "$SHARED/bench_locateanything.py" --row "$ROW" --scene "$SCENE" \
        --weights "$WEIGHTS" || warn "LocateAnything3D bench failed -- see above"
    fi
  }

  stage_detect() {
    step "detector check"
    # POGS feeds detector masks into training from lerf_step=2500 and asserts
    # 'No masks found' if the detector returns nothing
    # (full_images_datamanager.py:437), so a failed detection would otherwise
    # only surface most of the way into a long run. ~1 min here.
    if [ -n "$BLOCKED_MODELS" ]; then
      warn "row has blocked models ($BLOCKED_MODELS)"
      warn "the detection check can only validate the models that loaded"
    fi
    # return, not die, so one scene's detection failure is recorded by the sweep
    # rather than ending the run before the remaining scenes.
    python "$SHARED/detect_check.py" --row "$ROW" --scene "$INPUT" \
      || {
        echo "the detector found nothing in $SCENE -- see the message above" >&2
        return 1
      }
  }

  stage_validate() {
    step "detector accuracy on the authors' real frames"
    # Scene-independent on purpose: the 14 real ZED frames exist once, and IoU
    # against their seg labels says the same thing for every scene. Running it
    # inside a sweep would repeat the same number ten times.
    python "$SHARED/detect_check.py" --row "$ROW" --real || \
      warn "real-frame validation failed -- recorded as not run"
  }

  stage_track() {
    step "offline tracking on the generated sequence"
    if [ ! -d "$INPUT/tracking" ]; then
      warn "no tracking/ under $INPUT -- skipping"
      warn "  capture-only scene? regenerate it with the matching generator"
      return 0
    fi
    python "$SHARED/track_offline.py" --row "$ROW" \
      --input "$INPUT" --train-dir "$ART/train" "$@"
  }

  stage_report() {
    step "report"
    python "$SHARED/report.py" --row "$ROW"
    python "$SHARED/compare.py"
  }

  # --------------------------------------------------------------------------- #
  # the sweep: every scene, one row
  # --------------------------------------------------------------------------- #
  # Scoped so a failure names the scene it happened in. A 40-run sweep that dies
  # at run 27 with no context is the failure mode worth designing against.
  stage_sweep() {
    local stages="${1:-train track}"
    shift || true

    step "sweep: $ROW over every scene"
    echo "  stages per scene: $stages"
    echo "  scenes: $(python "$SHARED/download_weights.py" --list-scenes | tr '\n' ' ')"
    echo "  artefacts: $STUDY/artefacts/$ROW/<scene>/"

    local scene results=() t_all
    t_all=$(date +%s)
    while IFS= read -r scene; do
      [ -n "$scene" ] || continue
      local t0 rc=0
      t0=$(date +%s)
      printf '\n\033[1m---------- %s / %s ----------\033[0m\n' "$ROW" "$scene"
      # SCENE is exported rather than assigned so the sourced stages and every
      # python subprocess see the same value.
      SCENE="$scene"
      export SCENE
      # Re-derive INPUT and ART for this scene. Without this every scene would
      # read the paths computed at row_main entry, i.e. the first scene.
      set_scene
      local s
      for s in $stages; do
        case "$s" in
          train)    stage_train || rc=1 ;;
          track)    stage_track "$@" || rc=1 ;;
          detect)   stage_detect || rc=1 ;;
          bench)    stage_bench || rc=1 ;;
          smoke)    stage_smoke || rc=1 ;;
          validate) stage_validate || rc=1 ;;
          report)   stage_report || rc=1 ;;
          *) die "unknown sweep stage: $s (train|track|detect|bench|smoke)" ;;
        esac
        [ "$rc" -eq 0 ] || break
      done
      local mins
      mins=$(( ($(date +%s) - t0) / 60 ))
      if [ "$rc" -eq 0 ]; then
        printf '\033[32m  ok %s (%d min)\033[0m\n' "$scene" "$mins"
        results+=("$scene:ok:${mins}")
      else
        printf '\033[31m  FAILED %s (%d min) at stage %s\033[0m\n' "$scene" "$mins" "$s"
        results+=("$scene:FAILED:${mins}")
      fi
    done < <(python "$SHARED/download_weights.py" --list-scenes)

    step "sweep summary"
    printf '  %s\n' "${results[@]}"
    local total=$(( ($(date +%s) - t_all) / 60 ))
    echo "  total: ${total} min"

    mkdir -p "$STUDY/artefacts/$ROW"
    # v2 sweeps write sweep_v2.txt so they never replace the v1 record.
    local sfx=""
    [ "${SCENE_SET:-v1}" = "v1" ] || sfx="_${SCENE_SET}"
    {
      echo "row: $ROW"
      echo "stages: $stages"
      echo "total_minutes: $total"
      for r in "${results[@]}"; do echo "  $r"; done
    } > "$STUDY/artefacts/$ROW/sweep$sfx.txt"
    echo "  wrote $STUDY/artefacts/$ROW/sweep$sfx.txt"

    stage_report
    # Non-zero if any scene failed, so a sweep is safe to run from CI or a script.
    for r in "${results[@]}"; do
      case "$r" in *:FAILED:*) return 1 ;; esac
    done
    return 0
  }

  stage_all() {
    header
    stage_setup
    stage_weights
    stage_prefetch
    stage_smoke
    stage_detect
    if [ "${SKIP_TRAIN:-0}" = "1" ]; then
      warn "SKIP_TRAIN=1 -- skipping training and tracking"
      stage_bench
      stage_validate
      stage_report
      return 0
    fi
    stage_train
    stage_bench
    stage_validate
    stage_track
    stage_report
  }

  # activate_row_env returns quietly when the row's env does not exist, and
  # every row's `pip install -e` points the one shared `pogs` at its own tree.
  # Row 01 therefore ran for hours on row 00's code and DINOv2 and was reported
  # as a LocateAnything3D/dino.txt result. Refuse instead of falling back.
  verify_row_env() {
    [ "${CONDA_DEFAULT_ENV:-}" = "$ENVNAME" ] \
      || die "conda env '$ENVNAME' is not active (active: '${CONDA_DEFAULT_ENV:-none}'). Run 'bash run.sh setup' for this row first; running in another env would use another row's code."
    local where
    where="$(python - <<'EOF' 2>/dev/null
import importlib.util
s = importlib.util.find_spec("pogs")
print(";".join(s.submodule_search_locations or []) if s else "")
EOF
)"
    case "$where" in
      "$POGS"/*) ;;
      *) die "pogs resolves to '${where:-nothing}', not this row's tree ($POGS/pogs). Re-run 'bash run.sh setup' in this row." ;;
    esac
  }

  local stages="setup weights prefetch smoke detect train bench validate track sweep report all"
  # 'sweep bench' (and 'sweep report') import nothing from POGS. Rows 02 and 04
  # share an env whose editable `pogs` still points at row 00's tree, and a
  # re-setup there would silently re-point every other row's env. The guard
  # stays strict for everything that actually imports POGS.
  if [ "${1:-}" = "sweep" ]; then
    case "${2:-}" in
      bench|report) : ;;
      *) verify_row_env ;;
    esac
  else
    case "${1:-all}" in
      smoke|detect|train|validate|track|all) verify_row_env ;;
    esac
  fi
  case "${1:-all}" in
    setup)    header; stage_setup ;;
    weights)  stage_weights ;;
    prefetch) stage_prefetch ;;
    smoke)    stage_smoke ;;
    detect)   stage_detect ;;
    train)    stage_train ;;
    bench)    stage_bench ;;
    validate) stage_validate ;;
    track)    shift; stage_track "$@" ;;
    # Sweep takes its stage list next: 'sweep train track'. The remaining
    # arguments go to stage_track, which is the only stage that takes flags.
    sweep)
      shift
      local list="train track"
      [ "${1:-}" = "--" ] && { list=""; shift; }
      [ -n "${1:-}" ] && case "$1" in
        train|track|detect|bench|smoke|validate|report) list="$1"; shift ;;
      esac
      stage_sweep "$list" "$@" ;;
    report)   stage_report ;;
    all)      stage_all ;;
    *) die "unknown stage: $1 ($stages)" ;;
  esac
}