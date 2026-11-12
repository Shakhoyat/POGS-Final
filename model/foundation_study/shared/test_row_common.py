"""Checks for row_common.sh that need no conda, no GPU and no network.

stage_setup is a shell script, so nothing imports it -- which is exactly why it
broke twice: a pip line without `|| die` carried on past a Ctrl-C, and four
distinct missing dependencies were reported as one bogus import entry. These
are the invariants that would have caught both.

    python test_row_common.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "row_common.sh")

FAILED = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global FAILED
    if not ok:
        FAILED += 1
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def logical_lines(text: str) -> list[str]:
    """Statement-level lines: backslash continuations joined, comments dropped."""
    out: list[str] = []
    buf = ""
    for raw in text.splitlines():
        line = raw.rstrip()
        if buf:
            buf += " " + line.lstrip()
            if not line.endswith("\\"):
                out.append(buf)
                buf = ""
            continue
        if not line or line.lstrip().startswith("#"):
            continue
        if line.endswith("\\"):
            buf = line[:-1].rstrip()
        else:
            out.append(line)
    if buf:
        out.append(buf)
    return out


def heredoc(text: str) -> str:
    """The python block that implements the import chain check."""
    try:
        body = text.split("python - <<'PY'", 1)[1]
    except IndexError:
        return ""
    return body.split("\nPY", 1)[0]


def main() -> int:
    print("=" * 68)
    print("ROW_COMMON.SH - SETUP INVARIANTS")
    print("=" * 68)

    with open(SCRIPT, encoding="utf-8") as fh:
        text = fh.read()
    lines = logical_lines(text)

    # Matched by substring, not by first word: the two POGS installs are wrapped
    # in `( cd "$POGS" && ... )`, so a first-token test skips exactly the two
    # commands that pin the versions ROW_PIP exists to override.
    installs = [ln for ln in lines
                if "pip install" in ln or "conda install" in ln
                or "conda create" in ln]
    unguarded = [ln for ln in installs
                 if "|| die" not in ln and "|| warn" not in ln]
    check("every pip/conda install is guarded by || die",
          not unguarded and len(installs) >= 10,
          f"{len(installs)} installs, {len(unguarded)} unguarded"
          + (f": {unguarded[0]}" if unguarded else ""))

    missing_tools = [t for t in ("need gcc", "need g++", "need cmake", "need git")
                     if t not in text]
    check("stage_setup requires the compiler toolchain",
          not missing_tools, f"missing {missing_tools}")

    check("setuptools pinned below 81 for pkg_resources",
          "setuptools<81" in text)

    tcnn = [ln for ln in installs if "tiny-cuda-nn" in ln]
    check("tiny-cuda-nn built without build isolation",
          len(tcnn) == 1 and "--no-build-isolation" in tcnn[0],
          f"{tcnn[0] if tcnn else 'not installed'}")

    detectron2 = [ln for ln in installs if "detectron2" in ln]
    check("detectron2 installed, not merely symlinked",
          len(detectron2) == 1 and "--no-build-isolation" in detectron2[0]
          and "-e" in detectron2[0],
          f"{detectron2[0] if detectron2 else 'not installed'}")

    joined = " ".join(installs)
    check("sklearn and openai-clip installed (detic_dataloader needs both)",
          "scikit-learn" in joined and "openai-clip" in joined)

    # trimesh imports it lazily in simplify_quadric_decimation, which tracking
    # calls per part; installed on its own line, before the numpy pin.
    check("fast_simplification installed separately, before the numpy pin",
          "pip install --quiet fast_simplification==0.1.9" in text
          and text.index("fast_simplification==0.1.9")
          < text.index("pip install --quiet numpy==1.26.4"))

    check("numpy==1.26.4 re-pinned after nerfstudio",
          "pip install --quiet numpy==1.26.4" in text
          and text.index("pip install --quiet numpy==1.26.4")
          > text.index("dependencies/nerfstudio"))

    row_pip = text.index("pip install --quiet $ROW_PIP")
    check("ROW_PIP runs after the nerfstudio install",
          row_pip > text.index("dependencies/nerfstudio"))
    check("ROW_PIP runs after 'pip install -e .' so its pins survive",
          row_pip > text.index("python -m pip install --quiet -e ."))

    chain = heredoc(text)
    check("import chain gates on the real module, not a fictional 'Detic'",
          '("Detic",' not in chain
          and "pogs.data.utils.detic_dataloader" in chain)
    check("import chain uses importlib so dotted names work",
          "importlib.import_module" in chain)

    check("nvcc pinned to a host gcc <= 11 (CUDA 11.8 vs gcc 15)",
          "gxx_linux-64=11" in text
          and "NVCC_PREPEND_FLAGS" in text
          and "nvcc -c" in text)

    check("conda shell function dropped before installs (set -u + reactivation)",
          "unset -f conda" in text and "set +u" in text)

    check("every stage activates the row env, not only setup",
          text.count("activate_row_env") >= 3)

    check("libcuda findable for extensions that link -lcuda (WSL)",
          "LIBRARY_PATH" in text and "/usr/lib/wsl/lib" in text)

    check("CUDA JIT compile pinned to this GPU and serialized (W^X RAM)",
          "TORCH_CUDA_ARCH_LIST" in text and "MAX_JOBS" in text)

    total = 17 - FAILED
    print(f"\n{'PASS' if not FAILED else 'FAIL'}: {total} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
