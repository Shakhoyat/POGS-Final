"""
Software / hardware record written next to every result (PLAN §3.6).

``env.json`` = Python, PyTorch, CUDA, cuDNN, gsplat, GPU, driver, git commit and
whether the working tree was dirty. Timings and accuracy depend on all of them.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import subprocess
from typing import Dict

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE: Dict[str, object] = {}


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", _REPO, *args], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def _driver() -> str:
    for exe in ("nvidia-smi", "/usr/lib/wsl/lib/nvidia-smi"):
        try:
            out = subprocess.run([exe, "--query-gpu=driver_version", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
            if out:
                return out.splitlines()[0]
        except Exception:
            continue
    return ""


def collect_env() -> Dict[str, object]:
    """Static part is computed once per process; the timestamp is per call."""
    if not _CACHE:
        import torch

        info: Dict[str, object] = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "driver": _driver(),
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        }
        for mod in ("gsplat", "torchmetrics", "numpy", "sklearn"):
            try:
                info[mod] = __import__(mod).__version__
            except Exception:
                info[mod] = None
        _CACHE.update(info)
    return {**_CACHE, "written_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds")}


def write_env_json(directory: str) -> str:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "env.json")
    with open(path, "w") as f:
        json.dump(collect_env(), f, indent=2)
    return path
