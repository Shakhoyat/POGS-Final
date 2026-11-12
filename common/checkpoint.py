"""
Self-contained POGS checkpoints.

Everything the tracker needs travels in ONE file:

    gauss_params      per-Gaussian tensors (means, quats, ..., dino_feats)
    gaussian_field    hash-grid + CLIP / grouping heads
    nn                DINO projection head
    dino_pca          the PCA basis the DINO targets were projected with
    step              training step reached (drives the SH degree in use)
    config            the POGSConfig used for training (as a dict)

The PCA basis is the critical piece: rendered DINO features are distilled in
that basis, so live frames must be projected with the same matrix. The release
stores it as ``pca.npy`` next to the DINO cache for exactly this reason.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import torch

from .config import POGSConfig
from .model import POGSModel

CKPT_NAME = "pogs_checkpoint.pt"


def save_checkpoint(
    ckpt_dir: str,
    model: POGSModel,
    dino_pca: torch.Tensor,
    cfg: POGSConfig,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, CKPT_NAME)
    torch.save(
        {
            "gauss_params": {k: v.detach().cpu() for k, v in model.gauss_params.items()},
            "gaussian_field": model.gaussian_field.state_dict(),
            "nn": model.nn.state_dict(),
            "dino_pca": dino_pca.detach().cpu(),
            "step": int(model.step),
            "config": cfg.to_dict(),
            "extra": extra or {},
        },
        path,
    )
    return path


def load_checkpoint(
    ckpt_dir: str, cfg: POGSConfig, device: str = "cuda:0"
) -> Tuple[POGSModel, torch.Tensor, Dict[str, Any]]:
    """Returns ``(model, dino_pca, raw_checkpoint)``. ``cfg.model`` must match the
    architecture the checkpoint was trained with (hash size, dims)."""
    path = os.path.join(ckpt_dir, CKPT_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no POGS checkpoint at {path}")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    gp = ck["gauss_params"]
    model = POGSModel(
        seed_points=gp["means"], seed_colors=torch.zeros_like(gp["means"]) + 0.5,
        config=cfg.model, device=device, raster_backend=cfg.runtime.raster_backend,
    )
    model.gauss_params = torch.nn.ParameterDict(
        {k: torch.nn.Parameter(v.to(device)) for k, v in gp.items()}
    )
    model.gaussian_field.load_state_dict(ck["gaussian_field"])
    model.nn.load_state_dict(ck["nn"])
    model.step = int(ck.get("step", 0))
    model.eval()
    return model, ck["dino_pca"], ck


def checkpoint_exists(ckpt_dir: str) -> bool:
    return os.path.exists(os.path.join(ckpt_dir, CKPT_NAME))


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
