"""
``GaussianField`` -- the language + grouping feature field of POGS, ported from
tiny-cuda-nn to plain PyTorch.

The official field (``pogs/fields/gaussian_field.py``) is:

    clip_encs    : two tcnn HashGrid encodings
                     level 0 -> 12 levels, res  16 ->  128, 4 feats/level, 2^19
                     level 1 -> 12 levels, res 128 ->  512, 4 feats/level, 2^19
                   concatenated -> 96 dims
    clip_net     : CutlassMLP(96 + 1 -> 512), 3 hidden layers x 256, ReLU
                   (the "+1" is the LERF scale conditioning s in F_lang(x, s))
    instance_net : CutlassMLP(96 -> 128), 4 hidden layers x 256, ReLU

Both outputs are L2-normalised, matching the official ``get_outputs``.
Positions pass through nerfstudio's ``SceneContraction`` then are mapped to
[0, 1] by ``(x + 2) / 4`` -- reproduced verbatim below.

tiny-cuda-nn is not installable inside a Kaggle session in reasonable time,
so ``HashEncoding`` re-implements Muller et al.'s multiresolution hash grid
(ref [51] in the paper) in ~60 lines of PyTorch.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Instant-NGP's spatial hash primes.
_PRIMES = (1, 2_654_435_761, 805_459_861)


def scene_contraction(x: torch.Tensor) -> torch.Tensor:
    """nerfstudio ``SceneContraction`` with the default L2 order.

    ``|x| <= 1`` is left untouched, everything outside is squashed into the
    ball of radius 2, so the composed output lives in [-2, 2].
    """
    mag = x.norm(dim=-1, keepdim=True)
    return torch.where(mag < 1, x, (2.0 - 1.0 / mag.clamp(min=1e-8)) * (x / mag.clamp(min=1e-8)))


class HashEncoding(nn.Module):
    """Multiresolution hash grid (Instant-NGP).

    Like tiny-cuda-nn, each level stores ``min(2**log2_hashmap_size, (res+1)**3)``
    entries: a coarse level whose dense grid already fits is indexed *directly*
    (bijective, collision-free), and only the levels that overflow fall back to
    the spatial hash.  Storing ``2**19`` entries at every level instead would be
    both wrong (needless collisions in the coarse levels, which are the ones the
    grouping features rely on) and, at the official 2x12 levels, 203 MB of
    parameters -- 800 MB once Adam's two moments and a dense gradient are added.
    """

    def __init__(
        self,
        start_res: int,
        end_res: int,
        n_levels: int,
        log2_hashmap_size: int = 19,
        features_per_level: int = 4,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.fpl = features_per_level
        self.table_size = 2 ** log2_hashmap_size
        growth = math.exp((math.log(end_res) - math.log(start_res)) / (n_levels - 1))
        res = [int(round(start_res * growth**i)) for i in range(n_levels)]
        sizes = [min(self.table_size, (r + 1) ** 3) for r in res]
        offsets = [0]
        for s in sizes:
            offsets.append(offsets[-1] + s)

        self.register_buffer("res", torch.tensor(res, dtype=torch.long), persistent=False)
        self.register_buffer("sizes", torch.tensor(sizes, dtype=torch.long), persistent=False)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long), persistent=False)
        self.dense = [s < self.table_size for s in sizes]
        self.table = nn.Parameter(
            torch.empty(offsets[-1], features_per_level).uniform_(-1e-4, 1e-4)
        )

    @property
    def n_output_dims(self) -> int:
        return self.n_levels * self.fpl

    @property
    def n_params(self) -> int:
        return int(self.table.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 3) in [0, 1]. Returns (N, n_levels * features_per_level)."""
        N = x.shape[0]
        x = x.clamp(0.0, 1.0)
        outs = []
        # 8 corners of the unit cube
        corners = torch.tensor(
            [[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)],
            device=x.device, dtype=torch.long,
        )                                                   # (8, 3)
        for lv in range(self.n_levels):
            R = int(self.res[lv].item())
            size = int(self.sizes[lv].item())
            off = int(self.offsets[lv].item())
            xs = x * (R - 1)
            base = torch.floor(xs).long()                   # (N, 3)
            frac = xs - base                                # (N, 3)
            idx = (base.unsqueeze(1) + corners.unsqueeze(0)).clamp(0, R - 1)  # (N, 8, 3)

            if self.dense[lv]:
                # dense grid fits: index it directly, no collisions
                h = (idx[..., 0] * (R + 1) + idx[..., 1]) * (R + 1) + idx[..., 2]
            else:
                h = (idx[..., 0] * _PRIMES[0]) ^ (idx[..., 1] * _PRIMES[1]) ^ (idx[..., 2] * _PRIMES[2])
            h = h % size                                     # (N, 8)
            feat = self.table[off + h]                       # (N, 8, F)

            # trilinear weights
            w = torch.ones(N, 8, device=x.device, dtype=x.dtype)
            for d in range(3):
                cd = corners[:, d].to(x.dtype).unsqueeze(0)  # (1, 8)
                fd = frac[:, d : d + 1]                      # (N, 1)
                w = w * (cd * fd + (1 - cd) * (1 - fd))
            outs.append((feat * w.unsqueeze(-1)).sum(1))     # (N, F)
        return torch.cat(outs, dim=-1)


def he_init_(module: nn.Module) -> nn.Module:
    """He (Kaiming) init with the ReLU gain.

    PyTorch's default ``nn.Linear`` init is calibrated for LeakyReLU with
    ``a = sqrt(5)`` and shrinks activation variance by ~1/3 per layer.  Through a
    bias-free 4-5 layer head fed by a hash grid initialised to U(-1e-4, 1e-4),
    that drives the output below the epsilon in the L2 normalisation, so the
    "normalised" CLIP / instance features are not unit norm and the contrastive
    margins are meaningless early in training.  tiny-cuda-nn does not have this
    problem; a naive port does.
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return module


def _mlp(in_dim: int, out_dim: int, width: int, hidden_layers: int) -> nn.Sequential:
    """Mirrors tcnn ``CutlassMLP`` -- ReLU between, no output activation, no bias
    (tcnn networks are bias-free)."""
    layers: list[nn.Module] = [nn.Linear(in_dim, width, bias=False), nn.ReLU(inplace=True)]
    for _ in range(hidden_layers - 1):
        layers += [nn.Linear(width, width, bias=False), nn.ReLU(inplace=True)]
    layers += [nn.Linear(width, out_dim, bias=False)]
    return he_init_(nn.Sequential(*layers))


class GaussianField(nn.Module):
    """Port of ``pogs/fields/gaussian_field.py::GaussianField``."""

    def __init__(
        self,
        grid_layers: Sequence[int] = (12, 12),
        grid_sizes: Sequence[int] = (19, 19),
        grid_resolutions: Sequence[Tuple[int, int]] = ((16, 128), (128, 512)),
        n_features_level: int = 4,
        clip_n_dims: int = 512,
        instance_n_dims: int = 128,
        log2_hashmap_size: Optional[int] = None,
    ):
        """``log2_hashmap_size`` overrides ``grid_sizes`` for both encoders.  The
        official config is 19 for each; lowering it is the knob for machines that
        cannot hold ~800 MB of hash-grid parameters + Adam state."""
        super().__init__()
        if log2_hashmap_size is not None:
            grid_sizes = tuple(log2_hashmap_size for _ in grid_layers)
        self.encs = nn.ModuleList(
            [
                HashEncoding(
                    grid_resolutions[i][0], grid_resolutions[i][1], grid_layers[i],
                    log2_hashmap_size=grid_sizes[i], features_per_level=n_features_level,
                )
                for i in range(len(grid_layers))
            ]
        )
        self.hash_dim = sum(e.n_output_dims for e in self.encs)
        self.clip_net = _mlp(self.hash_dim + 1, clip_n_dims, 256, 3)
        self.instance_net = _mlp(self.hash_dim, instance_n_dims, 256, 4)
        self.instance_n_dims = instance_n_dims
        self.clip_n_dims = clip_n_dims

    # -- hash --------------------------------------------------------------- #

    def get_hash(self, positions: torch.Tensor) -> torch.Tensor:
        p = scene_contraction(positions)
        p = (p + 2.0) / 4.0
        return torch.cat([e(p.view(-1, 3)) for e in self.encs], dim=-1).to(torch.float32)

    # -- heads -------------------------------------------------------------- #

    def clip_from_hash(self, h: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        out = self.clip_net(torch.cat([h, scale.view(-1, 1)], dim=-1))
        # official: ``clip_pass / clip_pass.norm(...)`` with no epsilon
        return out / out.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    def instance_from_hash(self, h: torch.Tensor) -> torch.Tensor:
        out = self.instance_net(h)
        return out / (out.norm(dim=-1, keepdim=True) + 1e-5)

    def get_outputs(self, positions: torch.Tensor, clip_scales: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.get_hash(positions)
        return {
            "hashgrid": h,
            "clip": self.clip_from_hash(h, clip_scales),
            "instance": self.instance_from_hash(h),
        }


class DinoProjection(nn.Module):
    """``POGSModel.nn`` -- the 4-layer bias-free MLP that maps the stored
    ``gaussian_dim`` feature to the rendered ``dim`` feature."""

    def __init__(self, in_dim: int = 64, out_dim: int = 64, width: int = 64):
        super().__init__()
        self.net = he_init_(nn.Sequential(
            nn.Linear(in_dim, width, bias=False), nn.ReLU(),
            nn.Linear(width, width, bias=False), nn.ReLU(),
            nn.Linear(width, width, bias=False), nn.ReLU(),
            nn.Linear(width, out_dim, bias=False),
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)