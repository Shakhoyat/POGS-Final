"""
2D foundation-model encoders: DINOv2 (tracking features) and OpenCLIP
(language features), plus the LERF-style multi-scale CLIP pyramid.

Faithfulness notes
------------------
* ``DinoExtractor`` reproduces ``pogs/data/utils/dino_dataloader.py``:
  ``dinov2_vits14``, stride 14, ``get_intermediate_layers(..., reshape=True)[0]``
  divided by 10, then PCA to 64 dims with ``torch.pca_lowrank(q=64, niter=20)``.
* ``OpenCLIPNetwork.get_relevancy`` is a verbatim port of the official
  implementation (the paired positive/negative softmax from LERF).
* ``ClipPyramid`` reproduces ``PyramidEmbeddingDataloader`` +
  ``PatchEmbeddingDataloader``: ``patch_tile_size_range = (0.05, 0.5)`` over
  ``patch_tile_size_res = 10`` scales, stride = tile * stride_scaler.

Offline fallbacks
-----------------
Kaggle sessions may run without internet.  Each encoder therefore has a
deterministic frozen-random-feature fallback that keeps the *pipeline*
runnable; ``.is_pretrained`` reports which one is live so the notebook can say
so out loud instead of silently reporting numbers from random features.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_DINO_SIZE = 1260
"""``pogs/data/utils/dino_dataloader.py::MAX_DINO_SIZE``."""


def get_img_resolution(H: int, W: int, p: int = 14, max_size: int = MAX_DINO_SIZE) -> Tuple[int, int]:
    """Verbatim from the official dino_dataloader."""
    if H < W:
        new_W = max_size
        new_H = (int((H / W) * max_size) // p) * p
    else:
        new_H = max_size
        new_W = (int((W / H) * max_size) // p) * p
    return max(new_H, p), max(new_W, p)


# --------------------------------------------------------------------------- #
# Fallback: frozen random features
# --------------------------------------------------------------------------- #


class _FrozenRandomViT(nn.Module):
    """A frozen random CNN with a 14-px receptive stride.

    Random features are a legitimate (if weaker) stand-in: they are
    view-consistent, spatially discriminative and deterministic, which is all
    the POGS tracking loss structurally requires.  Accuracy will be lower than
    with real DINOv2 weights, and the notebook says so.
    """

    embed_dim = 384

    def __init__(self, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        chans = [3, 48, 96, 192, 384]
        layers: List[nn.Module] = []
        for i in range(4):
            conv = nn.Conv2d(chans[i], chans[i + 1], 3, stride=2 if i < 3 else 1, padding=1, bias=False)
            with torch.no_grad():
                conv.weight.copy_(torch.randn(conv.weight.shape, generator=g) * (2.0 / conv.weight[0].numel()) ** 0.5)
            layers += [conv, nn.GELU(), nn.GroupNorm(8, chans[i + 1])]
        self.net = nn.Sequential(*layers)
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) -> (B, C, H/14, W/14)
        f = self.net(x)
        target = (max(1, x.shape[-2] // 14), max(1, x.shape[-1] // 14))
        return F.interpolate(f, size=target, mode="bilinear", align_corners=False)


# --------------------------------------------------------------------------- #
# DINOv2
# --------------------------------------------------------------------------- #


class DinoExtractor:
    """DINOv2 ViT-S/14 patch features, PCA-reduced to ``pca_dim``."""

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        device: str = "cuda:0",
        pca_dim: int = 64,
        max_size: int = MAX_DINO_SIZE,
        model_type: str = "dinov2_vits14",
        allow_download: bool = True,
        dtype: torch.dtype = torch.float16,
    ):
        self.device = device
        self.pca_dim = pca_dim
        self.max_size = max_size
        self.dtype = dtype if "cuda" in str(device) else torch.float32
        self.pca_matrix: Optional[torch.Tensor] = None
        self.is_pretrained = False
        self.model = None
        self.backend = "none"

        if allow_download:
            try:
                # skip_validation=True bypasses torch.hub's _validate_not_a_forked_repo
                # GitHub API call.  That call is unauthenticated, so on a shared or
                # repeatedly-used host it returns HTTP 403 "rate limit exceeded",
                # which torch then mishandles as KeyError('Authorization').  The
                # result was a *silent* fall back to random features -- and every
                # tracking number in the paper depends on DINOv2 being real.
                self.model = torch.hub.load(
                    "facebookresearch/dinov2", model_type, verbose=False,
                    skip_validation=True,
                )
                self.backend = "torch.hub/dinov2"
                self.is_pretrained = True
            except Exception as exc:  # noqa: BLE001
                print(f"[dino] torch.hub failed ({type(exc).__name__}); trying timm")
                try:
                    import timm

                    self.model = timm.create_model(
                        "vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0
                    )
                    self.backend = "timm/dinov2"
                    self.is_pretrained = True
                except Exception as exc2:  # noqa: BLE001
                    print(f"[dino] timm failed ({type(exc2).__name__}); using frozen random features")

        if self.model is None:
            self.model = _FrozenRandomViT()
            self.backend = "frozen-random"
            self.is_pretrained = False

        self.model = self.model.to(device).eval()
        if self.dtype == torch.float16 and self.is_pretrained:
            self.model = self.model.half()
        for p in self.model.parameters():
            p.requires_grad_(False)

        mean = torch.tensor(self.IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(self.IMAGENET_STD, device=device).view(1, 3, 1, 1)
        self._mean, self._std = mean, std

    # -- raw features ------------------------------------------------------- #

    @torch.no_grad()
    def raw_feats(self, images: torch.Tensor, batch: int = 1) -> torch.Tensor:
        """images: (B, 3, H, W) in [0, 1] -> (B, h, w, 384)."""
        H, W = images.shape[-2:]
        h, w = get_img_resolution(H, W, 14, self.max_size)
        out = []
        for i in range(0, images.shape[0], batch):
            x = images[i : i + batch].to(self.device)
            x = F.interpolate(x, size=(h, w), mode="bicubic", align_corners=False, antialias=True)
            x = (x - self._mean) / self._std
            if self.dtype == torch.float16 and self.is_pretrained:
                x = x.half()
            if hasattr(self.model, "get_intermediate_layers"):
                f = self.model.get_intermediate_layers(x, reshape=True)[0]
            else:
                f = self.model.forward_features(x)
            f = f.permute(0, 2, 3, 1).float() / 10.0   # official: descriptors / 10
            out.append(f)
        return torch.cat(out, dim=0)

    # -- PCA ---------------------------------------------------------------- #

    @torch.no_grad()
    def fit_pca(self, images: torch.Tensor, batch: int = 1, max_samples: int = 400_000) -> None:
        feats = self.raw_feats(images, batch=batch)
        flat = feats.reshape(-1, feats.shape[-1])
        if flat.shape[0] > max_samples:
            sel = torch.randperm(flat.shape[0], device=flat.device)[:max_samples]
            flat = flat[sel]
        if self.pca_dim >= flat.shape[-1]:
            self.pca_matrix = torch.eye(flat.shape[-1], device=flat.device)
        else:
            self.pca_matrix = torch.pca_lowrank(flat, q=self.pca_dim, niter=20)[2]
        del feats, flat
        if "cuda" in str(self.device):
            torch.cuda.empty_cache()

    @torch.no_grad()
    def pca_feats(self, images: torch.Tensor, batch: int = 1) -> torch.Tensor:
        """(B, 3, H, W) in [0,1] -> (B, h, w, pca_dim)."""
        assert self.pca_matrix is not None, "call fit_pca() first"
        f = self.raw_feats(images, batch=batch)
        s = f.shape
        return (f.reshape(-1, s[-1]) @ self.pca_matrix.to(f)).reshape(*s[:-1], self.pca_dim)

    # -- PCA persistence ---------------------------------------------------- #
    # The rendered DINO features are distilled against features projected with
    # THIS basis. Tracking must project live frames with the very same matrix
    # (the official code saves it as ``pca.npy`` next to the DINO cache), so the
    # basis is part of the checkpoint and is never re-fitted implicitly.

    def get_pca(self) -> torch.Tensor:
        assert self.pca_matrix is not None, "PCA basis has not been fitted or loaded"
        return self.pca_matrix.detach().cpu().clone()

    def set_pca(self, matrix: torch.Tensor) -> None:
        assert matrix.ndim == 2 and matrix.shape[1] == self.pca_dim, matrix.shape
        self.pca_matrix = matrix.to(self.device).float()

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 3:
            images = images.unsqueeze(0)
        if self.pca_matrix is None:
            raise RuntimeError(
                "DinoExtractor has no PCA basis. Fit it on the training views "
                "(fit_pca) or load it from the checkpoint (set_pca). Fitting it on "
                "a tracking frame would put observed and rendered features in "
                "different spaces."
            )
        feats = self.pca_feats(images)
        if feats.shape[0] == 1:
            return feats.squeeze(0)
        return feats


# --------------------------------------------------------------------------- #
# OpenCLIP
# --------------------------------------------------------------------------- #


class _FrozenRandomCLIP(nn.Module):
    """Deterministic text/image embedder used when open_clip is unavailable.

    Text embeddings are a hash-seeded random unit vector per phrase; image
    embeddings are a frozen random projection of a colour/edge summary.  It
    preserves the *interface* (and lets grouping + relevancy code run), but it
    carries no semantics, so the notebook disables language-query claims.
    """

    def __init__(self, dim: int = 512, seed: int = 1234):
        super().__init__()
        self.dim = dim
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("proj", torch.randn(64, dim, generator=g) / math.sqrt(64))
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2, bias=False), nn.GELU(),
            nn.Conv2d(32, 64, 5, 2, 2, bias=False), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        with torch.no_grad():
            for m in self.stem.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.copy_(torch.randn(m.weight.shape, generator=g) * 0.1)
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode_text_str(self, phrase: str) -> torch.Tensor:
        g = torch.Generator().manual_seed(abs(hash(phrase)) % (2**31))
        v = torch.randn(self.dim, generator=g)
        return (v / v.norm()).to(self.proj.device)

    @torch.no_grad()
    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x) @ self.proj


class OpenCLIPNetwork:
    """Port of ``pogs/encoders/openclip_encoder.py``."""

    NEGATIVES = ("object", "things", "stuff", "texture")

    def __init__(
        self,
        device: str = "cuda:0",
        model_type: str = "ViT-B-16",
        pretrained: str = "laion2b_s34b_b88k",
        clip_n_dims: int = 512,
        allow_download: bool = True,
    ):
        self.device = device
        self.clip_n_dims = clip_n_dims
        self.is_pretrained = False
        self.model = None
        self.tokenizer = None
        self.backend = "none"

        if pretrained == "laion2b_s34b_b88k" and model_type == "ViT-B-32":
            pretrained = "laion2b_s34b_b79k"

        if allow_download:
            try:
                import open_clip

                model, _, _ = open_clip.create_model_and_transforms(
                    model_type, pretrained=pretrained,
                    precision="fp16" if "cuda" in str(device) else "fp32", device=device,
                )
                model.eval()
                self.model = model
                self.tokenizer = open_clip.get_tokenizer(model_type)
                self.is_pretrained = True
                self.backend = f"open_clip/{model_type}/{pretrained}"
            except Exception as exc:  # noqa: BLE001
                print(f"[clip] open_clip unavailable ({type(exc).__name__}: {exc}); "
                      f"using deterministic random embeddings")

        if self.model is None:
            self.model = _FrozenRandomCLIP(clip_n_dims).to(device).eval()
            self.backend = "frozen-random"

        # torchvision-free equivalent of the official Resize(224) + Normalize
        self._clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self._clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
        self.positives: List[str] = [""]
        self.pos_embeds = torch.zeros(1, clip_n_dims, device=device)
        self.neg_embeds = self._encode_text(list(self.NEGATIVES))

    @property
    def embedding_dim(self) -> int:
        return self.clip_n_dims

    # -- text --------------------------------------------------------------- #

    @torch.no_grad()
    def _encode_text(self, phrases: Sequence[str]) -> torch.Tensor:
        if self.is_pretrained:
            tok = torch.cat([self.tokenizer(p) for p in phrases]).to(self.device)
            e = self.model.encode_text(tok).float()
        else:
            e = torch.stack([self.model.encode_text_str(p) for p in phrases]).float()
        return e / e.norm(dim=-1, keepdim=True)

    def set_positives(self, text_list: Sequence[str]) -> None:
        self.positives = list(text_list)
        self.pos_embeds = self._encode_text(self.positives)

    # -- image -------------------------------------------------------------- #

    def process(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
        return (x - self._clip_mean) / self._clip_std

    @torch.no_grad()
    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) in [0, 1]."""
        p = self.process(x.to(self.device))
        if self.is_pretrained and "cuda" in str(self.device):
            p = p.half()
        return self.model.encode_image(p).float()

    # -- relevancy (verbatim port) ------------------------------------------ #

    def get_relevancy(self, embed: torch.Tensor, positive_id: int = 0) -> torch.Tensor:
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0).to(embed.device)
        p = phrases_embeds.to(embed.dtype)
        output = torch.mm(embed, p.T)
        positive_vals = output[..., positive_id : positive_id + 1]
        negative_vals = output[..., len(self.positives) :]
        repeated_pos = positive_vals.repeat(1, len(self.NEGATIVES))
        sims = torch.stack((repeated_pos, negative_vals), dim=-1)
        softmax = torch.softmax(10 * sims, dim=-1)
        best_id = softmax[..., 0].argmin(dim=1)
        return torch.gather(
            softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.NEGATIVES), 2)
        )[:, 0, :]


# --------------------------------------------------------------------------- #
# LERF-style multi-scale CLIP pyramid
# --------------------------------------------------------------------------- #


class ClipPyramid:
    """Dense per-pixel CLIP embeddings at ``n_scales`` physical scales.

    ``PatchEmbeddingDataloader`` slides a square tile of side
    ``tile_ratio * H`` with stride ``tile_ratio * H * stride_scaler`` over each
    image, embeds every tile, and stores the resulting coarse grid.  Sampling a
    pixel bilinearly interpolates that grid.  ``PyramidEmbeddingDataloader``
    stacks one such grid per scale and interpolates between the two bracketing
    scales at query time -- both reproduced here.
    """

    def __init__(
        self,
        clip: OpenCLIPNetwork,
        image_hw: Tuple[int, int],
        tile_size_range: Tuple[float, float] = (0.05, 0.5),
        n_scales: int = 10,
        stride_scaler: float = 0.5,
        device: str = "cuda:0",
    ):
        self.clip = clip
        self.H, self.W = image_hw
        self.device = device
        self.tile_ratios = torch.linspace(tile_size_range[0], tile_size_range[1], n_scales)
        self.stride_scalers = [
            float(np.interp(float(t), [0.05, 0.15], [1.0, stride_scaler])) for t in self.tile_ratios
        ]
        self.grids: List[torch.Tensor] = []      # per scale: (V, gh, gw, D)

    @torch.no_grad()
    def build(self, images: torch.Tensor, batch: int = 64, verbose: bool = True) -> None:
        """images: (V, 3, H, W) in [0, 1]."""
        V = images.shape[0]
        self.grids = []
        for si, ratio in enumerate(self.tile_ratios):
            k = max(8, int(self.H * float(ratio)))
            stride = max(4, int(k * self.stride_scalers[si]))
            pad = k // 2
            per_view = []
            for v in range(V):
                img = images[v : v + 1].to(self.device)
                patches = F.unfold(img, kernel_size=k, stride=stride, padding=pad)  # (1, 3*k*k, L)
                L = patches.shape[-1]
                patches = patches.transpose(1, 2).reshape(L, 3, k, k)
                embs = []
                for i in range(0, L, batch):
                    embs.append(self.clip.encode_image(patches[i : i + batch]))
                e = torch.cat(embs, 0)
                e = e / (e.norm(dim=-1, keepdim=True) + 1e-8)
                gh = (self.H + 2 * pad - k) // stride + 1
                gw = (self.W + 2 * pad - k) // stride + 1
                per_view.append(e.reshape(gh, gw, -1).cpu())
            self.grids.append(torch.stack(per_view))
            if verbose:
                print(f"  clip pyramid scale {si + 1}/{len(self.tile_ratios)} "
                      f"(tile={k}px, grid={self.grids[-1].shape[1]}x{self.grids[-1].shape[2]})")

    @torch.no_grad()
    def sample(self, view_idx: int, scale: float, out_hw: Tuple[int, int]) -> torch.Tensor:
        """Dense (h, w, D) embedding map for one view at a physical scale."""
        ratios = self.tile_ratios
        scale = float(np.clip(scale, float(ratios[0]), float(ratios[-1])))
        hi = int(torch.searchsorted(ratios, torch.tensor(scale)).item())
        hi = min(max(hi, 1), len(ratios) - 1)
        lo = hi - 1
        w = (scale - float(ratios[lo])) / (float(ratios[hi]) - float(ratios[lo]) + 1e-8)

        def _resample(si):
            g = self.grids[si][view_idx].to(self.device).permute(2, 0, 1).unsqueeze(0)
            return F.interpolate(g, size=out_hw, mode="bilinear", align_corners=False)[0].permute(1, 2, 0)

        e = (1 - w) * _resample(lo) + w * _resample(hi)
        return e / (e.norm(dim=-1, keepdim=True) + 1e-8)


def load_dino_extractor(
    model_name: str = "dinov2_vits14",
    device: str = "cuda:0",
    pca_dim: int = 64,
    allow_download: bool = True,
    **kwargs,
) -> DinoExtractor:
    """Load or initialize DINO feature extractor."""
    return DinoExtractor(
        device=device,
        pca_dim=pca_dim,
        model_type=model_name,
        allow_download=allow_download,
        **kwargs,
    )


def load_clip_extractor(
    model_name: str = "ViT-B-16",
    device: str = "cuda:0",
    allow_download: bool = True,
    **kwargs,
) -> OpenCLIPNetwork:
    """Load or initialize OpenCLIP network."""
    return OpenCLIPNetwork(
        device=device,
        model_type=model_name,
        allow_download=allow_download,
        **kwargs,
    )