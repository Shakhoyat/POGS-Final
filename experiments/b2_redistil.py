"""
Direction B, step 2 (LEDGER 8.B2): re-distil the DINO features ONLY, geometry and colour frozen, with another
2-D backbone, then re-run the B1 roll sweeps (training views, held-out views, tracking camera with / without
gripper and shadows) as the acceptance test before any tracking run.

Backbones (all ViT-S, 64-D PCA fitted on the 35 training views, as the release does):
    v2     DINOv2 ViT-S/14 (the release backbone), CONTROL: same feature-only loop, so any change in the
           other rows is the backbone's, not the re-distillation procedure's
    v2reg  DINOv2 ViT-S/14 with 4 registers (Darcet et al., ICLR 2024: removes high-norm artefact tokens)
    v3s    DINOv3 ViT-S/16 (timm `vit_small_patch16_dinov3.lvd1689m`; Gram anchoring for dense features)
The PCA basis is rescaled so the projected training features have the same mean squared norm as the release
basis (keeps the DINO : depth weight ratio of the tracking loss comparable). Geometry untouched, so the groups
(and ground-truth evaluation) are unchanged. Nothing here changes `model/pogs` (CLAUDE.md rule 2).

    source tools/cuda_env.sh && .venv/bin/python -m experiments.b2_redistil <kind> [--steps 3000]
Outputs: data_results/b2_redistil/<kind>/{features.pt, distil_log.json, views.json, context.json, summary.md}
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from data.scene_generator.pick_place_benchmark import load_capture
from data.scene_generator.tool_perturbation import TOOL_DATASET_DIR, build_tool_scene
from experiments import b_bias
from experiments.env_info import write_env_json
from experiments.o3_perturbation import setup_tool_context
from model.pogs import losses as L
from model.pogs.checkpoint import write_json
from model.pogs.encoders import DinoExtractor, get_img_resolution

OUT = "data_results/b2_redistil"


class TimmViTExtractor(DinoExtractor):
    """DinoExtractor interface (raw_feats / fit_pca / pca_feats / __call__) on a timm ViT with patch ``p``."""

    def __init__(self, name: str, device: str = "cuda:0", pca_dim: int = 64, patch: int = 16, max_size: int = 1248):
        import timm
        self.device, self.pca_dim, self.max_size, self.patch = device, pca_dim, max_size, patch
        self.dtype = torch.float16 if "cuda" in str(device) else torch.float32
        self.pca_matrix = None
        self.model = timm.create_model(name, pretrained=True, num_classes=0, dynamic_img_size=True).to(device).eval()
        if self.dtype == torch.float16:
            self.model = self.model.half()
        for q in self.model.parameters():
            q.requires_grad_(False)
        self.backend, self.is_pretrained = f"timm/{name}", True
        self._mean = torch.tensor(self.IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(self.IMAGENET_STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def raw_feats(self, images: torch.Tensor, batch: int = 1) -> torch.Tensor:
        H, W = images.shape[-2:]
        h, w = get_img_resolution(H, W, self.patch, self.max_size)
        out = []
        for i in range(0, images.shape[0], batch):
            x = F.interpolate(images[i:i + batch].to(self.device), size=(h, w), mode="bicubic",
                              align_corners=False, antialias=True)
            x = ((x - self._mean) / self._std).to(self.dtype)
            t = self.model.forward_features(x)[:, self.model.num_prefix_tokens:]
            out.append(t.reshape(x.shape[0], h // self.patch, w // self.patch, -1).float() / 10.0)
        return torch.cat(out, dim=0)


ANYUP_DIR = os.path.expanduser("~/third_party_pinned/anyup")              # git clone, read before use
ANYUP_COMMIT = "351807a9c4287368732cc247f26c7c81c9139af4"
ANYUP_CKPT = os.path.expanduser("~/third_party_pinned/anyup_weights/anyup_paper.pth")
ANYUP_SHA256 = "9d035c0f27114a6f32bdd3d8ed93b6cd39dad4b8f8bf94e69fddbfbf022901b2"


def _load_anyup(device: str):
    """AnyUp from a pinned, reviewed clone (no torch.hub fetch); state dict loaded with weights_only=True after
    checking the commit and the checkpoint's sha256."""
    import hashlib
    import subprocess
    import sys
    head = subprocess.run(["git", "-C", ANYUP_DIR, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != ANYUP_COMMIT:
        raise RuntimeError(f"AnyUp clone at {head}, expected pinned {ANYUP_COMMIT}")
    if hashlib.sha256(open(ANYUP_CKPT, "rb").read()).hexdigest() != ANYUP_SHA256:
        raise RuntimeError("AnyUp checkpoint checksum mismatch")
    sys.path.insert(0, ANYUP_DIR)
    from anyup.model import AnyUp
    m = AnyUp()
    m.load_state_dict(torch.load(ANYUP_CKPT, map_location="cpu", weights_only=True))
    return m.to(device).eval()


class AnyUpDinoExtractor(DinoExtractor):
    """DINOv2 ViT-S/14 features upsampled by AnyUp (checkpoint trained on DINOv2 ViT-S/14) to ``factor`` x the
    patch grid, guided by the ImageNet-normalised input image."""

    def __init__(self, device: str = "cuda:0", factor: int = 4):
        super().__init__(device=device, model_type="dinov2_vits14")
        self.factor = factor
        self.up = _load_anyup(device)
        for q in self.up.parameters():
            q.requires_grad_(False)
        self.backend = f"torch.hub/dinov2 + anyup@{ANYUP_COMMIT[:7]}"

    @torch.no_grad()
    def fit_pca(self, images: torch.Tensor, batch: int = 1, max_samples: int = 400_000) -> None:
        """Same PCA as the release, but sampled view by view (all upsampled views at once would need ~4 GB)."""
        per = max_samples // images.shape[0] + 1
        g = torch.Generator(device="cpu").manual_seed(0)
        picks = []
        for v in range(images.shape[0]):
            f = self.raw_feats(images[v:v + 1]).reshape(-1, 384)
            picks.append(f[torch.randperm(f.shape[0], generator=g)[:per].to(f.device)])
        flat = torch.cat(picks)
        self.pca_matrix = torch.pca_lowrank(flat, q=self.pca_dim, niter=20)[2]
        torch.cuda.empty_cache()

    @torch.no_grad()
    def raw_feats(self, images: torch.Tensor, batch: int = 1) -> torch.Tensor:
        lr = super().raw_feats(images, batch=batch) * 10.0           # undo /10 for the upsampler, redo below
        H, W = images.shape[-2:]
        h, w = get_img_resolution(H, W, 14, self.max_size)
        out = []
        for i in range(images.shape[0]):
            x = F.interpolate(images[i:i + 1].to(self.device).float(), size=(h, w), mode="bicubic",
                              align_corners=False, antialias=True)
            x = (x - self._mean) / self._std
            f = lr[i:i + 1].permute(0, 3, 1, 2).float()
            size = (lr.shape[1] * self.factor, lr.shape[2] * self.factor)
            u = self.up(x, f, output_size=size, q_chunk_size=2048)
            out.append(u.permute(0, 2, 3, 1).float() / 10.0)
        return torch.cat(out, dim=0)


def make_extractor(kind: str, device: str):
    if kind == "v2up":
        return AnyUpDinoExtractor(device=device)
    if kind == "v2":
        return DinoExtractor(device=device, model_type="dinov2_vits14")
    if kind == "v2reg":
        return DinoExtractor(device=device, model_type="dinov2_vits14_reg")
    if kind == "v3s":
        return TimmViTExtractor("vit_small_patch16_dinov3.lvd1689m", device=device)
    raise ValueError(kind)


@torch.no_grad()
def _targets(ext, imgs: torch.Tensor) -> torch.Tensor:
    return torch.stack([ext.pca_feats(imgs[v:v + 1]).squeeze(0).cpu() for v in range(imgs.shape[0])])


def redistil(ctx, kind: str, steps: int, device: str, out: str) -> dict:
    """Fits a new PCA, rescales it to the release basis' feature energy, and trains dino_feats + the DINO MLP
    only (Adam, release learning rates and schedule, nearest-neighbour variance term after 1000 steps)."""
    model, cfg = ctx["model"], ctx["cfg"]
    t = cfg.train
    cap = load_capture(TOOL_DATASET_DIR, device=device, objects=build_tool_scene())
    imgs = cap.images.permute(0, 3, 1, 2)                       # (V, 3, H, W)
    ref_energy = float((_targets(ctx["dino"], imgs) ** 2).sum(-1).mean())

    ext = make_extractor(kind, device)
    ext.fit_pca(imgs, batch=1)
    tg = _targets(ext, imgs)
    s = (ref_energy / float((tg ** 2).sum(-1).mean())) ** 0.5
    ext.set_pca(ext.get_pca() * s)
    tg = tg * s

    gp = model.gauss_params
    with torch.no_grad():
        gp["dino_feats"].data = torch.randn_like(gp["dino_feats"]) * cfg.model.dino_init_std
        for m in model.nn.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
    o_f = torch.optim.Adam([gp["dino_feats"]], lr=t.lr_dino_feats, eps=1e-15)
    o_n = torch.optim.Adam(list(model.nn.parameters()), lr=t.lr_nn_projection, eps=1e-15)
    frozen = [k for k in gp if k != "dino_feats"]
    for k in frozen:
        gp[k].requires_grad_(False)
    rng = np.random.default_rng(0)
    log, t0 = [], time.time()
    model.train()
    for step in range(steps):
        r = (t.lr_dino_feats_final / t.lr_dino_feats) ** min(step / 6000, 1.0)
        for o, lr0 in ((o_f, t.lr_dino_feats), (o_n, t.lr_nn_projection)):
            for g in o.param_groups:
                g["lr"] = lr0 * r
        v = int(rng.integers(0, imgs.shape[0]))
        o_f.zero_grad(set_to_none=True)
        o_n.zero_grad(set_to_none=True)
        o = model.get_outputs(cap.cameras[v].to(device), render_dino=True)
        loss = L.dino_loss(o["dino"], tg[v].to(device))
        if step > t.dino_nn_delay:
            loss = loss + L.dino_nn_variance_loss(gp["dino_feats"], model.ensure_nearest_ids())
        loss.backward()
        o_f.step()
        o_n.step()
        if step % 250 == 0 or step == steps - 1:
            log.append({"step": step, "loss": float(loss.detach()), "t": time.time() - t0})
            print(f"[b2:{kind}] step {step} loss {float(loss):.5f}", flush=True)
    model.eval()
    for k in frozen:
        gp[k].requires_grad_(True)
    torch.save({"dino_feats": gp["dino_feats"].detach().cpu(), "nn": model.nn.state_dict(),
                "pca": ext.get_pca(), "kind": kind, "pca_scale": s}, os.path.join(out, "features.pt"))
    info = {"kind": kind, "backend": ext.backend, "steps": steps, "pca_scale": s, "ref_energy": ref_energy,
            "final_loss": log[-1]["loss"], "time_s": time.time() - t0, "log": log}
    write_json(os.path.join(out, "distil_log.json"), info)
    ctx["dino"] = ext
    return info


def main(kind: str, steps: int, device: str = "cuda:0") -> None:
    out = os.path.join(OUT, kind)
    os.makedirs(out, exist_ok=True)
    write_env_json(out)
    ctx = setup_tool_context(device)
    fp = os.path.join(out, "features.pt")
    if os.path.exists(fp):                                         # reuse a finished distillation
        st = torch.load(fp, map_location=device)
        ctx["model"].gauss_params["dino_feats"].data = st["dino_feats"].to(device)
        ctx["model"].nn.load_state_dict(st["nn"])
        ext = make_extractor(kind, device)
        ext.set_pca(st["pca"])
        ctx["dino"] = ext
    else:
        redistil(ctx, kind, steps, device, out)
    b_bias.views(device, ctx=ctx, out=out)
    b_bias.context(device, ctx=ctx, out=out)
    b_bias.summarise(out=out, title=f"B2 acceptance sweeps: features re-distilled with `{kind}` (geometry frozen)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["v2", "v2reg", "v3s", "v2up"])
    ap.add_argument("--steps", type=int, default=3000)
    a = ap.parse_args()
    main(a.kind, a.steps)
