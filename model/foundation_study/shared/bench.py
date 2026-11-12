"""Latency and memory benchmark for the study's foundation models.

Deliberately standalone: it imports nothing from POGS, so Tier-1 models can be
timed without detectron2, Detic, gsplat or cuml installed. A model whose
dependency is missing is recorded as ``blocked`` rather than aborting the run --
one unavailable model must not cost us the other four numbers.

Protocol. Numbers from a naive loop are meaningless, so:

* ``cudnn.benchmark = True`` and a warmup pass, so autotuning and lazy
  allocation are not charged to the measurement
* ``torch.cuda.synchronize()`` on both sides of the timed region, because
  kernel launches are asynchronous and the clock would otherwise stop early
* median *and* p95, because a mean hides the tail that matters for a
  frame-rate budget
* ``reset_peak_memory_stats()`` per model, since peak VRAM is decided by the
  largest single model, not the sum
* fp32 and fp16 separately, because vendor tables are usually fp16 and mixing
  them makes the comparison dishonest

Usage::

    python bench.py --row 00_baseline_detic_dinov2_clip
    python bench.py --models clip_vitb16 --dtypes fp16
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_config as cfg  # noqa: E402


# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #
def describe_env() -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_total_mb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024**2
        )
        info["cudnn"] = torch.backends.cudnn.version()
    return info


# --------------------------------------------------------------------------- #
# loaders: each returns (callable, n_params)
# --------------------------------------------------------------------------- #
def _count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def load_dino(name: str, device: str):
    hub = cfg.MODELS[name]["spec"].split(":", 1)
    repo, entry = hub[0], hub[1]
    model = torch.hub.load(repo, entry).to(device).eval()

    def run(x):
        return model(x)

    return run, _count(model)


def load_dino_txt(name: str, device: str):
    repo, entry = cfg.MODELS[name]["spec"].split(":", 1)
    model = torch.hub.load(repo, entry).to(device).eval()

    def run(x):
        # DinoTxt has no forward_features. Its dense-feature path -- the role it
        # plays in place of POGS's DINOv2 -- is the class + patch tokens.
        return model.get_visual_class_and_patch_tokens(x)

    return run, _count(model)


def load_clip(name: str, device: str):
    import open_clip

    arch = cfg.MODELS[name]["spec"].split(" / ")
    model, _, _ = open_clip.create_model_and_transforms(
        arch[0], pretrained=arch[1]
    )
    model = model.to(device).eval()

    def run(x):
        return model.encode_image(x)

    return run, _count(model)


def local_weight(name: str, *candidates: str) -> str | None:
    """Path to an already-downloaded weight file, if there is one.

    Rows download weights up front via ``download_weights.py`` so a timed run
    never pays for a download. Preferring the local copy is what makes that
    worth doing: without this, ``pretrained=<repo id>`` silently re-derives
    through the HF cache and the timing includes whatever the cache misses.
    """
    mdir = os.path.join(cfg.WEIGHTS, name)
    if not os.path.isdir(mdir):
        return None
    for c in candidates:
        p = os.path.join(mdir, c)
        if os.path.exists(p):
            return p
    return None


def load_mobileclipv2(name: str, device: str):
    """MobileCLIP2 through the same open_clip entry point POGS uses for CLIP.

    timm ships an ``open_clip_config.json`` alongside the weights, so this is
    literally the baseline's loader with a different architecture string. That is
    the finding: in the alignment slot, MobileCLIP2 needs no adapter at all.
    """
    import open_clip

    # open_clip's registry calls it 'MobileCLIP2-B'; '-OpenCLIP' is the timm
    # repo suffix, not an architecture.
    arch = cfg.MODELS[name]["spec"].removesuffix("-OpenCLIP")
    local = local_weight(name, "open_clip_pytorch_model.bin")
    if local is None:
        raise FileNotFoundError(
            f"MobileCLIP2 weights not in {cfg.WEIGHTS}/{name}. Run "
            f"'python download_weights.py --models mobileclipv2' first."
        )
    model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=local)
    model = model.to(device).eval()

    def run(x):
        return model.encode_image(x)

    return run, _count(model)


def load_dinov3(name: str, device: str):
    """DINOv3 via transformers from the local snapshot.

    Not torch.hub: DINOv3 ships as a transformers checkpoint and is gated, so it
    needs ``HF_TOKEN`` and ``local_files_only`` once downloaded. Reports its own
    stride, because POGS' resolution math hardcodes 14 (dino_dataloader.py:24).
    """
    from transformers import AutoModel

    local = local_weight(name, "model.safetensors")
    if local is None:
        raise FileNotFoundError(
            f"DINOv3 weights not in {cfg.WEIGHTS}/{name}. It is gated: set "
            f"HF_TOKEN, then 'python download_weights.py --models dinov3'."
        )
    # The local folder, not the repo id: the id makes transformers look in its
    # own cache and miss the snapshot this script downloaded (see
    # load_locateanything3d for the same pattern).
    model = AutoModel.from_pretrained(os.path.dirname(local), local_files_only=True)
    model = model.to(device).eval()
    patch = getattr(model.config, "patch_size", 16)

    def run(x):
        return model(pixel_values=x).last_hidden_state

    run.stride = patch
    return run, _count(model)


def load_locateanything3d(name: str, device: str):
    """LocateAnything3D forward pass. Reported, but not POGS-compatible.

    Kept in the harness so the row produces a latency number instead of a bare
    "blocked", while the report states plainly that it emits 3D boxes and cannot
    satisfy POGS's per-instance 2D mask contract.
    """
    raise NotImplementedError(
        "LocateAnything3D is a vision-language model: a bare image forward has no "
        "text tokens (embedding(None)) and says nothing about detection speed. "
        "Its latency is measured by bench_locateanything.py (prompt + generate).")
    from transformers import AutoModel   # kept for reference; unreachable

    local = local_weight(name, "model-00001-of-00002.safetensors")
    if local is None:
        raise FileNotFoundError(
            f"LocateAnything3D weights not in {cfg.WEIGHTS}/{name}. They are "
            f"~9.5 GB; run 'python download_weights.py --models "
            f"locateanything3d'."
        )
    # The local folder, not the repo id: the id makes transformers look in its
    # own cache and online. custom_code: the model is built from .py files that
    # download_weights.py fetched next to the weights. torch_dtype, not dtype --
    # row 01 pins transformers 4.49, which predates the rename.
    model = AutoModel.from_pretrained(os.path.dirname(local), local_files_only=True,
                                      trust_remote_code=True,
                                      torch_dtype=torch.bfloat16,
                                      attn_implementation="sdpa")
    model = model.to(device).eval()

    def run(x):
        # pixel_values only: POGS' detector sees an image and nothing else.
        return model(pixel_values=x.to(torch.bfloat16))

    return run, _count(model)


def load_siglip2(name: str, device: str):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(cfg.MODELS[name]["spec"]).to(device).eval()

    def run(x):
        return model.get_image_features(pixel_values=x)

    return run, _count(model)


def load_yolo(name: str, device: str):
    from ultralytics import YOLO

    model = YOLO(cfg.MODELS[name]["spec"] + ".pt")
    model.to(device)

    def run(x):
        # Ultralytics takes uint8 HWC BGR at native resolution, so this is not
        # comparable tensor-for-tensor with the ViT models. We keep the tensor
        # move in the timed region because that is the real per-frame cost.
        img = (x[0].permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        return model.predict(img, verbose=False, device=0)

    return run, 0


def load_rfdetr(name: str, device: str):
    from rfdetr import RFDETRBase

    model = RFDETRBase()

    def run(x):
        return model.predict(x[0].permute(1, 2, 0).cpu().numpy(), verbose=False)

    return run, 0


LOADERS = {
    "dino_vits14": load_dino,
    "dino_vitl14": load_dino,
    "dino_txt": load_dino_txt,
    "dinov3": load_dinov3,
    "clip_vitb16": load_clip,
    "mobileclipv2": load_mobileclipv2,
    "siglip2": load_siglip2,
    "yolo": load_yolo,
    "rfdetr": load_rfdetr,
    "locateanything3d": load_locateanything3d,
}


def resolution_for(name: str, scene_w: int = cfg.IMAGE_W,
                   scene_h: int = cfg.IMAGE_H) -> tuple[int, int]:
    """The resolution POGS runs this model at, for a given scene.

    A substitution row does not necessarily use POGS' resolution, so stride-aware
    models are given their own grid. DINOv3 is stride 16 against POGS' stride 14,
    so reusing ``dino_resolution()`` would compare the two at different token
    counts and the latency figure would be meaningless.
    """
    stage = cfg.MODELS[name]["stage"]
    if name in ("clip_vitb16", "mobileclipv2"):
        # Both are open_clip encoders with position embeddings fixed at
        # 224x224 (196 tokens at patch 16); POGS' OpenCLIPNetwork resizes
        # every patch to that size before encoding
        # (encoders/openclip_encoder.py). A scene-sized grid fails with
        # "size of tensor a (300) must match the size of tensor b (196)".
        # The figure is per crop, not per frame.
        return 224, 224
    if "detection" in stage:
        return cfg.DETIC_RES, cfg.DETIC_RES
    if "features" in stage:
        stride = 16 if name == "dinov3" else cfg.DINO_STRIDE
        return cfg.dino_resolution(scene_w, scene_h, p=stride)
    return cfg.clip_resolution(scene_w, scene_h)


# --------------------------------------------------------------------------- #
# timing
# --------------------------------------------------------------------------- #
def time_model(run, x, warmup: int, iters: int) -> dict:
    cuda = x.is_cuda

    for _ in range(warmup):
        run(x)
    if cuda:
        torch.cuda.synchronize()

    if cuda:
        torch.cuda.reset_peak_memory_stats()
        base_mem = torch.cuda.memory_allocated() / 1024**2
        torch.cuda.reset_peak_memory_stats()

    samples = []
    for _ in range(iters):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        run(x)
        if cuda:
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)

    peak = (
        (torch.cuda.max_memory_allocated() / 1024**2 - base_mem)
        if cuda
        else 0.0
    )
    samples.sort()
    return {
        "median_ms": round(statistics.median(samples), 3),
        "mean_ms": round(statistics.fmean(samples), 3),
        "p95_ms": round(samples[int(0.95 * (len(samples) - 1))], 3),
        "min_ms": round(samples[0], 3),
        "max_ms": round(samples[-1], 3),
        "peak_delta_mb": round(peak, 1),
        "iters": iters,
    }


def bench_one(name: str, dtype_name: str, device: str, warmup: int,
              iters: int, scene_w: int, scene_h: int) -> dict:
    meta = cfg.MODELS[name]
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[
        dtype_name
    ]
    # Sized from the scene actually being benchmarked. The pendulum scenes are
    # 640x480 and pickplace is 1280x720, so a single hardcoded resolution would
    # time every model against an input size none of them will ever see -- and
    # the two scenes' numbers would not be comparable to each other.
    w, h = resolution_for(name, scene_w, scene_h)
    rec: dict = {
        "model": name,
        "stage": meta["stage"],
        "pogs_uses": meta["pogs_uses"],
        "open_vocab": meta["open_vocab"],
        "loader": meta["loader"],
        "spec": meta["spec"],
        "dtype": dtype_name,
        "resolution": f"{w}x{h}",
        "scene_resolution": f"{scene_w}x{scene_h}",
        "device": device,
    }
    try:
        t0 = time.perf_counter()
        run, n_params = LOADERS[name](name, device)
        rec["load_s"] = round(time.perf_counter() - t0, 2)
        rec["params"] = n_params

        # Weights stay fp32 and the reduced dtype is applied by autocast, so one
        # code path covers every loader (casting the input alone made fp32
        # weights meet half inputs: "Input type (Half) and bias type (float)").
        # Clamped for every dtype: the fp32 branch used to feed raw N(0,1)
        # (values above 1) and RF-DETR's range check blocked on it while the
        # clamped fp16 branch passed.
        x = torch.randn(1, 3, h, w, device=device, dtype=torch.float32).clamp(0, 1)
        if dtype != torch.float32:
            fp32_run = run

            def run(inp, _r=fp32_run):
                with torch.autocast(device_type=torch.device(device).type,
                                    dtype=dtype):
                    return _r(inp)

            rec["precision_mode"] = "autocast"
        rec.update(time_model(run, x, warmup, iters))
        rec["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - a missing dep must not abort
        rec["status"] = "blocked"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc(limit=3)
    # A failed load leaves the model reachable from the exception's traceback
    # frames (a reference cycle), so it stays on the GPU and the next model OOMs.
    # Drop every reference, collect the cycle, then return the memory.
    run = x = fp32_run = None
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rec


# --------------------------------------------------------------------------- #
def models_for_row(row: str) -> list[str]:
    """Which models a row's latency table should contain.

    The row's own models, not every model in the registry: row 02's table is
    about RF-DETR and MobileCLIP2, and including the baseline's DINOv2 there
    would make the two tables incomparable.
    """
    meta = cfg.ROWS.get(row)
    if not meta:
        return list(cfg.STANDALONE)
    wanted = [meta.get("detection"), meta.get("features"),
              meta.get("alignment")]
    if meta.get("available_models"):
        wanted += meta["available_models"]
    out = [m for m in dict.fromkeys(wanted) if m and m in cfg.MODELS]
    return [m for m in out if cfg.MODELS[m].get("loader") != "pogs"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark the study's models")
    ap.add_argument("--row", default=cfg.BASELINE_ROW,
                    help="which row's artefacts directory to write to")
    ap.add_argument("--models", nargs="+", default=None,
                    help="override the row's default model list")
    ap.add_argument("--dtypes", nargs="+", default=["fp32", "fp16"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--weights", default=None,
                    help="weights root (default: model_study/weights)")
    ap.add_argument("--scene", default=None,
                    help="scene whose resolution to benchmark at (default: "
                         "the primary scene). Latency depends on input size, "
                         "so a figure is only meaningful next to the scene it "
                         "was taken on.")
    args = ap.parse_args()

    if args.weights:
        cfg.WEIGHTS = os.path.abspath(args.weights)

    scene = args.scene or os.path.basename(cfg.PICKPLACE)
    try:
        scene_dir = cfg.scene_path(scene)
    except SystemExit:
        scene_dir = cfg.PICKPLACE
    scene_w, scene_h = cfg.scene_resolution(scene_dir)

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        print("WARNING: no CUDA visible. Absolute timings will not transfer "
              "to the 4070; relative ranking may still hold.", file=sys.stderr)

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(cfg.POGS_SEED)

    out_dir = os.path.join(cfg.ARTEFACTS, args.row, scene, "bench")
    os.makedirs(out_dir, exist_ok=True)

    results = {
        "environment": describe_env(),
        "scene": scene,
        "protocol": {
            "warmup": args.warmup,
            "iters": args.iters,
            "cudnn_benchmark": True,
            "synchronize": True,
            "seed": cfg.POGS_SEED,
            "scene_resolution": f"{scene_w}x{scene_h}",
        },
        "measurements": [],
    }

    print("=" * 72)
    print("POGS MODEL STUDY - LATENCY BENCHMARK")
    print("=" * 72)
    env = results["environment"]
    print(f"  gpu        : {env.get('gpu', 'cpu')}")
    print(f"  torch      : {env['torch']}  cuda {env.get('cuda', '-')}")
    print(f"  scene      : {scene}  ({scene_w}x{scene_h})")
    print(f"  protocol   : {args.warmup} warmup, {args.iters} iters, "
          f"{'+'.join(args.dtypes)}")
    print("=" * 72)

    names = args.models or models_for_row(args.row)
    if not names:
        print(f"  row {args.row} has no benchable model outside the POGS "
              f"environment. Nothing to time.")
    print(f"  models   : {', '.join(names) if names else '(none)'}")
    print("=" * 72)

    for name in names:
        if name not in cfg.MODELS:
            print(f"  SKIP  {name}: not in the registry")
            continue
        if cfg.MODELS[name].get("blocked"):
            note = cfg.MODELS[name].get("note", "no weights source")
            print(f"  BLOCKED {name}: {note}")
            results["measurements"].append({
                "model": name,
                "stage": cfg.MODELS[name]["stage"],
                "pogs_uses": False,
                "dtype": "-",
                "resolution": "-",
                "status": "blocked",
                "error": note,
            })
            continue
        for dtype_name in args.dtypes:
            rec = bench_one(name, dtype_name, args.device, args.warmup,
                            args.iters, scene_w, scene_h)
            results["measurements"].append(rec)
            if rec["status"] == "ok":
                star = " *" if rec["pogs_uses"] else "  "
                print(f"  {star} {name:<14} {dtype_name}  {rec['resolution']:>9}  "
                      f"median {rec['median_ms']:8.2f} ms  p95 {rec['p95_ms']:8.2f} ms"
                      f"  peak {rec['peak_delta_mb']:7.1f} MB")
            else:
                print(f"     {name:<14} {dtype_name}  BLOCKED: {rec['error']}")

    ok = [r for r in results["measurements"] if r["status"] == "ok"]
    # One entry per model, so a model measured in both fp32 and fp16 is not
    # counted twice. Take each model's fastest measured median; that keeps the
    # headline number comparable with vendor tables, which quote fp16.
    per_model_best: dict[str, float] = {}
    for m in sorted(ok, key=lambda r: r["median_ms"]):
        per_model_best.setdefault(m["model"], m["median_ms"])
    total = sum(per_model_best.values())
    results["total_pipeline_median_ms_fp16"] = round(total, 3)
    results["measured_models"] = len(ok)
    results["blocked_models"] = len(results["measurements"]) - len(ok)

    path = os.path.join(out_dir, "latency.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)
    print("=" * 72)
    print(f"  total pipeline (fastest per model): {total:.2f} ms")
    print(f"  wrote {path}")
    print("  * = stock POGS model")
    return 0


if __name__ == "__main__":
    sys.exit(main())