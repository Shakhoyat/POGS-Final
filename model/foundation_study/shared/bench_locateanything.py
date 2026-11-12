"""LocateAnything3D as a detector: latency per detection, and does it find the object.

``bench.py`` times a bare forward pass with a random image, which is meaningless
for a vision-language model: it needs a text prompt, and it spends its time
generating the box text token by token. This follows the model card instead --
chat template, real frame, ``generate`` -- and times that.

It also answers the question the latency number does not: are the boxes right?
Each frame's predicted boxes are compared with the box around the ground-truth
object mask (``tracking/obj_id``). The model emits boxes, not masks, so this
measures localisation only; POGS needs per-instance masks, which this row does
not supply (see the row's README section).

    python bench_locateanything.py --scene pendulum_v2_30 --frames 8
"""

import argparse
import glob
import json
import os
import re
import statistics
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_config as cfg  # noqa: E402

BOX = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")


def parse_boxes(answer: str, w: int, h: int) -> list[tuple[float, float, float, float]]:
    """Model coordinates are integers in [0, 1000]; convert to pixels."""
    return [(int(a) / 1000 * w, int(b) / 1000 * h, int(c) / 1000 * w, int(d) / 1000 * h)
            for a, b, c, d in BOX.findall(answer)]


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def gt_box(mask_path: str):
    m = np.array(Image.open(mask_path).convert("L")) > 127
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", default="01_locateanything_dinotxt")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--frames", type=int, default=8, help="frames spread over the sequence")
    ap.add_argument("--prompt", default=None, help="category; default by scene")
    ap.add_argument("--mode", default="hybrid", choices=["fast", "slow", "hybrid"])
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--weights", default=cfg.WEIGHTS)
    args = ap.parse_args()

    from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer

    model_dir = os.path.join(args.weights, "locateanything3d")
    category = args.prompt or ("mug" if "pickplace" in args.scene else "cylinder")
    track = os.path.join(cfg.scene_path(args.scene), "tracking")
    frames = sorted(glob.glob(os.path.join(track, "left", "*.png")))
    masks = sorted(glob.glob(os.path.join(track, "obj_id", "*.png")))
    pick = np.linspace(0, len(frames) - 1, args.frames).round().astype(int)

    print(f"scene {args.scene}   prompt {category!r}   {len(pick)} frames   mode {args.mode}")
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    proc = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    # The checkpoint's config asks for "magi" attention (a library we do not have).
    # Its language model accepts only magi or sdpa, and with transformers 4.51 the
    # attn_implementation argument reaches the outer model only, so the text model
    # would fall back to "eager" and raise. Set sdpa on the sub-configs directly.
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    for sub in (config, config.text_config, config.vision_config):
        sub._attn_implementation = "sdpa"
    model = AutoModel.from_pretrained(model_dir, config=config, torch_dtype=torch.bfloat16,
                                      trust_remote_code=True, local_files_only=True
                                      ).to("cuda").eval()
    load_s = time.perf_counter() - t0
    print(f"loaded in {load_s:.1f} s, "
          f"{sum(p.numel() for p in model.parameters()) / 1e9:.2f} B parameters")

    prompt = ("Locate all the instances that matches the following description: "
              f"{category}.")

    @torch.no_grad()
    def detect(img: Image.Image) -> str:
        messages = [{"role": "user", "content": [{"type": "image", "image": img},
                                                 {"type": "text", "text": prompt}]}]
        text = proc.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = proc.process_vision_info(messages)
        inputs = proc(text=[text], images=images, videos=videos, return_tensors="pt").to("cuda")
        out = model.generate(
            pixel_values=inputs["pixel_values"].to(torch.bfloat16),
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_hws=inputs.get("image_grid_hws", None),
            tokenizer=tok, max_new_tokens=args.max_new_tokens, use_cache=True,
            generation_mode=args.mode, verbose=False,
            # The model card's sampling settings. Greedy decoding without the
            # repetition penalty looped on one box until the token limit.
            do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.1)
        return out[0] if isinstance(out, tuple) else out

    torch.manual_seed(0)                                      # sampling is on; fix it
    detect(Image.open(frames[pick[0]]).convert("RGB"))        # warm-up, not timed
    torch.cuda.reset_peak_memory_stats()

    rows = []
    for k in pick:
        img = Image.open(frames[k]).convert("RGB")
        torch.cuda.synchronize()
        t = time.perf_counter()
        ans = detect(img)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        boxes = parse_boxes(ans, *img.size)
        g = gt_box(masks[k])
        ious_k = [iou(b, g) for b in boxes] if g else []
        best = max(ious_k, default=0.0) if g else None
        first = ious_k[0] if ious_k else 0.0
        precision = (sum(i >= 0.5 for i in ious_k) / len(ious_k)) if ious_k else 0.0
        n_tok = len(tok(ans, add_special_tokens=False).input_ids)
        hit_cap = n_tok >= args.max_new_tokens - 8      # stopped by the limit, not by itself
        rows.append({"frame": int(k), "seconds": round(dt, 3), "n_boxes": len(boxes),
                     "n_distinct_boxes": len(set(boxes)), "tokens": n_tok, "hit_token_cap": hit_cap,
                     "best_iou": None if best is None else round(best, 3),
                     "first_box_iou": round(first, 3), "precision": round(precision, 3),
                     "answer": ans[:200]})
        print(f"  frame {k:>3}  {dt:6.2f} s  {len(boxes)} box(es) ({len(set(boxes))} distinct)  "
              f"first IoU {first:.2f}  best IoU {'n/a' if best is None else f'{best:.2f}'}  "
              f"precision {precision:.2f}  {n_tok} tok{'  HIT TOKEN CAP' if hit_cap else ''}")

    secs = [r["seconds"] for r in rows]
    ious = [r["best_iou"] for r in rows if r["best_iou"] is not None]
    summary = {
        "model": "locateanything3d", "scene": args.scene, "prompt": category,
        "mode": args.mode, "dtype": "bf16", "frames": len(rows),
        "median_s": round(statistics.median(secs), 3), "max_s": round(max(secs), 3),
        "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "load_s": round(load_s, 1),
        "found_iou50": sum(i >= 0.5 for i in ious), "scored": len(ious),
        "mean_best_iou": round(float(np.mean(ious)), 3) if ious else None,
        "mean_first_box_iou": round(float(np.mean([r["first_box_iou"] for r in rows])), 3),
        "mean_precision": round(float(np.mean([r["precision"] for r in rows])), 3),
        "frames_hit_token_cap": sum(r["hit_token_cap"] for r in rows),
        "max_new_tokens": args.max_new_tokens,
        "output": "2D boxes only; no instance masks, so not usable as POGS's detector as is",
        "per_frame": rows,
    }
    out_dir = os.path.join(cfg.ARTEFACTS, args.row, args.scene, "bench")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "locateanything_detect.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nmedian {summary['median_s']} s per detection   peak {summary['peak_gpu_mb']} MB   "
          f"found (IoU>=0.5) {summary['found_iou50']}/{summary['scored']}   "
          f"mean best IoU {summary['mean_best_iou']}")
    print(f"first-box IoU {summary['mean_first_box_iou']}   precision {summary['mean_precision']}   "
          f"frames stopped by the {args.max_new_tokens}-token limit: "
          f"{summary['frames_hit_token_cap']}/{len(rows)}")
    print(f"wrote {out_dir}/locateanything_detect.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
