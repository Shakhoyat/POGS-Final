"""Warm each row's model caches so the first timed iteration is not a download.

``download_weights.py`` puts the bytes on disk, but several loaders then do their
own resolution work -- torch.hub unpacks an archive, open_clip builds a tokenizer,
transformers materialises a shard index -- and charging that to a latency
measurement makes the number describe the harness rather than the model.

Also the honest place to find out that a weight file is unusable: it reports each
model as ``ok`` or a reason, and a row with blocked models still completes.

Usage::

    python prefetch.py --row 00_baseline_detic_dinov2_clip
    python prefetch.py --row 01_locateanything_dinotxt --weights ~/pogs_weights
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_config as cfg  # noqa: E402
import bench as bench_mod  # noqa: E402


def prefetch_one(name: str, device: str,
                 scene_w: int = cfg.IMAGE_W, scene_h: int = cfg.IMAGE_H) -> dict:
    meta = cfg.MODELS.get(name, {})
    rec = {"model": name, "spec": meta.get("spec"), "stage": meta.get("stage")}
    loader = bench_mod.LOADERS.get(name)
    if loader is None:
        rec["status"] = "blocked"
        rec["reason"] = meta.get("note", "no loader in bench.LOADERS")
        return rec
    t0 = time.time()
    try:
        run, n_params = loader(name, device)
        # One real forward pass, not just construction: lazily-initialised buffers
        # and lazily-built masks only surface on the first call.
        import torch

        w, h = bench_mod.resolution_for(name, scene_w, scene_h)
        with torch.no_grad():
            run(torch.zeros(1, 3, h, w, device=device))
        rec["status"] = "ok"
        rec["params"] = n_params
        rec["seconds"] = round(time.time() - t0, 2)
        rec["resolution"] = f"{w}x{h}"
    except Exception as exc:  # noqa: BLE001 - one bad model must not stop the row
        rec["status"] = "blocked"
        rec["reason"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc(limit=2)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description="Warm model caches for a row")
    ap.add_argument("--row", default=cfg.BASELINE_ROW)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--scene", default=None,
                    help="scene whose resolution to warm at (default: the "
                         "primary scene)")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--device",
                    default="cuda" if _cuda() else "cpu")
    args = ap.parse_args()

    if args.weights:
        cfg.WEIGHTS = os.path.abspath(args.weights)
        bench_mod.cfg.WEIGHTS = cfg.WEIGHTS

    names = args.models or bench_mod.models_for_row(args.row)

    scene = args.scene or os.path.basename(cfg.PICKPLACE)
    try:
        scene_dir = cfg.scene_path(scene)
    except SystemExit:
        scene_dir = cfg.PICKPLACE
    scene_w, scene_h = cfg.scene_resolution(scene_dir)

    print("=" * 68)
    print(f"PREFETCH - {args.row} @ {scene} ({scene_w}x{scene_h})")
    print("=" * 68)
    if not names:
        print("  nothing to prefetch for this row")
        return 0

    results = []
    for name in names:
        rec = prefetch_one(name, args.device, scene_w, scene_h)
        results.append(rec)
        if rec["status"] == "ok":
            print(f"  ok       {name:<20} {rec['seconds']:6.2f}s  "
                  f"{rec.get('params', 0) / 1e6:8.1f}M params  {rec['resolution']}")
        else:
            print(f"  BLOCKED  {name:<20} {rec['reason']}")

    out_dir = os.path.join(cfg.ARTEFACTS, args.row, scene, "probe")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "prefetch.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"row": args.row, "scene": scene,
                   "results": results}, fh, indent=2)

    ok = [r for r in results if r["status"] == "ok"]
    blocked = [r for r in results if r["status"] != "ok"]
    print(f"\n  {len(ok)} ready, {len(blocked)} blocked -> {path}")
    return 0


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    sys.exit(main())