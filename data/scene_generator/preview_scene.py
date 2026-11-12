"""
Scene preview: render a handful of frames with the real generator code and tile
them into one contact sheet, to review the scene before a full generation.

    python -m data.scene_generator.preview_scene [--device cpu] [--scale 1.0]

Writes ``data_results/scene_preview/contact_sheet.png`` (+ the single frames).
Rows: two scan views and a depth map; tracking frames of ``base`` at the
approach / carry / place phases; ``tilt60``, ``occ_hand`` and ``multi`` mid-carry.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time

import cv2
import numpy as np
import torch

from data.scene_generator.capture_generator import hemisphere_trajectory
from data.scene_generator.photoreal_scene import render_photoreal_scene
from data.scene_generator.pick_place_benchmark import BenchmarkConfig, generate_sequence
from data.scene_generator.pick_place_scene import build_pick_place_scene
from data.scene_generator.scene_builder import FlatScene

OUT_DIR = "data_results/scene_preview"

# (condition, frame index at 5 Hz, caption)
TRACK_SHOTS = [
    ("base", 3, "base: approach"),
    ("base", 27, "base: carry"),
    ("base", 54, "base: placed on the table"),
    ("tilt60", 30, "tilt60: peak tilt"),
    ("occ_hand", 30, "occ_hand: hand occluder"),
    ("multi", 30, "multi: two objects moving"),
]


def _caption(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def main(device: str = "cpu", scale: float = 1.0, threads: int = 6) -> str:
    torch.set_num_threads(threads)
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = BenchmarkConfig()
    if scale != 1.0:
        import dataclasses
        cfg = dataclasses.replace(cfg, track_width=int(cfg.track_width * scale) // 2 * 2,
                                  track_height=int(cfg.track_height * scale) // 2 * 2,
                                  track_fx=cfg.track_fx * scale)
    W, H = cfg.track_width, cfg.track_height
    t0 = time.time()
    tiles = []

    # ---- scan views (what POGS is trained on) + one depth map ------------- #
    cams = hemisphere_trajectory(n_views=cfg.n_views, radius=cfg.capture_radius,
                                 elevation_deg=cfg.capture_elevation_deg, center=cfg.capture_center,
                                 width=int(cfg.capture_width * scale), height=int(cfg.capture_height * scale),
                                 fov_deg=cfg.capture_fov_deg, device=device)
    scene = FlatScene(build_pick_place_scene(), device=device)
    for i, vi in enumerate((4, 22)):
        out = render_photoreal_scene(scene, cams[vi], cast_shadows=True)
        rgb = (out["rgb"].cpu().numpy() * 255).astype(np.uint8)
        tiles.append(_caption(_fit(rgb, W, H), f"scan view {vi} (training image)"))
        if i == 0:
            d = out["depth"].cpu().numpy()
            dn = np.clip((d - 0.3) / 0.8, 0, 1)
            dv = cv2.applyColorMap((255 * (1 - dn)).astype(np.uint8), cv2.COLORMAP_TURBO)
            dv[d <= 0] = 0
            tiles.append(_caption(_fit(cv2.cvtColor(dv, cv2.COLOR_BGR2RGB), W, H), f"scan view {vi}: depth"))
        print(f"[preview] scan view {vi} done ({time.time() - t0:.0f}s)")

    # ---- tracking frames, rendered by the real sequence generator --------- #
    tmp = os.path.join(OUT_DIR, "_frames")
    for cond, f, cap in TRACK_SHOTS:
        sdir = generate_sequence(tmp, cond, 0, cfg, device=device, verbose=False, frames=[f])
        rgb = cv2.cvtColor(cv2.imread(os.path.join(sdir, "rgb", f"frame_{f:04d}.png")), cv2.COLOR_BGR2RGB)
        tiles.append(_caption(rgb, f"tracking camera | {cap}"))
        print(f"[preview] {cond} frame {f} done ({time.time() - t0:.0f}s)")
    shutil.rmtree(tmp, ignore_errors=True)

    for i, t in enumerate(tiles):
        cv2.imwrite(os.path.join(OUT_DIR, f"tile_{i:02d}.png"), cv2.cvtColor(t, cv2.COLOR_RGB2BGR))
    cols = 3
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(len(tiles) // cols)])
    path = os.path.join(OUT_DIR, "contact_sheet.png")
    cv2.imwrite(path, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"[preview] {path} in {time.time() - t0:.0f}s")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    main(a.device, a.scale, a.threads)
