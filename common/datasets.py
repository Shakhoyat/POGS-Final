"""
Loaders for **real** data, so POGS-Sim can be swapped out without touching any
other module.

Two entry points, matching the two phases:

``load_capture_folder``   Phase 1 -- a nerfstudio-style scan directory
                          (``transforms.json`` + images + depth), which is what
                          ``ns-train pogs --data ...`` consumes.

``load_tracking_folder``  Phase 3 -- the exact layout the official offline demo
                          reads in ``pogs/scripts/track_main_demo.py``::

                              offline_folder/
                                  left/    0000.png 0001.png ...
                                  depth/   0000.npy 0001.npy ...   (metres)

Recommended real datasets, and why (see the notebook's discussion section):

* **BOP-H3 model-free track** (BOP Challenge 2024) -- HOT3D, HOPEv2, HANDAL.
  The challenge's *static onboarding* videos are structurally identical to POGS's
  hemisphere scan, and the test videos are the manipulation phase. Closest match
  by construction.  https://bop.felk.cvut.cz/
* **DexYCB / HO-3D** -- humans manipulating YCB objects with per-frame 6-DoF
  ground truth; matches the paper's human-perturbation setting and its occlusion
  limitation.
* **YCB-Video (BOP core)** -- the standard RGB-D tracking reference; the scan
  phase has to be synthesised from the provided meshes.

Ground-truth poses are optional: without them the tracker still runs, you just
cannot score it.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .geometry import PinholeCamera
from .simscene import CaptureData, TrackingFrame, TrackingSequence, _dbscan_filter, deproject


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _read_image(path: str) -> np.ndarray:
    try:
        from PIL import Image

        return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    except ImportError:
        import imageio.v2 as imageio

        return imageio.imread(path)[..., :3].astype(np.float32) / 255.0


def _read_depth(path: str, scale: float = 1.0) -> np.ndarray:
    if path.endswith(".npy"):
        d = np.load(path).astype(np.float32)
    else:
        try:
            from PIL import Image

            d = np.asarray(Image.open(path)).astype(np.float32)
        except ImportError:
            import imageio.v2 as imageio

            d = imageio.imread(path).astype(np.float32)
    return d * scale


def _sorted_files(folder: str, exts: Sequence[str]) -> List[str]:
    return sorted(
        os.path.join(folder, f) for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in exts
    )


# --------------------------------------------------------------------------- #
# Phase 1
# --------------------------------------------------------------------------- #


def load_capture_folder(
    root: str,
    device: str = "cuda:0",
    depth_scale: float = 1.0,
    max_seed_points: int = 60_000,
    dbscan_eps: float = 0.012,
    dbscan_min_samples: int = 12,
    mask_dir: Optional[str] = "masks",
    downscale: float = 1.0,
    opengl_poses: bool = True,
    verbose: bool = True,
) -> CaptureData:
    """Read a nerfstudio-format scan directory into :class:`CaptureData`.

    ``transforms.json`` must provide ``fl_x, fl_y, cx, cy, w, h`` (or per-frame
    equivalents) and one ``transform_matrix`` per frame.  ``opengl_poses=True``
    treats those matrices as the nerfstudio/OpenGL convention (x right, y up,
    z backward) and converts them; set ``False`` if yours are already OpenCV.

    ``mask_dir`` supplies the 2D instance masks that Detic provides in the paper.
    Without it the grouping loss has nothing to learn from, so ``obj_ids`` comes
    back all -1 and you must plug in a segmenter before training.
    """
    with open(os.path.join(root, "transforms.json")) as f:
        meta = json.load(f)

    frames = meta["frames"]
    imgs, deps, oids, cams = [], [], [], []
    for i, fr in enumerate(frames):
        img_path = os.path.join(root, fr["file_path"])
        if not os.path.exists(img_path):
            for ext in (".png", ".jpg", ".jpeg"):
                if os.path.exists(img_path + ext):
                    img_path = img_path + ext
                    break
        rgb = _read_image(img_path)
        H, W = rgb.shape[:2]

        fx = fr.get("fl_x", meta.get("fl_x"))
        fy = fr.get("fl_y", meta.get("fl_y", fx))
        cx = fr.get("cx", meta.get("cx", W / 2))
        cy = fr.get("cy", meta.get("cy", H / 2))

        c2w = torch.as_tensor(np.asarray(fr["transform_matrix"], dtype=np.float32))
        if c2w.shape == (3, 4):
            c2w = torch.cat([c2w, torch.tensor([[0.0, 0, 0, 1]])], 0)
        cam = (PinholeCamera.from_nerfstudio(c2w, fx, fy, cx, cy, W, H) if opengl_poses
               else PinholeCamera(c2w, fx, fy, cx, cy, W, H))

        dpath = fr.get("depth_file_path")
        if dpath is None:
            base = os.path.splitext(os.path.basename(img_path))[0]
            for cand in (f"depth/{base}.npy", f"depths/{base}.npy", f"depth/{base}.png"):
                if os.path.exists(os.path.join(root, cand)):
                    dpath = cand
                    break
        depth = (torch.from_numpy(_read_depth(os.path.join(root, dpath), depth_scale))
                 if dpath else torch.zeros(H, W))

        mask = torch.full((H, W), -1, dtype=torch.long)
        if mask_dir:
            base = os.path.splitext(os.path.basename(img_path))[0]
            for cand in (f"{mask_dir}/{base}.npy", f"{mask_dir}/{base}.png"):
                mp = os.path.join(root, cand)
                if os.path.exists(mp):
                    m = np.load(mp) if cand.endswith(".npy") else _read_image(mp)[..., 0] * 255
                    mask = torch.from_numpy(np.asarray(m).astype(np.int64))
                    break

        t_rgb = torch.from_numpy(rgb)
        if downscale != 1.0:
            import torch.nn.functional as Fnn

            newH, newW = int(H * downscale), int(W * downscale)
            t_rgb = Fnn.interpolate(t_rgb.permute(2, 0, 1)[None], size=(newH, newW),
                                    mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
            depth = Fnn.interpolate(depth[None, None], size=(newH, newW), mode="nearest")[0, 0]
            mask = Fnn.interpolate(mask[None, None].float(), size=(newH, newW),
                                   mode="nearest")[0, 0].long()
            cam = cam.rescale(downscale)

        imgs.append(t_rgb); deps.append(depth); oids.append(mask); cams.append(cam.to(device))
        if verbose and (i % 10 == 0 or i == len(frames) - 1):
            print(f"  loaded view {i + 1}/{len(frames)}")

    images = torch.stack(imgs)
    depths = torch.stack(deps)
    obj_ids = torch.stack(oids)

    # fuse + DBSCAN, exactly as in the simulated capture
    pts, cols, ids = [], [], []
    stride = max(1, int(math.sqrt(images[0, ..., 0].numel() * len(cams) / max(max_seed_points * 4, 1))))
    for v, cam in enumerate(cams):
        d = depths[v].to(device)
        if float(d.max()) <= 0:
            continue
        p = deproject(d, cam)
        m = d > 0.05
        pts.append(p[m][::stride].cpu())
        cols.append(images[v].to(device)[m][::stride].cpu())
        ids.append(obj_ids[v].to(device)[m][::stride].cpu())

    if not pts:
        raise ValueError(
            "no valid depth found -- POGS needs metric depth to initialise the splat. "
            "Check depth_file_path / depth_scale."
        )
    P = torch.cat(pts).numpy(); C = torch.cat(cols).numpy(); I = torch.cat(ids).numpy()
    rng = np.random.default_rng(0)
    if len(P) > max_seed_points:
        sel = rng.choice(len(P), max_seed_points, replace=False)
        P, C, I = P[sel], C[sel], I[sel]
    keep = _dbscan_filter(P, dbscan_eps, dbscan_min_samples)
    if verbose:
        print(f"  fused {len(P)} pts -> DBSCAN kept {int(keep.sum())}")
    P, C, I = P[keep], C[keep], I[keep]

    names = ["background"] + [f"object_{i}" for i in range(1, int(max(I.max(), 0)) + 1)]
    return CaptureData(
        images=images, depths=depths, obj_ids=obj_ids, cameras=cams,
        seed_points=torch.from_numpy(P).float(),
        seed_colors=torch.from_numpy(C).float(),
        seed_obj_ids=torch.from_numpy(I).long(),
        object_names=names, scene=None,
    )


# --------------------------------------------------------------------------- #
# Phase 3
# --------------------------------------------------------------------------- #


def load_tracking_folder(
    root: str,
    camera: PinholeCamera,
    depth_scale: float = 1.0,
    max_frames: Optional[int] = None,
    gt_poses: Optional[Dict[int, np.ndarray]] = None,
    reset_frames: Sequence[int] = (),
    image_dir: str = "left",
    depth_dir: str = "depth",
    verbose: bool = True,
) -> TrackingSequence:
    """Read the official offline-demo layout into a :class:`TrackingSequence`.

    ``gt_poses`` maps object id -> ``(F, 4, 4)`` world-frame deltas relative to
    the scan pose; supply it to get quantitative metrics, omit it to just track.
    """
    img_files = _sorted_files(os.path.join(root, image_dir), (".png", ".jpg", ".jpeg"))
    dep_files = _sorted_files(os.path.join(root, depth_dir), (".npy", ".png"))
    if max_frames:
        img_files, dep_files = img_files[:max_frames], dep_files[:max_frames]
    if len(img_files) != len(dep_files):
        raise ValueError(f"{len(img_files)} images but {len(dep_files)} depth maps in {root}")

    frames: List[TrackingFrame] = []
    for i, (ip, dp) in enumerate(zip(img_files, dep_files)):
        rgb = torch.from_numpy(_read_image(ip))
        depth = torch.from_numpy(_read_depth(dp, depth_scale))
        delta = {} if gt_poses is None else {k: v[i] for k, v in gt_poses.items() if i < len(v)}
        frames.append(
            TrackingFrame(rgb=rgb, depth=depth,
                          obj_id=torch.full(rgb.shape[:2], -1, dtype=torch.long),
                          gt_delta=delta, hand_visible=False, reset_event=i in set(reset_frames))
        )
        if verbose and (i % 20 == 0 or i == len(img_files) - 1):
            print(f"  loaded frame {i + 1}/{len(img_files)}")

    ids = sorted(gt_poses.keys()) if gt_poses else []
    return TrackingSequence(frames=frames, camera=camera, object_ids=list(ids))


# --------------------------------------------------------------------------- #
# BOP convenience
# --------------------------------------------------------------------------- #


def bop_scene_to_capture(
    scene_dir: str,
    device: str = "cuda:0",
    depth_scale_mm_to_m: float = 1e-3,
    max_seed_points: int = 60_000,
    max_views: Optional[int] = None,
    verbose: bool = True,
) -> CaptureData:
    """Read one BOP scene directory (``rgb/``, ``depth/``, ``scene_camera.json``)
    as a POGS capture.

    Use this on a BOP-H3 **static onboarding** video: the object is still and the
    camera orbits, which is exactly the paper's scan phase. ``mask_visib/`` (when
    present) supplies the 2D instance masks the grouping loss needs.
    """
    with open(os.path.join(scene_dir, "scene_camera.json")) as f:
        scene_camera = json.load(f)

    keys = sorted(scene_camera.keys(), key=int)
    if max_views:
        keys = keys[:: max(len(keys) // max_views, 1)][:max_views]

    imgs, deps, oids, cams = [], [], [], []
    mask_dir = os.path.join(scene_dir, "mask_visib")
    has_masks = os.path.isdir(mask_dir)

    for k in keys:
        info = scene_camera[k]
        K = np.asarray(info["cam_K"], dtype=np.float64).reshape(3, 3)
        R = np.asarray(info["cam_R_w2c"], dtype=np.float64).reshape(3, 3)
        t = np.asarray(info["cam_t_w2c"], dtype=np.float64) * depth_scale_mm_to_m
        w2c = np.eye(4); w2c[:3, :3] = R; w2c[:3, 3] = t
        c2w = torch.as_tensor(np.linalg.inv(w2c), dtype=torch.float32)

        rgb_p = os.path.join(scene_dir, "rgb", f"{int(k):06d}.png")
        if not os.path.exists(rgb_p):
            rgb_p = os.path.join(scene_dir, "rgb", f"{int(k):06d}.jpg")
        rgb = _read_image(rgb_p)
        H, W = rgb.shape[:2]
        d = _read_depth(os.path.join(scene_dir, "depth", f"{int(k):06d}.png")) \
            * float(info.get("depth_scale", 1.0)) * depth_scale_mm_to_m

        mask = np.full((H, W), -1, dtype=np.int64)
        if has_masks:
            for oi in range(20):
                mp = os.path.join(mask_dir, f"{int(k):06d}_{oi:06d}.png")
                if not os.path.exists(mp):
                    break
                mm = _read_image(mp)[..., 0] > 0.5
                mask[mm] = oi + 1

        cams.append(PinholeCamera(c2w, K[0, 0], K[1, 1], K[0, 2], K[1, 2], W, H).to(device))
        imgs.append(torch.from_numpy(rgb))
        deps.append(torch.from_numpy(d.astype(np.float32)))
        oids.append(torch.from_numpy(mask))

    images = torch.stack(imgs); depths = torch.stack(deps); obj_ids = torch.stack(oids)
    pts, cols, ids = [], [], []
    stride = max(1, int(math.sqrt(images[0, ..., 0].numel() * len(cams) / max(max_seed_points * 4, 1))))
    for v, cam in enumerate(cams):
        dd = depths[v].to(device)
        p = deproject(dd, cam)
        m = dd > 0.05
        pts.append(p[m][::stride].cpu())
        cols.append(images[v].to(device)[m][::stride].cpu())
        ids.append(obj_ids[v].to(device)[m][::stride].cpu())

    P = torch.cat(pts).numpy(); C = torch.cat(cols).numpy(); I = torch.cat(ids).numpy()
    rng = np.random.default_rng(0)
    if len(P) > max_seed_points:
        sel = rng.choice(len(P), max_seed_points, replace=False)
        P, C, I = P[sel], C[sel], I[sel]
    keep = _dbscan_filter(P, 0.004, 12)
    if verbose:
        print(f"  BOP scene: {len(cams)} views, fused {len(P)} pts -> kept {int(keep.sum())}")
    P, C, I = P[keep], C[keep], I[keep]

    return CaptureData(
        images=images, depths=depths, obj_ids=obj_ids, cameras=cams,
        seed_points=torch.from_numpy(P).float(),
        seed_colors=torch.from_numpy(C).float(),
        seed_obj_ids=torch.from_numpy(I).long(),
        object_names=["background"] + [f"obj_{i}" for i in range(1, int(max(I.max(), 0)) + 1)],
        scene=None,
    )