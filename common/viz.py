"""Plotting and video helpers for the POGS notebook."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_PALETTE = np.array(
    [[228, 26, 28], [55, 126, 184], [77, 175, 74], [152, 78, 163],
     [255, 127, 0], [255, 255, 51], [166, 86, 40], [247, 129, 191]], dtype=np.uint8
)


def _plt():
    import matplotlib

    if matplotlib.get_backend().lower() not in ("agg", "module://matplotlib_inline.backend_inline"):
        pass
    import matplotlib.pyplot as plt

    return plt


def to_uint8(x) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def depth_to_rgb(d) -> np.ndarray:
    if torch.is_tensor(d):
        d = d.detach().cpu().numpy()
    d = d.squeeze()
    m = d > 0
    out = np.zeros_like(d)
    if m.any():
        out[m] = (d[m] - d[m].min()) / (np.ptp(d[m]) + 1e-9)
    plt = _plt()
    return (plt.get_cmap("turbo")(1.0 - out)[..., :3] * 255).astype(np.uint8)


def labels_to_rgb(labels) -> np.ndarray:
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    out = np.zeros(labels.shape + (3,), dtype=np.uint8)
    valid = labels >= 0
    out[valid] = _PALETTE[labels[valid] % len(_PALETTE)]
    return out


def feature_pca_rgb(feats) -> np.ndarray:
    """First 3 PCA components of a (H, W, C) feature map, for eyeballing DINO."""
    if torch.is_tensor(feats):
        feats = feats.detach().float().cpu()
    else:
        feats = torch.as_tensor(feats).float()
    h, w, c = feats.shape
    flat = feats.reshape(-1, c)
    flat = flat - flat.mean(0, keepdim=True)
    v = torch.pca_lowrank(flat, q=min(3, c))[2][:, :3]
    p = (flat @ v).reshape(h, w, 3)
    lo = p.reshape(-1, 3).quantile(0.02, dim=0)
    hi = p.reshape(-1, 3).quantile(0.98, dim=0)
    return to_uint8(((p - lo) / (hi - lo + 1e-8)).clamp(0, 1).numpy())


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #


def show_capture_grid(capture, n: int = 6, title: str = "Scene capture (Phase 1)"):
    plt = _plt()
    idx = np.linspace(0, capture.n_views() - 1, n).astype(int)
    fig, ax = plt.subplots(3, n, figsize=(2.6 * n, 7.2))
    for c, i in enumerate(idx):
        ax[0, c].imshow(to_uint8(capture.images[i])); ax[0, c].set_title(f"view {i}", fontsize=9)
        ax[1, c].imshow(depth_to_rgb(capture.depths[i]))
        ax[2, c].imshow(labels_to_rgb(capture.obj_ids[i]))
        for r in range(3):
            ax[r, c].axis("off")
    ax[0, 0].set_ylabel("RGB"); ax[1, 0].set_ylabel("depth"); ax[2, 0].set_ylabel("masks")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def show_pointcloud(points, colors=None, labels=None, title="Fused point cloud", elev=28, azim=-60, s=0.6):
    plt = _plt()
    P = points.detach().cpu().numpy() if torch.is_tensor(points) else np.asarray(points)
    if labels is not None:
        lab = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
        C = _PALETTE[np.clip(lab, 0, None) % len(_PALETTE)] / 255.0
        C[lab < 0] = 0.5
    elif colors is not None:
        C = colors.detach().cpu().numpy() if torch.is_tensor(colors) else np.asarray(colors)
    else:
        C = None
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=C, s=s, linewidths=0)
    ax.set_title(f"{title} ({len(P)} pts)")
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 0.55))
    for a, lbl in ((ax.set_xlabel, "x (m)"), (ax.set_ylabel, "y (m)"), (ax.set_zlabel, "z (m)")):
        a(lbl)
    fig.tight_layout()
    return fig


def plot_training(log, title="POGS training"):
    plt = _plt()
    fig, ax = plt.subplots(1, 4, figsize=(18, 3.6))
    ax[0].plot(log.step, log.psnr, "-o", ms=3); ax[0].set_title("PSNR (dB)")
    ax[1].plot(log.step, log.rgb, label="rgb")
    ax[1].plot(log.step, log.depth, label="depth (Pearson)")
    ax[1].plot(log.step, log.dino, label="dino")
    if any(v > 0 for v in getattr(log, "dino_nn", [])):
        ax[1].plot(log.step, log.dino_nn, label="dino NN-variance")
    ax[1].set_yscale("log"); ax[1].legend(fontsize=8); ax[1].set_title("geometry / feature losses")
    ax[2].plot(log.step, log.clip, label="clip"); ax[2].plot(log.step, log.group, label="grouping")
    ax[2].legend(fontsize=8); ax[2].set_title("language / grouping losses")
    ax[3].plot(log.step, log.n_gauss, "-o", ms=3); ax[3].set_title("# gaussians")
    for a in ax:
        a.set_xlabel("step"); a.grid(alpha=0.3)
    fig.suptitle(title); fig.tight_layout()
    return fig


def compare_render(model, capture, view: int, cfg, title="Render vs ground truth"):
    plt = _plt()
    cam = capture.cameras[view].to(cfg.runtime.device)
    rk = dict(max_per_tile=cfg.runtime.torch_raster_max_per_tile,
              tile_chunk=cfg.runtime.torch_raster_tile_chunk) if model.backend == "torch" else {}
    with torch.no_grad():
        out = model.get_outputs(cam, render_dino=True, raster_kwargs=rk)
    fig, ax = plt.subplots(1, 5, figsize=(21, 4))
    ax[0].imshow(to_uint8(capture.images[view])); ax[0].set_title("GT RGB")
    ax[1].imshow(to_uint8(out["rgb"])); ax[1].set_title("3DGS render")
    ax[2].imshow(depth_to_rgb(capture.depths[view])); ax[2].set_title("GT depth")
    ax[3].imshow(depth_to_rgb(out["depth"])); ax[3].set_title("rendered depth")
    ax[4].imshow(feature_pca_rgb(out["dino"])); ax[4].set_title("rendered DINO (PCA)")
    for a in ax:
        a.axis("off")
    fig.suptitle(title); fig.tight_layout()
    return fig


def plot_tracking(results: Dict[str, "object"], thresh_cm: float = 5.0, title="6-DoF tracking error"):
    plt = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(18, 4))
    for name, r in results.items():
        t = np.nanmean(r.trans_err_cm, axis=1)
        rot = np.nanmean(r.rot_err_deg, axis=1)
        ax[0].plot(t, label=name, lw=1.6)
        ax[1].plot(rot, label=name, lw=1.6)
    first = next(iter(results.values()))
    for f in first.reset_frames:
        for a in ax[:2]:
            a.axvline(f, color="k", ls=":", alpha=0.45, lw=1)
    if first.hand_frames:
        for a in ax[:2]:
            a.axvspan(min(first.hand_frames), max(first.hand_frames), color="orange", alpha=0.12)
    ax[0].axhline(thresh_cm, color="r", ls="--", lw=1, label=f"loss threshold ({thresh_cm} cm)")
    ax[0].set_ylabel("translation error (cm)"); ax[1].set_ylabel("rotation error (deg)")
    for a in ax[:2]:
        a.set_xlabel("frame"); a.grid(alpha=0.3); a.legend(fontsize=8)

    names = list(results.keys())
    means = [results[n].mean_position_error_cm() for n in names]
    stds = [results[n].std_position_error_cm() for n in names]
    ax[2].bar(range(len(names)), means, yerr=stds, capsize=4, color="#377eb8")
    ax[2].set_xticks(range(len(names))); ax[2].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax[2].set_ylabel("mean position error (cm)"); ax[2].grid(alpha=0.3, axis="y")
    ax[2].set_title("shaded band = hand occlusion; dotted = object reset", fontsize=9)
    fig.suptitle(title); fig.tight_layout()
    return fig




def plot_grasps(points, gs, n: int = 12, title="Grasp candidates on persistent geometry"):
    plt = _plt()
    from .grasp import gripper_wireframe

    P = np.asarray(points)
    fig = plt.figure(figsize=(7.5, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c="0.65", s=1.2, linewidths=0)
    top = gs.top(n)
    for i in range(len(top)):
        w = gripper_wireframe(top.poses[i])
        col = plt.get_cmap("viridis")(float(top.scores[i]))
        ax.plot(w[:4, 0], w[:4, 1], w[:4, 2], color=col, lw=1.8)
        ax.plot(w[4:, 0], w[4:, 1], w[4:, 2], color=col, lw=1.8)
    ax.set_title(f"{title}\n{len(gs)} candidates, best score {gs.scores.max():.3f}" if len(gs) else title)
    ax.set_box_aspect((1, 1, 0.7))
    fig.tight_layout()
    return fig


def montage(frames: Sequence[np.ndarray], cols: int = 6, title: Optional[str] = None):
    plt = _plt()
    n = len(frames)
    rows = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(2.4 * cols, 1.9 * rows))
    ax = np.atleast_2d(ax)
    for i in range(rows * cols):
        a = ax[i // cols, i % cols]
        a.axis("off")
        if i < n:
            a.imshow(frames[i])
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def save_video(frames: Sequence[np.ndarray], path: str, fps: int = 5) -> Optional[str]:
    try:
        import imageio.v2 as imageio

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        imageio.mimsave(path, list(frames), fps=fps)
        return path
    except Exception as exc:  # noqa: BLE001
        print(f"[viz] video write skipped ({type(exc).__name__}: {exc})")
        return None