"""
Dashboard videos for any experiment's tracking runs (v1, v2, v3).

Per frame:
    left    observed RGB, target's 3D box: ground truth (green) / prediction (cyan)
    right   the tracked objects' Gaussians rendered at the PREDICTED poses,
            blended over the observation (misalignment is directly visible)
    bottom  target position / rotation error so far, with the paper's
            3 cm / 5 deg success thresholds
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence

import cv2
import numpy as np
import torch

from common.config import POGSConfig
from common.evaluate import SUCCESS_POS_CM, SUCCESS_ROT_DEG, tracking_camera
from common.geometry import apply_group_deltas, quat_normalize
from common.raster import rasterize
from common.grouping import subset_groups
from experiments.pipeline import DATASET_DIR, RUNS_DIR, _case_groups
from scene_generator.pick_place_benchmark import Case, load_case

GREEN, CYAN, WHITE = (60, 220, 60), (255, 220, 0), (255, 255, 255)
_EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s).strip("_").lower()


def _project(pts_w: np.ndarray, cam) -> np.ndarray:
    w2c = cam.w2c.cpu().numpy()
    pc = pts_w @ w2c[:3, :3].T + w2c[:3, 3]
    z = np.clip(pc[:, 2], 1e-4, None)
    return np.stack([cam.fx * pc[:, 0] / z + cam.cx, cam.fy * pc[:, 1] / z + cam.cy], -1)


def _draw_box(img, corners_local, T, cam, color):
    pw = corners_local @ T[:3, :3].T + T[:3, 3]
    uv = _project(pw, cam).astype(int)
    for a, b in _EDGES:
        cv2.line(img, tuple(uv[a]), tuple(uv[b]), color, 2, cv2.LINE_AA)


def _plot(width, height, pos, rot, n_total):
    img = np.full((height, width, 3), 24, np.uint8)
    pad, h2 = 40, (height - 30) // 2
    for row, (vals, thr, unit, color, vmax) in enumerate(
        [(pos, SUCCESS_POS_CM, "cm", CYAN, max(6.0, np.nanmax(pos) * 1.1 if len(pos) else 6.0)),
         (rot, SUCCESS_ROT_DEG, "deg", (80, 160, 255), max(10.0, np.nanmax(rot) * 1.1 if len(rot) else 10.0))]
    ):
        y0 = 15 + row * h2
        cv2.rectangle(img, (pad, y0), (width - 10, y0 + h2 - 10), (70, 70, 70), 1)
        ty = int(y0 + (h2 - 10) * (1 - thr / vmax))
        cv2.line(img, (pad, ty), (width - 10, ty), (0, 0, 200), 1)
        cv2.putText(img, f"{'pos' if row == 0 else 'rot'} err ({unit}) | thr {thr:g}", (pad + 5, y0 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
        if len(vals) > 1:
            xs = pad + (np.arange(len(vals)) / max(n_total - 1, 1)) * (width - 10 - pad)
            ys = y0 + (h2 - 10) * (1 - np.clip(np.nan_to_num(vals) / vmax, 0, 1))
            cv2.polylines(img, [np.stack([xs, ys], -1).astype(np.int32)], False, color, 2, cv2.LINE_AA)
        cv2.putText(img, f"{vals[-1]:.2f}" if len(vals) else "", (5, y0 + h2 // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return img


@torch.no_grad()
def make_dashboard(model, grouping, cfg: POGSConfig, results_dir: str, case: Case, method: str,
                   dataset_dir: str = DATASET_DIR, runs_dir: str = RUNS_DIR) -> Optional[str]:
    run_dir = os.path.join(runs_dir, case.name, _slug(method))
    if not os.path.exists(os.path.join(run_dir, "telemetry.json")):
        return None
    tele = json.load(open(os.path.join(run_dir, "telemetry.json")))
    deltas = np.load(os.path.join(run_dir, "part_deltas.npy"))
    seq = load_case(dataset_dir, case)
    seq_name = case.name
    groups = grouping.groups
    dev = cfg.runtime.device
    cam = tracking_camera(seq.camera, cfg.tracking, dev)
    tg, moving, _ = _case_groups(seq, groups, grouping.group_to_gt)
    if deltas.shape[1] != groups.num_groups:          # run tracked the manipulated object(s) only
        groups, tg = subset_groups(groups, moving), 0

    gp = model.gauss_params
    tracked = groups.group_labels >= 0
    means0, quats0 = gp["means"].data[tracked], quat_normalize(gp["quats"].data[tracked])
    labels = groups.group_labels[tracked]
    scales = torch.exp(gp["scales"].data[tracked])
    opac = torch.sigmoid(gp["opacities"].data[tracked]).squeeze(-1)
    colors = model.colors_sh.data[tracked]
    local = (gp["means"].data[groups.group_masks[tg]] - groups.init_p2w[tg, :3]).cpu().numpy()
    lo, hi = np.percentile(local, 1, axis=0), np.percentile(local, 99, axis=0)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])

    frames, pos_hist, rot_hist = [], [], []
    n = min(len(tele), len(seq.frames), len(deltas))
    for fi, fr in enumerate(seq.frames[:n]):
        obs = cv2.resize((fr.rgb.numpy() * 255).astype(np.uint8), (cam.width, cam.height))
        o = tele[fi]["objects"][tg]
        pos_hist.append(o["trans_err_cm"])
        rot_hist.append(o["rot_err_deg"])
        left = obs.copy()
        _draw_box(left, corners, np.array(o["gt_pose"]), cam, GREEN)
        _draw_box(left, corners, np.array(o["pred_pose"]), cam, CYAN)
        m, q = apply_group_deltas(groups.init_p2w, torch.from_numpy(deltas[fi]).to(dev), labels, means0, quats0)
        out = rasterize(m, q, scales, opac, colors, cam, backend=model.backend,
                        sh_degree=model.sh_degree_to_use(), render_depth=False, background=None,
                        antialiased=cfg.model.rasterize_mode == "antialiased")
        a = out["alpha"].clamp(0, 1).cpu().numpy()
        rend = (out["render"].clamp(0, 1).cpu().numpy() * 255)
        right = (0.35 * obs + 0.65 * (a * rend + (1 - a) * obs)).astype(np.uint8)
        for img, title in ((left, f"{seq_name} | frame {fi:02d} | {fr.phase} | occl {fr.occlusion_frac:.0%}"),
                           (right, f"{method} | splats at predicted pose")):
            cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
            cv2.putText(img, title, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        top = np.hstack([left, right])
        frames.append(np.vstack([top, _plot(top.shape[1], 220, np.array(pos_hist), np.array(rot_hist), len(seq.frames))]))
    frames = [f[: f.shape[0] // 2 * 2, : f.shape[1] // 2 * 2] for f in frames]

    vdir = os.path.join(results_dir, "videos")
    os.makedirs(vdir, exist_ok=True)
    base = os.path.join(vdir, f"{seq_name}__{_slug(method)}")
    import imageio.v2 as imageio

    imageio.mimsave(base + ".mp4", frames, fps=seq.fps)
    small = [cv2.resize(f, (f.shape[1] // 2, f.shape[0] // 2)) for f in frames[::2]]
    imageio.mimsave(base + ".gif", small, duration=2.0 / seq.fps, loop=0)
    cv2.imwrite(base + "_sample.png", cv2.cvtColor(frames[len(frames) // 2], cv2.COLOR_RGB2BGR))
    return base + ".mp4"


def make_dashboards(model, grouping, cfg: POGSConfig, results_dir: str, cases: Sequence[Case],
                    methods: Sequence[str], seeds: Sequence[int] = (0,)) -> List[str]:
    """One video per (case, method) for the given seeds only."""
    out = []
    for c in cases:
        if c.seed not in seeds:
            continue
        for m in methods:
            p = make_dashboard(model, grouping, cfg, results_dir, c, m)
            if p:
                out.append(p)
                print(f"[video] {p}")
    return out
