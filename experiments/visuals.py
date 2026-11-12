"""
Dashboard videos for any experiment's tracking runs (v1, v2, v3, real-time).

Per video frame (2 x 2 panels + plot):
    top-left      observed camera image, target's 3D box: ground truth (green)
                  and the tracker's estimate (cyan)
    top-right     the tracked object's Gaussians at the estimated pose,
                  blended over the camera image (misalignment is visible)
    bottom-left   RAW SPLAT OUTPUT: the whole POGS scene (every Gaussian, the
                  tracked object moved to its estimated pose), no camera image
    bottom-right  RAW SPLAT OUTPUT: only the tracked object's Gaussians, on black
    bottom        position / rotation error so far with the paper's 3 cm / 5 deg
                  thresholds; ticks mark the frames the tracker actually processed
Speed overlay: per-frame compute time and rate; for real-time runs also the
camera rate, which frame the shown estimate came from, its age, and how many
camera frames were skipped.

Offline runs: one video frame per tracked frame (the tracker never falls behind).
Real-time runs (``Case.realtime``): one video frame per CAMERA frame, played at
camera rate, showing the estimate the tracker had published at that moment.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Sequence

import cv2
import numpy as np
import torch

from model.pogs.config import POGSConfig
from model.pogs.evaluate import SUCCESS_POS_CM, SUCCESS_ROT_DEG, tracking_camera
from model.pogs.geometry import apply_group_deltas, quat_normalize
from model.pogs.raster import rasterize
from model.pogs.grouping import subset_groups
from experiments.pipeline import DATASET_DIR, RUNS_DIR, _case_groups
from data.scene_generator.pick_place_benchmark import Case, load_case

# RGB (frames are RGB until written)
GREEN, CYAN, WHITE, ORANGE, GREY = (60, 220, 60), (0, 220, 255), (255, 255, 255), (255, 160, 40), (150, 150, 150)
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


def _label(img, lines, color=WHITE):
    """Black strip with one or more lines of text at the top of a panel."""
    h = 8 + 18 * len(lines)
    cv2.rectangle(img, (0, 0), (img.shape[1], h), (0, 0, 0), -1)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (8, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color if i == 0 else (215, 215, 215), 1, cv2.LINE_AA)


def _plot(width, height, pos, rot, n_total, ticks=None):
    """Error curves over the sequence; ``ticks`` = frame indices tracked so far."""
    img = np.full((height, width, 3), 24, np.uint8)
    pad, h2 = 40, (height - 30) // 2
    xs_of = lambda i: pad + (np.asarray(i) / max(n_total - 1, 1)) * (width - 10 - pad)
    for row, (vals, thr, unit, color, vmax) in enumerate(
        [(pos, SUCCESS_POS_CM, "cm", CYAN, max(6.0, np.nanmax(pos) * 1.1 if len(pos) else 6.0)),
         (rot, SUCCESS_ROT_DEG, "deg", ORANGE, max(10.0, np.nanmax(rot) * 1.1 if len(rot) else 10.0))]
    ):
        y0 = 15 + row * h2
        cv2.rectangle(img, (pad, y0), (width - 10, y0 + h2 - 10), (70, 70, 70), 1)
        ty = int(y0 + (h2 - 10) * (1 - thr / vmax))
        cv2.line(img, (pad, ty), (width - 10, ty), (200, 0, 0), 1)
        cv2.putText(img, f"{'pos' if row == 0 else 'rot'} err ({unit}) | thr {thr:g}", (pad + 5, y0 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
        if ticks is not None and row == 0:
            for x in xs_of(ticks).astype(int):
                cv2.line(img, (x, y0 + h2 - 16), (x, y0 + h2 - 10), GREY, 1)
        if len(vals) > 1:
            ys = y0 + (h2 - 10) * (1 - np.clip(np.nan_to_num(vals) / vmax, 0, 1))
            cv2.polylines(img, [np.stack([xs_of(np.arange(len(vals))), ys], -1).astype(np.int32)],
                          False, color, 2, cv2.LINE_AA)
        cv2.putText(img, f"{vals[-1]:.2f}" if len(vals) else "", (5, y0 + h2 // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return img


class _SplatRenderer:
    """Renders the trained POGS scene with the tracked groups moved."""

    def __init__(self, model, groups, cfg: POGSConfig, cam):
        gp = model.gauss_params
        self.model, self.groups, self.cfg, self.cam = model, groups, cfg, cam
        self.means = gp["means"].data
        self.quats = quat_normalize(gp["quats"].data)
        self.scales = torch.exp(gp["scales"].data)
        self.opac = torch.sigmoid(gp["opacities"].data).squeeze(-1)
        self.colors = model.colors_sh.data
        self.tracked = groups.group_labels >= 0
        self.labels = groups.group_labels[self.tracked]

    def _raster(self, m, q, sel=None):
        s, o, c = (self.scales, self.opac, self.colors) if sel is None else \
            (self.scales[sel], self.opac[sel], self.colors[sel])
        rt = self.cfg.runtime
        kw = (dict(max_per_tile=rt.torch_raster_max_per_tile, tile_chunk=rt.torch_raster_tile_chunk)
              if self.model.backend == "torch" else {})
        out = rasterize(m, q, s, o, c, self.cam, backend=self.model.backend,
                        sh_degree=self.model.sh_degree_to_use(), render_depth=False, background=None,
                        antialiased=self.cfg.model.rasterize_mode == "antialiased", **kw)
        return out["render"].clamp(0, 1).cpu().numpy(), out["alpha"].clamp(0, 1).cpu().numpy()

    @torch.no_grad()
    def __call__(self, deltas: torch.Tensor):
        """(object-only rgb, object-only alpha, full-scene rgb), float [0, 1]."""
        m_obj, q_obj = apply_group_deltas(self.groups.init_p2w, deltas, self.labels,
                                          self.means[self.tracked], self.quats[self.tracked])
        obj_rgb, obj_a = self._raster(m_obj, q_obj, self.tracked)
        m_all, q_all = self.means.clone(), self.quats.clone()
        m_all[self.tracked], q_all[self.tracked] = m_obj, q_obj
        full_rgb, _ = self._raster(m_all, q_all)
        return obj_rgb, obj_a, full_rgb


@torch.no_grad()
def make_dashboard(model, grouping, cfg: POGSConfig, results_dir: str, case: Case, method: str,
                   dataset_dir: str = DATASET_DIR, runs_dir: str = RUNS_DIR,
                   only_frames: Optional[Sequence[int]] = None) -> Optional[str]:
    """Writes ``<case>__<method>.mp4 / .gif / _sample.png``; with
    ``only_frames`` just those frames as ``_frame<i>.png`` (quick look)."""
    run_dir = os.path.join(runs_dir, case.name, _slug(method))
    if not os.path.exists(os.path.join(run_dir, "telemetry.json")):
        return None
    tele = json.load(open(os.path.join(run_dir, "telemetry.json")))
    deltas = np.load(os.path.join(run_dir, "part_deltas.npy"))
    sched_path = os.path.join(run_dir, "schedule.json")
    sched = json.load(open(sched_path)) if os.path.exists(sched_path) else None
    seq = load_case(dataset_dir, case)
    groups = grouping.groups
    dev = cfg.runtime.device
    cam = tracking_camera(seq.camera, cfg.tracking, dev)
    tg, moving, _ = _case_groups(seq, groups, grouping.group_to_gt)
    if deltas.shape[1] != groups.num_groups:          # run tracked the manipulated object(s) only
        groups, tg = subset_groups(groups, moving), 0
    splats = _SplatRenderer(model, groups, cfg, cam)

    local = (model.gauss_params["means"].data[groups.group_masks[tg]] - groups.init_p2w[tg, :3]).cpu().numpy()
    lo, hi = np.percentile(local, 1, axis=0), np.percentile(local, 99, axis=0)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])

    if sched is not None:
        ts = np.array(sched["timestamps"])
        proc = np.array(sched["proc_frames"])
        lat = np.array(sched["proc_done"]) - np.array(sched["proc_start"])
        src = np.array(sched["cam_src"])
        cam_hz = 1.0 / float(np.median(np.diff(ts)))
        play_fps = cam_hz
    else:
        ts = np.array([f.timestamp_sec for f in seq.frames])
        ts = ts - ts[0]
        play_fps = seq.fps

    frames = []
    n = min(len(tele), len(seq.frames), len(deltas))
    pos_all = np.array([t["objects"][tg]["trans_err_cm"] for t in tele[:n]])
    rot_all = np.array([t["objects"][tg]["rot_err_deg"] for t in tele[:n]])
    for fi in (range(n) if only_frames is None else [f for f in only_frames if f < n]):
        fr = seq.frames[fi]
        obs = cv2.resize((fr.rgb.numpy() * 255).astype(np.uint8), (cam.width, cam.height))
        o = tele[fi]["objects"][tg]

        if sched is not None:
            k = src[fi]
            done_so_far = proc[: int(np.searchsorted(np.array(sched["proc_done"]), ts[fi], side="right"))]
            if k >= 0:
                age = (ts[fi] - ts[proc[k]]) * 1000
                skipped = int(proc[k] - proc[k - 1] - 1) if k > 0 else 0
                speed = [f"camera {cam_hz:.0f} Hz | tracker {1.0 / lat[k]:.1f} Hz ({lat[k] * 1000:.0f} ms/frame)",
                         f"showing estimate of frame {proc[k]} | age {age:.0f} ms | skipped {skipped} frames"]
            else:
                speed = [f"camera {cam_hz:.0f} Hz | tracker busy on its first frame",
                         "showing the initial pose"]
            ticks = done_so_far
            head = f"{case.name} | t={ts[fi]:5.2f}s | frame {fi}/{len(seq.frames)} | {fr.phase}"
        else:
            ms = tele[fi].get("frame_ms", float("nan"))
            feed = (f"camera {seq.fps * case.speed:g} Hz, every {case.speed}th frame used" if case.speed > 1
                    else f"camera {seq.fps:g} Hz")
            speed = [f"compute {ms:.0f} ms/frame ({1000.0 / max(ms, 1e-6):.1f} Hz) | {feed}",
                     "offline: every frame is tracked, however long it takes"]
            ticks = np.arange(fi + 1)
            head = f"{case.name} | t={ts[fi]:5.2f}s | frame {fi}/{len(seq.frames)} | {fr.phase}"

        left = obs.copy()
        _draw_box(left, corners, np.array(o["gt_pose"]), cam, GREEN)
        _draw_box(left, corners, np.array(o["pred_pose"]), cam, CYAN)
        obj_rgb, obj_a, full_rgb = splats(torch.from_numpy(deltas[fi]).to(dev))
        right = (0.35 * obs + 0.65 * (obj_a * obj_rgb * 255 + (1 - obj_a) * obs)).astype(np.uint8)
        full = (full_rgb * 255).astype(np.uint8)
        alone = (obj_rgb * 255).astype(np.uint8)

        _label(left, [head, f"occlusion {fr.occlusion_frac:.0%} | green = ground truth, cyan = estimate"])
        _label(right, [f"{method} | object splats at the estimate"] + speed, CYAN)
        _label(full, ["RAW SPLATS: whole POGS scene (Gaussians only)",
                      "tracked object moved to the estimate; the rest stays at scan pose"])
        _label(alone, ["RAW SPLATS: tracked object only"])
        grid = np.vstack([np.hstack([left, right]), np.hstack([full, alone])])
        plot = _plot(grid.shape[1], 220, pos_all[: fi + 1], rot_all[: fi + 1], len(seq.frames), ticks)
        frames.append(np.vstack([grid, plot]))
    frames = [f[: f.shape[0] // 2 * 2, : f.shape[1] // 2 * 2] for f in frames]

    vdir = os.path.join(results_dir, "videos")
    os.makedirs(vdir, exist_ok=True)
    base = os.path.join(vdir, f"{case.name}__{_slug(method)}")
    if only_frames is not None:
        for fi, f in zip([f for f in only_frames if f < n], frames):
            cv2.imwrite(f"{base}_frame{fi}.png", cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        return f"{base}_frame{only_frames[0]}.png" if frames else None
    import imageio.v2 as imageio

    imageio.mimsave(base + ".mp4", frames, fps=play_fps, macro_block_size=1)
    step = max(1, int(round(play_fps / 5)))           # GIF at ~5 fps, half size
    small = [cv2.resize(f, (f.shape[1] // 2, f.shape[0] // 2)) for f in frames[::step]]
    imageio.mimsave(base + ".gif", small, duration=step / play_fps, loop=0)
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
