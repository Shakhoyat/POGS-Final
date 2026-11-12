"""
Shared experiment pipeline: train -> group -> track -> report.

v1 (base POGS), v2 (Kalman motion prior) and v3 (second target object) are thin
configurations of these functions, so they differ ONLY in what their configs
say -- never in a copy-pasted loop.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from model.pogs.checkpoint import checkpoint_exists, load_checkpoint, save_checkpoint, write_json
from model.pogs.config import POGSConfig
from model.pogs.encoders import DinoExtractor, OpenCLIPNetwork
from model.pogs.evaluate import (
    TrackingResult, adds_metric, adds_success_rate, object_diameter, object_model_points,
    static_baseline, track_realtime, track_sequence, variant_config,
)
from model.pogs.grouping import (
    ObjectGroups, assign_gt_ids, build_object_groups, cluster_purity, cluster_scene,
    group_gt_iou, match_clusters_to_gt, select_object_clusters, subset_groups,
)
from model.pogs.model import POGSModel
from model.pogs.trainer import evaluate_views, precompute_features, train_pogs
from data.scene_generator.pick_place_benchmark import (
    FACTORS, OBJECT_NAMES, OBJECT_QUERIES, REALTIME_FPS, SCENE_VERSION, Case, dataset_scene_version, load_capture, load_case,
)

DATASET_DIR = "datasets/synthetic_scenes/pick_place_floor_table"
SCENE_CKPT_DIR = "model_checkpoints/pick_place_floor_table"
RUNS_DIR = "data_results/benchmark_runs"


# --------------------------------------------------------------------------- #
# Encoders
# --------------------------------------------------------------------------- #


def load_encoders(cfg: POGSConfig, with_clip: bool = True) -> Tuple[DinoExtractor, Optional[OpenCLIPNetwork]]:
    dev = cfg.runtime.device
    dino = DinoExtractor(device=dev, pca_dim=cfg.model.gaussian_dim, max_size=cfg.runtime.dino_max_size,
                         model_type=cfg.encoders.dino_model)
    if not dino.is_pretrained:
        raise RuntimeError("DINOv2 weights could not be loaded; results would be meaningless")
    clip = None
    if with_clip:
        clip = OpenCLIPNetwork(device=dev, model_type=cfg.encoders.clip_model,
                               pretrained=cfg.encoders.clip_pretrained, clip_n_dims=cfg.model.clip_n_dims)
        if not clip.is_pretrained:
            raise RuntimeError("OpenCLIP weights could not be loaded")
    return dino, clip


# --------------------------------------------------------------------------- #
# Phase 1+2: train (or load) the persistent representation
# --------------------------------------------------------------------------- #


def train_or_load(
    cfg: POGSConfig,
    dataset_dir: str = DATASET_DIR,
    ckpt_dir: str = SCENE_CKPT_DIR,
    force_train: bool = False,
    verbose: bool = True,
):
    """Returns ``(model, capture, dino, clip)`` with ``dino``'s PCA basis set to
    the one the model was distilled with."""
    capture = load_capture(dataset_dir, device=cfg.runtime.device)
    dino, clip = load_encoders(cfg)
    version = dataset_scene_version(dataset_dir)

    if checkpoint_exists(ckpt_dir) and not force_train:
        model, pca, ck = load_checkpoint(ckpt_dir, cfg, device=cfg.runtime.device)
        ck_version = ck.get("extra", {}).get("scene_version", 1)
        if ck_version == version:
            dino.set_pca(pca)
            if verbose:
                print(f"[train] loaded checkpoint from {ckpt_dir}: {model.num_points} gaussians, step {model.step}")
            return model, capture, dino, clip
        print(f"[train] checkpoint is for scene v{ck_version}, data is v{version}: retraining")
        labels = os.path.join(ckpt_dir, "cluster_labels.npy")
        if os.path.exists(labels):
            os.remove(labels)

    if verbose:
        print(f"[train] {len(capture.cameras)} views, {len(capture.seed_points)} seed points")
    features = precompute_features(capture, cfg, dino, clip, verbose=verbose)
    model = POGSModel(capture.seed_points, capture.seed_colors, config=cfg.model,
                      device=cfg.runtime.device, raster_backend=cfg.runtime.raster_backend)
    if verbose:
        print(f"[train] rasteriser backend: {model.backend}")
    t0 = time.time()
    log = train_pogs(model, capture, features, cfg, verbose=verbose)
    train_time = time.time() - t0
    quality = evaluate_views(model, capture, cfg)
    save_checkpoint(ckpt_dir, model, features.dino_pca, cfg,
                    extra={"train_time_s": train_time, "scene_version": version, **quality})
    labels = os.path.join(ckpt_dir, "cluster_labels.npy")
    if os.path.exists(labels):                       # clusters of the previous model
        os.remove(labels)
    write_json(os.path.join(ckpt_dir, "train_summary.json"), {
        "n_gaussians": model.num_points, "steps": cfg.train.max_num_iterations,
        "train_time_s": train_time, "backend": model.backend, **quality, "log": log.to_dict(),
    })
    if verbose:
        print(f"[train] saved to {ckpt_dir}; train-view PSNR {quality['psnr_mean']:.2f} dB, "
              f"depth MAE {quality['depth_mae_m'] * 100:.2f} cm")
    return model, capture, dino, clip


# --------------------------------------------------------------------------- #
# Grouping (Sec. IV-B / IV-C)
# --------------------------------------------------------------------------- #


@dataclass
class GroupingResult:
    groups: ObjectGroups
    group_to_gt: Dict[int, int]
    report: Dict[str, object]


def build_groups(
    model: POGSModel,
    capture,
    cfg: POGSConfig,
    ckpt_dir: str = SCENE_CKPT_DIR,
    clip: Optional[OpenCLIPNetwork] = None,
    verbose: bool = True,
) -> GroupingResult:
    """HDBSCAN on the distilled grouping features, then object selection.
    Ground truth is used ONLY to score the result (purity, IoU, matching)."""
    labels_path = os.path.join(ckpt_dir, "cluster_labels.npy")
    if os.path.exists(labels_path) and np.load(labels_path).shape[0] == model.num_points:
        labels = torch.from_numpy(np.load(labels_path)).long().to(model.gauss_params["means"].device)
        if verbose:
            print(f"[group] loaded cluster labels from {labels_path}")
    else:
        labels = cluster_scene(model, cfg.grouping, verbose=verbose)
        np.save(labels_path, labels.cpu().numpy())

    queries = [OBJECT_QUERIES[k] for k in sorted(OBJECT_QUERIES)]
    keep, labels_sel = select_object_clusters(model, labels, cfg.grouping, clip=clip, queries=queries, verbose=verbose)
    groups = build_object_groups(model, labels_sel, keep_clusters=keep)

    gt = assign_gt_ids(model, capture.seed_points, capture.seed_obj_ids)
    group_to_gt = match_clusters_to_gt(groups, gt)
    groups.names = [OBJECT_NAMES.get(group_to_gt[g], f"cluster_{groups.cluster_ids[g]}") for g in range(groups.num_groups)]
    iou = group_gt_iou(groups, gt, group_to_gt)
    report = {
        "n_clusters": int(labels.max()) + 1,
        "tracked_clusters": keep,
        "purity_all_clusters": cluster_purity(labels, gt),
        "groups": [
            {"group": g, "cluster": groups.cluster_ids[g], "gt_object": group_to_gt[g],
             "name": groups.names[g], "n_gaussians": int(groups.group_masks[g].sum()), "iou_vs_gt": iou[g]}
            for g in range(groups.num_groups)
        ],
        "gt_objects_missing": sorted(set(OBJECT_NAMES) - set(group_to_gt.values())),
        "duplicate_matches": sorted({v for v in group_to_gt.values() if list(group_to_gt.values()).count(v) > 1}),
    }
    write_json(os.path.join(ckpt_dir, "object_clusters.json"), report)
    if verbose:
        for gr in report["groups"]:
            print(f"[group] group {gr['group']}: cluster {gr['cluster']} -> {gr['name']} "
                  f"({gr['n_gaussians']} gaussians, IoU vs GT {gr['iou_vs_gt']:.2f})")
        if report["gt_objects_missing"]:
            print(f"[group] WARNING: no tracked group for GT objects {report['gt_objects_missing']}")
    return GroupingResult(groups, group_to_gt, report)


# --------------------------------------------------------------------------- #
# Phase 3: tracking suite
# --------------------------------------------------------------------------- #


@dataclass
class MethodSpec:
    name: str
    variant: str = "POGS"
    """Passed to ``evaluate.variant_config``."""
    motion_prior: Optional[Callable[[int, float], object]] = None
    """Factory ``(K, dt) -> MotionPrior`` (v2); ``dt`` is the time between
    tracked frames, which changes with the speed factor."""


def run_suite(
    model: POGSModel,
    grouping: GroupingResult,
    dino: DinoExtractor,
    cfg: POGSConfig,
    cases: Sequence[Case],
    methods: Sequence[MethodSpec],
    out_dir: str,
    dataset_dir: str = DATASET_DIR,
    runs_dir: str = RUNS_DIR,
    include_static_baseline: bool = True,
    skip_existing: bool = True,
    track_all: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Runs every method on every benchmark case; writes per-run telemetry,
    ``summary.csv`` (one row per case x method) and the factor tables.
    Runs live in ``runs_dir/<case>/<method>`` shared by every experiment, so a
    method run once (e.g. base POGS by v1) is reused by the others; tables go
    to ``out_dir``. With ``skip_existing`` a finished run is read back instead
    of re-tracked, so an interrupted suite resumes where it stopped.

    Protocol: only the object(s) being manipulated are tracked, as when the
    operator clicks the object to follow (Sec. IV-C); ``track_all`` tracks
    every selected object instead and reports the drift of the static ones."""
    os.makedirs(out_dir, exist_ok=True)
    all_groups, all_g2gt = grouping.groups, grouping.group_to_gt
    groups, g2gt = all_groups, all_g2gt
    mp_all = mp = object_model_points(all_groups, model)
    rows = []
    for ci, case in enumerate(cases):
        seq = None
        names = (["Static (no motion)"] if include_static_baseline else []) + [m.name for m in methods]
        for name in names:
            run_dir = os.path.join(runs_dir, case.name, _slug(name))
            row_path = os.path.join(run_dir, "row.json")
            cached = _cached_row(row_path) if skip_existing else None
            if cached is not None:
                rows.append(cached)
                continue
            if seq is None:
                seq = load_case(dataset_dir, case)
                tg, moving, static_groups = _case_groups(seq, all_groups, all_g2gt)
                if not track_all:
                    groups = subset_groups(all_groups, moving)
                    g2gt = {i: all_g2gt[g] for i, g in enumerate(moving)}
                    mp = {i: mp_all[g] for i, g in enumerate(moving)} if mp_all is not None else None
                    tg, moving, static_groups = 0, list(range(len(moving))), []
            if name == "Static (no motion)":
                res = static_baseline(groups, seq, g2gt, mp)
            else:
                m = next(m for m in methods if m.name == name)
                if verbose:
                    print(f"\n=== [{ci + 1}/{len(cases)}] {case.name} :: {m.name} ===")
                tc = variant_config(cfg.tracking, m.variant)
                prior = m.motion_prior(groups.num_groups, 1.0 / seq.fps) if m.motion_prior is not None else None
                res = track_sequence(model, groups, seq, dino, cfg, g2gt, name=m.name,
                                     tracking_cfg=tc, model_points=mp, motion_prior=prior, verbose=verbose)
            os.makedirs(run_dir, exist_ok=True)
            write_json(os.path.join(run_dir, "telemetry.json"), res.telemetry())
            np.save(os.path.join(run_dir, "part_deltas.npy"), res.part_deltas)
            row = {"case": case.name, "condition": case.condition, "seed": case.seed, "speed": case.speed,
                   "depth": case.depth, "method": name, "target": groups.names[tg], "scene_version": SCENE_VERSION,
                   "n_frames": len(seq.frames), **_prefix(res.summary(tg), "")}
            row["lost_rate"] = lost_rate(res.trans_err_cm[:, tg])
            if mp is not None and tg in mp:
                row["adds_success"] = adds_success_rate(res.adds[:, tg], object_diameter(mp[tg]))
            for j, g in enumerate(moving[1:]):
                sm = res.summary(g)
                row["other_moving_success_rate"] = sm["success_rate"]
                row["other_moving_mean_pos_err_cm"] = sm["mean_pos_err_cm"]
            if static_groups:
                row["static_objs_mean_pos_err_cm"] = float(np.nanmean(res.trans_err_cm[:, static_groups]))
                row["static_objs_max_pos_err_cm"] = float(np.nanmax(res.trans_err_cm[:, static_groups]))
            write_json(row_path, row)
            rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    df = pd.DataFrame(rows)
    write_tables(df, out_dir)
    return df


# --------------------------------------------------------------------------- #
# Real-time protocol suite
# --------------------------------------------------------------------------- #

LATENCY_COMP = " + latency comp."
"""Suffix of the second row a motion-prior method yields under the real-time
protocol: its estimate moved forward to 'now' with the filter's velocity."""

REALTIME_COLS = [
    ("adds_success", "ADD-S success (now)", "{:.0%}"),
    ("lost_rate", "Lost (now)", "{:.0%}"),
    ("median_pos_err_cm", "Median pos (cm, now)", "{:.2f}"),
    ("mean_rot_err_deg", "Rot err (deg, now)", "{:.1f}"),
    ("proc_adds_success", "ADD-S success (tracked frames)", "{:.0%}"),
    ("latency_ms", "Latency (ms)", "{:.0f}"),
    ("tracker_hz", "Tracker Hz", "{:.1f}"),
    ("frames_skipped_mean", "Frames skipped", "{:.1f}"),
    ("estimate_age_ms", "Estimate age (ms)", "{:.0f}"),
]


def run_realtime_suite(
    model: POGSModel,
    grouping: GroupingResult,
    dino: DinoExtractor,
    cfg: POGSConfig,
    cases: Sequence[Case],
    methods: Sequence[MethodSpec],
    out_dir: str,
    dataset_dir: str = DATASET_DIR,
    runs_dir: str = RUNS_DIR,
    include_static_baseline: bool = True,
    skip_existing: bool = True,
    latency_fn=None,
    verbose: bool = True,
) -> pd.DataFrame:
    """``run_suite`` for real-time cases (``Case.realtime``): every method runs
    under ``evaluate.track_realtime`` and is scored at every camera frame on
    the pose it had published by then ("now"). Each run dir also stores
    ``schedule.json`` (which frames were tracked, when) for the videos.
    A method with a motion prior adds a ``<name> + latency comp.`` row from
    the same run. Target-only protocol, as in ``run_suite``."""
    os.makedirs(out_dir, exist_ok=True)
    all_groups, all_g2gt = grouping.groups, grouping.group_to_gt
    mp_all = object_model_points(all_groups, model)
    rows = []
    for ci, case in enumerate(cases):
        assert case.realtime, f"{case.name} is not a real-time case"
        seq = groups = g2gt = mp = None
        jobs = (["Static (no motion)"] if include_static_baseline else []) + [m.name for m in methods]
        for name in jobs:
            m = next((m for m in methods if m.name == name), None)
            out_names = [name] + ([name + LATENCY_COMP] if m is not None and m.motion_prior is not None else [])
            row_paths = [os.path.join(runs_dir, case.name, _slug(n), "row.json") for n in out_names]
            cached = [_cached_row(p) for p in row_paths] if skip_existing else [None]
            if all(c is not None for c in cached):
                rows += cached
                continue
            if seq is None:
                seq = load_case(dataset_dir, case)
                _, moving, _ = _case_groups(seq, all_groups, all_g2gt)
                groups = subset_groups(all_groups, moving)
                g2gt = {i: all_g2gt[g] for i, g in enumerate(moving)}
                mp = {i: mp_all[g] for i, g in enumerate(moving)}
            base = {"case": case.name, "condition": case.condition, "seed": case.seed, "speed": case.speed,
                    "depth": case.depth, "target": groups.names[0], "n_frames": len(seq.frames),
                    "scene_version": SCENE_VERSION}
            if m is None:
                res = static_baseline(groups, seq, g2gt, mp)
                outs = [(name, res, None, None)]
            else:
                if verbose:
                    print(f"\n=== [{ci + 1}/{len(cases)}] {case.name} :: {m.name} (real-time) ===")
                tc = variant_config(cfg.tracking, m.variant)
                prior = m.motion_prior(groups.num_groups, 1.0 / seq.fps) if m.motion_prior is not None else None
                rt = track_realtime(model, groups, seq, dino, cfg, g2gt, name=m.name, tracking_cfg=tc,
                                    model_points=mp, motion_prior=prior, latency_fn=latency_fn, verbose=verbose)
                outs = [(name, rt.now, rt.processed, rt)]
                if rt.now_extrap is not None:
                    outs.append((name + LATENCY_COMP, rt.now_extrap, rt.processed, rt))
            for n, res, proc, rt in outs:
                run_dir = os.path.join(runs_dir, case.name, _slug(n))
                os.makedirs(run_dir, exist_ok=True)
                write_json(os.path.join(run_dir, "telemetry.json"), res.telemetry())
                np.save(os.path.join(run_dir, "part_deltas.npy"), res.part_deltas)
                row = {**base, "method": n, **_prefix(res.summary(0), "")}
                row["lost_rate"] = lost_rate(res.trans_err_cm[:, 0])
                row["adds_success"] = adds_success_rate(res.adds[:, 0], object_diameter(mp[0]))
                if rt is not None:
                    write_json(os.path.join(run_dir, "schedule.json"), rt.schedule())
                    row["proc_adds_success"] = adds_success_rate(proc.adds[:, 0], object_diameter(mp[0]))
                    row["proc_mean_pos_err_cm"] = float(np.nanmean(proc.trans_err_cm[:, 0]))
                    row["n_tracked"] = int(len(rt.proc_frames))
                write_json(os.path.join(run_dir, "row.json"), row)
                rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    df = pd.DataFrame(rows)
    write_realtime_tables(df, out_dir)
    return df


def write_realtime_tables(df: pd.DataFrame, out_dir: str) -> None:
    """``summary.md``: one table per (condition, playback speed), mean ± std
    over seeds, then the mean over all real-time cases."""
    df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    methods = list(dict.fromkeys(df["method"]))
    cols = [c for c in REALTIME_COLS if c[0] in df]
    head = "| Method | " + " | ".join(h for _, h, _ in cols) + " |"
    sep = "|---|" + "---:|" * len(cols)
    lines = ["# Real-time tracking results", "",
             f"Camera stream at {REALTIME_FPS:.0f} Hz of scene time; at Nx playback the frames arrive N times "
             "faster (2x = 60 frames/s of wall time), so the objects move N times further per tracker step. "
             "The tracker always takes the newest frame "
             "once it is free; 'now' columns score, at EVERY camera frame, the pose the tracker had published "
             "by then. Latency is the measured compute time per tracked frame on this machine. "
             f"{df['seed'].nunique()} seed(s); mean ± std over seeds.", ""]
    for (cond, speed), sub in df.groupby(["condition", "speed"], sort=False):
        lines += [f"## {cond}, {speed}x playback speed", "", head, sep]
        for m in methods:
            r = sub[sub["method"] == m]
            if not r.empty:
                lines.append(f"| {m} | " + " | ".join(_fmt(r[k], k, f) for k, _, f in cols) + " |")
        lines.append("")
    lines += ["## all real-time cases", "", "| Method | Cases | " + " | ".join(h for _, h, _ in cols) + " |",
              "|---|---:|" + "---:|" * len(cols)]
    for m in methods:
        r = df[df["method"] == m]
        lines.append(f"| {m} | {len(r)} | " + " | ".join(
            (f.format(r[k].mean()) if r[k].notna().any() else "-") for k, _, f in cols) + " |")
    with open(os.path.join(out_dir, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _cached_row(path: str) -> Optional[Dict]:
    """A finished run's row, if it was produced on the current scene version."""
    if not os.path.exists(path):
        return None
    row = json.load(open(path))
    return row if row.get("scene_version", 1) == SCENE_VERSION else None


LOST_CM = 10.0
"""A frame counts as 'lost' when the position error exceeds this."""


def lost_rate(trans_err_cm: np.ndarray) -> float:
    v = trans_err_cm[~np.isnan(trans_err_cm)]
    return float((v > LOST_CM).mean()) if v.size else float("nan")


def rescore_runs(model: POGSModel, grouping: GroupingResult, runs_dir: str = RUNS_DIR, verbose: bool = True) -> int:
    """Adds ``adds_success`` to cached rows that predate it, from the poses
    saved in each run's telemetry (no re-tracking). Returns rows updated."""
    groups = grouping.groups
    mp = object_model_points(groups, model)
    by_name = {groups.names[g]: g for g in range(groups.num_groups)}
    n = 0
    if not os.path.isdir(runs_dir):
        return 0
    for case in sorted(os.listdir(runs_dir)):
        for meth in sorted(os.listdir(os.path.join(runs_dir, case))):
            d = os.path.join(runs_dir, case, meth)
            rp, tp = os.path.join(d, "row.json"), os.path.join(d, "telemetry.json")
            if not (os.path.exists(rp) and os.path.exists(tp)):
                continue
            row = json.load(open(rp))
            if "adds_success" in row and "lost_rate" in row:
                continue
            g = by_name.get(row["target"])
            if g is None:
                continue
            tele = json.load(open(tp))
            obj = next(o for o in tele[0]["objects"] if o["name"] == row["target"])
            k = tele[0]["objects"].index(obj)
            adds = np.array([adds_metric(mp[g], np.array(f["objects"][k]["pred_pose"]),
                                         np.array(f["objects"][k]["gt_pose"])) for f in tele])
            row["adds_success"] = adds_success_rate(adds, object_diameter(mp[g]))
            row["lost_rate"] = lost_rate(np.array([f["objects"][k]["trans_err_cm"] for f in tele]))
            write_json(rp, row)
            n += 1
    if verbose:
        print(f"[rescore] added adds_success to {n} cached rows")
    return n


def _case_groups(seq, groups: ObjectGroups, g2gt: Dict[int, int]):
    """(target group, groups of all moving objects, groups of static objects)."""
    def group_of(obj_id):
        gs = [g for g, o in g2gt.items() if o == obj_id]
        if not gs:
            raise RuntimeError(f"object {obj_id} of {seq.name} has no tracked group")
        return gs[0]
    moving = [group_of(o) for o in (seq.moving_obj_ids or [seq.target_obj_id])]
    static = [g for g in range(groups.num_groups) if g not in moving and g2gt.get(g, -1) >= 0]
    return moving[0], moving, static


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s).strip("_").lower()


def _prefix(d: Dict, p: str) -> Dict:
    return {f"{p}{k}": v for k, v in d.items() if k != "name"}


TABLE_COLS = [
    ("success_rate", "Success (3 cm & 5 deg)", "{:.0%}"),
    ("adds_success", "Success (ADD-S < 0.1d)", "{:.0%}"),
    ("median_pos_err_cm", "Median pos (cm)", "{:.2f}"),
    ("lost_rate", "Lost (>10 cm)", "{:.0%}"),
    ("mean_pos_err_cm", "Pos err (cm)", "{:.2f}"),
    ("mean_rot_err_deg", "Rot err (deg)", "{:.2f}"),
    ("mean_adds_cm", "ADD-S (cm)", "{:.2f}"),
    ("max_pos_err_cm", "Max pos (cm)", "{:.2f}"),
    ("final_pos_err_cm", "Final pos (cm)", "{:.2f}"),
    ("static_objs_mean_pos_err_cm", "Static drift (cm)", "{:.2f}"),
    ("mean_opt_ms", "Opt ms/frame", "{:.0f}"),
    ("fps", "FPS", "{:.1f}"),
]
"""Per-factor tables: mean over seeds, with the seed standard deviation for the
main accuracy columns."""
_WITH_STD = {"success_rate", "adds_success", "lost_rate", "mean_pos_err_cm", "mean_rot_err_deg",
             "proc_adds_success", "latency_ms"}


def _fmt(vals: pd.Series, key: str, fmt: str) -> str:
    v = vals.dropna()
    if v.empty:
        return "-"
    m = fmt.format(v.mean())
    if key in _WITH_STD and len(v) > 1:
        sd = v.std(ddof=1)
        m += " ± " + (f"{sd:.0%}" if key in ("success_rate", "adds_success", "lost_rate", "proc_adds_success")
                      else f"{sd:.0f}" if key == "latency_ms" else f"{sd:.2f}")
    return m


def write_tables(df: pd.DataFrame, out_dir: str) -> None:
    """``summary.md``: one table per benchmark factor (levels x methods,
    averaged over seeds), then the overall mean per method."""
    df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    methods = list(dict.fromkeys(df["method"]))
    head = "| Level | Method | " + " | ".join(h for _, h, _ in TABLE_COLS) + " |"
    sep = "|---|---|" + "---:|" * len(TABLE_COLS)
    lines = ["# Tracking benchmark results", "",
             f"{df['seed'].nunique()} seed(s) per level; values are mean ± std over seeds. "
             "Target object only, except 'Static drift' (objects that never move).", ""]
    for factor, levels in FACTORS.items():
        block = []
        for label, cond, speed, depth in levels:
            sub = df[(df["condition"] == cond) & (df["speed"] == speed) & (df["depth"] == depth)]
            for m in methods:
                r = sub[sub["method"] == m]
                if r.empty:
                    continue
                block.append(f"| {label} | {m} | " + " | ".join(_fmt(r[k], k, f) if k in r else "-"
                                                             for k, _, f in TABLE_COLS) + " |")
        if block:
            lines += [f"## {factor}", "", head, sep] + block + [""]
    lines += ["## all cases", "", "| Method | Cases | " + " | ".join(h for _, h, _ in TABLE_COLS) + " |",
              "|---|---:|" + "---:|" * len(TABLE_COLS)]
    for m in methods:
        r = df[df["method"] == m]
        lines.append(f"| {m} | {len(r)} | " + " | ".join(
            (fmt.format(r[k].mean()) if k in r and r[k].notna().any() else "-") for k, _, fmt in TABLE_COLS) + " |")
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
