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

from common.checkpoint import checkpoint_exists, load_checkpoint, save_checkpoint, write_json
from common.config import POGSConfig
from common.encoders import DinoExtractor, OpenCLIPNetwork
from common.evaluate import (
    TrackingResult, adds_metric, adds_success_rate, object_diameter, object_model_points,
    static_baseline, track_sequence, variant_config,
)
from common.grouping import (
    ObjectGroups, assign_gt_ids, build_object_groups, cluster_purity, cluster_scene,
    group_gt_iou, match_clusters_to_gt, select_object_clusters, subset_groups,
)
from common.model import POGSModel
from common.trainer import evaluate_views, precompute_features, train_pogs
from scene_generator.pick_place_benchmark import (
    FACTORS, OBJECT_NAMES, OBJECT_QUERIES, Case, load_capture, load_case,
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

    if checkpoint_exists(ckpt_dir) and not force_train:
        model, pca, _ = load_checkpoint(ckpt_dir, cfg, device=cfg.runtime.device)
        dino.set_pca(pca)
        if verbose:
            print(f"[train] loaded checkpoint from {ckpt_dir}: {model.num_points} gaussians, step {model.step}")
        return model, capture, dino, clip

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
                    extra={"train_time_s": train_time, **quality})
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
            if skip_existing and os.path.exists(row_path):
                rows.append(json.load(open(row_path)))
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
                   "depth": case.depth, "method": name, "target": groups.names[tg],
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
_WITH_STD = {"success_rate", "adds_success", "lost_rate", "mean_pos_err_cm", "mean_rot_err_deg"}


def _fmt(vals: pd.Series, key: str, fmt: str) -> str:
    v = vals.dropna()
    if v.empty:
        return "-"
    m = fmt.format(v.mean())
    if key in _WITH_STD and len(v) > 1:
        sd = v.std(ddof=1)
        m += " ± " + (f"{sd:.0%}" if key in ("success_rate", "adds_success", "lost_rate") else f"{sd:.2f}")
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
