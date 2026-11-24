"""8.OFF: roll sweeps of the OFFICIAL POGS tracking loss on the official splat (official venv).

    ~/venvs/pogs-official/bin/python -m experiments.off_repro.official_sweeps run --config <runs/.../config.yml>
    ~/venvs/pogs-official/bin/python -m experiments.off_repro.official_sweeps summarise

Grouping follows the release's interactive steps without the viewer: `POGSModel.cluster(eps=0.012)` (viewer
default), a scripted click (index.json), the `_crop_to_click` branch for precomputed labels (vote of the 10 nearest
Gaussians, DBSCAN eps 0.010 in xyz, points below table_z + 0.015 removed), then `Optimizer._cluster_from_file`
(gauss_params reduced to keep_inds, mapping, group masks). The loss is `RigidGroupOptimizer.get_optim_loss` with the
official config (ROI on), at part_deltas = rotation about world x (= drill bit axis, yaw 0) through the official group
centroid. Pre-registration: docs/note/off_reproduction_preregistration.md (OR1-OR3).
"""
import argparse
import glob
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from experiments.off_repro import shims  # noqa: E402,F401

import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

wp.init()  # as track_main_online_demo.py

FR = os.path.expanduser("~/pogs_external/off/frames")
OUT = os.path.join(ROOT, "data_results/off_repro")
GRID = np.arange(-20.0, 20.0 + 1e-9, 0.5)


def setup(config: str, click, table_z: float, eps: float = 0.012):
    import open3d as o3d
    import trimesh
    from pathlib import Path
    from nerfstudio.utils.eval_utils import eval_setup
    from pogs.tracking.rigid_group_optimizer import RigidGroupOptimizer, RigidGroupOptimizerConfig
    _, pipeline, _, _ = eval_setup(Path(config))
    model = pipeline.model
    model.eval()
    with torch.no_grad():
        labels = model.cluster(eps)
    model.cluster_labels = torch.Tensor(labels)
    means = model.gauss_params["means"].detach()
    pts = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(means.cpu().numpy()))
    kd = o3d.geometry.KDTreeFlann(pts)
    _, inds, _ = kd.search_knn_vector_3d(np.asarray(click, float).reshape(3, 1), 10)
    sphere = list(inds)
    vote = int(torch.tensor(model.cluster_labels[sphere].mode())[0].item())
    keep_list = torch.where(model.cluster_labels == vote)[0]
    keep_o3d = pts.select_by_index(keep_list.tolist())
    sphere_vote = torch.where(model.cluster_labels[sphere] == vote)[0]
    sphere_keep = [(torch.where(keep_list == torch.tensor(sphere)[i])[0]).item() for i in sphere_vote.tolist()]
    gc = keep_o3d.cluster_dbscan(eps=0.010, min_points=1)
    inner = torch.tensor(gc)[sphere_keep].mode()[0].item()
    keep = keep_list[torch.where(torch.tensor(gc) == inner)[0]]
    keep = keep[torch.where(means[keep][:, 2].cpu() > table_z + 0.015)[0]]
    info = {"n_clusters": int(labels.max()) + 1, "vote": vote, "n_keep": int(len(keep)),
            "n_gaussians": int(means.shape[0])}
    # Optimizer._cluster_from_file
    import copy  # pipeline._queue_state() without its viewer-button side effect (no viewer here)
    pipeline.state_stack.append(copy.deepcopy({k: v.detach() for k, v in model.gauss_params.items()}))
    prev = pipeline.state_stack[-1]
    for name in model.gauss_params.keys():
        model.gauss_params[name] = prev[name][keep]
    model.keep_inds = keep
    cl = model.cluster_labels[keep].to(torch.int32)
    model.mapping, cl_keep = torch.unique(cl, return_inverse=True)
    group_masks = [(cid == cl_keep).cuda() for cid in range(cl_keep.max().item() + 1)]
    centroid = model.gauss_params["means"][group_masks[0]].mean(0).detach().cpu().numpy()
    model.cgtf_stack = np.array([[1.0, 0, 0, 0, *centroid]])
    pipeline.cgtf_stack = model.cgtf_stack
    scale = pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_scale
    opt = RigidGroupOptimizer(RigidGroupOptimizerConfig(), model, group_masks=group_masks,
                              group_labels=cl_keep.int().cuda(), dataset_scale=scale)
    opt.is_initialized = True
    info.update(centroid=centroid.tolist(), dataset_scale=float(scale))
    return pipeline, opt, info


def ns_camera(c2w_cv, K, w, h, scale):
    import trimesh
    from nerfstudio.cameras.cameras import Cameras
    m = torch.from_numpy(np.asarray(c2w_cv, np.float32)) @ torch.from_numpy(
        trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])).float()
    m = m[None, :3, :].clone()
    m[:, :3, 3] *= scale
    cam = Cameras(camera_to_worlds=m, fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]),
                  width=int(w), height=int(h))
    cam.rescale_output_resolution(500 / min(int(w), int(h)))  # Optimizer.MATCH_RESOLUTION
    return cam


def roll_deltas(deg):
    h = math.radians(deg) / 2
    d = torch.zeros(1, 7, device="cuda")
    d[0, 3:] = torch.tensor([math.cos(h), math.sin(h), 0.0, 0.0])
    return d


def run(config: str) -> None:
    import pogs.tracking.rigid_group_optimizer  # noqa: F401  (official import order; observation <-> utils2 cycle)
    from pogs.tracking.observation import PosedObservation
    os.makedirs(OUT, exist_ok=True)
    os.chdir(os.path.expanduser("~/pogs_external/off"))  # official caches: ./outputs/<data name> (as in training)
    idx = json.load(open(os.path.join(FR, "index.json")))
    pipeline, opt, info = setup(config, idx["click_xyz"], idx["table_z"])
    print("setup:", info, flush=True)
    dino_fn = pipeline.datamanager.dino_dataloader.get_pca_feats
    rows = []
    for fr in idx["frames"]:
        z = np.load(os.path.join(FR, fr["file"]))
        cam = ns_camera(z["c2w_cv"], z["K"], z["width"], z["height"], info["dataset_scale"])
        rgb = torch.from_numpy(z["rgb"]).float().cuda() / 255.0
        depth = torch.from_numpy(z["depth"]).float().cuda()[..., None]
        if rgb.shape[0] != int(cam.height) or rgb.shape[1] != int(cam.width):
            raise RuntimeError(f"frame {rgb.shape} vs camera {int(cam.height)}x{int(cam.width)}")
        obs = PosedObservation(rgb=rgb, camera=cam, dino_fn=dino_fn, metric_depth_img=depth)
        opt.set_observation(obs)
        curves = {"dino": [], "dino_depth": []}
        for g in GRID:
            for key, ud in (("dino", False), ("dino_depth", True)):
                with torch.no_grad():
                    res = opt.get_optim_loss(opt.frame, roll_deltas(g), True, ud, False, False, False, False, True)
                curves[key].append(float(res[0]) if res is not None else float("nan"))
        am = {k: float(GRID[int(np.nanargmin(v))]) for k, v in curves.items()}
        on1 = np.isclose(GRID % 1.0, 0.0)
        am1 = {k: float(GRID[on1][int(np.nanargmin(np.asarray(v)[on1]))]) for k, v in curves.items()}
        rows.append({**fr, "curves": curves, "argmin": am, "argmin_1deg": am1})
        print(f"[off] {fr['view']:20s} argmin dino {am['dino']:+.1f} (1deg {am1['dino']:+.0f})  "
              f"dino+depth {am['dino_depth']:+.1f}", flush=True)
    json.dump({"grid_deg": GRID.tolist(), "setup": info, "config": config, "rows": rows},
              open(os.path.join(OUT, "official_sweeps.json"), "w"))


def summarise() -> None:
    from scipy.stats import spearmanr
    d = json.load(open(os.path.join(OUT, "official_sweeps.json")))
    rows = d["rows"]
    dose = [r for r in rows if r["kind"] == "dose"]
    grip1 = next(r for r in dose if abs(r["shadow_strength"] - 1.0) < 1e-9)
    or1 = grip1["argmin_1deg"]["dino"] <= -5.0
    s = [r["shadow_strength"] for r in dose]
    a = [r["argmin"]["dino"] for r in dose]
    rho = float(spearmanr(s, a).statistic)
    a0 = next(r["argmin"]["dino"] for r in dose if r["shadow_strength"] == 0.0)
    shift = grip1["argmin"]["dino"] - a0
    or2 = rho <= -0.8 and shift <= -4.0
    g1 = json.load(open(os.path.join(ROOT, "data_results/g1_generality/sweeps_seed0.json")))["rows"]
    ours = {r["view"]: r["argmin"]["L"] for r in g1 if r.get("object") == "power_drill" and r["axis"] == "x"}
    pairs = [(grip1["argmin_1deg"]["dino"], ours.get("track, gripper (frame 0)"))]
    for r in rows:
        if r["kind"] == "ring":
            pairs.append((r["argmin_1deg"]["dino"], ours.get(r["view"])))
    pairs = [(x, y) for x, y in pairs if y is not None and np.isfinite(x) and np.isfinite(y)]
    rho3 = float(spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).statistic) if len(pairs) >= 3 else np.nan
    mad = float(np.median([abs(x - y) for x, y in pairs])) if pairs else np.nan
    or3 = rho3 >= 0.5 and mad <= 3.0
    L = ["# 8.OFF: official POGS code, static roll sweeps", "",
         "Pre-registered in `docs/note/off_reproduction_preregistration.md`. Static (no motion) is not applicable: no"
         " tracking is run. Rules first.", "", f"Setup: {json.dumps(d['setup'])}", "",
         "## Pre-registered results", "", "| Rule | Statistic | Threshold | Result |", "|---|---|---|---|",
         f"| OR1 | roll argmin, gripper + shadow (1° grid) = {grip1['argmin_1deg']['dino']:+.0f}° | ≤ −5° | "
         f"{'pass' if or1 else 'FAIL'} |",
         f"| OR2 | Spearman(s, argmin) = {rho:+.2f}; shift s 0→1 = {shift:+.1f}° | ≤ −0.8 and ≤ −4° | "
         f"{'pass' if or2 else 'FAIL'} |",
         f"| OR3 | Spearman vs ours = {rho3:+.2f}; median abs diff = {mad:.1f}° (n {len(pairs)}) | ≥ 0.5 and ≤ 3° | "
         f"{'pass' if or3 else 'FAIL'} |", "", "## Rows", "",
         "| View | s | argmin DINO (official) | DINO+depth (official, exploratory) | ours (G1 seed 0, release L) |",
         "|---|---|---|---|---|"]
    for r in rows:
        key = "track, gripper (frame 0)" if r["view"] == "gripper s=1" else r["view"]
        o = ours.get(key)
        L.append(f"| {r['view']} | {r['shadow_strength']:g} | {r['argmin']['dino']:+.1f} | "
                 f"{r['argmin']['dino_depth']:+.1f} | {'' if o is None else f'{o:+.0f}'} |")
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:14]))
    print(f"FINDING: OR1 | official POGS loss, drill roll argmin with gripper shadow {grip1['argmin_1deg']['dino']:+.0f} deg"
          f" ({'pass' if or1 else 'FAIL'}, rule <= -5) | data_results/off_repro/summary.md")
    print(f"FINDING: OR2 | official loss shadow dose-response rho {rho:+.2f}, shift {shift:+.1f} deg "
          f"({'pass' if or2 else 'FAIL'}) | data_results/off_repro/summary.md")
    print(f"FINDING: OR3 | official vs re-implementation argmins: rho {rho3:+.2f}, median |diff| {mad:.1f} deg "
          f"({'pass' if or3 else 'FAIL'}) | data_results/off_repro/summary.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "summarise"])
    ap.add_argument("--config")
    a = ap.parse_args()
    run(a.config) if a.step == "run" else summarise()
