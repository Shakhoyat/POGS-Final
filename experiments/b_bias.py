"""
Direction B (LEDGER 2026-11-23 decision; plan rev. 2 §8 "features or loss mismatch"): find the cause of the
DINO-term roll bias (`docs/results/e2_loss_bias_2026-11-23.md`) with cheap, ordered checks. Evaluation /
diagnostics only: ground truth is used to place the sweep, never by the tracker.

    B0 gt     Is the ground truth itself right? (sim analogue of an independent ArUco check)
              (a) metadata: drill GT motion is exactly identity over every approach frame of all O4 cases, and
                  capture and sequences are built from the same scene placement;
              (b) splat-independent of DINO: 1-D sweeps of the RGB-only and depth-only terms along all 6 DOF of
                  the object frame on the tracking camera, frame 0. If GT and splat agree, every RGB/depth
                  argmin is at 0 within the grid.
    B1 views  Does the -10 deg DINO minimum stay put or move with viewpoint / image grid?
              (a) roll sweeps (DINO, depth, RGB separately) on all 35 training views (drill at its scanned pose,
                  no gripper; the features were distilled on exactly these views) and on the tracking camera;
              (b) patch-grid test on the tracking camera: shift the ROI origin by 0..13 px (less than one DINO
                  patch) and re-sweep. A minimum that moves with the grid points to positional artefacts.
              (c) weights: argmin of DINO + w * depth and DINO + w * RGB for a ladder of w, from the same sweeps.

    source tools/cuda_env.sh && .venv/bin/python -m experiments.b_bias gt|views|grid|summarise
Outputs: data_results/b_bias/{gt,views,grid}.json and summary.md.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import numpy as np
import torch

from data.scene_generator.pick_place_benchmark import load_capture, load_sequence
from data.scene_generator.tool_perturbation import TOOL_DATASET_DIR, build_tool_scene, o4_cases
from experiments.env_info import write_env_json
from experiments.o3_perturbation import setup_tool_context
from model.pogs.checkpoint import write_json
from model.pogs.evaluate import tracking_camera, variant_config
from model.pogs.information import perturbed_deltas
from model.pogs.tracking import PosedObservation, RigidGroupOptimizer

OUT = "data_results/b_bias"
ROLL_GRID = np.arange(-30.0, 30.0 + 1e-9, 1.0)              # deg
FINE_ROT = np.arange(-6.0, 6.0 + 1e-9, 0.5)                 # deg
FINE_TRANS = np.arange(-6.0, 6.0 + 1e-9, 0.5) * 1e-3        # m
DEPTH_W = [0.0, 1.0, 3.7, 10.0, 30.0, 100.0, 1000.0]
RGB_W = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]


def _optimizer(ctx, tcfg):
    opt = RigidGroupOptimizer(tcfg, ctx["model"], ctx["groups"])
    opt.is_initialized = True                      # pose = identity = the scanned pose (GT here)
    return opt


def _terms(opt, obs, base, d6: np.ndarray) -> Dict[str, float]:
    """Unweighted DINO, depth and RGB terms at base * Exp(d6) (right perturbation, object frame)."""
    d = torch.as_tensor(d6, dtype=torch.float64)
    with torch.no_grad():
        _, st = opt.get_optim_loss(obs, perturbed_deltas(base, 0, d), use_dino=True, use_depth=True,
                                   use_rgb=True, use_roi=opt.config.use_roi)
    return {k: float(st.get(k, np.nan)) for k in ("dino", "depth", "rgb")}


def _sweep(opt, obs, axis6: np.ndarray, grid: np.ndarray, scale: float) -> Dict[str, List[float]]:
    base = opt.part_deltas.detach().clone()
    out = {k: [] for k in ("dino", "depth", "rgb")}
    for g in grid:
        t = _terms(opt, obs, base, axis6 * g * scale)
        for k in out:
            out[k].append(t[k])
    return out


def _argmin(grid, v) -> float:
    v = np.asarray(v, dtype=float)
    return float(grid[int(np.nanargmin(v))]) if np.isfinite(v).any() else float("nan")


def _obs(rgb, depth, cam, ctx, tcfg, opt):
    obs = PosedObservation(rgb, cam, ctx["dino"], depth, tcfg.max_roi_resolution)
    opt.set_observation(obs)
    return obs


def _roll6(ctx) -> np.ndarray:
    a = np.asarray(ctx["bit_axis_local"], dtype=float)
    return np.concatenate([a / np.linalg.norm(a), np.zeros(3)])


# --------------------------------------------------------------------------- #
# B0
# --------------------------------------------------------------------------- #

def gt_check(device="cuda:0", n_cases: int = 8) -> None:
    os.makedirs(OUT, exist_ok=True)
    write_env_json(OUT)
    # (a) metadata, all cases
    worst_rot, worst_t, n_frames, n_cases_md = 0.0, 0.0, 0, 0
    for p in sorted(glob.glob(os.path.join(TOOL_DATASET_DIR, "grasp_*", "tracking_metadata.json"))):
        m = json.load(open(p))
        n_cases_md += 1
        for f in m["frames"]:
            if f["phase"] != "approach":
                continue
            T = np.asarray(f["gt_deltas"]["6"])
            c = np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)
            worst_rot = max(worst_rot, float(np.degrees(np.arccos(c))))
            worst_t = max(worst_t, float(np.linalg.norm(T[:3, 3])))
            n_frames += 1
    drill = next(o for o in build_tool_scene() if o.obj_id == 6)
    cap_meta = json.load(open(os.path.join(TOOL_DATASET_DIR, "capture", "capture_metadata.json")))
    md = {"n_cases": n_cases_md, "n_approach_frames": n_frames, "max_rot_deg": worst_rot, "max_trans_m": worst_t,
          "scene_drill_position": list(drill.position), "scene_drill_yaw": float(drill.yaw),
          "capture_metadata_keys": sorted(cap_meta)[:30]}
    print("[b0] metadata:", json.dumps(md))

    # (b) 6-DOF RGB / depth sweeps on the tracking camera, frame 0 (drill at scan pose, gripper above)
    ctx = setup_tool_context(device)
    tcfg = variant_config(ctx["cfg"].tracking, "POGS")
    a = np.asarray(ctx["bit_axis_local"], float)
    a = a / np.linalg.norm(a)
    z = np.array([0.0, 0.0, 1.0])
    b = np.cross(z, a)                                  # horizontal, perpendicular to the bit axis
    axes = {"rot_bit(roll)": np.r_[a, 0, 0, 0], "rot_perp": np.r_[b, 0, 0, 0], "rot_z": np.r_[z, 0, 0, 0],
            "t_bit": np.r_[0, 0, 0, a], "t_perp": np.r_[0, 0, 0, b], "t_z": np.r_[0, 0, 0, z]}
    res = []
    for c in o4_cases(range(n_cases), angles=(15.0,)):
        seq = load_sequence(TOOL_DATASET_DIR, c.name)
        cam = tracking_camera(seq.camera, tcfg, device)
        opt = _optimizer(ctx, tcfg)
        fr = seq.frames[0]
        obs = _obs(fr.rgb.to(device), fr.depth.to(device), cam, ctx, tcfg, opt)
        row = {"case": c.name}
        for name, ax in axes.items():
            rot = name.startswith("rot")
            grid = FINE_ROT if rot else FINE_TRANS
            sw = _sweep(opt, obs, ax, grid, np.radians(1.0) if rot else 1.0)
            unit = 1.0 if rot else 1e3
            row[name] = {k: _argmin(grid, sw[k]) * unit for k in sw}
        res.append(row)
        print(f"[b0] {c.name}: " + ", ".join(f"{n}: rgb {row[n]['rgb']:+.1f} depth {row[n]['depth']:+.1f} "
                                             f"dino {row[n]['dino']:+.1f}" for n in axes), flush=True)
    write_json(os.path.join(OUT, "gt.json"), {"metadata": md, "sweeps": res,
               "units": "rotation deg, translation mm; grid ±6 at 0.5", "frame": 0})


# --------------------------------------------------------------------------- #
# B1
# --------------------------------------------------------------------------- #

def views(device="cuda:0") -> None:
    os.makedirs(OUT, exist_ok=True)
    ctx = setup_tool_context(device)
    tcfg = variant_config(ctx["cfg"].tracking, "POGS")
    cap = load_capture(TOOL_DATASET_DIR, device=device, objects=build_tool_scene())
    roll = _roll6(ctx)
    rows = []

    def one(label, rgb, depth, cam):
        opt = _optimizer(ctx, tcfg)
        try:
            obs = _obs(rgb, depth, cam, ctx, tcfg, opt)
            sw = _sweep(opt, obs, roll, ROLL_GRID, np.radians(1.0))
        except RuntimeError as e:                      # drill outside the view
            print(f"[b1] {label}: skipped ({e})")
            return
        r = {"view": label, "cam_pos": cam.c2w[:3, 3].tolist(), **{k: sw[k] for k in sw}}
        r["argmin"] = {k: _argmin(ROLL_GRID, sw[k]) for k in sw}
        rows.append(r)
        print(f"[b1] {label}: argmin dino {r['argmin']['dino']:+.0f}  depth {r['argmin']['depth']:+.0f}  "
              f"rgb {r['argmin']['rgb']:+.0f}", flush=True)

    for i, cam in enumerate(cap.cameras):
        cam_t = tracking_camera(cam, tcfg, device)
        one(f"train_{i:02d}", cap.images[i].to(device), cap.depths[i].to(device), cam_t)
    for c in o4_cases(range(3), angles=(15.0,)):
        seq = load_sequence(TOOL_DATASET_DIR, c.name)
        fr = seq.frames[0]
        one(f"track_{c.name}_f0", fr.rgb.to(device), fr.depth.to(device), tracking_camera(seq.camera, tcfg, device))
    write_json(os.path.join(OUT, "views.json"), {"roll_grid_deg": ROLL_GRID.tolist(), "rows": rows})


def grid_test(device="cuda:0", case: str = "grasp_+15_s0") -> None:
    os.makedirs(OUT, exist_ok=True)
    ctx = setup_tool_context(device)
    tcfg = variant_config(ctx["cfg"].tracking, "POGS")
    roll = _roll6(ctx)
    seq = load_sequence(TOOL_DATASET_DIR, case)
    cam = tracking_camera(seq.camera, tcfg, device)
    fr = seq.frames[0]
    rows = []
    for dx, dy in [(0, 0), (3, 0), (7, 0), (10, 0), (13, 0), (0, 7), (7, 7), (-7, 0), (0, -7)]:
        opt = _optimizer(ctx, tcfg)
        obs = _obs(fr.rgb.to(device), fr.depth.to(device), cam, ctx, tcfg, opt)
        bx0, bx1, by0, by1 = obs.roi_frames[0].bbox
        W, H = cam.width, cam.height
        x0, y0 = max(bx0 + dx, 0), max(by0 + dy, 0)
        x1, y1 = min(bx1 + dx, W - 1), min(by1 + dy, H - 1)
        obs.update_roi(0, x0 / (W - 1), x1 / (W - 1), y0 / (H - 1), y1 / (H - 1))
        sw = _sweep(opt, obs, roll, ROLL_GRID, np.radians(1.0))
        r = {"shift_px": [dx, dy], "bbox": list(obs.roi_frames[0].bbox), **sw,
             "argmin": {k: _argmin(ROLL_GRID, sw[k]) for k in sw}}
        rows.append(r)
        print(f"[b1-grid] shift {dx:+d},{dy:+d}: argmin dino {r['argmin']['dino']:+.0f} depth "
              f"{r['argmin']['depth']:+.0f} rgb {r['argmin']['rgb']:+.0f}", flush=True)
    write_json(os.path.join(OUT, "grid.json"), {"case": case, "roll_grid_deg": ROLL_GRID.tolist(), "rows": rows})


def _render(objs, cam, transforms, shadows: bool, device):
    from data.scene_generator.photoreal_scene import render_photoreal_scene
    from data.scene_generator.scene_builder import FlatScene
    out = render_photoreal_scene(FlatScene(objs, device=device).clone_with_transforms(transforms), cam,
                                 cast_shadows=shadows)
    return out["rgb"].float(), out["depth"].float()


def context(device="cuda:0") -> None:
    """B1d: separate 'novel viewpoint' from 'gripper in the scene' for the tracking camera. Clean re-renders
    (no depth noise) of the scanned scene: tracking camera with / without the frame-0 gripper and with / without
    shadows, and held-out viewpoints (never distilled on) looking at the drill, no gripper."""
    import math
    from data.scene_generator.pick_place_benchmark import GRIPPER_ID, BenchmarkConfig
    from data.scene_generator.pick_place_benchmark import tracking_camera as bench_cam
    from data.scene_generator.photoreal_scene import make_robot_gripper
    from data.scene_generator.tool_perturbation import (APPROACH_FRAMES, DRILL_ID, GRASP_APPROACH_H,
                                                        GRASP_PIVOT_LOCAL, _smooth, _translate)
    from model.pogs.geometry import PinholeCamera, look_at

    os.makedirs(OUT, exist_ok=True)
    ctx = setup_tool_context(device)
    tcfg = variant_config(ctx["cfg"].tracking, "POGS")
    roll = _roll6(ctx)
    bcfg = BenchmarkConfig()
    objects = build_tool_scene()
    drill = next(o for o in objects if o.obj_id == DRILL_ID)
    c = np.asarray(drill.position) + GRASP_PIVOT_LOCAL          # drill yaw 0 at scan
    g = make_robot_gripper(scale=1.0)
    g.position, g.yaw, g.obj_id, g.movable = (c[0], c[1], drill.position[2] + 0.138), drill.yaw + math.pi / 2, \
        GRIPPER_ID, True
    G0 = _translate([0, 0, GRASP_APPROACH_H * (1 - _smooth(1 / APPROACH_FRAMES))])
    track = bench_cam(bcfg, device)
    rows = []

    def one(label, objs, tr, cam, shadows):
        rgb, depth = _render(objs, cam, tr, shadows, device)
        cam_t = tracking_camera(cam, tcfg, device)
        opt = _optimizer(ctx, tcfg)
        try:
            obs = _obs(rgb, depth, cam_t, ctx, tcfg, opt)
            sw = _sweep(opt, obs, roll, ROLL_GRID, np.radians(1.0))
        except RuntimeError as e:
            print(f"[b1d] {label}: skipped ({e})")
            return
        r = {"view": label, "cam_pos": cam.c2w[:3, 3].tolist(), "shadows": shadows, **sw,
             "argmin": {k: _argmin(ROLL_GRID, sw[k]) for k in sw}}
        rows.append(r)
        print(f"[b1d] {label}: argmin dino {r['argmin']['dino']:+.0f} depth {r['argmin']['depth']:+.0f} "
              f"rgb {r['argmin']['rgb']:+.0f}", flush=True)

    for shadows in (True, False):
        sfx = "" if shadows else ", no shadows"
        one("track, gripper (frame 0)" + sfx, objects + [g], {GRIPPER_ID: G0}, track, shadows)
        one("track, no gripper" + sfx, objects, {}, track, shadows)
    # held-out viewpoints: rings around the drill (look-at the drill centre), no gripper
    tgt = np.asarray(drill.position) + np.array([0.0, 0.0, 0.07])
    for radius, elev in [(0.87, 30.0), (0.68, 30.0), (0.68, 55.0)]:
        for az in range(0, 360, 30):
            a, e = math.radians(az), math.radians(elev)
            eye = tgt + radius * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
            c2w = torch.from_numpy(look_at(eye, tgt)).float().to(device)
            cam = PinholeCamera(c2w, bcfg.track_fx, bcfg.track_fx, bcfg.track_width / 2.0, bcfg.track_height / 2.0,
                                bcfg.track_width, bcfg.track_height)
            one(f"heldout r{radius:.2f} e{elev:.0f} az{az:03d}", objects, {}, cam, True)
    write_json(os.path.join(OUT, "context.json"), {"roll_grid_deg": ROLL_GRID.tolist(), "rows": rows})


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #

def _weights_table(rows, grid) -> List[str]:
    L = ["| View set | n | " + " | ".join(f"DINO+{w:g}·depth" for w in DEPTH_W) + " |",
         "|---|---|" + "---|" * len(DEPTH_W)]
    for label, sel in rows:
        if not sel:
            continue
        cells = []
        for w in DEPTH_W:
            am = [_argmin(grid, np.asarray(r["dino"]) + w * np.asarray(r["depth"])) for r in sel]
            cells.append(f"{np.median(am):+.0f} [{np.min(am):+.0f}, {np.max(am):+.0f}]")
        L.append(f"| {label} | {len(sel)} | " + " | ".join(cells) + " |")
    L += ["", "| View set | n | " + " | ".join(f"DINO+{w:g}·RGB" for w in RGB_W) + " |",
          "|---|---|" + "---|" * len(RGB_W)]
    for label, sel in rows:
        if not sel:
            continue
        cells = []
        for w in RGB_W:
            am = [_argmin(grid, np.asarray(r["dino"]) + w * np.asarray(r["rgb"])) for r in sel]
            cells.append(f"{np.median(am):+.0f} [{np.min(am):+.0f}, {np.max(am):+.0f}]")
        L.append(f"| {label} | {len(sel)} | " + " | ".join(cells) + " |")
    return L


def summarise() -> None:
    L = ["# Direction B diagnostics (B0 ground truth, B1 viewpoint / grid / weights)", "",
         "Evaluation only. Single splat `tool_table`. Argmins in degrees of roll about the bit axis "
         "(median [min, max] over views).", ""]
    p = os.path.join(OUT, "gt.json")
    if os.path.exists(p):
        g = json.load(open(p))
        md = g["metadata"]
        L += ["## B0: is the ground truth right?", "",
              f"Metadata: {md['n_approach_frames']} approach frames in {md['n_cases']} cases; max GT rotation "
              f"{md['max_rot_deg']:.2e} deg, max translation {md['max_trans_m']:.2e} m (drill at the scanned pose).", "",
              "Argmin of each term along each object-frame DOF, tracking camera frame 0 (deg / mm, grid ±6 at 0.5):", "",
              "| DOF | RGB median [min, max] | depth | DINO |", "|---|---|---|---|"]
        for dof in [k for k in g["sweeps"][0] if k != "case"]:
            cells = []
            for term in ("rgb", "depth", "dino"):
                v = [r[dof][term] for r in g["sweeps"]]
                cells.append(f"{np.median(v):+.1f} [{np.min(v):+.1f}, {np.max(v):+.1f}]")
            L.append(f"| {dof} | " + " | ".join(cells) + " |")
        L.append("")
    p = os.path.join(OUT, "views.json")
    if os.path.exists(p):
        v = json.load(open(p))
        grid = np.asarray(v["roll_grid_deg"])
        tr = [r for r in v["rows"] if r["view"].startswith("train")]
        tk = [r for r in v["rows"] if r["view"].startswith("track")]
        L += ["## B1a: roll argmin by viewpoint (grid ±30 at 1)", "", "| View set | n | DINO | depth | RGB |",
              "|---|---|---|---|---|"]
        for label, sel in [("training views (distilled on)", tr), ("tracking camera, frame 0", tk)]:
            if sel:
                cells = []
                for k in ("dino", "depth", "rgb"):
                    am = [r["argmin"][k] for r in sel]
                    cells.append(f"{np.median(am):+.0f} [{np.min(am):+.0f}, {np.max(am):+.0f}]")
                L.append(f"| {label} | {len(sel)} | " + " | ".join(cells) + " |")
        L += ["", "Per training view (DINO argmin): " +
              ", ".join(f"{r['view'][-2:]}:{r['argmin']['dino']:+.0f}" for r in tr), "",
              "## B1c: argmin of the combined loss vs term weight", ""]
        L += _weights_table([("training views", tr), ("tracking camera", tk)], grid)
        L.append("")
    p = os.path.join(OUT, "grid.json")
    if os.path.exists(p):
        g = json.load(open(p))
        L += [f"## B1b: patch-grid shift ({g['case']}, frame 0; DINO patch ≈ 10 px at this ROI scale)", "",
              "| ROI shift (px) | DINO | depth | RGB |", "|---|---|---|---|"]
        for r in g["rows"]:
            L.append(f"| {r['shift_px'][0]:+d}, {r['shift_px'][1]:+d} | {r['argmin']['dino']:+.0f} | "
                     f"{r['argmin']['depth']:+.0f} | {r['argmin']['rgb']:+.0f} |")
        L.append("")
    p = os.path.join(OUT, "context.json")
    if os.path.exists(p):
        g = json.load(open(p))
        L += ["## B1d: gripper / shadow context and held-out viewpoints (clean re-renders)", "",
              "| View | DINO | depth | RGB |", "|---|---|---|---|"]
        for r in g["rows"]:
            if not r["view"].startswith("heldout"):
                L.append(f"| {r['view']} | {r['argmin']['dino']:+.0f} | {r['argmin']['depth']:+.0f} | "
                         f"{r['argmin']['rgb']:+.0f} |")
        ho = [r for r in g["rows"] if r["view"].startswith("heldout")]
        if ho:
            L += ["", "| Held-out ring | n | DINO median [min, max] | mean abs DINO argmin | depth | RGB |",
                  "|---|---|---|---|---|---|"]
            for ring in sorted({r["view"].split(" az")[0] for r in ho}):
                sel = [r for r in ho if r["view"].startswith(ring + " ")]
                cells = []
                for k in ("dino", "depth", "rgb"):
                    am = [r["argmin"][k] for r in sel]
                    cells.append(f"{np.median(am):+.0f} [{np.min(am):+.0f}, {np.max(am):+.0f}]")
                mad = np.mean([abs(r["argmin"]["dino"]) for r in sel])
                L.append(f"| {ring} | {len(sel)} | {cells[0]} | {mad:.1f} | {cells[1]} | {cells[2]} |")
            L += ["", "Per held-out view (DINO argmin): " +
                  ", ".join(f"{r['view'].replace('heldout ', '')}:{r['argmin']['dino']:+.0f}" for r in ho)]
        L.append("")
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["gt", "views", "grid", "context", "summarise"])
    a = ap.parse_args()
    {"gt": gt_check, "views": views, "grid": grid_test, "context": context, "summarise": summarise}[a.step]()
