"""
E5: static translation sweeps at the ground-truth pose along the true O4 lift trajectory, release objective vs.
the masked (`maskobj`) objective. Pre-registered in `docs/note/e5_dynamic_failure_preregistration.md` (D1, D2).
Evaluation only: ground truth places the sweep.

    source tools/cuda_env.sh
    .venv/bin/python -m experiments.e5_dynamic run
    .venv/bin/python -m experiments.e5_dynamic summarise
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from data.scene_generator.pick_place_benchmark import load_sequence
from data.scene_generator.tool_perturbation import TOOL_DATASET_DIR
from experiments.b_bias import _argmin, _obs, _optimizer
from experiments.env_info import write_env_json
from experiments.g1_generality import TRANS_GRID, _curves, _kappa
from experiments.o3_perturbation import setup_tool_context
from model.pogs.checkpoint import write_json
from model.pogs.evaluate import gt_pose_for_group, tracking_camera, variant_config
from model.pogs.information import pose_to_part_delta

OUT = "data_results/e5_dynamic"
CASES = [f"grasp_+15_s{s}" for s in range(10)] + [f"grasp_-15_s{s}" for s in range(10)]
WORLD = {"x": np.r_[1.0, 0, 0], "y": np.r_[0, 1.0, 0], "z": np.r_[0, 0, 1.0]}


def run(device: str = "cuda:0", phases=("lift", "approach")) -> None:
    os.makedirs(OUT, exist_ok=True)
    write_env_json(OUT)
    ctx = setup_tool_context(device)
    cfg = ctx["cfg"]
    tcfg = variant_config(cfg.tracking, "POGS")
    wd = float(tcfg.depth_loss_mult)
    c0 = ctx["groups"].init_p2w[0, :3].detach().cpu().numpy()
    gt_id = ctx["g2gt"][0]
    rows = []
    for name in CASES:
        seq = load_sequence(TOOL_DATASET_DIR, name, lazy=True)
        cam = tracking_camera(seq.camera, tcfg, device)
        for fi, fm in enumerate(seq.meta["frames"]):
            if fm["phase"] not in phases:
                continue
            T = gt_pose_for_group(seq, fi, gt_id, c0)
            opt = _optimizer(ctx, tcfg)
            with torch.no_grad():
                opt.part_deltas.data[0] = pose_to_part_delta(T, c0).to(opt.part_deltas)
            fr = seq.frames[fi]
            rgb = torch.as_tensor(np.asarray(fr.rgb)).float().to(device)
            if rgb.max() > 1.5:
                rgb = rgb / 255.0
            depth = torch.as_tensor(np.asarray(fr.depth)).float().to(device)
            try:
                obs = _obs(rgb, depth, cam, ctx, tcfg, opt)
            except RuntimeError as e:
                print(f"[e5] {name} f{fi}: skipped ({e})", flush=True)
                continue
            R = np.asarray(T)[:3, :3]
            for ax, w in WORLD.items():
                a6 = np.concatenate([np.zeros(3), R.T @ w])        # world axis expressed in the object frame
                cv = _curves(opt, obs, a6, TRANS_GRID, with_depth=True, scale=1e-3)
                L, ins, dep = (np.asarray(cv[k], float) for k in ("L", "inside_mean", "depth"))
                J_rel, J_mask = L + wd * dep, ins + wd * dep
                r = dict(case=name, frame=fi, phase=fm["phase"], occlusion_frac=fm.get("occlusion_frac"), axis=ax,
                         argmin={"J_release": _argmin(TRANS_GRID, J_rel), "J_mask": _argmin(TRANS_GRID, J_mask),
                                 "L": _argmin(TRANS_GRID, L), "inside": _argmin(TRANS_GRID, ins),
                                 "depth": _argmin(TRANS_GRID, dep)},
                         kappa={"J_release": _kappa(TRANS_GRID, J_rel, 7.5), "J_mask": _kappa(TRANS_GRID, J_mask, 7.5),
                                "L": _kappa(TRANS_GRID, L, 7.5), "inside": _kappa(TRANS_GRID, ins, 7.5)},
                         curves={"J_release": J_rel.tolist(), "J_mask": J_mask.tolist()})
                rows.append(r)
            last = rows[-3:]
            print(f"[e5] {name} f{fi:02d} {fm['phase']:8s} argmin J_rel " + " ".join(
                f"{r['argmin']['J_release']:+.1f}" for r in last) + " | J_mask " + " ".join(
                f"{r['argmin']['J_mask']:+.1f}" for r in last) + " mm", flush=True)
    write_json(os.path.join(OUT, "sweeps.json"), {"grid_mm": TRANS_GRID.tolist(), "depth_weight": wd, "rows": rows})


def summarise() -> None:
    d = json.load(open(os.path.join(OUT, "sweeps.json")))
    rows = d["rows"]

    def stats(phase):
        R = [r for r in rows if r["phase"] == phase]
        frames = {}
        for r in R:
            frames.setdefault((r["case"], r["frame"]), []).append(r)
        mx = {k: [max(abs(r["argmin"][k]) for r in fr) for fr in frames.values()] for k in ("J_release", "J_mask")}
        ratios = [r["kappa"]["J_mask"] / r["kappa"]["J_release"] for r in R
                  if r["kappa"]["J_release"] and r["kappa"]["J_release"] > 0 and np.isfinite(r["kappa"]["J_mask"])]
        return len(frames), {k: float(np.median(v)) if v else float("nan") for k, v in mx.items()}, \
            (float(np.median(ratios)) if ratios else float("nan")), len(ratios)

    n, med, kr, nk = stats("lift")
    D1 = med["J_mask"] >= 5.0 and med["J_mask"] >= 2 * med["J_release"]
    D2 = kr < 0.5
    na, meda, kra, _ = stats("approach")
    L = ["# E5: static sweeps along the true lift trajectory (release vs. masked objective)", "",
         "Pre-registered in `docs/note/e5_dynamic_failure_preregistration.md`. Static (no motion) is not applicable: "
         "the tracker is not run; sweeps at the ground-truth pose. Rules first.", "",
         f"Lift frames: {n}; depth weight {d['depth_weight']}.", "", "## Pre-registered results", "",
         "| Rule | Statistic | Threshold | Result |", "|---|---|---|---|",
         f"| D1 | median max-axis abs(argmin): J_mask {med['J_mask']:.1f} mm vs J_release {med['J_release']:.1f} mm | "
         f"≥ 5 mm and ≥ 2× | {'pass' if D1 else 'FAIL'} |",
         f"| D2 | median κ(J_mask)/κ(J_release) = {kr:.2f} (n {nk}) | < 0.5 | {'pass' if D2 else 'FAIL'} |", "",
         "## Exploratory", "",
         f"- Approach frames ({na}): median max-axis abs(argmin) J_mask {meda['J_mask']:.1f} mm, J_release "
         f"{meda['J_release']:.1f} mm; κ ratio {kra:.2f}."]
    lift = [r for r in rows if r["phase"] == "lift"]
    for k in ("L", "inside", "depth"):
        v = [abs(r["argmin"][k]) for r in lift]
        L.append(f"- Lift, term {k}: median abs(argmin) per axis {np.median(v):.1f} mm.")
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"FINDING: E5-D1 | lift frames: masked objective median max-axis translation argmin {med['J_mask']:.1f} mm vs "
          f"release {med['J_release']:.1f} mm ({'pass' if D1 else 'FAIL'}) | {OUT}/summary.md")
    print(f"FINDING: E5-D2 | lift frames: median stiffness ratio masked/release {kr:.2f} "
          f"({'pass' if D2 else 'FAIL'}, rule < 0.5) | {OUT}/summary.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "summarise"])
    a = ap.parse_args()
    run() if a.step == "run" else summarise()
