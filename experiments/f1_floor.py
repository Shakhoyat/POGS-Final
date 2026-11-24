"""
F1: origin of the shared approach roll floor (pre-registered, `docs/note/f1_floor_origin_preregistration.md`).
(a) whole-image DINO loss at frame 0 (release init stage 1); (b) ROI DINO loss at approach frames 1-3. Both at the true
pose, roll about object x, official splat and our seed-0 splat, through our loss code (agrees with the official one).

    source tools/cuda_env.sh && .venv/bin/python -m experiments.f1_floor
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import torch

from data.scene_generator.pick_place_benchmark import load_sequence
from data.scene_generator.tool_perturbation import DRILL_ID, TOOL_DATASET_DIR
from experiments.b_bias import _argmin, _obs, _optimizer
from experiments.g1_generality import AXES, DOSE_GRID, _ctx, _curves, load_splat
from experiments.off_repro.swap_ours_on_official import official_ctx
from model.pogs.evaluate import tracking_camera, variant_config
from model.pogs.information import perturbed_deltas

OUT = "data_results/f1_floor"
CASES = [f"grasp_+15_s{s}" for s in range(10)] + [f"grasp_-15_s{s}" for s in range(10)]


def whole_image_curve(opt, obs, axis6, grid):
    base = opt.part_deltas.detach().clone()
    L = []
    with torch.no_grad():
        for g in grid:
            pd = perturbed_deltas(base, 0, torch.as_tensor(axis6 * g * math.pi / 180.0, dtype=torch.float64))
            loss, _ = opt.get_optim_loss(obs, pd, use_dino=True, use_depth=False, use_rgb=False, use_roi=False)
            L.append(float(loss) if loss is not None else float("nan"))
    return L


def run_splat(name, ctx, device):
    tcfg = variant_config(ctx["cfg"].tracking, "POGS")
    rows = []
    for case in CASES:
        seq = load_sequence(TOOL_DATASET_DIR, case, lazy=True)
        cam = tracking_camera(seq.camera, tcfg, device)
        app = [i for i, fm in enumerate(seq.meta["frames"]) if fm["phase"] == "approach"][:4]
        for fi in app:
            fr = seq.frames[fi]
            opt = _optimizer(ctx, tcfg)
            obs = _obs(fr.rgb.float().to(device), fr.depth.float().to(device), cam, ctx, tcfg, opt)
            if fi == app[0]:
                L = whole_image_curve(opt, obs, AXES["x"], DOSE_GRID)
                rows.append({"splat": name, "case": case, "frame": fi, "loss": "whole_image",
                             "argmin": _argmin(DOSE_GRID, L)})
            else:
                cv = _curves(opt, obs, AXES["x"], DOSE_GRID, with_depth=False)
                rows.append({"splat": name, "case": case, "frame": fi, "loss": "roi",
                             "argmin": _argmin(DOSE_GRID, cv["L"])})
        last = [r for r in rows if r["case"] == case]
        print(f"[f1] {name} {case}: " + " ".join(f"f{r['frame']}({r['loss'][:2]}) {r['argmin']:+.1f}" for r in last),
              flush=True)
    return rows


def main(device="cuda:0"):
    os.makedirs(OUT, exist_ok=True)
    rows = []
    ctx_o, _ = official_ctx(device)
    rows += run_splat("official", ctx_o, device)
    del ctx_o
    torch.cuda.empty_cache()
    cfg, model, dino, grouping = load_splat(0, device)
    rows += run_splat("ours_seed0", _ctx(cfg, model, dino, grouping, DRILL_ID), device)
    json.dump(rows, open(os.path.join(OUT, "f1.json"), "w"), indent=1)
    res = {}
    for sp in ("official", "ours_seed0"):
        a = [r["argmin"] for r in rows if r["splat"] == sp and r["loss"] == "whole_image"]
        b = [r["argmin"] for r in rows if r["splat"] == sp and r["loss"] == "roi"]
        res[sp] = (float(np.median(a)), float(np.median(b)), a, b)
    F1a = all(v[0] <= -5.0 for v in res.values())
    F1b = all(v[1] <= -10.0 for v in res.values())
    L = ["# F1: origin of the shared approach roll floor", "",
         "Pre-registered in `docs/note/f1_floor_origin_preregistration.md`. Static (no motion) has 0° here (the drill "
         "does not move in the approach); sweeps at the true pose. Rules first.", "",
         "| Rule | official splat | our seed-0 splat | Result |", "|---|---|---|---|",
         f"| F1a median whole-image argmin at frame 0 ≤ −5° | {res['official'][0]:+.1f}° | {res['ours_seed0'][0]:+.1f}° | "
         f"{'pass' if F1a else 'FAIL'} |",
         f"| F1b median ROI argmin at approach frames 1–3 ≤ −10° | {res['official'][1]:+.1f}° | {res['ours_seed0'][1]:+.1f}° | "
         f"{'pass' if F1b else 'FAIL'} |", "", "## Exploratory: per-frame medians", ""]
    for sp in ("official", "ours_seed0"):
        for f in (1, 2, 3):
            v = [r["argmin"] for r in rows if r["splat"] == sp and r["loss"] == "roi" and r["frame"] == f]
            if v:
                L.append(f"- {sp} frame {f}: median ROI argmin {np.median(v):+.1f}° (range {min(v):+.1f} to {max(v):+.1f})")
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"FINDING: F1a | whole-image init-loss roll argmin at frame 0: official {res['official'][0]:+.1f}, ours "
          f"{res['ours_seed0'][0]:+.1f} ({'pass' if F1a else 'FAIL'}) | {OUT}/summary.md")
    print(f"FINDING: F1b | ROI roll argmin at approach frames 1-3: official {res['official'][1]:+.1f}, ours "
          f"{res['ours_seed0'][1]:+.1f} ({'pass' if F1b else 'FAIL'}) | {OUT}/summary.md")


if __name__ == "__main__":
    main()
