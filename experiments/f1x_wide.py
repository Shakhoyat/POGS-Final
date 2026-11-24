"""F1x (exploratory, no pass rule): wide roll sweeps (±60° at 1°) of the ROI DINO loss, with and without depth, at the
true pose on approach frames 0-3 (identical across O4 seeds, so one sequence), official and our seed-0 splats.
    source tools/cuda_env.sh && .venv/bin/python -m experiments.f1x_wide"""
import json, os
import numpy as np, torch
from data.scene_generator.pick_place_benchmark import load_sequence
from data.scene_generator.tool_perturbation import DRILL_ID, TOOL_DATASET_DIR
from experiments.b_bias import _argmin, _obs, _optimizer
from experiments.g1_generality import AXES, _ctx, _curves, load_splat
from experiments.off_repro.swap_ours_on_official import official_ctx
from model.pogs.evaluate import tracking_camera, variant_config

OUT = "data_results/f1_floor"
GRID = np.arange(-60.0, 60.0 + 1e-9, 1.0)


def run(name, ctx, device="cuda:0"):
    tcfg = variant_config(ctx["cfg"].tracking, "POGS")
    seq = load_sequence(TOOL_DATASET_DIR, "grasp_+15_s0", lazy=True)
    cam = tracking_camera(seq.camera, tcfg, device)
    rows = []
    for fi in range(4):
        fr = seq.frames[fi]
        opt = _optimizer(ctx, tcfg)
        obs = _obs(fr.rgb.float().to(device), fr.depth.float().to(device), cam, ctx, tcfg, opt)
        cv = _curves(opt, obs, AXES["x"], GRID, with_depth=True)
        L, D = np.asarray(cv["L"]), np.asarray(cv["depth"])
        J = L + float(tcfg.depth_loss_mult) * D
        rows.append({"splat": name, "frame": fi, "L": L.tolist(), "J": J.tolist(),
                     "argmin_L": _argmin(GRID, L), "argmin_J": _argmin(GRID, J)})
        loc = [GRID[i] for i in range(1, len(L) - 1) if L[i] < L[i - 1] and L[i] < L[i + 1]]
        print(f"[f1x] {name} frame {fi}: argmin L {rows[-1]['argmin_L']:+.0f}  J {rows[-1]['argmin_J']:+.0f}  "
              f"local minima of L at {loc}  L(0)={L[60]:.4f} L(-20)={L[40]:.4f} L(-40)={L[20]:.4f}", flush=True)
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    c, _ = official_ctx()
    rows += run("official", c)
    del c; torch.cuda.empty_cache()
    cfg, model, dino, grouping = load_splat(0)
    rows += run("ours_seed0", _ctx(cfg, model, dino, grouping, DRILL_ID))
    json.dump({"grid_deg": GRID.tolist(), "rows": rows}, open(os.path.join(OUT, "f1x_wide.json"), "w"))


if __name__ == "__main__":
    main()
