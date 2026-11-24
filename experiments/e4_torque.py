"""
E4: does the boundary torque of the implicit silhouette term predict the roll bias? Pre-registered in
`docs/note/e4_boundary_torque_preregistration.md` (E4a-c). Static, evaluation only: ground truth places the sweep.

    source tools/cuda_env.sh
    .venv/bin/python -m experiments.e4_torque run --seed 0      # 20 frames per splat (seeds 0, 1, 2)
    .venv/bin/python -m experiments.e4_torque summarise
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np
import torch

from data.scene_generator.tool_perturbation import DRILL_ID, build_tool_scene
from experiments.b_bias import _argmin, _obs, _optimizer
from experiments.env_info import write_env_json
from experiments.g1_generality import (AXES, DOSE_LEVELS, _ctx, _gripper_frame0, _kappa, _ring_cam, load_splat)
from model.pogs import losses as LL
from model.pogs.checkpoint import write_json
from model.pogs.evaluate import tracking_camera, variant_config
from model.pogs.information import perturbed_deltas

OUT = "data_results/e4_torque"
GRID = np.arange(-20.0, 20.0 + 1e-9, 0.5)
DELTA = 1.0  # deg, central difference for T


def _render_s(objs, cam, tr, shadows, s, device):
    from data.scene_generator.photoreal_scene import render_photoreal_scene
    from data.scene_generator.scene_builder import FlatScene
    o = render_photoreal_scene(FlatScene(objs, device=device).clone_with_transforms(tr), cam,
                               cast_shadows=shadows, shadow_strength=s)
    return o["rgb"].float(), o["depth"].float()


def _measure(opt, obs, axis6):
    """L on GRID (release DINO term, as G1 dose) and the torque quantities at the true pose."""
    rf = obs.roi_frames[0]
    real = rf.dino_feats.to(opt.init_means.device)
    base = opt.part_deltas.detach().clone()
    k = opt.config.blur_kernel_size
    g = real.norm(dim=-1)
    valid = torch.isfinite(g)
    n = valid.sum().clamp(min=1)
    gmean = g[valid].mean()

    def render(deg):
        pd = perturbed_deltas(base, 0, torch.as_tensor(axis6 * deg * math.pi / 180.0, dtype=torch.float64))
        return opt._render(rf.camera, pd, 0, True)

    L, e_out_terms = [], {}
    with torch.no_grad():
        for d in GRID:
            o = render(d)
            e = (real - LL.gaussian_blur2d(o["dino"], k)).norm(dim=-1)
            v = torch.isfinite(e)
            L.append(float(e[v].sum() / v.sum().clamp(min=1)))
        B = {}
        for d in (-DELTA, DELTA):
            o = render(d)
            a = o["accumulation"].squeeze(-1).clamp(0, 1)
            e = (real - LL.gaussian_blur2d(o["dino"], k)).norm(dim=-1)
            B[d] = dict(g=float(((1 - a) * g)[valid].sum() / n), c=float(((1 - a) * gmean)[valid].sum() / n),
                        e=float(((1 - a) * e)[valid & torch.isfinite(e)].sum() / n))
    T = {key: (B[DELTA][key] - B[-DELTA][key]) / (2 * DELTA) for key in ("g", "c", "e")}
    kap = _kappa(GRID, L, 5.0)
    pred = {key: (-T[key] / (2 * kap) if kap > 0 else float("nan")) for key in T}
    return dict(L=L, argmin=_argmin(GRID, L), kappa=kap, T=T, pred=pred, g_mean=float(gmean))


def run(seed: int, device: str = "cuda:0") -> None:
    from data.scene_generator.pick_place_benchmark import BenchmarkConfig
    from data.scene_generator.pick_place_benchmark import tracking_camera as bench_cam
    os.makedirs(OUT, exist_ok=True)
    write_env_json(OUT)
    cfg, model, dino, grouping = load_splat(seed, device)
    tcfg = variant_config(cfg.tracking, "POGS")
    ctx = _ctx(cfg, model, dino, grouping, DRILL_ID)
    bcfg = BenchmarkConfig()
    scene = build_tool_scene()
    grip, grip_tr = _gripper_frame0(scene)
    track = bench_cam(bcfg, device)
    drill = next(o for o in scene if o.obj_id == DRILL_ID)
    target = np.asarray(drill.position, float) + np.array([0.0, 0.0, 0.05])
    conds = [(f"gripper s={s:g}", scene + [grip], grip_tr, track, True, s, "dose") for s in DOSE_LEVELS]
    conds += [("track", scene, {}, track, True, 1.0, "track"), ("track, no shadows", scene, {}, track, False, 1.0,
                                                                  "track")]
    conds += [(f"heldout az{az:03d}", scene, {}, _ring_cam(target, az, bcfg, device), True, 1.0, "ring")
              for az in range(0, 360, 30)]
    rows = []
    for label, objs, tr, cam, sh, s, kind in conds:
        rgb, depth = _render_s(objs, cam, tr, sh, s, device)
        opt = _optimizer(ctx, tcfg)
        try:
            obs = _obs(rgb, depth, tracking_camera(cam, tcfg, device), ctx, tcfg, opt)
        except RuntimeError as e:
            print(f"[e4] s{seed} {label}: skipped ({e})", flush=True)
            continue
        for ax in ("x", "y", "z"):          # x = roll (pre-registered); y, z exploratory
            m = _measure(opt, obs, AXES[ax])
            rows.append({"seed": seed, "view": label, "kind": kind, "shadow_strength": s, "shadows": sh,
                         "axis": ax, **m})
            if ax == "x":
                print(f"[e4] s{seed} {label:20s} argmin {m['argmin']:+.1f}  pred_g {m['pred']['g']:+.1f}  "
                      f"pred_c {m['pred']['c']:+.1f}  pred_e {m['pred']['e']:+.1f}  kappa {m['kappa']:.2e}",
                      flush=True)
    write_json(os.path.join(OUT, f"torque_seed{seed}.json"), {"grid_deg": GRID.tolist(), "rows": rows})


def _spearman(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[ok], b[ok]).statistic) if ok.sum() >= 3 else float("nan"), int(ok.sum())


def summarise() -> None:
    rows = []
    for p in sorted(glob.glob(f"{OUT}/torque_seed*.json")):
        rows += json.load(open(p))["rows"]
    roll = [r for r in rows if r["axis"] == "x"]
    a = [r["argmin"] for r in roll]
    pg = [r["pred"]["g"] for r in roll]
    pc = [r["pred"]["c"] for r in roll]
    pe = [r["pred"]["e"] for r in roll]
    rg, n = _spearman(pg, a)
    rc, _ = _spearman(pc, a)
    big = [(x, y) for x, y in zip(pg, a) if abs(y) >= 2 and np.isfinite(x)]
    sign = sum(np.sign(x) == np.sign(y) for x, y in big) / max(len(big), 1)
    E4a, E4b, E4c = rg >= 0.5, sign >= 0.75, (rg - rc) >= 0.2
    L = ["# E4: boundary torque vs. roll bias (static)", "",
         "Pre-registered in `docs/note/e4_boundary_torque_preregistration.md`. Static (no motion) is not applicable:"
         " no tracking is run. Rules first, then exploratory.", "",
         f"Conditions (roll): {len(roll)} (expected 60); finite predictions: {n}.", "",
         "## Pre-registered results", "",
         "| Rule | Statistic | Threshold | Result |", "|---|---|---|---|",
         f"| E4a | Spearman(â_g, a) = {rg:+.2f} | ≥ 0.5 | {'pass' if E4a else 'FAIL'} |",
         f"| E4b | sign agreement {sign:.0%} over {len(big)} conditions with abs(a) ≥ 2° | ≥ 75% | "
         f"{'pass' if E4b else 'FAIL'} |",
         f"| E4c | Spearman gain over constant-g predictor {rg - rc:+.2f} (â_c: {rc:+.2f}) | ≥ 0.2 | "
         f"{'pass' if E4c else 'FAIL'} |", "",
         "## Exploratory", ""]
    re_, _ = _spearman(pe, a)
    L.append(f"- Predictor with the exact outside residual e(p): Spearman {re_:+.2f}.")
    for ax in ("y", "z"):
        sub = [r for r in rows if r["axis"] == ax]
        if sub:
            r_, n_ = _spearman([r["pred"]["g"] for r in sub], [r["argmin"] for r in sub])
            L.append(f"- Axis {ax}: Spearman(â_g, a) {r_:+.2f} (n {n_}).")
    for sd in sorted({r["seed"] for r in roll}):
        d = [r for r in roll if r["seed"] == sd and r["kind"] == "dose"]
        if d:
            r_, _ = _spearman([r["shadow_strength"] for r in d], [r["T"]["g"] for r in d])
            L.append(f"- Seed {sd} dose rows: Spearman(s, T_g) {r_:+.2f}; argmin "
                     + ", ".join(f"{r['argmin']:+.1f}" for r in d) + "; â_g "
                     + ", ".join(f"{r['pred']['g']:+.1f}" for r in d))
    L += ["", "## Rows (roll)", "", "| Seed | View | a (deg) | â_g | â_c | â_e | κ |", "|---|---|---|---|---|---|---|"]
    L += [f"| {r['seed']} | {r['view']} | {r['argmin']:+.1f} | {r['pred']['g']:+.1f} | {r['pred']['c']:+.1f} | "
          f"{r['pred']['e']:+.1f} | {r['kappa']:.2e} |" for r in roll]
    open(f"{OUT}/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L[:16]))
    print(f"FINDING: E4a | boundary-torque predictor vs measured roll argmin: Spearman {rg:+.2f} over {n} "
          f"conditions ({'pass' if E4a else 'FAIL'}, rule >= 0.5) | {OUT}/summary.md")
    print(f"FINDING: E4b | sign agreement {sign:.0%} over {len(big)} conditions with abs(a)>=2 deg "
          f"({'pass' if E4b else 'FAIL'}, rule >= 75%) | {OUT}/summary.md")
    print(f"FINDING: E4c | spatial-asymmetry gain over constant-g predictor {rg - rc:+.2f} "
          f"({'pass' if E4c else 'FAIL'}, rule >= 0.2) | {OUT}/summary.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "summarise"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.seed) if a.step == "run" else summarise()
