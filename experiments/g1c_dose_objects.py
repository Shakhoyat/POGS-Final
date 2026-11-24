"""
G1c: shadow dose-response on the four primary objects without the gripper. Pre-registered in
`docs/note/g1c_dose_objects_preregistration.md` (C1, C2). Static, evaluation only.

    source tools/cuda_env.sh
    .venv/bin/python -m experiments.g1c_dose_objects run --seed 0
    .venv/bin/python -m experiments.g1c_dose_objects summarise
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from data.scene_generator.tool_perturbation import TOOL_OBJECT_NAMES, build_tool_scene
from experiments.b_bias import _argmin, _obs, _optimizer
from experiments.e4_torque import _render_s
from experiments.env_info import write_env_json
from experiments.g1_generality import AXES, DOSE_GRID, DOSE_LEVELS, PRIMARY, _ctx, _curves, load_splat
from model.pogs.checkpoint import write_json
from model.pogs.evaluate import tracking_camera, variant_config

OUT = "data_results/g1c_dose_objects"


def run(seed: int, device: str = "cuda:0") -> None:
    from data.scene_generator.pick_place_benchmark import BenchmarkConfig
    from data.scene_generator.pick_place_benchmark import tracking_camera as bench_cam
    os.makedirs(OUT, exist_ok=True)
    write_env_json(OUT)
    cfg, model, dino, grouping = load_splat(seed, device)
    tcfg = variant_config(cfg.tracking, "POGS")
    scene = build_tool_scene()
    cam = bench_cam(BenchmarkConfig(), device)
    name_to_id = {v: k for k, v in TOOL_OBJECT_NAMES.items()}
    frames = {s: _render_s(scene, cam, {}, True, s, device) for s in DOSE_LEVELS}
    rows = []
    for name in PRIMARY:
        ctx = _ctx(cfg, model, dino, grouping, name_to_id[name])
        if ctx is None:
            print(f"[g1c] s{seed} {name}: no group", flush=True)
            continue
        for s in DOSE_LEVELS:
            rgb, depth = frames[s]
            opt = _optimizer(ctx, tcfg)
            try:
                obs = _obs(rgb, depth, tracking_camera(cam, tcfg, device), ctx, tcfg, opt)
            except RuntimeError as e:
                print(f"[g1c] s{seed} {name} s={s}: skipped ({e})", flush=True)
                continue
            for ax in ("x", "y", "z"):
                cv = _curves(opt, obs, AXES[ax], DOSE_GRID, with_depth=False)
                r = {"seed": seed, "object": name, "shadow_strength": s, "axis": ax,
                     "argmin": {"L": _argmin(DOSE_GRID, cv["L"]), "bg_rendered": _argmin(DOSE_GRID, cv["bg_rendered"])},
                     "curves": {"L": cv["L"], "bg_rendered": cv["bg_rendered"]}}
                rows.append(r)
            print(f"[g1c] s{seed} {name:14s} s={s:<4g} argmin L x/y/z " + " ".join(
                f"{r['argmin']['L']:+.1f}" for r in rows[-3:]), flush=True)
    write_json(os.path.join(OUT, f"dose_objects_seed{seed}.json"), {"grid_deg": DOSE_GRID.tolist(), "rows": rows})


def summarise() -> None:
    from scipy.stats import spearmanr
    rows = []
    for p in sorted(glob.glob(f"{OUT}/dose_objects_seed*.json")):
        rows += json.load(open(p))["rows"]
    cases = {}
    for r in rows:
        cases.setdefault((r["seed"], r["object"], r["axis"]), []).append(r)
    stats = []
    for (sd, ob, ax), rs in sorted(cases.items()):
        rs = sorted(rs, key=lambda r: r["shadow_strength"])
        s = [r["shadow_strength"] for r in rs]
        a = [r["argmin"]["L"] for r in rs]
        b = [r["argmin"]["bg_rendered"] for r in rs]
        if len(rs) < len(DOSE_LEVELS):
            continue
        rho = float(spearmanr(s, a).statistic) if np.ptp(a) > 0 else 0.0
        a0, a1 = a[s.index(0.0)], a[s.index(1.0)]
        b0, b1 = b[s.index(0.0)], b[s.index(1.0)]
        stats.append(dict(seed=sd, object=ob, axis=ax, rho=rho, shift=a1 - a0, bg_shift=b1 - b0, argmins=a, bg=b,
                          meets=abs(rho) >= 0.8 and abs(a1 - a0) >= 2.0))
    per = {}
    for st in stats:
        per.setdefault((st["object"], st["seed"]), False)
        per[(st["object"], st["seed"])] |= st["meets"]
    obj_ok = {ob: sum(per.get((ob, sd), False) for sd in (0, 1, 2)) >= 2 for ob in PRIMARY}
    C1 = sum(obj_ok.values()) >= 3
    met = [st for st in stats if st["meets"]]
    c2_frac = (sum(abs(st["bg_shift"]) < 0.5 * abs(st["shift"]) for st in met) / len(met)) if met else float("nan")
    C2 = bool(met) and c2_frac >= 0.75
    L = ["# G1c: shadow dose-response on four objects (no gripper)", "",
         "Pre-registered in `docs/note/g1c_dose_objects_preregistration.md`. Static (no motion) is not applicable:"
         " no tracking is run. Rules first.", "", "## Pre-registered results", "",
         "| Rule | Statistic | Result |", "|---|---|---|",
         f"| C1 | objects with a qualifying axis on ≥ 2/3 splats: {sum(obj_ok.values())}/4 "
         f"({', '.join(o for o, k in obj_ok.items() if k) or 'none'}) | {'pass' if C1 else 'FAIL'} |",
         f"| C2 | rendered background moves < half the release shift in {c2_frac:.0%} of {len(met)} qualifying cases"
         f" (rule ≥ 75%) | {'pass' if C2 else ('FAIL' if met else 'not testable')} |", "",
         "## All cases", "", "| Splat | Object | Axis | ρ(s, argmin) | shift 0→1 (°) | bg shift (°) | argmins (s = "
         + ", ".join(f"{s:g}" for s in DOSE_LEVELS) + ") | qualifies |", "|---|---|---|---|---|---|---|---|"]
    L += [f"| {st['seed']} | {st['object']} | {st['axis']} | {st['rho']:+.2f} | {st['shift']:+.1f} | "
          f"{st['bg_shift']:+.1f} | {', '.join(f'{x:+.1f}' for x in st['argmins'])} | {'yes' if st['meets'] else ''} |"
          for st in stats]
    open(f"{OUT}/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L[:12]))
    print(f"FINDING: G1c-C1 | shadow dose-response without gripper: {sum(obj_ok.values())}/4 objects qualify on >= 2/3 "
          f"splats ({'pass' if C1 else 'FAIL'}) | {OUT}/summary.md")
    print(f"FINDING: G1c-C2 | rendered background moves < half in {c2_frac:.0%} of {len(met)} qualifying cases "
          f"({'pass' if C2 else 'FAIL or not testable'}) | {OUT}/summary.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "summarise"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.seed) if a.step == "run" else summarise()
