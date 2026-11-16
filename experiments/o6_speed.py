"""
PLAN Phase 4 (O6): can POGS track at >= 5 Hz, and what does it cost in accuracy?

4.1  Profile: torch.profiler over a few tracking steps of the base case -> top CUDA ops
     (data_results/o6_speed/profile.txt) plus mean DINO / optimiser ms from the runs.
4.2  Candidates on the base condition, 3 seeds, each as plain POGS AND with the Kalman
     prior (+check) at the SAME iteration count, so a speed claim always has an equally
     cheap baseline (Claude.md "Rules"). Settings change only TrackingConfig fields passed
     to the shared ``track_sequence``; the loop is not copied.
4.3  Summary table: ADD-S success / lost / median error / Hz / meets 5 Hz, mean +- seed std.
     Hz = 1 / mean(DINO + optimiser time per frame) on this machine (see env.json).

    source tools/cuda_env.sh && .venv/bin/python -m experiments.o6_speed
"""

from __future__ import annotations

import copy
import json
import os

import numpy as np
import pandas as pd
import torch

from experiments import pipeline
from experiments.env_info import write_env_json
from experiments.pipeline import DATASET_DIR, _slug, build_groups, lost_rate, train_or_load
from experiments.v1_base_pogs.config import get_v1_config
from experiments.v2_kalman_pogs.config import _kf
from data.scene_generator.pick_place_benchmark import N_SEEDS, Case, load_case
from metrics.pose import adds_success_rate, object_diameter
from metrics.stats import mean_std
from model.pogs.checkpoint import write_json
from model.pogs.evaluate import object_model_points, static_baseline, track_sequence
from model.pogs.grouping import subset_groups
from experiments.pipeline import _case_groups

OUT = "data_results/o6_speed"
TARGET_HZ = 5.0
# (label, track_niter, match_resolution short side px)
CANDIDATES = [
    ("25 it @500 (base)", 25, 500), ("15 it @500", 15, 500), ("10 it @500", 10, 500), ("5 it @500", 5, 500),
    ("10 it @350", 10, 350), ("5 it @350", 5, 350), ("10 it @250", 10, 250), ("5 it @250", 5, 250),
]
PRIORS = [("POGS", None), ("POGS + KF + check", _kf(True))]


def profile_one_step(model, groups, seq, dino, cfg, g2gt, mp, n_frames: int = 6) -> str:
    """Op-level CUDA profile of the first ``n_frames`` tracked frames of ``seq``."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        track_sequence(model, groups, seq, dino, cfg, g2gt, name="profile", model_points=mp,
                       max_frames=n_frames, verbose=False)
    return prof.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=60)


def main(device: str = "cuda:0") -> pd.DataFrame:
    os.makedirs(OUT, exist_ok=True)
    write_env_json(OUT)
    cfg = get_v1_config(device)
    model, capture, dino, clip = train_or_load(cfg)
    grouping = build_groups(model, capture, cfg, clip=clip)
    mp_all = object_model_points(grouping.groups, model)
    cases = [Case("base", s) for s in range(N_SEEDS)]

    rows = []
    for ci, case in enumerate(cases):
        seq = load_case(DATASET_DIR, case)
        _, moving, _ = _case_groups(seq, grouping.groups, grouping.group_to_gt)
        groups = subset_groups(grouping.groups, moving)
        g2gt = {i: grouping.group_to_gt[g] for i, g in enumerate(moving)}
        mp = {i: mp_all[g] for i, g in enumerate(moving)}
        if ci == 0 and not os.path.exists(os.path.join(OUT, "profile.txt")):
            txt = profile_one_step(model, groups, seq, dino, cfg, g2gt, mp)
            open(os.path.join(OUT, "profile.txt"), "w").write(txt)
        rs = static_baseline(groups, seq, g2gt, mp)
        rows.append({"case": case.name, "seed": case.seed, "prior": "Static (no motion)", "setting": "-",
                     "ckpt_id": pipeline.CKPT_ID, "hz": float("nan"),
                     "adds_success": adds_success_rate(rs.adds[:, 0], object_diameter(mp[0])),
                     "lost_rate": lost_rate(rs.trans_err_cm[:, 0]),
                     "median_pos_err_cm": float(np.nanmedian(rs.trans_err_cm[:, 0]))})
        for label, niter, res in CANDIDATES:
            for pname, factory in PRIORS:
                name = f"{pname} | {label}"
                run_dir = os.path.join(OUT, "runs", case.name, _slug(name))
                rp = os.path.join(run_dir, "row.json")
                if os.path.exists(rp):
                    row = json.load(open(rp))
                    if row.get("ckpt_id") == pipeline.CKPT_ID:
                        rows.append(row)
                        continue
                tc = copy.deepcopy(cfg.tracking)
                tc.track_niter, tc.match_resolution = niter, res
                prior = factory(groups.num_groups, 1.0 / seq.fps) if factory else None
                r = track_sequence(model, groups, seq, dino, cfg, g2gt, name=name, tracking_cfg=tc,
                                   model_points=mp, motion_prior=prior, verbose=False)
                ms = float(np.nanmean(r.dino_ms + r.opt_ms))
                row = {"case": case.name, "seed": case.seed, "prior": pname, "setting": label, "niter": niter,
                       "match_res": res, "ckpt_id": pipeline.CKPT_ID, "ms_per_frame": ms, "hz": 1000.0 / ms,
                       "dino_ms": float(np.nanmean(r.dino_ms)), "opt_ms": float(np.nanmean(r.opt_ms)),
                       "adds_success": adds_success_rate(r.adds[:, 0], object_diameter(mp[0])),
                       "lost_rate": lost_rate(r.trans_err_cm[:, 0]),
                       "median_pos_err_cm": float(np.nanmedian(r.trans_err_cm[:, 0])),
                       "mean_rot_err_deg": float(np.nanmean(r.rot_err_deg[:, 0]))}
                os.makedirs(run_dir, exist_ok=True)
                write_json(rp, row)
                rows.append(row)
                print(f"[o6] {case.name} {name}: {row['hz']:.1f} Hz, ADD-S {row['adds_success']:.0%}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    write_summary(df)
    return df


def write_summary(df: pd.DataFrame) -> None:
    lines = ["| Prior | Setting | Hz | ADD-S success | Lost | Median pos (cm) | Meets 5 Hz |",
             "|---|---|---|---|---|---|---|"]
    for (prior, setting), g in df.groupby(["prior", "setting"], sort=False):
        hz, hzs = mean_std(g.hz)
        a, As = mean_std(g.adds_success)
        lo, los = mean_std(g.lost_rate)
        m, ms_ = mean_std(g.median_pos_err_cm)
        lines.append(f"| {prior} | {setting} | {'-' if np.isnan(hz) else f'{hz:.1f} ± {hzs:.1f}'} | {a:.0%} ± {As:.0%} | {lo:.0%} ± {los:.0%} | "
                     f"{m:.2f} ± {ms_:.2f} | {'n/a' if np.isnan(hz) else ('yes' if hz >= TARGET_HZ else 'no')} |")
    lines.append(f"\nn = {df.seed.nunique()} seeds per row (base condition, 5 Hz camera). Hz = 1 / mean(DINO + optimiser) "
                 "per tracked frame; hardware in env.json.")
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
