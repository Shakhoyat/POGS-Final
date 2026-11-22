"""
Plan rev. 2, E2 (+ GPU part of E1): does the tracker's own loss "see" roll about the drill-bit axis?

For O4 trials, runs plain POGS (paper variant) with an analysis-only probe on the motion-prior hook
(``observe``; no warm start, no fusion, so tracking is identical to POGS). After every frame's
optimisation, the probe computes the 6x6 Hessian of the tracker loss for the drill group
(``model.pogs.information.tracker_loss_hessian``) at the estimate and at the ground-truth pose
(GT used for evaluation only), plus the loss at both.

Outputs data_results/e2_info/{trials/<case>.npz, summary.md}:
  P1: Spearman(log roll_info_rel at the estimate, |roll error|) and the share of frames whose weakest
      rotation direction is within 15 deg of the bit axis (pass: rho <= -0.5 and share >= 70%).
  E1: share of frames with loss(estimate) < loss(GT): if most, the loss/features prefer the wrong pose
      (feature/loss mismatch); if loss(estimate) > loss(GT), the optimiser stops short.

    source tools/cuda_env.sh && .venv/bin/python -m experiments.e2_information --trials 10 [--shard k --nshards n]
Accuracy-only: may share the GPU, but never during a latency stage (o5, o6, realtime).
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
from scipy import stats

from data.scene_generator.pick_place_benchmark import load_sequence
from data.scene_generator.tool_perturbation import TOOL_DATASET_DIR, o4_cases
from experiments.env_info import write_env_json
from experiments.o3_perturbation import setup_tool_context
from metrics.pose import roll_swing_error_deg
from model.pogs.evaluate import gt_pose_for_group, track_sequence, variant_config
from model.pogs.information import observability_summary, pose_to_part_delta, tracker_loss_hessian

OUT = "data_results/e2_info"
COS15 = float(np.cos(np.radians(15.0)))
SCALE_ROT, SCALE_TRANS, N_FIT = float(np.radians(1.0)), 3e-3, 12
"""Regression Hessian at 1 deg / 3 mm (24 gradients). A central-difference Hessian was step-dependent on
the real loss (weak-axis alignment 0.25-0.87 across 0.1-2 deg steps; LEDGER 8.E2), so curvature is
measured at the scale of a typical per-frame motion instead."""


class InfoProbe:
    """Motion-prior-shaped analysis probe: leaves tracking untouched, records information."""
    warm_start = False
    fuse = False
    check_loss = False

    def __init__(self, seq, gt_id: int, centroid: np.ndarray, use_depth: bool):
        self.seq, self.gt_id, self.c0, self.use_depth = seq, gt_id, centroid, use_depth
        self.fi = 0
        self.rec = {k: [] for k in ("H_est", "H_gt", "loss_est", "loss_gt", "fit_resid")}
        self._last = None

    def initialize(self, posevecs):
        self._last = posevecs.detach().clone()

    def predict(self, device=None, dt=None):
        return self._last

    def update(self, posevecs, device=None):
        self._last = posevecs.detach().clone()
        return posevecs

    def observe(self, opt, obs):
        T_gt = gt_pose_for_group(self.seq, self.fi, self.gt_id, self.c0)
        base_gt = opt.part_deltas.detach().clone()
        base_gt[0] = pose_to_part_delta(T_gt, self.c0).to(base_gt)
        try:
            kw = dict(use_depth=self.use_depth, h_rot=SCALE_ROT, h_trans=SCALE_TRANS, method="regression", n=N_FIT)
            he = tracker_loss_hessian(opt, obs, 0, **kw)
            hg = tracker_loss_hessian(opt, obs, 0, base=base_gt, **kw)
            vals = (he["H"], hg["H"], he["loss"], hg["loss"], he["fit_resid"])
        except RuntimeError:
            vals = (np.full((6, 6), np.nan), np.full((6, 6), np.nan), np.nan, np.nan, np.nan)
        for k, v in zip(self.rec, vals):
            self.rec[k].append(v)
        self.fi += 1


def run(trials: int, shard: int, nshards: int, device: str = "cuda:0") -> None:
    os.makedirs(os.path.join(OUT, "trials"), exist_ok=True)
    if shard == 0:
        write_env_json(OUT)
    ctx = setup_tool_context(device)
    cfg, model, dino, groups, g2gt, mp = (ctx[k] for k in ("cfg", "model", "dino", "groups", "g2gt", "mp"))
    tcfg = variant_config(cfg.tracking, "POGS")
    c0 = groups.init_p2w[0, :3].detach().cpu().numpy()
    cases = [c for i, c in enumerate(o4_cases(range(trials))) if i % nshards == shard]
    for c in cases:
        fp = os.path.join(OUT, "trials", f"{c.name}.npz")
        if os.path.exists(fp):
            continue
        seq = load_sequence(TOOL_DATASET_DIR, c.name)
        probe = InfoProbe(seq, g2gt[0], c0, tcfg.use_depth)
        res = track_sequence(model, groups, seq, dino, cfg, g2gt, name="POGS", tracking_cfg=tcfg,
                             model_points=mp, motion_prior=probe, verbose=False)
        P, G = res.pred_poses[:, 0], res.gt_poses[:, 0]
        rs = np.array([roll_swing_error_deg(P[f], G[f], ctx["bit_axis_local"]) for f in range(len(P))])
        np.savez_compressed(fp, phase=np.array([fm["phase"] for fm in seq.meta["frames"]]),
                            angle=c.angle, seed=c.seed, pred_T=P, gt_T=G, roll=rs[:, 0], swing=rs[:, 1],
                            rot=res.rot_err_deg[:, 0], **{k: np.asarray(v) for k, v in probe.rec.items()},
                            bit_axis=ctx["bit_axis_local"])
        print(f"[e2] {c.name} done", flush=True)


def summarise(radius_m: float = 0.10) -> None:
    files = sorted(glob.glob(os.path.join(OUT, "trials", "*.npz")))
    if not files:
        print("[e2] no trials")
        return
    rows = []
    for f in files:
        d = np.load(f)
        for i in range(len(d["roll"])):
            if not np.isfinite(d["H_est"][i]).all():
                continue
            se = observability_summary(d["H_est"][i], radius_m, d["bit_axis"])
            sg = observability_summary(d["H_gt"][i], radius_m, d["bit_axis"])
            rows.append((str(d["phase"][i]), float(d["angle"]), abs(d["roll"][i]), d["swing"][i],
                         se["roll_info_rel"], sg["roll_info_rel"], se["weak_rot_axis_cos"], sg["weak_rot_axis_cos"],
                         float(d["loss_est"][i]), float(d["loss_gt"][i]), se["axis_in_weak2"], sg["axis_in_weak2"],
                         float(d["fit_resid"][i])))
    a = np.array([r[1:] for r in rows], dtype=np.float64)
    phase = np.array([r[0] for r in rows])
    roll, rel_e, rel_g, cos_e, cos_g, le, lg = a[:, 1], a[:, 3], a[:, 4], a[:, 5], a[:, 6], a[:, 7], a[:, 8]
    w2_e, w2_g, fres = a[:, 9], a[:, 10], a[:, 11]
    rho, p = stats.spearmanr(np.log(np.maximum(rel_e, 1e-12)), roll)
    rho_g, p_g = stats.spearmanr(np.log(np.maximum(rel_g, 1e-12)), roll)
    L = ["# E2: information of the POGS loss about drill-bit roll (evaluation only)", "",
         f"{len(files)} trials, {len(rows)} frames (POGS paper variant; probe does not alter tracking). "
         f"Rotation scaled by radius {radius_m} m.", "",
         "| Quantity | At estimate | At ground truth |", "|---|---|---|",
         f"| Spearman(log roll information, abs roll error) [P1: <= -0.5] | {rho:.2f} (p={p:.1e}) | {rho_g:.2f} (p={p_g:.1e}) |",
         f"| Weakest rotation direction within 15° of bit axis [P1: >= 70%] | {np.mean(cos_e > COS15):.0%} | {np.mean(cos_g > COS15):.0%} |",
         f"| Median roll information / strongest rotation information | {np.median(rel_e):.3g} | {np.median(rel_g):.3g} |",
         f"| Median share of bit axis in the 2-D weak rotation subspace (1 = fully weak) | {np.median(w2_e):.2f} | {np.median(w2_g):.2f} |",
         f"| Median quadratic-fit residual of the loss (1°/3 mm) | {np.median(fres):.2f} | – |",
         "", "## E1 (GPU part): loss at the estimate vs at the ground truth", "",
         "| Phase | n | loss(est) < loss(GT) | median loss(est) − loss(GT) |", "|---|---|---|---|"]
    for ph in ["approach", "lift", "rest_x", "perturbed"]:
        m = phase == ph
        if m.any():
            L.append(f"| {ph} | {m.sum()} | {np.mean(le[m] < lg[m]):.0%} | {np.median(le[m] - lg[m]):.4f} |")
    L += ["", "Reading: loss(est) < loss(GT) on most frames means the loss itself prefers the wrong pose "
          "(features / depth cannot separate roll) — not an optimiser failure. P1 verdict: "
          + ("PASS" if (rho <= -0.5 and np.mean(cos_e > COS15) >= 0.7) else "FAIL") + " (at the estimate).",
          "Static baseline: not applicable (information of the loss, not a tracker comparison)."]
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args()
    if a.summarise:
        summarise()
    else:
        run(a.trials, a.shard, a.nshards)
        if a.nshards == 1:
            summarise()
