"""8.OFF OR4: the OFFICIAL tracking loop on the O4 approach (+ lift) frames (official venv).
Pre-registered in `docs/note/off_or4_tracking_preregistration.md`.

    ~/venvs/pogs-official/bin/python -m experiments.off_repro.official_track run --config <config.yml>
    ~/venvs/pogs-official/bin/python -m experiments.off_repro.official_track summarise

Loop as `track_main_online_demo.py`: set_frame(Frame) on frame 0 -> initialize_obj_pose() (release defaults)
-> per later frame set_observation(PosedObservation) + step(niter=25) (no per-frame depth, release).
Ground truth is used only to score (roll/swing about the drill-bit axis = object x).
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from experiments.off_repro import shims  # noqa: E402,F401

import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

wp.init()  # as track_main_online_demo.py

from experiments.off_repro.official_sweeps import FR, OUT, ns_camera, setup  # noqa: E402

SEQ = os.path.expanduser("~/pogs_external/off/sequences")
PROBE = os.environ.get("OFF_PROBE", "0") == "1"


def _swing_twist(R_err, axis):
    """Signed twist about `axis` and swing of the axis (deg) for a rotation matrix (as metrics.pose)."""
    from scipy.spatial.transform import Rotation
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    q = Rotation.from_matrix(R_err).as_quat()          # x y z w
    v, w = q[:3], q[3]
    p = np.dot(v, a) * a
    twist = np.r_[p, w]
    n = np.linalg.norm(twist)
    if n < 1e-12:
        twist_angle = 0.0
    else:
        twist = twist / n
        twist_angle = 2 * np.degrees(np.arctan2(np.dot(twist[:3], a), twist[3]))
        twist_angle = (twist_angle + 180) % 360 - 180
    swing_axis = R_err @ a
    swing = np.degrees(np.arccos(np.clip(np.dot(swing_axis, a), -1, 1)))
    return float(twist_angle), float(swing)


def _pose_from_deltas(d, c0):
    from scipy.spatial.transform import Rotation
    d = d.detach().cpu().numpy().astype(float)
    T = np.eye(4)
    q = d[3:] / np.linalg.norm(d[3:])                  # wxyz
    T[:3, :3] = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    T[:3, 3] = c0 + d[:3]
    return T


def run(config: str) -> None:
    from pogs.tracking.rigid_group_optimizer import RigidGroupOptimizer, RigidGroupOptimizerConfig
    from pogs.tracking.observation import Frame, PosedObservation
    os.makedirs(OUT, exist_ok=True)
    os.chdir(os.path.expanduser("~/pogs_external/off"))
    idx = json.load(open(os.path.join(FR, "index.json")))
    pipeline, opt0, info = setup(config, idx["click_xyz"], idx["table_z"])
    model = pipeline.model
    dino_fn = pipeline.datamanager.dino_dataloader.get_pca_feats
    c0 = np.asarray(info["centroid"], float)
    axis = np.r_[1.0, 0, 0]
    out = []
    for p in sorted(glob.glob(os.path.join(SEQ, "grasp_*.npz"))):
        z = np.load(p)
        name = os.path.splitext(os.path.basename(p))[0]
        cam = ns_camera(z["c2w_cv"], z["K"], z["width"], z["height"], info["dataset_scale"])
        opt = RigidGroupOptimizer(RigidGroupOptimizerConfig(), model, group_masks=opt0.group_masks,
                                  group_labels=opt0.group_labels, dataset_scale=info["dataset_scale"])
        rows = []
        for k in range(len(z["frame"])):
            rgb = torch.from_numpy(z["rgb"][k]).float().cuda() / 255.0
            depth = torch.from_numpy(z["depth"][k]).float().cuda()          # (H, W): official Frame adds the channel
            if k == 0:
                opt.set_frame(Frame(rgb=rgb, camera=cam.to("cuda"), dino_fn=dino_fn, metric_depth_img=depth))
                opt.initialize_obj_pose(niter=100, n_seeds=7)
            else:
                opt.set_observation(PosedObservation(rgb=rgb, camera=cam, dino_fn=dino_fn, metric_depth_img=depth))
                opt.step(niter=25)
            P = _pose_from_deltas(opt.part_deltas[0], c0)
            Tg = z["gt_delta"][k]
            G = np.eye(4)
            G[:3, :3] = Tg[:3, :3]
            G[:3, 3] = Tg[:3, :3] @ c0 + Tg[:3, 3]
            roll, swing = _swing_twist(G[:3, :3].T @ P[:3, :3], axis)
            row = dict(frame=int(z["frame"][k]), phase=str(z["phase"][k]), roll=roll, swing=swing,
                       trans_err_cm=float(np.linalg.norm(P[:3, 3] - G[:3, 3]) * 100))
            if PROBE and k > 0:          # 8.OFF-P (exploratory): official loss at the estimate vs at the GT pose
                from scipy.spatial.transform import Rotation
                qg = Rotation.from_matrix(G[:3, :3]).as_quat()            # xyzw
                dg = torch.tensor([[*(G[:3, 3] - c0), qg[3], qg[0], qg[1], qg[2]]], dtype=torch.float32, device="cuda")
                with torch.no_grad():
                    le = opt.get_optim_loss(opt.frame, opt.part_deltas.detach().clone(), True, False, False, False,
                                            False, False, True)[0]
                    lg = opt.get_optim_loss(opt.frame, dg, True, False, False, False, False, False, True)[0]
                row.update(loss_est=float(le), loss_gt=float(lg))
            rows.append(row)
        last_app = [r for r in rows if r["phase"] == "approach"][-1]
        print(f"[or4] {name}: last approach frame {last_app['frame']} roll {last_app['roll']:+.1f} swing "
              f"{last_app['swing']:.1f} trans {last_app['trans_err_cm']:.2f} cm", flush=True)
        out.append(dict(case=name, rows=rows))
        opt.reset_transforms()
    json.dump({"setup": info, "config": config, "cases": out},
              open(os.path.join(OUT, "official_track_probe.json" if PROBE else "official_track.json"), "w"))


def summarise() -> None:
    d = json.load(open(os.path.join(OUT, "official_track.json")))
    last = [[r for r in c["rows"] if r["phase"] == "approach"][-1] for c in d["cases"]]
    med = float(np.median([abs(r["roll"]) for r in last]))
    OR4 = med >= 5.0
    lift = [r for c in d["cases"] for r in c["rows"] if r["phase"] == "lift"]
    L = ["# 8.OFF OR4: official tracking loop, roll before contact", "",
         "Pre-registered in `docs/note/off_or4_tracking_preregistration.md`. Static (no motion) has 0° roll and 0 cm "
         "error in the approach (the drill does not move).", "",
         "| Rule | Statistic | Threshold | Result |", "|---|---|---|---|",
         f"| OR4 | median abs(roll) at the last approach frame = {med:.1f}° (n {len(last)}) | ≥ 5° | "
         f"{'pass' if OR4 else 'FAIL'} |", "", "## Exploratory", "",
         f"- Signed roll at last approach frame: median {np.median([r['roll'] for r in last]):+.1f}°, "
         f"negative in {sum(r['roll'] < 0 for r in last)}/{len(last)}.",
         f"- Swing at last approach frame: median {np.median([r['swing'] for r in last]):.1f}°; translation error "
         f"median {np.median([r['trans_err_cm'] for r in last]):.2f} cm.",
         f"- Lift frames: median abs(roll) {np.median([abs(r['roll']) for r in lift]):.1f}°, median translation error "
         f"{np.median([r['trans_err_cm'] for r in lift]):.2f} cm." if lift else "- no lift frames", "",
         "| Case | frame | roll (°) | swing (°) | trans (cm) |", "|---|---|---|---|---|"]
    L += [f"| {c['case']} | {r['frame']} | {r['roll']:+.1f} | {r['swing']:.1f} | {r['trans_err_cm']:.2f} |"
          for c, r in zip(d["cases"], last)]
    open(os.path.join(OUT, "summary_or4.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:12]))
    print(f"FINDING: OR4 | official tracker median abs roll at end of approach {med:.1f} deg over {len(last)} sequences "
          f"({'pass' if OR4 else 'FAIL'}, rule >= 5) | data_results/off_repro/summary_or4.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "summarise"])
    ap.add_argument("--config")
    a = ap.parse_args()
    run(a.config) if a.step == "run" else summarise()
