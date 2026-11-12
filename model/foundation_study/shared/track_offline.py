"""Offline tracking on generated sequences, with no viewer and no ZED.

``pogs/scripts/track_main_demo.py`` cannot be pointed at our data as-is:

* it hardcodes ``world_to_extrinsic_zed.tf`` and a ZED calibration matrix
* it opens a ``viser`` server and a nerfstudio ``Viewer``
* its clustering step blocks on ``input()`` for a human crop

All three are interactive or hardware-specific. This runs the same
``Optimizer`` headlessly, which is what a repeatable n=1 measurement needs --
the whole point of fixing seed 42 was that a human pressing enter at a different
moment would make the number unreproducible.

Pipeline, all from POGS' own code:

    eval_setup(config)  ->  POGS model trained on the capture
    model.cluster(eps)  ->  HDBSCAN labels over all gaussians   (no crop needed)
    keep_inds           ->  the mug, identified by projecting the
                            generator's GT obj_id mask onto frame 0
    Optimizer(...)      ->  init_obj_pose(), then step_opt() per frame

Why keep_inds is not taken from the largest cluster: the mug is ~3% of frame
and the gripper is a separate instance the study is deliberately *not* tracking,
so "largest cluster" would frequently select the gripper instead. The ground
truth mask is exact here, which is the one thing synthetic data gives us over
the authors' capture.

Use it on the GPU host, after a completed ``ns-train pogs``::

    python track_offline.py --row 00_baseline_detic_dinov2_clip \
        --cluster-eps 0.1 --opt-iters 15
"""

from __future__ import annotations

import argparse
import functools
import glob
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_config as cfg  # noqa: E402


def find_config(run_dir: str) -> str | None:
    """The nerfstudio ``config.yml`` a finished training run leaves behind.

    Training writes ``<output-dir>/<timestamp>/config.yml`` and the checkpoint
    under a ``nerfstudio_models/`` sibling subdirectory (verified on
    nerfstudio 1.1.1: ``step-000003999.ckpt``). Older builds wrote
    ``step-*-train.ckpt`` beside the config. Both layouts are accepted, but a
    bare glob for ``config.yml`` must not match a run that never finished --
    hence requiring a checkpoint to exist.
    """
    pats = (os.path.join(run_dir, "**", "config.yml"),
            os.path.join(run_dir, "config.yml"))
    # The newest finished run: sorted() put the oldest timestamp first, so a
    # retrain into the same train/ was silently ignored and the old run tracked.
    done = []
    for pat in pats:
        for p in glob.glob(pat, recursive=True):
            base = os.path.dirname(p)
            if (glob.glob(os.path.join(base, "nerfstudio_models", "step-*.ckpt"))
                    or glob.glob(os.path.join(base, "step-*.ckpt"))):
                done.append(p)
    return max(done, key=os.path.getmtime) if done else None


def load_sequence(tracking_dir: str):
    """``left/`` + ``depth/``, the intrinsics, and the camera pose.

    The tracking folder has no ``transforms.json`` -- that lives in ``capture/``,
    describing the 36-view ring rather than this one fixed camera. What the
    tracking sequence carries instead is ``gt_poses.json``, with the same
    ``camera_intrinsics`` dict and ``camera_opengl_c2w``.
    """
    images = sorted(glob.glob(os.path.join(tracking_dir, "left", "*.png")))
    depths = sorted(glob.glob(os.path.join(tracking_dir, "depth", "*.npy")))
    if not images or len(images) != len(depths):
        raise SystemExit(
            f"expected matching left/ and depth/ under {tracking_dir}, "
            f"got {len(images)} and {len(depths)}"
        )

    with open(os.path.join(tracking_dir, "gt_poses.json")) as fh:
        gt = json.load(fh)
    intr = gt["camera_intrinsics"]
    K = np.array([[intr["fl_x"], 0.0, intr["cx"]],
                  [0.0, intr["fl_y"], intr["cy"]],
                  [0.0, 0.0, 1.0]], dtype=np.float64)

    # OpenGL to OpenCV, same rebase the visualiser uses: OpenGL cameras look
    # along -Z with +Y up, OpenCV along +Z with +Y down. Flipping the rotation's
    # two right-hand columns is what converts the convention; the translation
    # column is already the camera centre in world coordinates and is shared.
    c2w = np.asarray(gt["camera_opengl_c2w"], dtype=np.float64) @ np.diag(
        [1.0, -1.0, -1.0, 1.0]
    )
    return images, depths, K, int(intr["w"]), int(intr["h"]), c2w, gt


def mug_gaussians(model, K, width, height, mask_path: str, depth_path: str,
                  device: str, c2w: np.ndarray, dataset_scale: float,
                  margin: float = 0.02, max_px: int = 40000):
    """The gaussians of the object the ground-truth mask covers in frame 0.

    Stands in for the manual crop box the authors drag out in the viewer.

    1. *Seeds*: the mask's pixels are back-projected at their own depth, moved
       into the model's frame with the tracking camera pose, and every gaussian
       whose mean lies within 3 mm of one of those surface points is a seed.
       Seeds only cover the side of the object the camera sees.
    2. *Crop box*: the box around the seeds, grown by ``margin`` and pushed
       back along the viewing direction by the object's own width, so the far
       side (never visible, but part of the object) is inside it.

    Returns ``(seeds, in_box)`` as index tensors; the caller intersects
    ``in_box`` with the seeds' cluster(s), the way the authors' crop is
    followed by their instance-feature clustering.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise SystemExit(f"no ground-truth mask at {mask_path}")
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    fg = np.argwhere(mask > 127)
    if len(fg) > max_px:
        fg = fg[np.random.default_rng(0).choice(len(fg), max_px, replace=False)]

    # Depth comes from the tracking frame, not the capture ring: the capture
    # cameras orbit at 0.70 m and the tracking camera sits 0.50 m out, so
    # reusing capture depth would place the query cloud tens of millimetres off.
    depths = np.load(depth_path, allow_pickle=True)
    d = depths[fg[:, 0], fg[:, 1]].astype(np.float64)
    keep = d > 0
    fg, d = fg[keep], d[keep]

    pts = np.stack([
        (fg[:, 1] + 0.5 - K[0, 2]) * d / K[0, 0],
        (fg[:, 0] + 0.5 - K[1, 2]) * d / K[1, 1],
        d,
    ], axis=1)

    # Camera frame -> metric world (the tracking camera's pose). The model sits
    # in metres * dataparser_scale (pogs_config trains with no re-orientation,
    # no re-centring and no pose scaling, which Optimizer also assumes).
    pts_w = pts @ c2w[:3, :3].T + c2w[:3, 3]
    scale = float(dataset_scale)

    from scipy.spatial import cKDTree
    means_w = model.gauss_params["means"].detach().cpu().numpy().astype(np.float64) / scale
    dist, _ = cKDTree(pts_w).query(means_w, k=1)
    seeds = np.nonzero(dist < 0.003)[0]
    if len(seeds) == 0:
        raise SystemExit(
            "no trained gaussian falls inside the ground-truth mask. Check "
            "that training ran on this capture and that the mask and depth "
            "come from the same frame 0 (and that the dataparser did not "
            "re-orient/re-centre the scene)."
        )

    # Box in a gravity-aligned frame: a = along the horizontal viewing
    # direction, b = across it, c = up.
    fwd = c2w[:3, 2].copy()
    fwd[2] = 0.0
    fwd /= max(np.linalg.norm(fwd), 1e-9)
    lat = np.array([-fwd[1], fwd[0], 0.0])

    def to_abc(x):
        return np.stack([x @ fwd, x @ lat, x[:, 2]], axis=1)

    s_abc, m_abc = to_abc(pts_w), to_abc(means_w)
    lo, hi = s_abc.min(0), s_abc.max(0)
    width_b = hi[1] - lo[1]
    box_lo = np.array([lo[0] - margin, lo[1] - margin, lo[2] + 0.002])  # floor just above the table plane
    box_hi = np.array([lo[0] + max(hi[0] - lo[0], width_b) + margin,
                       hi[1] + margin, hi[2] + margin])
    in_box = np.nonzero(((m_abc >= box_lo) & (m_abc <= box_hi)).all(1))[0]

    as_t = lambda i: torch.as_tensor(i, dtype=torch.long, device=device)
    return as_t(seeds), as_t(in_box)


def score_frame(i: int, est: np.ndarray, gt_frame: dict, est0: np.ndarray,
                gt0: np.ndarray) -> dict:
    """Per-frame tracking error: displacement error since frame 0, in mm.

    The score is ``|(est_i - est_0) - (gt_i - gt_0)|`` per part, not a difference
    of distances. Differencing norms measures error only along the ray from the
    ground truth to the estimate, so a 50 mm sideways slide reads as roughly
    zero; differencing the displacements themselves measures the motion that
    actually happened, which is what POGS claims to do.

    ``est0`` and ``gt0`` are the frame-0 reference, passed in rather than
    recomputed per frame -- calibrating on the current frame would subtract the
    answer and report zero error for a tracker that drifted a centimetre on
    every frame.
    """
    est_d = est[:, :3, 3]
    est0_d = est0[:, :3, 3]
    gt_xyz = np.array(gt_frame["focus"]["centroid"], dtype=np.float64)
    gt_xyz0 = np.asarray(gt0, dtype=np.float64)

    # Displacements on both sides, then their difference. est_0 and gt_0 anchor
    # the same point differently (part centroid vs mug base), but that constant
    # cancels out of a displacement, which is exactly why this form works.
    errs = np.linalg.norm(
        (est_d - est0_d) - (gt_xyz - gt_xyz0), axis=1) * 1000.0
    rec = {
        "frame": i,
        "centroid_err_mm": [round(float(v), 3) for v in errs],
        "centroid_err_mm_mean": round(float(np.mean(errs)), 3),
        "gt_centroid": gt_xyz.tolist(),
        "gt_displacement_mm": round(
            float(np.linalg.norm(gt_xyz - gt_xyz0)) * 1000.0, 3),
        "est_centroids": est_d.tolist(),
        # Full 3x4 pose per part (rotation + translation) so a later analysis can
        # score orientation without rerunning the tracker.
        "est_poses": est[:, :3, :].tolist(),
    }
    # Both scenes label their frames differently -- pickplace has phases
    # (hold/lift/carry/place), pendulum has a signed angle. Kept when present so a
    # report can group by either, rather than this function guessing a schema.
    if "phase" in gt_frame:
        rec["phase"] = gt_frame["phase"]
    if "angle_deg" in gt_frame:
        rec["angle_deg"] = gt_frame["angle_deg"]
    return rec


def build_clusters(model, keep_inds, labels: np.ndarray) -> np.ndarray:
    """``clusters.npy`` in the layout ``Optimizer._cluster_from_file`` expects.

    ``[cluster_labels, keep_inds, cgtf_stack]``. ``cgtf_stack`` is unused by the
    optimizer until grasps are generated, but the file must be a length-3 object
    array, so it is written as an empty (0, 7) placeholder rather than omitted.
    """
    labels = np.asarray(labels)
    if len(labels) != len(model.gauss_params["means"]):
        raise SystemExit(
            f"cluster_labels has {len(labels)} entries but the model has "
            f"{len(model.gauss_params['means'])} gaussians"
        )
    keep = (keep_inds.detach().cpu().numpy() if torch.is_tensor(keep_inds)
            else np.asarray(keep_inds))
    n_clusters = int(labels.max()) + 1
    cgtf = np.zeros((n_clusters, 7), dtype=np.float32)
    cgtf[:, 0] = 1.0   # identity wxyz; an all-zero quaternion makes viser divide by zero
    return np.array([labels, keep, cgtf], dtype=object)


def main() -> int:
    ap = argparse.ArgumentParser(description="Headless POGS tracking")
    ap.add_argument("--row", default=cfg.BASELINE_ROW)
    ap.add_argument("--input", default=None,
                    help="generated scene root (default: the row's input)")
    ap.add_argument("--train-dir", default=None,
                    help="training output dir (default: artefacts/<row>/train)")
    ap.add_argument("--cluster-eps", type=float, default=0.1)
    ap.add_argument("--opt-iters", type=int, default=cfg.POGS_TRACK_OPT_ITERS,
                    help="optimizer iterations per tracking frame; "
                         "track_main_demo.py uses 15")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--min-part", type=int, default=100,
                    help="smallest cluster, in gaussians, tracked as a part; "
                         "smaller ones cannot be rendered to find their ROI")
    ap.add_argument("--merge-parts", action="store_true",
                    help="track every kept cluster as one rigid part instead "
                         "of one part per cluster (the POGS behaviour)")
    args = ap.parse_args()

    root = os.path.abspath(args.input) if args.input else cfg.PICKPLACE
    train_dir = os.path.abspath(
        args.train_dir or os.path.join(cfg.ARTEFACTS, args.row, "train"))
    tracking_dir = os.path.join(root, "tracking")
    scene_name = os.path.basename(os.path.normpath(root))
    config_path = find_config(train_dir)
    if config_path is None:
        raise SystemExit(
            f"no finished training config under {train_dir}. Run "
            f"'bash run.sh train' first -- eval_setup needs a checkpoint next "
            f"to the config."
        )
    config_path = Path(config_path).resolve()
    print(f"config : {config_path}")

    # POGS caches DINO features *and the PCA basis they are projected with* at
    # outputs/<scene>/{dino.npy,pca.npy}, relative to the current directory
    # (full_images_datamanager.py:148). Training runs from the row's POGS/ tree
    # (row_common.sh: cd "$POGS" && ns-train). Run from anywhere else and the
    # loader finds no cache, recomputes it, and torch.pca_lowrank -- randomised
    # -- returns a different basis. The gaussians' DINO features were fit in
    # the training basis, so the tracking loss compares two unrelated spaces
    # and the pose optimiser pushes the object out of view ("Object left ROI",
    # or get_optim_loss returning None once nothing renders).
    pogs_dir = os.path.join(cfg.STUDY_ROOT, args.row, "POGS")
    pca_path = os.path.join(pogs_dir, "outputs", scene_name, "pca.npy")
    if not os.path.exists(pca_path):
        raise SystemExit(
            f"no training DINO cache at {pca_path}. Tracking must reuse the "
            f"PCA basis training fit; a recomputed one does not match the "
            f"trained features. Re-run 'bash run.sh train' for this scene."
        )
    os.chdir(pogs_dir)
    print(f"cwd    : {pogs_dir}  (training's DINO/PCA cache)")

    images, depths, K, width, height, c2w_cv, gt = load_sequence(tracking_dir)
    if args.max_frames:
        images, depths = images[:args.max_frames], depths[:args.max_frames]
    print(f"frames : {len(images)}   {width}x{height}")

    device = "cuda:0"
    from nerfstudio.utils.eval_utils import eval_setup
    from pogs.tracking.optim import Optimizer

    # our own checkpoint, trained minutes ago on this machine; PyTorch 2.6
    # defaults weights_only=True, which rejects the numpy scalars nerfstudio
    # pickles. eval_load_checkpoint has no kwarg to override this.
    torch.load = functools.partial(torch.load, weights_only=False)

    # optim.py's wait loop (lines 261-270) exists so the interactive demo's
    # browser render can populate best_scales. Headless, nothing renders.
    # _cluster_from_file never reads best_scales, so a dummy lets the
    # cached-cluster path proceed without a viewer.
    _orig_setup = Optimizer._setup_crops_and_groups

    def _setup_no_wait(self):
        if self.cluster_from_file is not None:
            # _cluster_from_file does torch.zeros_like(cluster_labels) and
            # indexes gauss_params with keep_inds; the cached file stores
            # numpy arrays, so convert before handing it over.
            cf = self.cluster_from_file
            self.cluster_from_file = [
                torch.as_tensor(cf[0], dtype=torch.long),
                torch.as_tensor(cf[1], dtype=torch.long),
                cf[2],
            ]
            m = self.pipeline.model
            if getattr(m, "best_scales", None) is None:
                dev = m.gauss_params["means"].device
                m.best_scales = torch.zeros(1, device=dev)
        return _orig_setup(self)

    Optimizer._setup_crops_and_groups = _setup_no_wait

    # warp must be initialised before the first kernel launch; the authors'
    # demo calls wp.init() explicitly.
    import warp as wp
    wp.init()

    # init_obj_pose(render=True) keeps a GS render of every optimisation step
    # and writes test_camopt*.mp4 into the cwd -- for the demo's viewer. Neither
    # affects the pose, and the video write needs ffmpeg, so render nothing.
    from pogs.tracking.rigid_group_optimizer import RigidGroupOptimizer
    _orig_init_pose = RigidGroupOptimizer.initialize_obj_pose

    def _init_pose_no_render(self, niter=100, n_seeds=6, render=False):
        return _orig_init_pose(self, niter=niter, n_seeds=n_seeds, render=False)

    RigidGroupOptimizer.initialize_obj_pose = _init_pose_no_render

    train_config, pipeline, _, _ = eval_setup(config_path)
    model = pipeline.model

    print(f"clustering (HDBSCAN, eps={args.cluster_eps}) -- the same "
          f"cuml call _cluster_scene makes")
    labels = model.cluster(args.cluster_eps)
    labels = np.asarray(labels)
    n_clusters = len(set(labels.tolist())) - (1 if -1 in labels else 0)
    print(f"  {n_clusters} clusters over {len(labels)} gaussians")
    if n_clusters < 2:
        raise SystemExit(
            f"{n_clusters} cluster(s) found. The mug and the gripper must be "
            f"separate for this study; try a smaller --cluster-eps."
        )

    mask0 = os.path.join(tracking_dir, "obj_id", os.path.basename(images[0]))
    if not os.path.exists(mask0):
        raise SystemExit(
            f"no ground-truth obj_id mask at {mask0}. Regenerate the scene with "
            f"generate_pickplace.py -- the tracker identifies the mug from this "
            f"mask rather than by picking a cluster by hand."
        )
    seeds, in_box = mug_gaussians(
        model, K, width, height, mask0, depths[0], device, c2w_cv,
        pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_scale)
    seeds_np, box_np = seeds.cpu().numpy(), in_box.cpu().numpy()

    # The seeds' cluster(s) say which instance the mug is; the box says where.
    # Intersecting them mirrors the authors' flow (crop, then take the
    # clusters inside the crop) and drops neighbours such as the gripper.
    seed_counts = np.bincount(labels[seeds_np][labels[seeds_np] >= 0],
                              minlength=int(labels.max()) + 1)
    seed_clusters = [int(c) for c in np.nonzero(
        seed_counts >= 0.10 * len(seeds_np))[0]]
    # Seeds are filtered too: a handful of them can land in a neighbouring
    # cluster, and a part of a few gaussians cannot be rendered at the 0.9
    # accumulation RigidGroupOptimizer needs to find its ROI ("Object left ROI").
    cand = np.union1d(seeds_np, box_np)
    cand = cand[np.isin(labels[cand], seed_clusters)]
    sizes = np.bincount(labels[cand], minlength=int(labels.max()) + 1)
    big = [c for c in seed_clusters if sizes[c] >= args.min_part]
    if not big:
        raise SystemExit(
            f"no cluster has >= {args.min_part} mug gaussians "
            f"(sizes {sizes.tolist()}); try a different --cluster-eps"
        )
    keep_np = cand[np.isin(labels[cand], big)]
    keep = torch.as_tensor(keep_np, dtype=torch.long, device=device)

    kept_labels = labels[keep_np]
    kept_clusters = set(np.unique(kept_labels).tolist())
    print(f"  mug seeds (visible surface): {len(seeds_np)}   "
          f"box: {len(box_np)}   kept: {len(keep_np)}   "
          f"seed clusters: {seed_clusters}")
    print(f"  kept gaussians per cluster: "
          f"{ {int(c): int((kept_labels == c).sum()) for c in kept_clusters} }")
    _ds = float(pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_scale)
    _m = model.gauss_params["means"].detach().cpu().numpy().astype(np.float64) / _ds
    for _name, _ix in (("seeds", seeds_np), ("kept ", keep_np)):
        _p = _m[_ix]
        print(f"  {_name} extent (m, xyz): {np.round(_p.max(0) - _p.min(0), 3).tolist()}"
              f"   min {np.round(_p.min(0), 3).tolist()}   max {np.round(_p.max(0), 3).tolist()}")
    if len(kept_clusters) != 1:
        if args.merge_parts:
            major = int(np.bincount(kept_labels[kept_labels >= 0]).argmax())
            labels = labels.copy()
            labels[keep_np] = major
            kept_clusters = {major}
            print(f"  --merge-parts: merged into cluster {major}, tracked as "
                  f"one rigid part.")
        else:
            print("  note: the mug spans several clusters; each is tracked as "
                  "its own part, as POGS does. --merge-parts treats it as one "
                  "rigid body instead.")

    clusters_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(config_path))),
        "clusters.npy",
    )
    os.makedirs(os.path.dirname(clusters_path), exist_ok=True)
    np.save(clusters_path, build_clusters(model, keep, labels), allow_pickle=True)
    print(f"  wrote {clusters_path}")

    # Optimizer wants the initial camera pose as a (1, 3, 4) OpenCV matrix in
    # world coordinates. The generator stored it OpenGL; load_sequence rebased it.
    cam2world = torch.tensor(c2w_cv, dtype=torch.float32)[None, :3, :]

    toad = Optimizer(config_path, K, width, height, cam2world)  # optim.py builds the OpenGL camera on CPU (torch.cat with a CPU row)

    first_rgb = cv2.imread(images[0], cv2.IMREAD_COLOR)
    first_depth = torch.from_numpy(np.load(depths[0]).astype(np.float32))
    toad.set_frame(
        torch.from_numpy(cv2.cvtColor(first_rgb, cv2.COLOR_BGR2RGB)).to(device),
        toad.cam2world_ns_ds,
        first_depth.to(device),
    )

    print("initializing object pose")
    init_failed = False
    try:
        toad.init_obj_pose()
    except RuntimeError as exc:
        # Same "Object left ROI" as below, but raised while registering frame 0:
        # the render of the object at its starting pose came back empty, i.e. the
        # splat's object cluster was unusable (a thin pendulum can end up with a
        # single seed Gaussian on an unlucky training run). That is a measured
        # failure of the pipeline on this scene, not a harness fault, so record it
        # and let the sweep go on instead of dying with no tracking.json.
        if "Object left ROI" not in str(exc):
            raise
        init_failed = True
        print(f"  INIT FAILED: {exc} -- object could not be registered on frame 0")

    # Frame 0 is the reference: init_obj_pose optimises against the frame set by
    # set_frame, so its output is the estimate *for frame 0*. Scoring from there
    # makes the first row exactly zero by construction rather than pretending an
    # absolute registration we never measured.
    gt_frames = gt["frames"]
    if init_failed:
        # No estimate exists. Use identity poses, which scores every frame as
        # "the tracker never moved the object", the same frozen-estimate convention
        # as a track lost part-way through.
        est0 = np.tile(np.eye(4), (max(1, len(kept_clusters)), 1, 1))
    else:
        est0 = np.asarray([tf.as_matrix() for tf in toad.get_parts2world()])
    gt0 = np.array(gt_frames[0]["focus"]["centroid"])
    traj = [score_frame(0, est0, gt_frames[0], est0, gt0)]

    lost_at = 0 if init_failed else None
    est = est0
    for i in range(1, 1 if init_failed else len(images)):
        rgb = cv2.imread(images[i], cv2.IMREAD_COLOR)
        depth = torch.from_numpy(np.load(depths[i]).astype(np.float32))
        try:
            toad.set_observation(
                torch.from_numpy(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)).to(device),
                toad.cam2world_ns,
                depth.to(device),
            )
        except RuntimeError as exc:
            # "Object left ROI": the optimiser renders the object at its own
            # current estimate to find the crop, and that render came back empty
            # because the estimate has drifted off the object. That is the tracker
            # losing the object -- a result, not a harness fault. Crashing here
            # discarded the whole trajectory and tracking.json with it, so the
            # scene showed up as a failed run instead of a measured failure.
            if "Object left ROI" not in str(exc):
                raise
            lost_at = i
            print(f"  TRACK LOST at frame {i}/{len(images)}: {exc}")
            break
        t0 = time.time()
        toad.step_opt(niter=args.opt_iters)
        seconds = time.time() - t0
        est = np.asarray([tf.as_matrix() for tf in toad.get_parts2world()])
        rec = score_frame(i, est, gt_frames[i], est0, gt0)
        rec["seconds"] = round(seconds, 4)
        traj.append(rec)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(images)} frames   "
                  f"last {rec['centroid_err_mm_mean']:.1f} mm mean")

    # Frames after a loss are scored with the estimate frozen where the tracker
    # last had it, and flagged. Averaging only the frames before the loss would
    # rank a tracker that lost the object at frame 12 above one that followed it
    # to the end with a few millimetres of error.
    if lost_at is not None:
        for j in range(max(lost_at, 1), len(images)):
            rec = score_frame(j, est, gt_frames[j], est0, gt0)
            rec["lost"] = True
            traj.append(rec)

    # Next to train/, i.e. artefacts/<row>/<scene>/track -- where report.py and
    # compare.py read it. artefacts/<row>/track was shared by every scene, so a
    # sweep kept only the last scene's result and the report found none.
    out_dir = os.path.join(os.path.dirname(train_dir), "track")
    os.makedirs(out_dir, exist_ok=True)

    # Score only the frames where the object is actually moving, decided by the
    # ground truth rather than by a scene-specific label.
    #
    # pickplace names its phases (hold/lift/carry/place) and pendulum carries a
    # signed angle, so keying off either would silently score every frame for the
    # other scene -- including the stationary ones, where error is near zero and
    # would dilute the result. Asking "did the object move?" works for both, and
    # is what the number is supposed to mean.
    moving = [f for f in traj if f.get("gt_displacement_mm", 0.0) > 1.0]
    means = [f["centroid_err_mm_mean"] for f in moving] or \
            [f["centroid_err_mm_mean"] for f in traj]
    if not moving:
        note = ("every frame is stationary by ground truth; the error figure is "
                "over the whole sequence and is not a motion measurement")
    else:
        note = ("displacement error against the generator's ground truth, "
                "frame 0 as reference, averaged over the frames where the "
                "object actually moves")

    if init_failed:
        note += ("; INIT FAILED -- the object could not be registered on frame 0, so "
                 "every frame is scored with no motion estimated (identity pose)")
    elif lost_at is not None:
        note += (f"; TRACKING LOST at frame {lost_at} of {len(images)} -- frames "
                 f"from {lost_at} are scored with the estimate frozen at its last "
                 f"pose")

    result = {
        "row": args.row,
        "scene": scene_name,
        "input": root,
        "config": str(config_path),
        "seed": int(os.environ.get("SEED", cfg.POGS_SEED)),
        "opt_iters": args.opt_iters,
        "cluster_eps": args.cluster_eps,
        "n_frames": len(images),
        "n_gaussians_total": int(len(labels)),
        "n_gaussians_kept": int(len(keep)),
        "clusters": sorted(int(c) for c in kept_clusters),
        "summary": {
            "centroid_err_mm_mean": round(float(np.mean(means)), 3),
            "centroid_err_mm_max": round(float(np.max(means)), 3),
            "frames_scored": len(means),
            "frames_total": len(traj),
            "lost_at_frame": lost_at,
            "init_failed": init_failed,
            "note": note,
        },
        "frames": traj,
    }
    path = os.path.join(out_dir, "tracking.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"wrote {path}")
    print(f"  {result['summary']['centroid_err_mm_mean']:.1f} mm mean over "
          f"{len(means)} moving frame(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())