"""Regression tests for the pieces whose bugs invalidated earlier results.

    .venv/bin/python -m pytest -q tests
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.geometry import (  # noqa: E402
    apply_group_deltas, identity_posevec, posevec_to_matrix, quat_to_rotmat, so3_exp,
)


def _rand_quat(n, gen):
    q = torch.randn(n, 4, generator=gen, dtype=torch.float64)
    return q / q.norm(dim=-1, keepdim=True)


def _mat(t, q):
    T = torch.eye(4, dtype=torch.float64)
    T[:3, :3] = quat_to_rotmat(q[None])[0]
    T[:3, 3] = t
    return T


def test_apply_group_deltas_matches_release_kernel():
    """Release (warp kernel): new_g2w = p2w * pdelta * inv(p2w) * g2w."""
    g = torch.Generator().manual_seed(0)
    K, N = 3, 50
    p2w = torch.zeros(K, 7, dtype=torch.float64)
    p2w[:, :3] = torch.randn(K, 3, generator=g, dtype=torch.float64)
    p2w[:, 3] = 1.0
    deltas = torch.cat([torch.randn(K, 3, generator=g, dtype=torch.float64) * 0.1, _rand_quat(K, g)], -1)
    labels = torch.randint(0, K, (N,), generator=g)
    means = torch.randn(N, 3, generator=g, dtype=torch.float64)
    quats = _rand_quat(N, g)
    m, q = apply_group_deltas(p2w, deltas, labels, means, quats)
    for i in range(N):
        k = int(labels[i])
        P = _mat(p2w[k, :3], p2w[k, 3:])
        D = _mat(deltas[k, :3], deltas[k, 3:])
        G = _mat(means[i], quats[i])
        ref = P @ D @ torch.linalg.inv(P) @ G
        assert torch.allclose(m[i], ref[:3, 3], atol=1e-9)
        assert torch.allclose(quat_to_rotmat(q[i][None])[0], ref[:3, :3], atol=1e-9)


def test_apply_group_deltas_gradient():
    g = torch.Generator().manual_seed(1)
    p2w = identity_posevec(1, dtype=torch.float64)
    p2w[0, :3] = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)
    means = torch.randn(20, 3, generator=g, dtype=torch.float64)
    quats = _rand_quat(20, g)
    labels = torch.zeros(20, dtype=torch.long)
    d = torch.cat([torch.randn(1, 3, generator=g, dtype=torch.float64) * 0.05, _rand_quat(1, g)], -1).requires_grad_()

    def f(x):
        m, _ = apply_group_deltas(p2w, x, labels, means, quats)
        return (m ** 2).sum()

    assert torch.autograd.gradcheck(f, (d,), eps=1e-6, atol=1e-5)


def test_pose_semantics_match_gt_convention():
    """Predicting the exact GT motion must give zero error: POGS's delta rotates
    about the scan centroid c and translates it, GT is x' = R x + t."""
    from common.evaluate import gt_pose_for_group, rotation_error_deg, translation_error_cm
    from common.tracking import RigidGroupOptimizer  # noqa: F401  (import check)

    c = np.array([-0.08, -0.06, 0.05])
    R = quat_to_rotmat(so3_exp(torch.tensor([[0.0, 0.0, 0.7]], dtype=torch.float64)))[0].numpy()
    pivot = np.array([-0.07, -0.05, 0.048])          # GT pivot differs from the cluster centroid
    t = np.array([0.2, 0.1, 0.06])
    Tgt = np.eye(4)
    Tgt[:3, :3] = R
    Tgt[:3, 3] = pivot - R @ pivot + t

    class _F:
        gt_delta = {1: Tgt}

    class _S:
        frames = [_F()]

    T_gt = gt_pose_for_group(_S(), 0, 1, c)
    # POGS pose with delta translation = R c + t' - c, rotation R
    P = np.eye(4)
    P[:3, :3] = R
    P[:3, 3] = c + (Tgt[:3, :3] @ c + Tgt[:3, 3] - c)
    assert translation_error_cm(P, T_gt) < 1e-9
    assert rotation_error_deg(P, T_gt) < 1e-6


def test_dino_extractor_refuses_without_pca_and_checkpoint_roundtrip(tmp_path):
    from common.checkpoint import load_checkpoint, save_checkpoint
    from common.config import POGSConfig
    from common.encoders import DinoExtractor
    from common.model import POGSModel

    dino = DinoExtractor(device="cpu", allow_download=False)
    with pytest.raises(RuntimeError):
        dino(torch.rand(1, 3, 56, 56))

    cfg = POGSConfig(device="cpu")
    cfg.model.log2_hashmap_size = 12
    cfg.runtime.raster_backend = "torch"
    pts = torch.rand(64, 3)
    model = POGSModel(pts, torch.rand(64, 3), config=cfg.model, device="cpu", raster_backend="torch")
    model.step = 1234
    pca = torch.randn(384, 64)
    save_checkpoint(str(tmp_path), model, pca, cfg)
    m2, pca2, _ = load_checkpoint(str(tmp_path), cfg, device="cpu")
    assert torch.equal(pca, pca2)
    assert m2.step == 1234
    for k in model.gauss_params:
        assert torch.equal(model.gauss_params[k].data, m2.gauss_params[k].data)
    dino.set_pca(pca2)
    assert dino(torch.rand(1, 3, 56, 56)).shape[-1] == 64


def test_kalman_constant_velocity():
    from common.kalman import KalmanConfig, MultiObjectKalman

    kf = MultiObjectKalman(1, KalmanConfig(meas_pos_std=1e-4, meas_rot_std=1e-4))
    kf.initialize(torch.tensor([[0, 0, 0, 1, 0, 0, 0.0]]))
    for k in range(1, 30):
        kf.predict()
        q = so3_exp(torch.tensor([[0.0, 0.0, 0.05 * k]]))[0]
        kf.update(torch.cat([torch.tensor([0.01 * k, 0.0, 0.0]), q])[None].double())
    p = kf.predict()[0]
    assert abs(float(p[0]) - 0.30) < 2e-3
    yaw = 2 * np.arctan2(float(p[6]), float(p[3]))
    assert abs(yaw - 1.5) < 0.02


def test_build_masks_negative_last_and_background_excluded():
    from common.trainer import build_masks

    ids = torch.zeros(40, 40, dtype=torch.long)
    ids[5:15, 5:15] = 1
    ids[20:30, 20:30] = 9          # support table
    ids[0:3, 30:40] = 2
    masks = build_masks(ids, background_ids=(0, 9))
    assert len(masks) == 3
    neg = masks[-1]
    assert bool(neg[25, 25]) and bool(neg[0, 0]) and not bool(neg[10, 10])


def test_dbscan_keeps_every_surface():
    from scene_generator.capture_generator import dbscan_filter

    rng = np.random.default_rng(0)
    blobs = [rng.normal(loc=(i * 0.5, 0, 0), scale=0.005, size=(200, 3)) for i in range(8)]
    pts = np.concatenate(blobs + [rng.uniform(-1, 4, size=(5, 3))])
    _, _, keep = dbscan_filter(pts, np.zeros_like(pts), eps=0.02, min_samples=5, return_mask=True)
    assert keep[:1600].all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_torch_and_gsplat_rasterisers_agree():
    from common.geometry import PinholeCamera, look_at
    from common.raster import gsplat_available, rasterize

    if not gsplat_available():
        pytest.skip("gsplat unavailable")
    g = torch.Generator().manual_seed(0)
    N = 300
    means = (torch.rand(N, 3, generator=g) - 0.5) * 0.2
    quats = torch.randn(N, 4, generator=g)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    scales = torch.rand(N, 3, generator=g) * 0.01 + 0.003
    opac = torch.rand(N, generator=g) * 0.8 + 0.1
    cols = torch.rand(N, 8, generator=g)
    c2w = torch.from_numpy(look_at(np.array([0.4, -0.4, 0.4]), np.zeros(3))).float()
    cam = PinholeCamera(c2w.cuda(), 200, 200, 80, 60, 160, 120)
    args = [t.cuda() for t in (means, quats, scales, opac, cols)]
    a = rasterize(*args, cam, backend="gsplat", render_depth=True)
    b = rasterize(*args, cam, backend="torch", render_depth=True, max_per_tile=4096)
    assert (a["alpha"] - b["alpha"]).abs().mean() < 0.02
    assert (a["render"] - b["render"]).abs().mean() < 0.02


def test_benchmark_trajectory_and_cases():
    """Placed object ends upright on the table at the sampled point; tilt
    returns to upright; cases are unique and every factor has the base case."""
    import scene_generator.pick_place_benchmark as b

    cfg = b.BenchmarkConfig()
    objs = b.build_pick_place_scene()
    rng = np.random.default_rng(0)
    for cname in ("base", "tilt60", "combined45"):
        cond = b.CONDITIONS[cname]
        prm = b.sample_params(rng, cond.target, cfg)
        plan, _ = b.plan_pick_and_place(objs, b.TARGETS[cond.target], cond, prm, cfg)
        assert np.allclose(plan[0]["delta"], np.eye(4))
        D = plan[-1]["delta"]
        mug = next(o for o in objs if o.obj_id == 1)
        base_after = D[:3, :3] @ np.array(mug.position) + D[:3, 3]
        assert np.allclose(base_after, [*prm.place_xy, b.TABLE_TOP_Z], atol=1e-9)
        assert abs(D[2, 2] - 1.0) < 1e-9                      # upright at the end
        assert abs(plan[-1]["rot_deg"] - (0.0 if cond.rot_axis == "tilt" else cond.rot_deg)) < 1e-6
        assert max(p["rot_deg"] for p in plan) >= cond.rot_deg - 1.0
    cases = b.benchmark_cases()
    assert len(cases) == len(set(cases))
    for levels in b.FACTORS.values():
        assert any(c == "base" and s == 1 and d == "stereo" for _, c, s, d in levels)


def test_degrade_depth_deterministic():
    from scene_generator.pick_place_benchmark import degrade_depth

    d = np.full((200, 300), 0.8, np.float32)
    d[:20] = 0.0
    a, b2 = degrade_depth(d, "noisy", "k"), degrade_depth(d, "noisy", "k")
    assert np.array_equal(a, b2)
    assert not np.array_equal(a, degrade_depth(d, "noisy", "other"))
    assert (a[:20] == 0).all()
    holes = (a[20:] == 0).mean()
    assert 0.02 < holes < 0.09
    assert np.array_equal(degrade_depth(d, "clean", "k"), d)


def test_kalman_damping_and_speed_limit():
    """Without fresh evidence the velocity decays; it is never above the cap."""
    from common.kalman import KalmanConfig, SE3KalmanFilter

    f = SE3KalmanFilter(KalmanConfig(dt=0.2, vel_damping_tau=1.0, max_speed=0.5))
    f.initialize(np.array([0, 0, 0, 1, 0, 0, 0.0]))
    for k in range(1, 40):                       # measurements racing away at 2 m/s
        f.predict()
        f.update(np.array([0.4 * k, 0, 0, 1, 0, 0, 0.0]))
        assert np.linalg.norm(f.v) <= 0.5 + 1e-9
    v0 = np.linalg.norm(f.v)
    for _ in range(10):
        f.predict()
    assert np.linalg.norm(f.v) < 0.2 * v0


# --------------------------------------------------------------------------- #
# Scene v2 (materials, layout) and the real-time protocol
# --------------------------------------------------------------------------- #


def test_texture_moves_with_object():
    """Textures are evaluated in the primitive's own frame (the first generator
    used world coordinates, so moving objects slid under a fixed pattern)."""
    from scene_generator.materials import surface_color
    from scene_generator.pick_place_scene import build_pick_place_scene
    from scene_generator.scene_builder import FlatScene

    sc = FlatScene(build_pick_place_scene())
    for oid in (1, 3, 4, 5):                                  # mug, soup, apple, cereal
        i = int((sc.obj_id == oid).nonzero()[0])
        idx = torch.tensor([i] * 3)
        pts = sc.t[idx] + torch.tensor([[0.02, 0.0, 0.01], [0.0, 0.025, -0.01], [-0.015, 0.01, 0.02]])
        D = np.eye(4)
        D[:3, :3] = quat_to_rotmat(so3_exp(torch.tensor([[0.3, -0.5, 1.1]], dtype=torch.float64)))[0].numpy()
        D[:3, 3] = [0.07, -0.03, 0.05]
        moved = sc.clone_with_transforms({oid: D})
        R, t = torch.tensor(D[:3, :3], dtype=torch.float32), torch.tensor(D[:3, 3], dtype=torch.float32)
        assert torch.allclose(surface_color(sc, idx, pts), surface_color(moved, idx, pts @ R.T + t), atol=1e-4)


def test_scene_layout_no_overlap():
    """Table top exactly at TABLE_TOP_Z; floor objects clear of the table
    footprint and of each other (the soup can used to sit inside the table)."""
    import scene_generator.pick_place_benchmark as b

    objs = b.build_pick_place_scene()
    table = next(o for o in objs if o.obj_id == b.TABLE_ID)
    top = table.prims[0]
    assert abs(table.position[2] + top.offset[2] + top.scale[2] - b.TABLE_TOP_Z) < 1e-12
    tx0, tx1 = table.position[0] - top.scale[0], table.position[0] + top.scale[0]
    ty0, ty1 = table.position[1] - top.scale[1], table.position[1] + top.scale[1]

    def footprint(o, n=24):
        """Top-down points covering every primitive of ``o`` (world xy)."""
        from scene_generator.primitives import BOX, euler_to_R
        c, sn = np.cos(o.yaw), np.sin(o.yaw)
        Ro = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1.0]])
        g = np.linspace(-1, 1, n)
        u, v = np.meshgrid(g, g)
        uv = np.stack([u.ravel(), v.ravel()], -1)
        out = []
        for p in o.prims:
            loc = uv if p.kind == BOX else uv[(uv ** 2).sum(-1) <= 1.0]
            pts = np.c_[loc * np.array(p.scale[:2]), np.zeros(len(loc))]
            w = (Ro @ (euler_to_R(*p.rot_euler) @ pts.T + np.array(p.offset)[:, None])).T + np.array(o.position)
            out.append(w[:, :2])
        return np.concatenate(out)

    movers = [o for o in objs if o.obj_id in b.OBJECT_NAMES]
    fps = {o.name: footprint(o) for o in movers}
    for o in movers:
        x, y = fps[o.name][:, 0], fps[o.name][:, 1]
        gap = np.maximum.reduce([tx0 - x, x - tx1, ty0 - y, y - ty1])
        assert gap.min() > 0.02, (o.name, gap.min())               # >= 2 cm off the table footprint
    for i, a in enumerate(movers):
        for c in movers[i + 1:]:
            d = np.linalg.norm(fps[a.name][:, None] - fps[c.name][None], axis=-1).min()
            assert d > 0.02, (a.name, c.name, d)


def test_realtime_sequences_share_trajectories():
    """Frame 6k of the 30 Hz rendering is frame k of the 5 Hz one."""
    import scene_generator.pick_place_benchmark as b

    cfg = b.BenchmarkConfig()
    rt = b.realtime_config(cfg)
    assert rt.fps == 30.0 and rt.n_frames == 355
    objs = b.build_pick_place_scene()
    cond = b.CONDITIONS["base"]
    prm = b.sample_params(np.random.default_rng(1), "mug", cfg)
    p5, _ = b.plan_pick_and_place(objs, 1, cond, prm, cfg)
    p30, _ = b.plan_pick_and_place(objs, 1, cond, prm, rt)
    for k in range(cfg.n_frames):
        assert np.allclose(p5[k]["delta"], p30[6 * k]["delta"])
    c = b.Case("base", 0, 2, realtime=True)
    assert c.sequence == "base_s0_30hz" and c.name == "base_s0_30hz__rt2x_stereo"
    assert b.Case("base", 0, 2).name == "base_s0__2x_stereo"     # offline names unchanged


def test_pick_next_frame_and_sources():
    from common.evaluate import estimate_sources, pick_next_frame

    ts = np.arange(10) / 10.0                                  # 10 Hz camera
    assert pick_next_frame(ts, 0.0, -1) == 0
    assert pick_next_frame(ts, 0.35, 0) == 3                   # newest arrived, 1-2 skipped
    assert pick_next_frame(ts, 0.05, 0) == 1                   # nothing new yet: wait for 1
    assert pick_next_frame(ts, 5.0, 9) is None
    src = estimate_sources(ts, np.array([0.25, 0.62]))
    assert src.tolist() == [-1, -1, -1, 0, 0, 0, 0, 1, 1, 1]


def test_kalman_variable_dt_and_extrapolate():
    from common.kalman import KalmanConfig, SE3KalmanFilter

    cfg = KalmanConfig(dt=0.2, vel_damping_tau=0.0, max_speed=0.0, max_ang_speed=0.0)
    a, b2 = SE3KalmanFilter(cfg), SE3KalmanFilter(cfg)
    for f in (a, b2):
        f.initialize(np.array([0, 0, 0, 1, 0, 0, 0.0]))
        f.v = np.array([0.1, 0.0, 0.0])
    a.predict(0.1), a.predict(0.1)
    b2.predict(0.2)
    assert np.allclose(a.p, b2.p) and np.allclose(a.p, [0.02, 0, 0])
    assert np.trace(a.P_t) < np.trace(b2.P_t) + 1e-6           # same process noise order
    x = a.extrapolate(np.array([1.0, 0, 0, 1, 0, 0, 0]), 0.5)
    assert np.allclose(x[:3], [1.05, 0, 0], atol=1e-6)
    assert np.allclose(a.p, [0.02, 0, 0])                       # state untouched
    d = SE3KalmanFilter(KalmanConfig(dt=0.2, vel_damping_tau=1.0))
    d.initialize(np.zeros(3).tolist() + [1, 0, 0, 0])
    d.v = np.array([0.1, 0, 0])
    assert np.isclose(d.extrapolate(np.array([0, 0, 0, 1, 0, 0, 0.0]), 10.0)[0], 0.1 * (1 - np.exp(-10)), atol=1e-5)


def test_track_realtime_with_fake_tracker(monkeypatch):
    """The real-time loop on a fake tracker with a fixed 0.25 s latency on a
    10 Hz stream: frames 0, 2, 5, 7 are tracked (the newest arrived when the
    tracker frees up; 9 would finish after the stream ends), and every camera
    frame is scored on the last estimate finished by then."""
    import types
    import common.evaluate as ev

    class FakeTracker:
        def __init__(self, *a, **k):
            self.d = torch.zeros(1, 7)
            self.d[0, 3] = 1.0
            self.prior = None

        def initialize(self, fr):
            return {"init_time": 0.0}

        def step(self, fr, first, dt=None):
            self.d = self.d.clone()
            self.d[0, 0] = float(fr.frame_idx)                 # "estimate" = frame index
            return {"time_s": 0.25, "dino_ms": 1.0, "opt_ms": 1.0, "loss": 0.0}

        def deltas(self):
            return self.d.clone()

        def poses(self, d=None):
            d = self.d if d is None else d
            P = np.tile(np.eye(4), (1, 1, 1))
            P[0, 0, 3] = float(d[0, 0])
            return P

        def finish(self, n, verbose, name):
            return {}

    monkeypatch.setattr(ev, "_FrameTracker", FakeTracker)
    frames = [types.SimpleNamespace(frame_idx=i, timestamp_sec=i / 10.0, occluded=False,
                                    gt_delta={1: np.eye(4)}) for i in range(10)]
    seq = types.SimpleNamespace(frames=frames, camera=None)
    groups = types.SimpleNamespace(num_groups=1, init_p2w=torch.zeros(1, 7), names=["mug"])
    rt = ev.track_realtime(None, groups, seq, None, None, {0: 1}, tracking_cfg=object(), verbose=False)
    assert rt.proc_frames.tolist() == [0, 2, 5, 7]
    assert np.allclose(rt.proc_done, [0.25, 0.5, 0.75, 1.0])
    # held estimate x at each camera frame (frame index it came from; init = 0)
    held = rt.now.pred_poses[:, 0, 0, 3]
    assert held.tolist() == [0, 0, 0, 0, 0, 2, 2, 2, 5, 5]
    assert rt.now_extrap is None
    st = rt.now.diagnostics
    assert np.isclose(st["latency_ms"], 250) and np.isclose(st["tracker_hz"], 4.0)
    assert np.isclose(st["frames_skipped_mean"], 4 / 3)
    assert np.isclose(st["estimate_age_ms"], np.mean([300, 400, 300, 400, 500, 300, 400]))   # from frame 3 on
