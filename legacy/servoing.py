"""
Tool servoing -- Sec. V-B and Table II of the paper.

    "We consider a tabletop workspace with a tool object and an ArUco marker
     fixed to a target object surface. In these experiments, the tool object is
     a drill and the target object is a wooden platform. The tool is manually
     annotated with a coordinate frame at the tool tip to indicate which
     component to align with the target (i.e. align the drill tip
     perpendicularly). The robot then grasps the tool and aligns it to the
     tracked ArUco marker. The robot then performs 6DoF visual servoing such
     that the tool tip remains in its relative target pose to the object. We
     leave tool rotation about its target axis (i.e drill bit axis)
     unconstrained, and pick the orientation during servoing which minimizes
     tool motion. During each trial, the target object is moved to five random
     poses and we record the success rate for how often the tool follows the
     object. At each pose, a human moves the drill 15-30 degrees clockwise or
     counter clockwise about the grasp axis, and records how often the robot
     can adapt to this perturbation and locate the drill tip within 3cm and 5
     degrees of the target."

This is a **closed-loop** experiment, so unlike Sec. V-A it cannot run on a
pre-rendered video: every frame is rendered from the state the previous frame's
control command produced.  The loop here is

    render -> POGS tracking -> tip/marker poses from the *estimate*
           -> servo command -> apply to the *true* tool pose -> render ...

which means every bit of tracking error feeds straight into servo error, the
same coupling the real system has.

Two substitutions, both hardware-only:

* the UR5 is replaced by a rate-limited SE(3) command applied to the tool
  (``max_step_m`` / ``max_step_deg`` stand in for the arm's velocity limit);
* the ArUco marker is replaced by an annotated frame on the target object,
  because POGS tracks the target object itself -- the marker only ever served
  to give the *human* a ground-truth readout.

Tier definitions are the paper's:

    Tier 1  target stays on the tabletop plane and can move anywhere within a
            55 cm x 50 cm square; in-grasp tool rotation up to 15 degrees.
    Tier 2  target moves in 3D within a 55 cm x 72 cm x 10 cm box; in-grasp
            tool rotation up to 30 degrees.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import POGSConfig, TrackingConfig
from .geometry import PinholeCamera
from .grouping import ObjectGroups
from .model import POGSModel
from .simscene import (
    CaptureData,
    FlatScene,
    SceneObject,
    make_hand,
    render_scene,
    target_marker_frame,
    tool_tip_frame,
)
from .tracking import DinoFn, PosedObservation, RigidGroupOptimizer

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

TIER_WORKSPACE_M: Dict[int, Tuple[float, float, float]] = {
    1: (0.55, 0.50, 0.00),
    2: (0.55, 0.72, 0.10),
}
"""Paper Sec. V-B: 'Tier 1 experiments have the target object stay on the
tabletop plane and can move anywhere within a 55 cm by 50 cm square. Tier 2
experiments have the target moving in 3D space where it can move anywhere
within a 55 cm by 72 cm by 10 cm box.'"""

TIER_PERTURB_DEG: Dict[int, float] = {1: 15.0, 2: 30.0}
"""'the tool orientation in-grasp can be changed up to 15 degrees' (Tier 1);
'When the tool-in-gripper rotation increased to 30 degrees ...' (Tier 2)."""


@dataclass
class ServoConfig:
    """Everything the Sec. V-B protocol needs."""

    tier: int = 1
    n_trials: int = 5
    """'We run this experiment on two tiers with five trials each tier.'"""
    n_target_poses: int = 5
    """'the target object is moved to five random poses'."""

    pos_tol_cm: float = 3.0
    rot_tol_deg: float = 5.0
    """'locate the drill tip within 3cm and 5 degrees of the target'."""

    standoff_m: float = 0.03
    """Relative target pose of the tip: this far above the marker, along the
    marker's outward normal, with the tip axis pointing into the surface."""

    dt: float = 0.2
    """5 Hz -- the paper's reported online tracking rate (Sec. VI)."""
    settle_frames: int = 25
    """Servo frames allowed to converge onto a new target pose.  With a perfect
    pose estimate the controller needs 13-23 frames to cross the tier
    workspaces at ``max_step_m``, so this is the smallest budget that does not
    itself cause failures."""
    recovery_frames: int = 16
    """Servo frames allowed to recover from an in-grasp perturbation (2-4 with
    a perfect estimate)."""

    gain_pos: float = 0.55
    gain_rot: float = 0.55
    max_step_m: float = 0.030
    max_step_deg: float = 7.0
    """Rate limits standing in for the arm's Cartesian velocity limit; they are
    also what keeps object motion slow enough for POGS to track (Sec. VI)."""

    target_move_frames: int = 12
    """The human does not teleport the target: it slides into its new pose over
    this many frames while the servo loop keeps running.  POGS is explicitly a
    slow-motion tracker (Sec. VI: 'objects had to be moved slowly with no sudden
    motions'), so a one-frame jump would fail for reasons that have nothing to do
    with the servo controller.  These frames count toward the pose's time budget,
    which is what makes 'Follow Target' a following experiment.

    At the default 8 frames a full-width tier-2 move is ~4 cm/frame (~20 cm/s at
    5 Hz), which is brisk. Raise it to slow the human down and separate tracking
    failures from controller failures; lower it to stress the tracker."""

    with_gripper: bool = True
    """Render a gripper/hand occluding the tool handle. Sec. VI names this as a
    known weakness ('objects that are partially occluded ... have less robust
    tracking'), so leaving it on is the honest setting."""

    reach_box_m: Tuple[float, float, float, float, float, float] = (
        -0.45, 0.45, -0.40, 0.40, 0.0, 0.45
    )
    """(xmin, xmax, ymin, ymax, zmin, zmax) the tool tip can physically reach.

    Stands in for the UR5's reach envelope.  It matters because this is a closed
    loop: without it, a diverging pose estimate commands the tool further away
    every frame, the tracker falls further behind, and the tool ends up a metre
    outside the scene -- a failure the real rig cannot produce.  With the limit,
    a bad estimate still fails the trial, but it fails inside the workspace and
    can recover.  ``ServoResult.summary()['reach_limited_frames']`` reports how
    often the limit was hit."""

    workspace_clamp: Tuple[float, float] = (0.36, 0.26)
    """The paper's tier boxes are larger than POGS-Sim's tabletop, so target
    positions are clamped to |x| and |y| below these values to keep the target
    on the table and inside the camera frustum. The achieved displacement range
    is reported in the result so the clamp is never silent."""

    supersample: int = 1
    depth_noise_std: float = 0.0035
    seed: int = 0


# --------------------------------------------------------------------------- #
# Small SE(3) helpers (numpy; the tracker's own algebra stays in torch)
# --------------------------------------------------------------------------- #


def _rodrigues(omega: np.ndarray) -> np.ndarray:
    """Rotation vector -> 3x3 rotation matrix."""
    theta = float(np.linalg.norm(omega))
    if theta < 1e-12:
        return np.eye(3)
    k = omega / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def _delta_about(pivot: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """World-frame 4x4 that rotates by ``R`` about ``pivot`` then translates."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pivot - R @ pivot + t
    return T


def _apply(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    return T[:3, :3] @ p + T[:3, 3]


def _axis_angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation vector taking unit vector ``a`` onto unit vector ``b``.

    Using the *minimal* rotation is what leaves rotation about the tool axis
    unconstrained, exactly as the paper specifies.
    """
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(np.dot(a, b))
    if s < 1e-9:
        if c > 0:
            return np.zeros(3)
        # antiparallel: any orthogonal axis, pi radians
        ortho = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            ortho = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, ortho)
        return math.pi * axis / np.linalg.norm(axis)
    return math.atan2(s, c) * v / s


def world_delta_from_part2world(T: np.ndarray, scan_centroid: np.ndarray) -> np.ndarray:
    """POGS reports a pose as ``[R | transformed centroid]``.  Recover the
    world-frame rigid delta that maps scan-time geometry to that pose."""
    R = T[:3, :3]
    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3] = T[:3, 3] - R @ scan_centroid
    return out


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass
class ServoAttempt:
    trial: int
    pose_idx: int
    kind: str                    # "follow" | "clockwise" | "ccw"
    success: bool
    time_s: float
    final_pos_err_cm: float
    final_rot_err_deg: float
    n_frames: int
    pos_err_trace: List[float] = field(default_factory=list)
    rot_err_trace: List[float] = field(default_factory=list)
    track_pos_err_cm: List[float] = field(default_factory=list)
    """Per-frame POGS pose error on the tool, i.e. the estimation error that
    the servo loop is actually reacting to."""


@dataclass
class ServoResult:
    tier: int
    kind: str
    attempts: List[ServoAttempt]
    target_displacements_cm: List[float] = field(default_factory=list)
    wall_time_s: float = 0.0
    reach_limited_frames: int = 0
    """Frames where the commanded tool tip hit the arm's reach envelope. A large
    value means the pose estimate was driving the tool out of the workspace."""

    @property
    def n(self) -> int:
        return len(self.attempts)

    @property
    def n_success(self) -> int:
        return sum(a.success for a in self.attempts)

    def success_rate(self) -> float:
        return self.n_success / max(self.n, 1)

    def mean_time_s(self) -> float:
        ok = [a.time_s for a in self.attempts if a.success]
        return float(np.mean(ok)) if ok else float("nan")

    def summary(self) -> Dict[str, object]:
        return {
            "tier": self.tier,
            "kind": self.kind,
            "success": f"{self.n_success}/{self.n}",
            "success_rate": round(self.success_rate(), 3),
            "time_s": round(self.mean_time_s(), 2),
            "mean_final_pos_err_cm": round(
                float(np.mean([a.final_pos_err_cm for a in self.attempts])), 2
            ),
            "mean_final_rot_err_deg": round(
                float(np.mean([a.final_rot_err_deg for a in self.attempts])), 2
            ),
            "mean_tool_track_err_cm": round(
                float(np.mean([np.mean(a.track_pos_err_cm) for a in self.attempts
                               if a.track_pos_err_cm])), 2
            ) if any(a.track_pos_err_cm for a in self.attempts) else float("nan"),
            "max_target_disp_cm": round(max(self.target_displacements_cm), 1)
            if self.target_displacements_cm else 0.0,
            "reach_limited_frames": self.reach_limited_frames,
        }


# --------------------------------------------------------------------------- #
# The servo rig
# --------------------------------------------------------------------------- #


class ToolServoRig:
    """Holds the closed-loop state: the true scene, the POGS tracker, and the
    annotated tool-tip / marker frames."""

    def __init__(
        self,
        model: POGSModel,
        groups: ObjectGroups,
        capture: CaptureData,
        dino_fn: DinoFn,
        cfg: POGSConfig,
        group_to_obj: Dict[int, int],
        tool_name: str = "power drill",
        target_name: str = "wooden shelf",
        camera: Optional[PinholeCamera] = None,
        servo_cfg: Optional[ServoConfig] = None,
        tracking_cfg: Optional[TrackingConfig] = None,
    ):
        if capture.scene is None:
            raise ValueError(
                "tool servoing is a closed-loop experiment: it re-renders the scene "
                "every frame, so it needs POGS-Sim's analytic scene "
                "(CaptureData.scene). Real-data captures cannot drive it."
            )
        self.model = model
        self.groups = groups
        self.capture = capture
        self.dino_fn = dino_fn
        self.cfg = cfg
        self.scfg = servo_cfg or ServoConfig()
        self.tcfg = tracking_cfg or cfg.tracking
        self.device = cfg.runtime.device

        objs = {o.name: o for o in capture.scene.objects}
        if tool_name not in objs or target_name not in objs:
            raise KeyError(
                f"scene has {sorted(objs)}; tool servoing needs '{tool_name}' and "
                f"'{target_name}'"
            )
        self.tool_obj: SceneObject = objs[tool_name]
        self.target_obj: SceneObject = objs[target_name]

        self.tip_scan, self.tip_axis_scan = tool_tip_frame(self.tool_obj)
        self.marker_scan, self.marker_normal_scan = target_marker_frame(self.target_obj)
        # the drill's handle runs along the object's +z; the human twists about it
        R_obj = np.asarray(
            [[math.cos(self.tool_obj.yaw), -math.sin(self.tool_obj.yaw), 0.0],
             [math.sin(self.tool_obj.yaw), math.cos(self.tool_obj.yaw), 0.0],
             [0.0, 0.0, 1.0]]
        )
        self.grasp_axis_scan = R_obj @ np.array([0.0, 0.0, 1.0])

        # POGS group index <-> sim object id
        obj_to_group = {v: k for k, v in group_to_obj.items()}
        if self.tool_obj.obj_id not in obj_to_group or self.target_obj.obj_id not in obj_to_group:
            raise KeyError(
                f"clustering did not recover both objects: group_to_obj={group_to_obj}, "
                f"tool id={self.tool_obj.obj_id}, target id={self.target_obj.obj_id}"
            )
        self.gi_tool = obj_to_group[self.tool_obj.obj_id]
        self.gi_target = obj_to_group[self.target_obj.obj_id]
        self.centroids = {
            gi: groups.init_p2w[gi, :3].detach().cpu().numpy() for gi in range(groups.num_groups)
        }

        self.camera = (camera or self._default_camera()).to(self.device)
        scale = self.tcfg.match_resolution / min(self.camera.width, self.camera.height)
        self.track_camera = self.camera.rescale(min(scale, 1.0)) if scale < 1.0 else self.camera

        self.raster_kwargs = (
            dict(max_per_tile=cfg.runtime.torch_raster_max_per_tile,
                 tile_chunk=cfg.runtime.torch_raster_tile_chunk)
            if model.backend == "torch" else {}
        )

        # true world-frame deltas relative to the scan pose
        self.true_delta: Dict[int, np.ndarray] = {
            self.tool_obj.obj_id: np.eye(4),
            self.target_obj.obj_id: np.eye(4),
        }
        self._snapshot = model.state_snapshot()
        self.opt: Optional[RigidGroupOptimizer] = None
        self.n_reach_limited = 0

    # -- scene -------------------------------------------------------------- #

    def _default_camera(self) -> PinholeCamera:
        from .geometry import look_at

        eye = np.array([0.30, -0.52, 0.42])
        return PinholeCamera.from_fov(
            torch.as_tensor(look_at(eye, np.array([0.0, 0.0, 0.06])), dtype=torch.float32),
            62.0, self.capture.images.shape[2], self.capture.images.shape[1],
        )

    def _gripper(self) -> SceneObject:
        """A hand wrapped around the drill handle, moving with the tool."""
        h = make_hand(scale=0.9)
        D = self.true_delta[self.tool_obj.obj_id]
        p = _apply(D, self._grasp_point_scan())
        h.position = (float(p[0]), float(p[1] - 0.045), float(p[2]))
        h.yaw = float(self.tool_obj.yaw + math.atan2(D[1, 0], D[0, 0]) + math.pi / 2)
        return h

    @torch.no_grad()
    def render_frame(self) -> Dict[str, torch.Tensor]:
        objs = list(self.capture.scene.objects)
        if self.scfg.with_gripper:
            objs = objs + [self._gripper()]
        scene = FlatScene(objs, device=self.device).clone_with_transforms(self.true_delta)
        out = render_scene(scene, self.camera, supersample=self.scfg.supersample)
        d = out["depth"]
        noise = torch.randn_like(d) * self.scfg.depth_noise_std * (d / 0.7).clamp(min=0.2)
        out["depth"] = torch.where(d > 0, d + noise, d)
        return out

    def _observation(self, out: Dict[str, torch.Tensor]) -> PosedObservation:
        return PosedObservation(
            out["rgb"].to(self.device), self.track_camera, self.dino_fn,
            out["depth"].to(self.device), self.tcfg.max_roi_resolution,
        )

    # -- tracking ----------------------------------------------------------- #

    def start(self, verbose: bool = False) -> Dict[str, float]:
        """Render the first frame and run POGS's 100+70-step initialisation."""
        self.model.load_snapshot(self._snapshot)
        self.opt = RigidGroupOptimizer(
            self.tcfg, self.model, self.groups, raster_kwargs=self.raster_kwargs
        )
        obs = self._observation(self.render_frame())
        # Same seeding policy as the Sec. V-A driver (``evaluate.track_sequence``):
        # ``init_n_seeds=1`` is the official behaviour, and it is the correct one --
        # Sec. IV-C canonicalises each object frame so its scan-time pose is identity.
        info = self.opt.initialize_obj_pose(
            obs, n_seeds=self.tcfg.init_n_seeds, verbose=verbose
        )
        return info

    def _track_once(self, niter: Optional[int] = None) -> None:
        obs = self._observation(self.render_frame())
        if self.tcfg.use_roi:
            for i in range(self.groups.num_groups):
                try:
                    obs.add_roi(*self.opt.calculate_roi(i, obs.frame.camera))
                except RuntimeError:
                    obs.add_roi(0.0, 1.0, 0.0, 1.0)
        self.opt.set_frame(obs)
        self.opt.step(niter=niter)

    def estimated_delta(self, gi: int) -> np.ndarray:
        T = self.opt.get_part2world(gi).detach().cpu().numpy()
        return world_delta_from_part2world(T, self.centroids[gi])

    # -- frames ------------------------------------------------------------- #

    def tip_pose(self, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return _apply(delta, self.tip_scan), delta[:3, :3] @ self.tip_axis_scan

    def marker_pose(self, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return _apply(delta, self.marker_scan), delta[:3, :3] @ self.marker_normal_scan

    def desired_tip(self, target_delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """'the tool tip remains in its relative target pose to the object'."""
        p, n = self.marker_pose(target_delta)
        n = n / np.linalg.norm(n)
        return p + self.scfg.standoff_m * n, -n

    def true_error(self) -> Tuple[float, float]:
        """(position error cm, tip-axis error deg) against the *true* poses."""
        tip, axis = self.tip_pose(self.true_delta[self.tool_obj.obj_id])
        des_p, des_a = self.desired_tip(self.true_delta[self.target_obj.obj_id])
        pos = float(np.linalg.norm(tip - des_p) * 100.0)
        axis = axis / np.linalg.norm(axis)
        rot = float(np.degrees(math.acos(float(np.clip(np.dot(axis, des_a), -1.0, 1.0)))))
        return pos, rot

    def tool_tracking_error_cm(self) -> float:
        est = self.estimated_delta(self.gi_tool)
        true = self.true_delta[self.tool_obj.obj_id]
        c = self.centroids[self.gi_tool]
        return float(np.linalg.norm(_apply(est, c) - _apply(true, c)) * 100.0)

    # -- control ------------------------------------------------------------ #

    def servo_command(self) -> np.ndarray:
        """One rate-limited SE(3) command, computed entirely from POGS's
        *estimated* poses (never from ground truth)."""
        s = self.scfg
        tool_est = self.estimated_delta(self.gi_tool)
        targ_est = self.estimated_delta(self.gi_target)
        tip, axis = self.tip_pose(tool_est)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        des_p, des_a = self.desired_tip(targ_est)

        e_p = s.gain_pos * (des_p - tip)
        n = float(np.linalg.norm(e_p))
        if n > s.max_step_m:
            e_p = e_p * (s.max_step_m / n)

        w = s.gain_rot * _axis_angle_between(axis, des_a)
        a = float(np.linalg.norm(w))
        cap = math.radians(s.max_step_deg)
        if a > cap:
            w = w * (cap / a)

        # rotate about the *estimated* tip so the correction is tip-centred
        return _delta_about(tip, _rodrigues(w), e_p)

    def apply_to_tool(self, delta: np.ndarray) -> None:
        """Execute one command, subject to the arm's reach envelope."""
        oid = self.tool_obj.obj_id
        new = delta @ self.true_delta[oid]
        lo = np.array(self.scfg.reach_box_m[0::2], dtype=np.float64)
        hi = np.array(self.scfg.reach_box_m[1::2], dtype=np.float64)
        tip = _apply(new, self.tip_scan)
        clamped = np.clip(tip, lo, hi)
        if not np.allclose(clamped, tip):
            self.n_reach_limited += 1
            new = _delta_about(np.zeros(3), np.eye(3), clamped - tip) @ new
        self.true_delta[oid] = new

    def perturb_tool(self, degrees: float) -> None:
        """A human twists the tool in the gripper about the grasp axis."""
        D = self.true_delta[self.tool_obj.obj_id]
        axis = D[:3, :3] @ self.grasp_axis_scan
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        pivot = _apply(D, self._grasp_point_scan())
        R = _rodrigues(math.radians(degrees) * axis)
        self.true_delta[self.tool_obj.obj_id] = _delta_about(pivot, R, np.zeros(3)) @ D

    def _grasp_point_scan(self) -> np.ndarray:
        R_obj = np.array(
            [[math.cos(self.tool_obj.yaw), -math.sin(self.tool_obj.yaw), 0.0],
             [math.sin(self.tool_obj.yaw), math.cos(self.tool_obj.yaw), 0.0],
             [0.0, 0.0, 1.0]]
        )
        return R_obj @ np.array([-0.010, 0.0, 0.036]) + np.asarray(self.tool_obj.position)

    # -- target motion ------------------------------------------------------ #

    def sample_target_pose(self, rng: np.random.Generator) -> Tuple[float, np.ndarray]:
        """A random (yaw, translation) inside the tier workspace."""
        s = self.scfg
        ex, ey, ez = TIER_WORKSPACE_M[s.tier]
        base = np.asarray(self.target_obj.position, dtype=np.float64)
        cx, cy = s.workspace_clamp
        t = np.array(
            [
                float(np.clip(base[0] + rng.uniform(-ex / 2, ex / 2), -cx, cx)) - base[0],
                float(np.clip(base[1] + rng.uniform(-ey / 2, ey / 2), -cy, cy)) - base[1],
                float(rng.uniform(0.0, ez)),
            ]
        )
        return float(rng.uniform(-math.pi, math.pi)), t

    def set_target_pose(self, yaw: float, t: np.ndarray) -> None:
        base = np.asarray(self.target_obj.position, dtype=np.float64)
        R = _rodrigues(np.array([0.0, 0.0, yaw]))
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t + base - R @ base
        self.true_delta[self.target_obj.obj_id] = T

    def target_pose(self) -> Tuple[float, np.ndarray]:
        """Current (yaw, translation) of the target."""
        base = np.asarray(self.target_obj.position, dtype=np.float64)
        T = self.true_delta[self.target_obj.obj_id]
        yaw = math.atan2(T[1, 0], T[0, 0])
        return yaw, T[:3, 3] - (base - T[:3, :3] @ base)

    def marker_position(self) -> np.ndarray:
        return _apply(self.true_delta[self.target_obj.obj_id], self.marker_scan)

    def restore(self) -> None:
        self.model.load_snapshot(self._snapshot)


# --------------------------------------------------------------------------- #
# The Table II protocol
# --------------------------------------------------------------------------- #


def _smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _servo_frame(rig: ToolServoRig, attempt: ServoAttempt) -> Tuple[float, float]:
    """One closed-loop frame: command from the last estimate, then re-observe."""
    rig.apply_to_tool(rig.servo_command())
    rig._track_once()
    pos, rot = rig.true_error()
    attempt.pos_err_trace.append(pos)
    attempt.rot_err_trace.append(rot)
    attempt.track_pos_err_cm.append(rig.tool_tracking_error_cm())
    attempt.n_frames += 1
    return pos, rot


def _glide_target(
    rig: ToolServoRig, yaw_goal: float, t_goal: np.ndarray, attempt: ServoAttempt
) -> float:
    """Slide the target into its new pose while the tool keeps servoing.
    Returns how far the marker travelled, in cm."""
    yaw0, t0 = rig.target_pose()
    start = rig.marker_position()
    dyaw = (yaw_goal - yaw0 + math.pi) % (2 * math.pi) - math.pi
    n = max(rig.scfg.target_move_frames, 1)
    for k in range(n):
        a = _smoothstep((k + 1) / n)
        rig.set_target_pose(yaw0 + a * dyaw, (1 - a) * t0 + a * t_goal)
        _servo_frame(rig, attempt)
    return float(np.linalg.norm(rig.marker_position() - start) * 100.0)


def _servo_until(
    rig: ToolServoRig, max_frames: int, attempt: ServoAttempt
) -> Tuple[bool, float]:
    """Run the closed loop until the *true* tip pose is inside tolerance.
    Returns (converged, seconds taken)."""
    s = rig.scfg
    for k in range(max_frames):
        pos, rot = _servo_frame(rig, attempt)
        if pos <= s.pos_tol_cm and rot <= s.rot_tol_deg:
            return True, (k + 1) * s.dt
    return False, max_frames * s.dt


def run_tool_servoing(
    model: POGSModel,
    groups: ObjectGroups,
    capture: CaptureData,
    dino_fn: DinoFn,
    cfg: POGSConfig,
    group_to_obj: Dict[int, int],
    kind: str = "clockwise",
    tier: int = 1,
    servo_cfg: Optional[ServoConfig] = None,
    tool_name: str = "power drill",
    target_name: str = "wooden shelf",
    verbose: bool = True,
) -> ServoResult:
    """One row x one tier of Table II.

    ``kind='follow'``      the target moves; success = the tool follows it.
    ``kind='clockwise'``   after converging, the tool is twisted +theta in the
                           gripper; success = it recovers within tolerance.
    ``kind='ccw'``         same, -theta.
    """
    if kind not in ("follow", "clockwise", "ccw"):
        raise ValueError(f"kind must be follow/clockwise/ccw, got {kind!r}")

    scfg = servo_cfg or ServoConfig()
    scfg.tier = tier
    rig = ToolServoRig(
        model, groups, capture, dino_fn, cfg, group_to_obj,
        tool_name=tool_name, target_name=target_name, servo_cfg=scfg,
    )
    theta = TIER_PERTURB_DEG[tier] * (1.0 if kind == "clockwise" else -1.0)
    kind_salt = {"follow": 0, "clockwise": 1, "ccw": 2}[kind]

    result = ServoResult(tier=tier, kind=kind, attempts=[])
    t_all = time.time()

    for trial in range(scfg.n_trials):
        # deterministic across processes: str.__hash__ is salted per interpreter
        rng = np.random.default_rng(scfg.seed + 1000 * tier + 100 * trial + 7 * kind_salt)
        rig.true_delta = {rig.tool_obj.obj_id: np.eye(4), rig.target_obj.obj_id: np.eye(4)}
        info = rig.start(verbose=False)
        if verbose:
            print(f"  [{kind} tier{tier}] trial {trial + 1}/{scfg.n_trials} "
                  f"init fine_loss={info['fine_loss']:.4f}")

        # --- initial convergence: bring tool near target before first motion ---
        init_attempt = ServoAttempt(
            trial=trial, pose_idx=-1, kind=kind, success=False,
            time_s=float("nan"), final_pos_err_cm=float("nan"),
            final_rot_err_deg=float("nan"), n_frames=0,
        )
        _servo_until(rig, scfg.settle_frames, init_attempt)

        for pose_idx in range(scfg.n_target_poses):
            yaw_goal, t_goal = rig.sample_target_pose(rng)

            attempt = ServoAttempt(
                trial=trial, pose_idx=pose_idx, kind=kind, success=False,
                time_s=float("nan"), final_pos_err_cm=float("nan"),
                final_rot_err_deg=float("nan"), n_frames=0,
            )

            # the human slides the target into place; the tool follows as it goes
            disp = _glide_target(rig, yaw_goal, t_goal, attempt)
            result.target_displacements_cm.append(disp)

            settled, t_settle = _servo_until(rig, scfg.settle_frames, attempt)
            t_settle += scfg.target_move_frames * scfg.dt

            if kind == "follow":
                ok, took = settled, t_settle
            else:
                # the human twist happens whether or not the settle converged,
                # which is the paper's protocol ('At each pose, a human moves
                # the drill 15-30 degrees ...')
                rig.perturb_tool(theta)
                rig._track_once()
                ok, took = _servo_until(rig, scfg.recovery_frames, attempt)

            pos, rot = rig.true_error()
            attempt.success = bool(ok)
            attempt.time_s = float(took)
            attempt.final_pos_err_cm = pos
            attempt.final_rot_err_deg = rot
            result.attempts.append(attempt)

            if verbose:
                print(f"      pose {pose_idx + 1}/{scfg.n_target_poses} "
                      f"(target moved {disp:.1f} cm): "
                      f"{'OK ' if ok else 'FAIL'} in {took:.2f}s  "
                      f"final {pos:.2f} cm / {rot:.1f} deg  "
                      f"(tool track err {np.mean(attempt.track_pos_err_cm):.2f} cm)")

    rig.restore()
    result.wall_time_s = time.time() - t_all
    result.reach_limited_frames = rig.n_reach_limited
    return result


def tool_servoing_table(results: Sequence[ServoResult]):
    """Table II shape: rows = Clockwise / CCW / Follow Target, columns = tier."""
    label = {"clockwise": "Clockwise", "ccw": "CCW", "follow": "Follow Target"}
    rows: Dict[str, Dict[str, object]] = {}
    for r in results:
        row = rows.setdefault(label[r.kind], {})
        row[f"Tier {r.tier} Success Rate"] = f"{r.n_success}/{r.n}"
        row[f"Tier {r.tier} Time (s)"] = round(r.mean_time_s(), 2)
    order = [label[k] for k in ("clockwise", "ccw", "follow") if label[k] in rows]
    try:
        import pandas as pd

        return pd.DataFrame([rows[k] for k in order], index=order)
    except ImportError:
        return {k: rows[k] for k in order}


def run_tool_servoing_suite(
    model: POGSModel,
    groups: ObjectGroups,
    capture: CaptureData,
    dino_fn: DinoFn,
    cfg: POGSConfig,
    group_to_obj: Dict[int, int],
    tiers: Sequence[int] = (1, 2),
    kinds: Sequence[str] = ("clockwise", "ccw", "follow"),
    servo_cfg: Optional[ServoConfig] = None,
    verbose: bool = True,
    **kwargs,
) -> List[ServoResult]:
    """The whole of Table II."""
    import copy

    out: List[ServoResult] = []
    for tier in tiers:
        for kind in kinds:
            sc = copy.deepcopy(servo_cfg) if servo_cfg is not None else ServoConfig()
            if verbose:
                print(f"\n--- tool servoing: {kind}, tier {tier} "
                      f"({sc.n_trials} trials x {sc.n_target_poses} poses) ---")
            out.append(
                run_tool_servoing(
                    model, groups, capture, dino_fn, cfg, group_to_obj,
                    kind=kind, tier=tier, servo_cfg=sc, verbose=verbose, **kwargs,
                )
            )
    return out