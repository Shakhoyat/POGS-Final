"""
SE(3) Kinematic Extended Kalman Filter for Persistent Object Tracking (v2).

Provides constant-velocity kinematic prediction on the SE(3) Lie group:
1. Warm-starts gradient-based Gaussian Splatting tracking within the basin of attraction.
2. Bridges occlusions (e.g. robotic gripper or human hand grasping) via kinematic momentum.
3. Filters out noisy optimization outliers and clamps velocity to physical manipulation limits.
4. Protects static scene groups (e.g. background table) from motion drift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .geometry import (
    identity_posevec,
    quat_multiply,
    quat_normalize,
    quat_to_rotmat,
    rotmat_to_quat,
    so3_exp,
    so3_log,
)


@dataclass
class KalmanConfig:
    dt: float = 0.2                     # Time step in seconds (5 Hz default)
    # Physical velocity limits (tabletop robotics)
    max_linear_vel: float = 0.35        # Maximum linear velocity in m/s (35 cm/s)
    max_angular_vel: float = 2.0        # Maximum angular velocity in rad/s (~115 deg/s)
    # Dynamics damping
    vel_decay: float = 0.88             # Linear velocity damping factor per step
    omega_decay: float = 0.82           # Angular velocity damping factor per step
    # Process & Measurement noise
    q_pos: float = 0.015                # Position process noise (m)
    q_rot: float = 0.040                # Orientation process noise (rad)
    r_pos: float = 0.020                # Nominal position measurement noise (m)
    r_rot: float = 0.050                # Nominal rotation measurement noise (rad)
    r_pos_occluded: float = 0.400       # Inflated measurement noise under occlusion (m)
    r_rot_occluded: float = 0.900       # Inflated rotation measurement noise under occlusion (rad)
    # Innovation gating
    gate_threshold: float = 16.8        # Mahalanobis chi-sq threshold (6 DoF, p=0.01)
    use_gating: bool = True
    warmstart_pogs: bool = True         # Seed POGS gradient descent with Kalman prediction


class SingleObjectSE3KalmanFilter:
    """EKF tracking 6-DoF pose and 6-DoF spatial velocity for a single rigid object."""

    def __init__(self, config: KalmanConfig, device: str = "cpu", dtype: torch.dtype = torch.float32, is_static: bool = False):
        self.cfg = config
        self.device = device
        self.dtype = dtype
        self.is_static = is_static

        # State:
        # pos: (3,), quat: (4,) wxyz, vel: (3,), omega: (3,)
        self.pos = torch.zeros(3, device=device, dtype=dtype)
        self.quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)
        self.vel = torch.zeros(3, device=device, dtype=dtype)
        self.omega = torch.zeros(3, device=device, dtype=dtype)

        # 6x6 pose covariance
        self.P_pos = torch.eye(3, device=device, dtype=dtype) * (self.cfg.q_pos**2)
        self.P_rot = torch.eye(3, device=device, dtype=dtype) * (self.cfg.q_rot**2)

        self.prev_pos = torch.zeros(3, device=device, dtype=dtype)
        self.prev_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)
        self.is_initialized = False

    def reset(self, init_pos: torch.Tensor, init_quat: torch.Tensor):
        """Initialize filter with first observation."""
        self.pos = init_pos.detach().clone().to(self.device, self.dtype)
        self.quat = quat_normalize(init_quat.detach().clone().to(self.device, self.dtype).unsqueeze(0))[0]
        self.prev_pos = self.pos.clone()
        self.prev_quat = self.quat.clone()
        self.vel = torch.zeros(3, device=self.device, dtype=self.dtype)
        self.omega = torch.zeros(3, device=self.device, dtype=self.dtype)
        self.P_pos = torch.eye(3, device=self.device, dtype=self.dtype) * (self.cfg.q_pos**2)
        self.P_rot = torch.eye(3, device=self.device, dtype=self.dtype) * (self.cfg.q_rot**2)
        self.is_initialized = True

    def predict(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Constant-velocity kinematic prediction on SE(3). Returns (pred_pos, pred_quat)."""
        if self.is_static or not self.is_initialized:
            return self.pos, self.quat

        dt = self.cfg.dt
        # Clamp velocity to physical manipulation limits
        v_norm = torch.norm(self.vel)
        if v_norm > self.cfg.max_linear_vel:
            self.vel = self.vel * (self.cfg.max_linear_vel / (v_norm + 1e-8))

        w_norm = torch.norm(self.omega)
        if w_norm > self.cfg.max_angular_vel:
            self.omega = self.omega * (self.cfg.max_angular_vel / (w_norm + 1e-8))

        # Position propagation
        pred_pos = self.pos + self.vel * dt

        # Orientation propagation using Lie algebra exponential map
        rot_delta = so3_exp((self.omega * dt).unsqueeze(0))[0]
        pred_quat = quat_normalize(quat_multiply(rot_delta.unsqueeze(0), self.quat.unsqueeze(0)))[0]

        # Velocity damping
        self.vel = self.vel * self.cfg.vel_decay
        self.omega = self.omega * self.cfg.omega_decay

        # Covariance propagation
        self.P_pos = self.P_pos + torch.eye(3, device=self.device, dtype=self.dtype) * (self.cfg.q_pos**2)
        self.P_rot = self.P_rot + torch.eye(3, device=self.device, dtype=self.dtype) * (self.cfg.q_rot**2)

        self.pos = pred_pos
        self.quat = pred_quat
        return self.pos, self.quat

    def update(
        self,
        meas_pos: torch.Tensor,
        meas_quat: torch.Tensor,
        is_occluded: bool = False,
        quality_score: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Measurement update using converged POGS pose."""
        if self.is_static or not self.is_initialized:
            return self.pos, self.quat

        meas_pos = meas_pos.to(self.device, self.dtype)
        meas_quat = quat_normalize(meas_quat.to(self.device, self.dtype).unsqueeze(0))[0]

        # Measurement uncertainties
        r_p = self.cfg.r_pos_occluded if is_occluded else self.cfg.r_pos / max(quality_score, 0.1)
        r_r = self.cfg.r_rot_occluded if is_occluded else self.cfg.r_rot / max(quality_score, 0.1)
        R_pos = torch.eye(3, device=self.device, dtype=self.dtype) * (r_p**2)
        R_rot = torch.eye(3, device=self.device, dtype=self.dtype) * (r_r**2)

        # Innovation residuals
        pos_res = meas_pos - self.pos

        # Geodesic rotation error in Lie algebra so(3)
        q_pred_inv = torch.cat([self.quat[:1], -self.quat[1:]])
        q_diff = quat_multiply(meas_quat.unsqueeze(0), q_pred_inv.unsqueeze(0))[0]
        rot_res = so3_log(q_diff.unsqueeze(0))[0]

        # Innovation gating: if residual exceeds 15 cm or 45 deg unexpectedly, treat as occlusion/outlier
        if self.cfg.use_gating:
            if float(torch.norm(pos_res)) > 0.15 or float(torch.norm(rot_res)) > 0.8:
                is_occluded = True
                R_pos *= 10.0
                R_rot *= 10.0

        # Position Kalman update
        S_pos = self.P_pos + R_pos
        K_pos = self.P_pos @ torch.linalg.inv(S_pos)
        self.pos = self.pos + K_pos @ pos_res
        self.P_pos = (torch.eye(3, device=self.device, dtype=self.dtype) - K_pos) @ self.P_pos

        # Orientation Kalman update
        S_rot = self.P_rot + R_rot
        K_rot = self.P_rot @ torch.linalg.inv(S_rot)
        dq = so3_exp((K_rot @ rot_res).unsqueeze(0))[0]
        self.quat = quat_normalize(quat_multiply(dq.unsqueeze(0), self.quat.unsqueeze(0)))[0]
        self.P_rot = (torch.eye(3, device=self.device, dtype=self.dtype) - K_rot) @ self.P_rot

        # Smooth velocity estimation from filtered displacement if not occluded
        if not is_occluded:
            dt = self.cfg.dt
            meas_v = (self.pos - self.prev_pos) / dt
            # Angular velocity from delta quaternion
            dq_step = quat_multiply(self.quat.unsqueeze(0), torch.cat([self.prev_quat[:1], -self.prev_quat[1:]]).unsqueeze(0))[0]
            meas_w = so3_log(dq_step.unsqueeze(0))[0] / dt

            # Alpha filter (0.7 momentum, 0.3 measurement)
            self.vel = 0.70 * self.vel + 0.30 * meas_v
            self.omega = 0.70 * self.omega + 0.30 * meas_w

            # Clamp velocities
            v_norm = torch.norm(self.vel)
            if v_norm > self.cfg.max_linear_vel:
                self.vel = self.vel * (self.cfg.max_linear_vel / (v_norm + 1e-8))
            w_norm = torch.norm(self.omega)
            if w_norm > self.cfg.max_angular_vel:
                self.omega = self.omega * (self.cfg.max_angular_vel / (w_norm + 1e-8))

        self.prev_pos = self.pos.clone()
        self.prev_quat = self.quat.clone()

        return self.pos, self.quat

    def get_pose_matrix(self) -> torch.Tensor:
        T = torch.eye(4, device=self.device, dtype=self.dtype)
        T[:3, :3] = quat_to_rotmat(self.quat.unsqueeze(0))[0]
        T[:3, 3] = self.pos
        return T


class SE3MultiObjectKalmanFilter:
    """Manages independent SE(3) Kalman Filters for each object group."""

    def __init__(
        self,
        num_objects: int,
        config: Optional[KalmanConfig] = None,
        device: str = "cpu",
        active_groups: Optional[List[int]] = None,
    ):
        self.num_objects = num_objects
        self.cfg = config or KalmanConfig()
        self.device = device
        self.active_groups = set(active_groups) if active_groups is not None else set(range(num_objects))
        self.filters = [
            SingleObjectSE3KalmanFilter(
                self.cfg,
                device=device,
                is_static=(i not in self.active_groups),
            )
            for i in range(num_objects)
        ]

    def initialize_from_poses(self, posevecs_7: torch.Tensor):
        for i in range(self.num_objects):
            self.filters[i].reset(posevecs_7[i, :3], posevecs_7[i, 3:])

    def predict_all(self) -> torch.Tensor:
        pred_poses = torch.zeros(self.num_objects, 7, device=self.device)
        for i in range(self.num_objects):
            p, q = self.filters[i].predict()
            pred_poses[i, :3] = p
            pred_poses[i, 3:] = q
        return pred_poses

    def update_all(
        self,
        meas_poses_7: torch.Tensor,
        is_occluded: bool = False,
        quality_scores: Optional[List[float]] = None,
    ) -> torch.Tensor:
        filtered_poses = torch.zeros(self.num_objects, 7, device=self.device)
        for i in range(self.num_objects):
            q_score = quality_scores[i] if quality_scores is not None else 1.0
            p, q = self.filters[i].update(
                meas_poses_7[i, :3],
                meas_poses_7[i, 3:],
                is_occluded=is_occluded,
                quality_score=q_score,
            )
            filtered_poses[i, :3] = p
            filtered_poses[i, 3:] = q
        return filtered_poses

    def get_matrices(self) -> torch.Tensor:
        return torch.stack([f.get_pose_matrix() for f in self.filters], dim=0)
