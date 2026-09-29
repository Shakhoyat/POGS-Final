"""
Extended 6-DoF Kalman Filter for POGS Visual Tracking.

Implements a constant-velocity kinematic model with state estimation:
- State vector: x = [p_x, p_y, p_z, v_x, v_y, v_z, yaw, omega_z]^T (8-dim)
- Differentiable SE(3) pose translation + continuous vertical yaw angle
- Predict-Step warm-starts the 3D Gaussian Splatting optimization
- Update-Step fuses POGS gradient descent pose with temporal covariance
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Tuple, Optional


class PoseKalmanFilter6DoF:
    """Kalman Filter with Constant-Velocity kinematics for Tabletop Pick-and-Place."""

    def __init__(
        self,
        dt: float = 0.2, # default 5 Hz
        process_noise_pos: float = 1e-4,
        process_noise_vel: float = 1e-3,
        process_noise_yaw: float = 1e-3,
        process_noise_omega: float = 1e-2,
        measure_noise_pos: float = 5e-4,
        measure_noise_yaw: float = 5e-3,
    ):
        self.dt = dt
        self.dim_x = 8  # [x, y, z, vx, vy, vz, yaw, omega]
        self.dim_z = 4  # [x, y, z, yaw]

        # State transition matrix F
        self.F = np.eye(self.dim_x, dtype=np.float32)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt
        self.F[6, 7] = dt

        # Measurement matrix H
        self.H = np.zeros((self.dim_z, self.dim_x), dtype=np.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 6] = 1.0

        # Process noise covariance Q
        self.Q = np.zeros((self.dim_x, self.dim_x), dtype=np.float32)
        # Position block
        q_pos = process_noise_pos
        q_vel = process_noise_vel
        self.Q[0, 0] = self.Q[1, 1] = self.Q[2, 2] = q_pos * (dt**3) / 3.0
        self.Q[3, 3] = self.Q[4, 4] = self.Q[5, 5] = q_vel * dt
        self.Q[0, 3] = self.Q[1, 4] = self.Q[2, 5] = q_pos * (dt**2) / 2.0
        self.Q[3, 0] = self.Q[4, 1] = self.Q[5, 2] = self.Q[0, 3]
        # Yaw block
        q_yaw = process_noise_yaw
        q_omega = process_noise_omega
        self.Q[6, 6] = q_yaw * (dt**3) / 3.0
        self.Q[7, 7] = q_omega * dt
        self.Q[6, 7] = self.Q[7, 6] = q_yaw * (dt**2) / 2.0

        # Measurement noise covariance R
        self.R = np.eye(self.dim_z, dtype=np.float32)
        self.R[0, 0] = self.R[1, 1] = self.R[2, 2] = measure_noise_pos
        self.R[3, 3] = measure_noise_yaw

        # State vector x and covariance P
        self.x = np.zeros(self.dim_x, dtype=np.float32)
        self.P = np.eye(self.dim_x, dtype=np.float32) * 0.01

        self.is_initialized = False

    def initialize(self, pos: np.ndarray, yaw_rad: float = 0.0):
        """Initializes the filter with the initial scan pose and zero velocity."""
        self.x[:] = 0.0
        self.x[0:3] = pos
        self.x[6] = yaw_rad
        self.P = np.eye(self.dim_x, dtype=np.float32) * 1e-4
        self.is_initialized = True

    def predict(self) -> Tuple[np.ndarray, float]:
        """Runs the time update step: x_pred = F * x, P_pred = F * P * F^T + Q.
        Returns:
            predicted_pos: (3,)
            predicted_yaw: float (radians)
        """
        if not self.is_initialized:
            raise RuntimeError("Kalman filter must be initialized first.")

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0:3].copy(), float(self.x[6])

    def update(self, measured_pos: np.ndarray, measured_yaw: float) -> Tuple[np.ndarray, float]:
        """Runs the measurement update step with the POGS gradient descent estimate.
        Returns:
            fused_pos: (3,)
            fused_yaw: float (radians)
        """
        if not self.is_initialized:
            self.initialize(measured_pos, measured_yaw)
            return measured_pos.copy(), float(measured_yaw)

        z = np.zeros(self.dim_z, dtype=np.float32)
        z[0:3] = measured_pos
        # Continuous angle unwrap for innovation
        yaw_diff = (measured_yaw - self.x[6] + np.pi) % (2.0 * np.pi) - np.pi
        z[3] = self.x[6] + yaw_diff

        # Innovation
        y = z - self.H @ self.x

        # Innovation covariance S
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain K
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Updated state
        self.x = self.x + K @ y

        # Updated covariance (Joseph form for numerical stability)
        I_KH = np.eye(self.dim_x, dtype=np.float32) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return self.x[0:3].copy(), float(self.x[6])

    def get_state(self) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Returns (position, velocity, yaw, angular_velocity)."""
        return self.x[0:3].copy(), self.x[3:6].copy(), float(self.x[6]), float(self.x[7])
