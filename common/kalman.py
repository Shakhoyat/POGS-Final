"""
Constant-velocity Kalman filter on SE(3), one per tracked object (used by v2).

State of object i, expressed in exactly the parameters POGS optimises
(``RigidGroupOptimizer.part_deltas``: translation of the centroid in the world
frame + world-frame rotation about the centroid):

    nominal:  p (3), q (4, wxyz), v (3), w (3)
    error:    [dp, dv] (6)   and   [dtheta, dw] (6)

Translation and rotation are filtered as two independent constant-velocity
models driven by white-noise acceleration (the textbook "discrete white noise
acceleration" model, Bar-Shalom et al. Sec. 6.3.2). Rotation uses an error-state
formulation: the nominal quaternion is propagated with ``exp(w dt) * q`` and the
3-D error lives in the tangent space, so there is no Euler-angle singularity and
no yaw-only restriction.

Measurements are full 6-DoF poses (the optimiser's converged ``part_deltas``).
The normalised innovation squared (NIS) of every update is recorded so filter
consistency can be checked: for a well-tuned filter its mean is ~3 per block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from .geometry import quat_multiply, quat_normalize, quat_conjugate, so3_exp, so3_log


@dataclass
class KalmanConfig:
    dt: float = 0.2
    """Frame period (5 Hz, paper Sec. VI)."""
    accel_std: float = 0.5
    """White-noise linear acceleration (m/s^2)."""
    ang_accel_std: float = 4.0
    """White-noise angular acceleration (rad/s^2)."""
    meas_pos_std: float = 0.01
    """Std of the optimiser's converged centroid estimate (m)."""
    meas_rot_std: float = 0.06
    """Std of the optimiser's converged rotation estimate (rad)."""
    init_vel_std: float = 0.05
    init_omega_std: float = 0.2
    init_pos_std: float = 0.005
    init_rot_std: float = 0.02
    gate_chi2: float = 0.0
    """Reject an update block whose NIS exceeds this (0 = no gating).
    chi2(3 dof, p=0.999) = 16.27."""
    vel_damping_tau: float = 1.0
    """Velocity decays as exp(-dt / tau) per step unless measurements sustain
    it (0 = pure constant velocity). Without it a warm-started optimiser,
    whose output is not independent of the prediction, can feed a runaway
    velocity estimate (observed: metres of drift once tracking slipped)."""
    max_speed: float = 0.5
    """Clamp on |v| (m/s); hand-held / robot manipulation speeds (0 = off)."""
    max_ang_speed: float = 3.0
    """Clamp on |w| (rad/s) (0 = off)."""


def _cv_matrices(dt: float, sigma: float, damping: float = 1.0):
    F = np.eye(6)
    F[:3, 3:] = np.eye(3) * dt
    F[3:, 3:] = np.eye(3) * damping
    q11, q12, q22 = dt**4 / 4.0, dt**3 / 2.0, dt**2
    Q = np.zeros((6, 6))
    Q[:3, :3] = np.eye(3) * q11
    Q[:3, 3:] = np.eye(3) * q12
    Q[3:, :3] = np.eye(3) * q12
    Q[3:, 3:] = np.eye(3) * q22
    return F, Q * sigma**2


class SE3KalmanFilter:
    """Single-object constant-velocity filter (float64, CPU)."""

    H = np.hstack([np.eye(3), np.zeros((3, 3))])

    def __init__(self, cfg: KalmanConfig):
        self.cfg = cfg
        self.alpha = float(np.exp(-cfg.dt / cfg.vel_damping_tau)) if cfg.vel_damping_tau > 0 else 1.0
        self.F_t, self.Q_t = _cv_matrices(cfg.dt, cfg.accel_std, self.alpha)
        self.F_r, self.Q_r = _cv_matrices(cfg.dt, cfg.ang_accel_std, self.alpha)
        self.R_t = np.eye(3) * cfg.meas_pos_std**2
        self.R_r = np.eye(3) * cfg.meas_rot_std**2
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.w = np.zeros(3)
        self.P_t = np.eye(6)
        self.P_r = np.eye(6)
        self.nis: List[float] = []
        self.n_rejected = 0
        self.initialized = False

    # -- helpers ----------------------------------------------------------- #

    @staticmethod
    def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return quat_multiply(torch.from_numpy(a)[None], torch.from_numpy(b)[None])[0].numpy()

    @staticmethod
    def _exp(v: np.ndarray) -> np.ndarray:
        return so3_exp(torch.from_numpy(v)[None])[0].numpy()

    @staticmethod
    def _log(q: np.ndarray) -> np.ndarray:
        return so3_log(torch.from_numpy(q)[None])[0].numpy()

    @staticmethod
    def _conj(q: np.ndarray) -> np.ndarray:
        return quat_conjugate(torch.from_numpy(q)[None])[0].numpy()

    # -- API --------------------------------------------------------------- #

    def initialize(self, posevec: np.ndarray) -> None:
        c = self.cfg
        self.p = np.asarray(posevec[:3], dtype=np.float64).copy()
        q = np.asarray(posevec[3:], dtype=np.float64)
        self.q = q / np.linalg.norm(q)
        self.v = np.zeros(3)
        self.w = np.zeros(3)
        self.P_t = np.diag([c.init_pos_std**2] * 3 + [c.init_vel_std**2] * 3)
        self.P_r = np.diag([c.init_rot_std**2] * 3 + [c.init_omega_std**2] * 3)
        self.initialized = True

    def predict(self) -> np.ndarray:
        assert self.initialized
        dt = self.cfg.dt
        self.p = self.p + self.v * dt
        self.q = self._qmul(self._exp(self.w * dt), self.q)
        self.v = self.v * self.alpha
        self.w = self.w * self.alpha
        self.q /= np.linalg.norm(self.q)
        self.P_t = self.F_t @ self.P_t @ self.F_t.T + self.Q_t
        self.P_r = self.F_r @ self.P_r @ self.F_r.T + self.Q_r
        return self.posevec()

    def _block_update(self, P, R, y):
        S = self.H @ P @ self.H.T + R
        S_inv = np.linalg.inv(S)
        nis = float(y @ S_inv @ y)
        if self.cfg.gate_chi2 > 0 and nis > self.cfg.gate_chi2:
            return None, P, nis
        K = P @ self.H.T @ S_inv
        I_KH = np.eye(6) - K @ self.H
        P_new = I_KH @ P @ I_KH.T + K @ R @ K.T      # Joseph form
        return K @ y, P_new, nis

    def update(self, posevec: np.ndarray) -> np.ndarray:
        if not self.initialized:
            self.initialize(posevec)
            return self.posevec()
        pm = np.asarray(posevec[:3], dtype=np.float64)
        qm = np.asarray(posevec[3:], dtype=np.float64)
        qm = qm / np.linalg.norm(qm)

        dx, self.P_t, nis_t = self._block_update(self.P_t, self.R_t, pm - self.p)
        if dx is not None:
            self.p = self.p + dx[:3]
            self.v = self.v + dx[3:]
        else:
            self.n_rejected += 1

        y_r = self._log(self._qmul(qm, self._conj(self.q)))      # world-frame error
        dx, self.P_r, nis_r = self._block_update(self.P_r, self.R_r, y_r)
        if dx is not None:
            self.q = self._qmul(self._exp(dx[:3]), self.q)
            self.q /= np.linalg.norm(self.q)
            self.w = self.w + dx[3:]
        else:
            self.n_rejected += 1
        self.v = self._clamp(self.v, self.cfg.max_speed)
        self.w = self._clamp(self.w, self.cfg.max_ang_speed)
        self.nis.append(nis_t + nis_r)
        return self.posevec()

    @staticmethod
    def _clamp(x: np.ndarray, limit: float) -> np.ndarray:
        n = float(np.linalg.norm(x))
        return x * (limit / n) if limit > 0 and n > limit else x

    def posevec(self) -> np.ndarray:
        return np.concatenate([self.p, self.q]).astype(np.float32)


class MultiObjectKalman:
    """Independent filters, one per tracked object; (K, 7) tensors in and out."""

    def __init__(self, num_objects: int, cfg: Optional[KalmanConfig] = None):
        self.cfg = cfg or KalmanConfig()
        self.filters = [SE3KalmanFilter(self.cfg) for _ in range(num_objects)]

    def initialize(self, posevecs: torch.Tensor) -> None:
        pv = posevecs.detach().cpu().double().numpy()
        for f, p in zip(self.filters, pv):
            f.initialize(p)

    def predict(self, device=None) -> torch.Tensor:
        out = torch.from_numpy(np.stack([f.predict() for f in self.filters]))
        return out.to(device) if device is not None else out

    def update(self, posevecs: torch.Tensor, device=None) -> torch.Tensor:
        pv = posevecs.detach().cpu().double().numpy()
        out = torch.from_numpy(np.stack([f.update(p) for f, p in zip(self.filters, pv)]))
        return out.to(device) if device is not None else out

    def mean_nis(self) -> float:
        vals = [v for f in self.filters for v in f.nis]
        return float(np.mean(vals)) if vals else float("nan")

    def n_rejected(self) -> int:
        return sum(f.n_rejected for f in self.filters)


class KalmanMotionPrior(MultiObjectKalman):
    """Adapter for ``evaluate.track_sequence(motion_prior=...)``.

    warm_start  start each frame's optimisation from the filter's prediction
                instead of the previous frame's estimate.
    fuse        replace the optimiser's output with the filtered estimate (it
                then also seeds the next frame). Note that the optimiser's
                output is not an independent measurement when it was
                warm-started from the prediction, so fusing double-counts the
                prior; this is reported, not hidden (see ``mean_nis``).
    check_loss  before optimising, keep the predicted start for an object only
                if it lowers the tracking loss compared with the previous
                estimate (greedy per object; ROIs come from the previous
                estimate, as in base POGS). Guards against bad predictions.
    """

    def __init__(self, num_objects: int, cfg: Optional[KalmanConfig] = None,
                 warm_start: bool = True, fuse: bool = False, check_loss: bool = False):
        super().__init__(num_objects, cfg)
        self.warm_start = warm_start
        self.fuse = fuse
        self.check_loss = check_loss
