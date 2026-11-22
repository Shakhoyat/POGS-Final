"""
Information-adaptive measurement noise for the POGS Kalman prior (plan rev. 2, M3 / task T4).

``InfoKalmanPrior`` is a ``KalmanMotionPrior`` whose measurement covariance R is set every frame from
the tracker loss's own curvature (``model.pogs.information``), through the ``observe`` hook of
``evaluate.track_sequence``. Nothing in the tracking loop is copied (CLAUDE.md rule 2).

Ladder (compact pose graph, arXiv 2610.11815, Table 6, used the same structure):
  fixed   R = config constants (= the existing v2 KF; reference)
  trace   R = isotropic per block, sized by tr(Sigma_block) / 3
  full    R = Sigma blocks (anisotropic; rotation and translation blocks, cross terms dropped because
          the filter is block-decoupled)
  oracle  R = squared ground-truth error per block + floor. Upper bound only: uses GT, so it is an
          evaluation ablation, never a method (CLAUDE.md rule 3).

Defaults (warm_start=True, fuse=False, check_loss=False) equal the v2 "POGS + KF" prior
(experiments/v2_kalman_pogs/config.py::_kf), so mode "fixed" reproduces it exactly and the other modes
change only R. With fuse=False, R acts through the prediction used as the next warm start; fusing the
output is a separate, secondary rung.

Sigma = c * (H + lam I)^-1, H = regression Hessian of the loss at 1 deg / 3 mm. The scale c is not
known for a robust feature loss; it is fitted on calibration seeds (NEES = dof) and frozen before test
seeds are run (plan rev. 2, sec. 5). Frames: H is in the object frame (right perturbation), the filter's
rotation error is world-frame (left), so blocks are rotated by the current object rotation R_obj.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
import torch

from .geometry import quat_to_rotmat
from .information import tracker_loss_hessian
from .kalman import KalmanConfig, KalmanMotionPrior

MODES = ("fixed", "trace", "full", "oracle")


def body_cov_to_world(Sigma6: np.ndarray, R_obj: np.ndarray):
    """(6, 6) body-frame covariance (rot first) -> world-frame (3, 3) rotation and translation blocks."""
    Rr = R_obj @ Sigma6[:3, :3] @ R_obj.T
    Rt = R_obj @ Sigma6[3:, 3:] @ R_obj.T
    return 0.5 * (Rr + Rr.T), 0.5 * (Rt + Rt.T)


def information_to_cov(H: np.ndarray, scale: float, rel_lam: float = 1e-3) -> np.ndarray:
    """Sigma = scale * (H_psd + lam I)^-1 with lam = rel_lam * lambda_max (keeps null directions finite
    but very uncertain, so the filter keeps its prior there, X-ICP style)."""
    w, V = np.linalg.eigh(0.5 * (H + H.T))
    w = np.clip(w, 0.0, None)
    lam = rel_lam * max(float(w.max()), 1e-12)
    return scale * (V / (w + lam)) @ V.T


class InfoKalmanPrior(KalmanMotionPrior):
    """KalmanMotionPrior with per-frame R from the loss information (see module docstring).

    ``oracle_err_fn(k) -> (rot_err_world(3), trans_err_world(3))`` supplies GT errors for the oracle
    mode only. ``r_floor`` bounds R from below (rad^2, m^2) so a very sharp curvature cannot freeze the
    filter. Records per frame the blocks used and the information summary for calibration (E4)."""

    def __init__(self, num_objects: int, mode: str = "full", scale: float = 1.0,
                 cfg: Optional[KalmanConfig] = None, warm_start: bool = True, fuse: bool = False,
                 check_loss: bool = False, use_depth: bool = True, scale_rot: float = float(np.radians(1.0)),
                 scale_trans: float = 3e-3, n_fit: int = 12, r_floor=(1e-6, 1e-8),
                 oracle_err_fn: Optional[Callable[[int], tuple]] = None):
        assert mode in MODES, mode
        super().__init__(num_objects, cfg, warm_start=warm_start, fuse=fuse, check_loss=check_loss)
        self.mode, self.scale, self.use_depth = mode, scale, use_depth
        self.hk = dict(h_rot=scale_rot, h_trans=scale_trans, method="regression", n=n_fit)
        self.r_floor = r_floor
        self.oracle_err_fn = oracle_err_fn
        self.log: List[dict] = []

    # -- R from a body-frame covariance (pure, testable) -------------------- #

    def set_measurement_cov(self, k: int, Sigma6: np.ndarray, R_obj: np.ndarray) -> None:
        f = self.filters[k]
        Rr, Rt = body_cov_to_world(Sigma6, R_obj)
        if self.mode == "trace":
            Rr = np.eye(3) * np.trace(Rr) / 3.0
            Rt = np.eye(3) * np.trace(Rt) / 3.0
        f.R_r = Rr + np.eye(3) * self.r_floor[0]
        f.R_t = Rt + np.eye(3) * self.r_floor[1]

    def set_oracle_cov(self, k: int, rot_err: np.ndarray, trans_err: np.ndarray) -> None:
        f = self.filters[k]
        f.R_r = np.diag(np.asarray(rot_err, float) ** 2) + np.eye(3) * self.r_floor[0]
        f.R_t = np.diag(np.asarray(trans_err, float) ** 2) + np.eye(3) * self.r_floor[1]

    # -- hook --------------------------------------------------------------- #

    def observe(self, opt, obs) -> None:
        if self.mode == "fixed":
            return
        rec = {}
        for k in range(len(self.filters)):
            if self.mode == "oracle":
                if self.oracle_err_fn is None:
                    raise ValueError("oracle mode needs oracle_err_fn (evaluation-only ablation)")
                self.set_oracle_cov(k, *self.oracle_err_fn(k))
                continue
            try:
                h = tracker_loss_hessian(opt, obs, k, use_depth=self.use_depth, **self.hk)
            except RuntimeError:
                continue                      # keep last R (object out of view)
            q = opt.part_deltas.detach()[k, 3:].float().cpu()
            R_obj = quat_to_rotmat(q.unsqueeze(0))[0].double().numpy()
            Sigma = information_to_cov(h["H"], self.scale)
            self.set_measurement_cov(k, Sigma, R_obj)
            rec[k] = {"H": h["H"], "fit_resid": h["fit_resid"], "R_r": self.filters[k].R_r.copy(),
                      "R_t": self.filters[k].R_t.copy()}
        self.log.append(rec)
