"""
SE(3) / SO(3) utilities and the pinhole camera used throughout POGS.

The official release performs the per-Gaussian rigid transform inside an
NVIDIA-Warp kernel (``pogs/tracking/utils.py::apply_to_model``) and uses
``viser.transforms`` for the SE(3) algebra.  Neither is available in a stock
Kaggle image, so both are re-implemented here in plain differentiable PyTorch.
The maths is identical: a group pose is stored as the 7-vector

    p = [t_x, t_y, t_z, q_w, q_x, q_y, q_z]

and a Gaussian at world position ``mu`` belonging to group ``g`` is mapped by

    T_new = T_p2w(g) @ Delta(g) @ T_p2w(g)^-1 @ T_g2w

which is exactly the composition in the Warp kernel.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# Quaternion helpers (w, x, y, z convention -- same as nerfstudio / viser)
# --------------------------------------------------------------------------- #


def quat_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / (q.norm(dim=-1, keepdim=True) + eps)


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """(..., 4) wxyz -> (..., 3, 3). Differentiable, normalises internally."""
    q = quat_normalize(q)
    w, x, y, z = torch.unbind(q, dim=-1)
    mat = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    )
    return mat.reshape(q.shape[:-1] + (3, 3))


def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 4) wxyz. Shepperd's method (branch-free enough)."""
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    trace = m00 + m11 + m22

    q = torch.zeros(R.shape[:-2] + (4,), dtype=R.dtype, device=R.device)

    c0 = trace > 0
    s = torch.sqrt(torch.clamp(trace + 1.0, min=1e-12)) * 2
    q0 = torch.stack([0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s], -1)

    c1 = (~c0) & (m00 > m11) & (m00 > m22)
    s1 = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=1e-12)) * 2
    q1 = torch.stack(
        [(m21 - m12) / s1, 0.25 * s1, (m01 + m10) / s1, (m02 + m20) / s1], -1
    )

    c2 = (~c0) & (~c1) & (m11 > m22)
    s2 = torch.sqrt(torch.clamp(1.0 + m11 - m00 - m22, min=1e-12)) * 2
    q2 = torch.stack(
        [(m02 - m20) / s2, (m01 + m10) / s2, 0.25 * s2, (m12 + m21) / s2], -1
    )

    c3 = (~c0) & (~c1) & (~c2)
    s3 = torch.sqrt(torch.clamp(1.0 + m22 - m00 - m11, min=1e-12)) * 2
    q3 = torch.stack(
        [(m10 - m01) / s3, (m02 + m20) / s3, (m12 + m21) / s3, 0.25 * s3], -1
    )

    for cond, val in ((c0, q0), (c1, q1), (c2, q2), (c3, q3)):
        q = torch.where(cond.unsqueeze(-1), val, q)
    return quat_normalize(q)


def quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product, wxyz."""
    aw, ax, ay, az = torch.unbind(a, dim=-1)
    bw, bx, by, bz = torch.unbind(b, dim=-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate (..., 3) vectors by (..., 4) quaternions without building R."""
    qv = q[..., 1:]
    w = q[..., :1]
    t = 2.0 * torch.cross(qv, v, dim=-1)
    return v + w * t + torch.cross(qv, t, dim=-1)


# --------------------------------------------------------------------------- #
# so(3) / SE(3) exp & log
# --------------------------------------------------------------------------- #


def so3_exp(omega: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """(..., 3) rotation vector -> (..., 4) quaternion wxyz."""
    theta = omega.norm(dim=-1, keepdim=True)
    half = 0.5 * theta
    small = theta < eps
    # sin(theta/2)/theta, second-order Taylor when theta -> 0
    k = torch.where(small, 0.5 - theta**2 / 48.0, torch.sin(half) / theta.clamp(min=eps))
    w = torch.where(small, 1.0 - theta**2 / 8.0, torch.cos(half))
    return quat_normalize(torch.cat([w, k * omega], dim=-1))


def so3_log(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """(..., 4) quaternion wxyz -> (..., 3) rotation vector."""
    q = quat_normalize(q)
    # canonicalise to the hemisphere with w >= 0 so the log is the short way round
    q = torch.where(q[..., :1] < 0, -q, q)
    w = q[..., :1].clamp(-1.0, 1.0)
    v = q[..., 1:]
    vnorm = v.norm(dim=-1, keepdim=True)
    small = vnorm < eps
    scale = torch.where(small, 2.0 + (2.0 / 3.0) * vnorm**2, 2.0 * torch.atan2(vnorm, w) / vnorm.clamp(min=eps))
    return scale * v


def posevec_to_matrix(p: torch.Tensor) -> torch.Tensor:
    """(N, 7) [t | q_wxyz] -> (N, 4, 4)."""
    assert p.shape[-1] == 7, p.shape
    n = p.shape[0]
    out = torch.eye(4, dtype=p.dtype, device=p.device).unsqueeze(0).repeat(n, 1, 1)
    out[:, :3, :3] = quat_to_rotmat(p[:, 3:])
    out[:, :3, 3] = p[:, :3]
    return out


def matrix_to_posevec(T: torch.Tensor) -> torch.Tensor:
    """(N, 4, 4) -> (N, 7)."""
    return torch.cat([T[:, :3, 3], rotmat_to_quat(T[:, :3, :3])], dim=-1)


def identity_posevec(n: int = 1, device="cpu", dtype=torch.float32) -> torch.Tensor:
    p = torch.zeros(n, 7, device=device, dtype=dtype)
    p[:, 3] = 1.0
    return p


def se3_compose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compose two (N, 7) pose vectors: returns a @ b."""
    qa, ta = a[:, 3:], a[:, :3]
    qb, tb = b[:, 3:], b[:, :3]
    q = quat_multiply(qa, qb)
    t = ta + quat_rotate(qa, tb)
    return torch.cat([t, q], dim=-1)


def se3_inverse(a: torch.Tensor) -> torch.Tensor:
    qi = quat_conjugate(quat_normalize(a[:, 3:]))
    ti = -quat_rotate(qi, a[:, :3])
    return torch.cat([ti, qi], dim=-1)


# --------------------------------------------------------------------------- #
# The Warp kernel, in torch
# --------------------------------------------------------------------------- #


def apply_group_deltas(
    init_p2w_7vec: torch.Tensor,   # (K, 7) part->world at scan time (identity rot)
    part_deltas: torch.Tensor,     # (K, 7) learnable delta, world frame
    group_labels: torch.Tensor,    # (N,) int64 in [0, K)
    means: torch.Tensor,           # (N, 3)
    quats: torch.Tensor,           # (N, 4) wxyz
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable port of ``pogs/tracking/utils.py::apply_to_model``.

    new_g2w = p2w @ delta @ p2w^-1 @ g2w
    """
    # Compose the K per-group transforms first, then gather them: identical
    # maths, but the gather's backward is a K-row scatter-add (index_select)
    # instead of advanced-indexing's slow accumulate over N rows.
    T = se3_compose(se3_compose(init_p2w_7vec, part_deltas), se3_inverse(init_p2w_7vec))   # (K, 7)
    T_n = torch.index_select(T, 0, group_labels)                                            # (N, 7)
    g2w = torch.cat([means, quat_normalize(quats)], dim=-1)
    new_g2w = se3_compose(T_n, g2w)
    return new_g2w[:, :3], new_g2w[:, 3:]


# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #


class PinholeCamera:
    """Minimal stand-in for ``nerfstudio.cameras.Cameras``.

    ``c2w`` is stored in the OpenCV convention (x right, y down, z forward),
    which is what the rasteriser expects.  ``from_nerfstudio`` converts from
    the OpenGL/nerfstudio convention (x right, y up, z backward).
    """

    def __init__(
        self,
        c2w: torch.Tensor,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
    ):
        assert c2w.shape == (4, 4), c2w.shape
        self.c2w = c2w
        self.fx, self.fy, self.cx, self.cy = float(fx), float(fy), float(cx), float(cy)
        self.width, self.height = int(width), int(height)

    # -- constructors ------------------------------------------------------ #

    @staticmethod
    def from_fov(
        c2w: torch.Tensor, fov_x_deg: float, width: int, height: int
    ) -> "PinholeCamera":
        fx = 0.5 * width / math.tan(0.5 * math.radians(fov_x_deg))
        return PinholeCamera(c2w, fx, fx, width * 0.5, height * 0.5, width, height)

    @staticmethod
    def from_nerfstudio(
        c2w_gl: torch.Tensor, fx, fy, cx, cy, width, height
    ) -> "PinholeCamera":
        flip = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=c2w_gl.dtype, device=c2w_gl.device))
        return PinholeCamera(c2w_gl @ flip, fx, fy, cx, cy, width, height)

    # -- accessors --------------------------------------------------------- #

    @property
    def device(self):
        return self.c2w.device

    @property
    def w2c(self) -> torch.Tensor:
        R = self.c2w[:3, :3]
        t = self.c2w[:3, 3]
        out = torch.eye(4, dtype=self.c2w.dtype, device=self.c2w.device)
        out[:3, :3] = R.T
        out[:3, 3] = -R.T @ t
        return out

    @property
    def K(self) -> torch.Tensor:
        K = torch.eye(3, dtype=self.c2w.dtype, device=self.c2w.device)
        K[0, 0], K[1, 1], K[0, 2], K[1, 2] = self.fx, self.fy, self.cx, self.cy
        return K

    def to(self, device) -> "PinholeCamera":
        return PinholeCamera(
            self.c2w.to(device), self.fx, self.fy, self.cx, self.cy, self.width, self.height
        )

    def clone(self) -> "PinholeCamera":
        return PinholeCamera(
            self.c2w.clone(), self.fx, self.fy, self.cx, self.cy, self.width, self.height
        )

    def rescale(self, factor: float) -> "PinholeCamera":
        """Uniform output-resolution rescale, matching
        ``Cameras.rescale_output_resolution``."""
        return PinholeCamera(
            self.c2w.clone(),
            self.fx * factor,
            self.fy * factor,
            self.cx * factor,
            self.cy * factor,
            max(1, int(round(self.width * factor))),
            max(1, int(round(self.height * factor))),
        )

    def rescale_to(self, height: int, width: int) -> "PinholeCamera":
        """Resample to an exact output size (independent x / y factors)."""
        sx, sy = width / self.width, height / self.height
        return PinholeCamera(
            self.c2w.clone(), self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy,
            int(width), int(height),
        )

    def crop(self, xmin: int, xmax: int, ymin: int, ymax: int) -> "PinholeCamera":
        """Port of ``pogs/tracking/utils.py::crop_camera``."""
        return PinholeCamera(
            self.c2w.clone(),
            self.fx,
            self.fy,
            self.cx - xmin,
            self.cy - ymin,
            xmax - xmin,
            ymax - ymin,
        )

    def __repr__(self):
        return (
            f"PinholeCamera({self.width}x{self.height}, f=({self.fx:.1f},{self.fy:.1f}),"
            f" c=({self.cx:.1f},{self.cy:.1f}))"
        )


def look_at(eye: np.ndarray, target: np.ndarray, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """OpenCV-convention camera-to-world matrix looking from ``eye`` at ``target``."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    fwd = target - eye
    fwd /= np.linalg.norm(fwd) + 1e-12
    if abs(float(np.dot(fwd, up))) > 0.999:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right) + 1e-12
    down = np.cross(fwd, right)

    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = fwd
    c2w[:3, 3] = eye
    return c2w


def geodesic_rotation_error(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Angle in degrees between two rotation matrices."""
    R = np.asarray(R_a, dtype=np.float64).T @ np.asarray(R_b, dtype=np.float64)
    cos = (np.trace(R) - 1.0) / 2.0
    sin = 0.5 * np.linalg.norm([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    # atan2 is accurate near 0 and 180 deg, where arccos loses ~1e-2 deg
    return float(np.degrees(np.arctan2(sin, cos)))