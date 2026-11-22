"""
Pose information of the tracking loss (plan rev. 2, M1 / task T2). Evaluation and analysis only:
nothing here changes the v1 tracker.

Perturbation model. A group's pose is part_deltas = [t | q] with part2world T = [R(q) | c0 + t]
(``RigidGroupOptimizer.get_part2world``; identity rotation at scan). We perturb on the right, in the
object frame (origin = scan centroid c0):

    R' = R Exp(omega),   t' = t + R v,     delta = (omega, v) in R^6   [rad, m]

so eigenvectors are expressed in the object's own axes (e.g. the drill-bit axis is object +x).

Two estimators of the 6x6 information matrix, both from closures, so they apply unchanged to the real
render loss (``RigidGroupOptimizer.get_optim_loss``) and to synthetic test problems:

* ``gauss_newton_info(res_fn, ...)``: J by central finite differences of a residual vector
  (12 evaluations), H = J^T J, sigma^2 = r^T r / (m - 6), Sigma = sigma^2 (H + lam I)^-1.
* ``loss_hessian(loss_fn, ...)``: Hessian of a scalar loss by central differences of autograd
  gradients (12 backward passes), symmetrised and projected to PSD. This is what the tracker's
  scalar mean loss admits directly.

``observability_summary`` reports scale-free quantities (rotation rows scaled by the object radius,
so all six coordinates are metres; X-ICP-style separate rotation / translation blocks), because raw
eigenvalues do not transfer across objects or images (X-ICP, T-RO 2024, Sec. II-B).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import torch

from .geometry import quat_multiply, quat_normalize, quat_to_rotmat, so3_exp


def perturb_part_delta(pd: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Right-perturb one (7,) part delta [t | q_wxyz] by delta = (omega, v) in the object frame."""
    t, q = pd[:3], pd[3:]
    R = quat_to_rotmat(q.unsqueeze(0))[0]
    delta = delta.to(device=q.device, dtype=q.dtype)
    q_new = quat_normalize(quat_multiply(q, so3_exp(delta[:3])))
    t_new = t + R @ delta[3:]
    return torch.cat([t_new, q_new])


def perturbed_deltas(part_deltas: torch.Tensor, k: int, delta: torch.Tensor) -> torch.Tensor:
    """(K, 7) copy of ``part_deltas`` with group k right-perturbed by ``delta``."""
    base = part_deltas.detach().clone()
    rows = [perturb_part_delta(base[i], delta) if i == k else base[i] for i in range(base.shape[0])]
    return torch.stack(rows)        # out-of-place, so it stays differentiable in delta


def _steps(h_rot: float, h_trans: float, dtype=torch.float64) -> torch.Tensor:
    return torch.tensor([h_rot] * 3 + [h_trans] * 3, dtype=dtype)


def gauss_newton_info(res_fn: Callable[[torch.Tensor], torch.Tensor], h_rot: float = 2e-3,
                      h_trans: float = 1e-3, lam: float = 0.0) -> Dict[str, np.ndarray]:
    """H = J^T J for the residual vector ``res_fn(delta)`` (delta in R^6, float64).

    The residual vector must have a fixed length (use a fixed pixel set, e.g. the ROI at the
    estimate). Returns H, J, r0, sigma2 and Sigma = sigma2 (H + lam I)^-1 (pinv if singular)."""
    z = torch.zeros(6, dtype=torch.float64)
    r0 = res_fn(z).detach().double().reshape(-1)
    hs = _steps(h_rot, h_trans)
    cols = []
    for i in range(6):
        e = torch.zeros(6, dtype=torch.float64)
        e[i] = hs[i]
        rp = res_fn(e).detach().double().reshape(-1)
        rm = res_fn(-e).detach().double().reshape(-1)
        cols.append((rp - rm) / (2.0 * hs[i]))
    J = torch.stack(cols, dim=1).cpu().numpy()
    H = J.T @ J
    m = r0.numel()
    sigma2 = float(r0 @ r0) / max(m - 6, 1)
    Sigma = sigma2 * np.linalg.pinv(H + lam * np.eye(6))
    return {"H": H, "J": J, "r0": r0.cpu().numpy(), "sigma2": sigma2, "Sigma": Sigma}


def loss_hessian(loss_fn: Callable[[torch.Tensor], torch.Tensor], h_rot: float = 2e-3,
                 h_trans: float = 1e-3) -> Dict[str, np.ndarray]:
    """Hessian of a scalar ``loss_fn(delta)`` by central differences of its autograd gradient.

    ``loss_fn`` must be differentiable in delta (it is, through ``perturbed_deltas`` and the
    rasteriser). Returns the symmetrised Hessian, its PSD projection and the gradient at 0."""
    def grad_at(d: torch.Tensor) -> np.ndarray:
        with torch.enable_grad():   # callers (tracking hooks) may run under no_grad
            d = d.clone().requires_grad_(True)
            L = loss_fn(d)
            (g,) = torch.autograd.grad(L, d)
        return g.detach().double().cpu().numpy()

    z = torch.zeros(6, dtype=torch.float64)
    g0 = grad_at(z)
    hs = _steps(h_rot, h_trans).numpy()
    Hc = np.zeros((6, 6))
    for i in range(6):
        e = np.zeros(6)
        e[i] = hs[i]
        gp = grad_at(torch.tensor(e))
        gm = grad_at(torch.tensor(-e))
        Hc[:, i] = (gp - gm) / (2.0 * hs[i])
    Hs = 0.5 * (Hc + Hc.T)
    w, V = np.linalg.eigh(Hs)
    H_psd = (V * np.clip(w, 0.0, None)) @ V.T
    return {"H": Hs, "H_psd": H_psd, "grad": g0, "asym": float(np.abs(Hc - Hc.T).max())}


def loss_hessian_regression(loss_fn: Callable[[torch.Tensor], torch.Tensor], scale_rot: float = 0.01745,
                            scale_trans: float = 3e-3, n: int = 24, seed: int = 0) -> Dict[str, np.ndarray]:
    """Hessian at a stated *scale*: least-squares fit of g(delta) = g0 + H delta over ``n`` random
    perturbations delta ~ N(0, diag(scale^2)), symmetrised.

    The POGS render loss is non-smooth at sub-pixel scales (ROI / valid masks, alpha thresholds), so a
    central-difference Hessian depends on the step (E2 smoke test, LEDGER 8.E2). Fitting over a
    neighbourhood of the size of a typical per-frame motion gives the curvature that matters for
    tracking. Also returns the relative fit residual as a quality flag."""
    rng = np.random.default_rng(seed)

    def grad_at(d: np.ndarray) -> np.ndarray:
        with torch.enable_grad():
            dt = torch.tensor(d, dtype=torch.float64, requires_grad=True)
            (g,) = torch.autograd.grad(loss_fn(dt), dt)
        return g.detach().double().cpu().numpy()

    sc = np.array([scale_rot] * 3 + [scale_trans] * 3)
    D = rng.normal(size=(n, 6)) * sc
    D = np.concatenate([D, -D])                       # antithetic pairs cancel odd terms
    g0 = grad_at(np.zeros(6))
    G = np.stack([grad_at(d) for d in D]) - g0
    Ht, *_ = np.linalg.lstsq(D, G, rcond=None)        # G ~ D H^T
    H = 0.5 * (Ht.T + Ht)
    resid = float(np.linalg.norm(G - D @ H) / max(np.linalg.norm(G), 1e-300))
    return {"H": H, "grad": g0, "fit_resid": resid, "asym": float(np.abs(Ht - Ht.T).max())}


def observability_summary(H: np.ndarray, radius_m: float,
                          axis_local: Optional[np.ndarray] = None) -> Dict[str, object]:
    """Scale-free observability of a 6x6 information matrix (rotation first).

    Rotation coordinates are rescaled by the object radius (delta_theta * radius = metres of surface
    motion), so eigenvalues of the full matrix are comparable across directions. Also returns the
    X-ICP-style 3x3 block spectra and, if ``axis_local`` is given, how well the weakest rotation
    direction aligns with it (|cos|; 1 = the weak direction is roll about that axis)."""
    S = np.diag([1.0 / radius_m] * 3 + [1.0] * 3)
    Hn = S @ H @ S
    w, V = np.linalg.eigh(0.5 * (Hn + Hn.T))
    wr, Vr = np.linalg.eigh(0.5 * (H[:3, :3] + H[:3, :3].T))
    wt, Vt = np.linalg.eigh(0.5 * (H[3:, 3:] + H[3:, 3:].T))
    lam_max = float(max(w[-1], 1e-300))
    out = {
        "eig": w, "eigvec": V, "cond": float(lam_max / max(w[0], 1e-300)),
        "rel_min": float(max(w[0], 0.0) / lam_max),
        "eig_rot": wr, "eigvec_rot": Vr, "eig_trans": wt, "eigvec_trans": Vt,
        "rel_min_rot": float(max(wr[0], 0.0) / max(wr[-1], 1e-300)),
    }
    if axis_local is not None:
        a = np.asarray(axis_local, dtype=np.float64)
        a = a / np.linalg.norm(a)
        out["weak_rot_axis_cos"] = float(abs(Vr[:, 0] @ a))
        # information along roll about the axis, relative to the strongest rotation direction
        out["roll_info_rel"] = float(max(a @ H[:3, :3] @ a, 0.0) / max(wr[-1], 1e-300))
        # share of the axis inside the 2-D weak rotation subspace (robust when the two smallest
        # rotation eigenvalues nearly tie, which makes the single weakest eigenvector ill-defined)
        out["axis_in_weak2"] = float(np.linalg.norm(Vr[:, :2].T @ a))
        out["weak_gap_rot"] = float(max(wr[1], 0.0) / max(wr[0], 1e-300)) if wr[0] > 0 else float("inf")
    return out


def tracker_loss_hessian(opt, obs, k: int, use_depth: bool = True, h_rot: float = 2e-3,
                         h_trans: float = 1e-3, base: Optional[torch.Tensor] = None,
                         method: str = "fd", n: int = 24) -> Dict[str, np.ndarray]:
    """Hessian of the tracker's own loss (``RigidGroupOptimizer.get_optim_loss``) w.r.t. a right
    perturbation of group k, at the current estimate. ``obs`` must already carry its ROIs
    (``set_observation``), so the pixel set is the one the tracker used. 12 backward passes.
    Not yet exercised on a GPU run; E2 (LEDGER 8.E2) is its first real use."""
    base = opt.part_deltas.detach() if base is None else base.detach().to(opt.part_deltas)

    def loss_fn(d: torch.Tensor) -> torch.Tensor:
        L, _ = opt.get_optim_loss(obs, perturbed_deltas(base, k, d), use_depth=use_depth,
                                  use_roi=opt.config.use_roi)
        if L is None:
            raise RuntimeError("tracking loss undefined at this pose (object left the view)")
        return L.double()

    if method == "regression":
        out = loss_hessian_regression(loss_fn, scale_rot=h_rot, scale_trans=h_trans, n=n)
    else:
        out = loss_hessian(loss_fn, h_rot=h_rot, h_trans=h_trans)
    with torch.no_grad():
        out["loss"] = float(loss_fn(torch.zeros(6, dtype=torch.float64)))
    return out


def pose_to_part_delta(T: np.ndarray, centroid: np.ndarray) -> torch.Tensor:
    """Inverse of ``get_part2world`` for one group: (4, 4) pose [R | c'] -> (7,) [c' - c0 | q]."""
    from .geometry import rotmat_to_quat
    R = torch.as_tensor(np.asarray(T)[:3, :3], dtype=torch.float32)
    t = torch.as_tensor(np.asarray(T)[:3, 3] - np.asarray(centroid), dtype=torch.float32)
    return torch.cat([t, quat_normalize(rotmat_to_quat(R.unsqueeze(0))[0])])
