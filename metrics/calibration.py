"""
Calibration and recovery metrics (plan rev. 2, task T6; evaluation only, ground truth allowed).

* Gaussian belief on se(3): NEES, coverage of the 1-alpha credible ellipsoid, NLL, and a
  reliability curve of predicted vs observed error.
* Circular (1-D weak-axis) belief: NLL of the true angle under a gridded density on (-pi, pi],
  with the uniform reference log(2 pi) (Implicit-PDF reports the SO(3) analogue, -2.29).
* Recovery: RRTrack-style event metric (recovered if any of the next W frames passes), and a
  Kaplan-Meier time-to-recover curve with right-censoring at the horizon.
* Hierarchical bootstrap over nested units (scene -> splat seed -> trial), because trials that share
  one trained splat are not independent (plan rev. 2, sec. 6).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
# Gaussian beliefs
# --------------------------------------------------------------------------- #

def nees(err: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Normalised estimation error squared e^T P^-1 e for (N, d) errors and (N, d, d) covariances.
    For a consistent estimator, NEES ~ chi2(d) with mean d."""
    err = np.atleast_2d(np.asarray(err, dtype=np.float64))
    cov = np.asarray(cov, dtype=np.float64).reshape(err.shape[0], err.shape[1], err.shape[1])
    sol = np.linalg.solve(cov, err[..., None])[..., 0]
    return np.einsum("nd,nd->n", err, sol)


def nees_consistency(nees_vals: np.ndarray, dof: int, alpha: float = 0.05) -> Dict[str, float]:
    """Mean NEES and the two-sided chi2 acceptance interval for the mean of N samples
    (Bar-Shalom et al., 'Estimation with Applications to Tracking and Navigation', Sec. 5.4)."""
    v = np.asarray(nees_vals, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = v.size
    lo = stats.chi2.ppf(alpha / 2, n * dof) / n
    hi = stats.chi2.ppf(1 - alpha / 2, n * dof) / n
    m = float(v.mean()) if n else float("nan")
    return {"mean_nees": m, "lo": float(lo), "hi": float(hi), "n": n, "dof": dof,
            "verdict": "consistent" if lo <= m <= hi else ("overconfident" if m > hi else "underconfident")}


def coverage(nees_vals: np.ndarray, dof: int, level: float = 0.90) -> float:
    """Fraction of true poses inside the `level` credible ellipsoid (nominal = level)."""
    v = np.asarray(nees_vals, dtype=np.float64)
    v = v[np.isfinite(v)]
    return float((v <= stats.chi2.ppf(level, dof)).mean()) if v.size else float("nan")


def gaussian_nll(err: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Negative log-density of the true error under N(0, cov)."""
    err = np.atleast_2d(np.asarray(err, dtype=np.float64))
    d = err.shape[1]
    cov = np.asarray(cov, dtype=np.float64).reshape(err.shape[0], d, d)
    _, logdet = np.linalg.slogdet(cov)
    return 0.5 * (nees(err, cov) + logdet + d * math.log(2 * math.pi))


def reliability_curve(pred_sigma: np.ndarray, abs_err: np.ndarray, n_bins: int = 10) -> Dict[str, np.ndarray]:
    """Bin by predicted sigma (quantile bins); per bin, the RMS observed error. Calibrated: rms ~ sigma."""
    s = np.asarray(pred_sigma, dtype=np.float64)
    e = np.asarray(abs_err, dtype=np.float64)
    ok = np.isfinite(s) & np.isfinite(e)
    s, e = s[ok], e[ok]
    edges = np.unique(np.quantile(s, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, len(edges) - 2)
    ms, rms, cnt = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            ms.append(s[m].mean())
            rms.append(math.sqrt(float((e[m] ** 2).mean())))
            cnt.append(int(m.sum()))
    return {"pred_sigma": np.array(ms), "rms_err": np.array(rms), "count": np.array(cnt)}


# --------------------------------------------------------------------------- #
# 1-D circular belief on the weak axis
# --------------------------------------------------------------------------- #

UNIFORM_CIRCLE_NLL = math.log(2 * math.pi)


def circular_nll(grid_rad: np.ndarray, log_density_unnorm: np.ndarray, true_rad: float) -> float:
    """NLL of ``true_rad`` under a density given on a uniform grid over (-pi, pi]
    (linear interpolation of the normalised density, periodic). Uniform belief = log(2 pi)."""
    g = np.asarray(grid_rad, dtype=np.float64)
    order = np.argsort(g)
    g = g[order]
    ld = np.asarray(log_density_unnorm, dtype=np.float64)[order]
    p = np.exp(ld - ld.max())
    dg = 2 * math.pi / g.size
    p = p / (p.sum() * dg)
    x = (true_rad + math.pi) % (2 * math.pi) - math.pi
    gp = np.concatenate([g, [g[0] + 2 * math.pi]])
    pp = np.concatenate([p, [p[0]]])
    if x < g[0]:
        x += 2 * math.pi
    return float(-math.log(max(np.interp(x, gp, pp), 1e-300)))


# --------------------------------------------------------------------------- #
# Recovery
# --------------------------------------------------------------------------- #

def event_recovered(success_per_frame: Sequence[bool], event_frame: int, window: int = 10) -> Dict[str, object]:
    """RRTrack event metric (Sec. IV-A-b): recovered if any of the ``window`` frames after the event
    passes; delay = frames to the first pass (None if not recovered)."""
    s = np.asarray(success_per_frame, dtype=bool)[event_frame + 1: event_frame + 1 + window]
    hits = np.nonzero(s)[0]
    return {"recovered": bool(hits.size), "delay": int(hits[0] + 1) if hits.size else None}


def kaplan_meier(times: Sequence[Optional[int]], horizon: int) -> Dict[str, np.ndarray]:
    """Survival S(t) = P(not yet recovered after t frames); None = censored at ``horizon``."""
    t = np.array([horizon if x is None else x for x in times], dtype=np.float64)
    ev = np.array([x is not None for x in times], dtype=bool)
    grid = np.arange(0, horizon + 1)
    S, s = [], 1.0
    for k in grid:
        at_risk = (t >= k).sum()
        d = ((t == k) & ev).sum()
        if at_risk > 0 and d > 0:
            s *= 1 - d / at_risk
        S.append(s)
    return {"t": grid, "survival": np.array(S)}


# --------------------------------------------------------------------------- #
# Hierarchical bootstrap
# --------------------------------------------------------------------------- #

def hierarchical_bootstrap(values: np.ndarray, groups: Sequence[np.ndarray], n_boot: int = 2000,
                           stat=np.mean, seed: int = 0, ci: float = 0.95) -> Dict[str, float]:
    """Resample nested units with replacement, top level first.

    ``groups`` lists label arrays from the outermost unit inward (e.g. [scene, splat_seed]); trials
    are resampled within the innermost unit. Returns the point estimate and a percentile interval."""
    v = np.asarray(values, dtype=np.float64)
    labels = [np.asarray(g) for g in groups]
    rng = np.random.default_rng(seed)

    def draw(idx: np.ndarray, level: int) -> np.ndarray:
        if level == len(labels):
            return rng.choice(idx, size=idx.size, replace=True)
        units = np.unique(labels[level][idx])
        pick = rng.choice(units, size=units.size, replace=True)
        return np.concatenate([draw(idx[labels[level][idx] == u], level + 1) for u in pick])

    all_idx = np.arange(v.size)
    boots = np.array([stat(v[draw(all_idx, 0)]) for _ in range(n_boot)])
    a = (1 - ci) / 2
    return {"estimate": float(stat(v)), "lo": float(np.quantile(boots, a)), "hi": float(np.quantile(boots, 1 - a)),
            "n_boot": n_boot}


def paired_difference_bootstrap(a: np.ndarray, b: np.ndarray, groups: Sequence[np.ndarray], **kw) -> Dict[str, float]:
    """Bootstrap of mean(a - b) for methods evaluated on the same trials (paired)."""
    return hierarchical_bootstrap(np.asarray(a, np.float64) - np.asarray(b, np.float64), groups, **kw)
