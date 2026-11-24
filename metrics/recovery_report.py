"""
Recovery-metric report for tool / object pose tracking (analysis note §6, "evaluation recommendations").

One call scores a set of perturbation trials and the matching no-motion (Static) trials under every rule that
the literature uses, side by side, and decomposes the rotation error about the tool axis:

    rep = recovery_report(trials, static_trials, axis_local, tip_local, perturbation_sign)
    print(format_report(rep))

A trial is a dict with 4x4 object-to-world poses: ``T_pred`` / ``T_gt`` after the recovery window and
``T_pred_pre`` / ``T_gt_pre`` just before the perturbation (only the relative rule needs the ``_pre`` poses).

Rules
- ``full``      tip within ``pos_cm`` and geodesic rotation error within ``rot_deg`` (all 3 rotational DOF).
- ``roll_free`` tip within ``pos_cm`` and swing of the tool axis within ``rot_deg``; roll about the axis is free
                (POGS Sec. V-B). Report which DOF a rule leaves free.
- ``relative``  error after recovery no larger than before the perturbation (+ a tolerance). Biased by any
                pre-existing signed error; reported stratified by whether that error has the same sign as the
                perturbation (``perturbation_sign``), and never without Static.
Every rule is also scored on Static: a rule a tracker that never moves can pass is not measuring recovery.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from metrics.pose import SUCCESS_POS_CM, SUCCESS_ROT_DEG, rotation_error_deg, roll_swing_error_deg
from metrics.stats import wilson_interval


def _tip_err_cm(T_pred, T_gt, tip_local) -> float:
    p = np.r_[np.asarray(tip_local, float), 1.0]
    return float(np.linalg.norm((T_pred @ p)[:3] - (T_gt @ p)[:3]) * 100.0)


def score_trial(trial: Dict, axis_local, tip_local, pos_cm: float = SUCCESS_POS_CM,
                rot_deg: float = SUCCESS_ROT_DEG, rel_tol_deg: float = 1.0) -> Dict[str, float]:
    Tp, Tg = np.asarray(trial["T_pred"], float), np.asarray(trial["T_gt"], float)
    tip = _tip_err_cm(Tp, Tg, tip_local)
    geo = rotation_error_deg(Tp, Tg)
    roll, swing = roll_swing_error_deg(Tp, Tg, axis_local)
    out = {"tip_cm": tip, "geodesic_deg": geo, "roll_deg": roll, "swing_deg": swing,
           "full": float(tip < pos_cm and geo < rot_deg), "roll_free": float(tip < pos_cm and swing < rot_deg)}
    if "T_pred_pre" in trial and "T_gt_pre" in trial:
        Tpp, Tgp = np.asarray(trial["T_pred_pre"], float), np.asarray(trial["T_gt_pre"], float)
        geo_pre = rotation_error_deg(Tpp, Tgp)
        roll_pre, _ = roll_swing_error_deg(Tpp, Tgp, axis_local)
        out.update(geodesic_pre_deg=geo_pre, roll_pre_deg=roll_pre,
                   relative=float(geo <= geo_pre + rel_tol_deg and tip < pos_cm))
    return out


def _rate(xs: Sequence[float]) -> Dict[str, float]:
    k, n = int(round(sum(xs))), len(xs)
    lo, hi = wilson_interval(k, n) if n else (float("nan"), float("nan"))
    return {"k": k, "n": n, "rate": k / n if n else float("nan"), "ci95": (lo, hi)}


def recovery_report(trials: List[Dict], static_trials: List[Dict], axis_local, tip_local,
                    perturbation_sign: Optional[float] = None, **kw) -> Dict:
    S = [score_trial(t, axis_local, tip_local, **kw) for t in trials]
    Z = [score_trial(t, axis_local, tip_local, **kw) for t in static_trials]
    rep: Dict = {"rules": {}, "error": {}}
    for rule in ("full", "roll_free", "relative"):
        if all(rule in s for s in S) and S:
            rep["rules"][rule] = {"method": _rate([s[rule] for s in S]),
                                  "static": _rate([z[rule] for z in Z if rule in z]) if Z else None}
    if perturbation_sign is not None and S and all("roll_pre_deg" in s for s in S):
        same = [s for s in S if np.sign(s["roll_pre_deg"]) == np.sign(perturbation_sign)]
        opp = [s for s in S if np.sign(s["roll_pre_deg"]) != np.sign(perturbation_sign)]
        rep["relative_by_sign"] = {"pre-error same sign as perturbation": _rate([s["relative"] for s in same]),
                                   "opposite sign": _rate([s["relative"] for s in opp])}
    for k in ("tip_cm", "geodesic_deg", "roll_deg", "swing_deg"):
        v = np.asarray([s[k] for s in S], float)
        if v.size:
            rep["error"][k] = {"median": float(np.median(v)), "mean": float(np.mean(v)),
                               "mean_abs": float(np.mean(np.abs(v))),
                               "share_positive": float(np.mean(v > 0)) if k == "roll_deg" else None}
    if S:
        r = np.abs([s["roll_deg"] for s in S])
        g = np.asarray([s["geodesic_deg"] for s in S])
        rep["roll_share_of_rotation_error"] = float(np.median(r / np.maximum(g, 1e-9)))
    rep["free_dof"] = {"full": "none", "roll_free": "rotation about the tool axis", "relative": "none (relative)"}
    return rep


def format_report(rep: Dict, title: str = "Recovery") -> str:
    L = [f"| {title}: rule | free DOF | method k/n [95% CI] | Static k/n [95% CI] |", "|---|---|---|---|"]
    for rule, v in rep["rules"].items():
        m, s = v["method"], v["static"]

        def cell(x):
            return "n/a" if not x else f"{x['k']}/{x['n']} [{x['ci95'][0]:.2f}, {x['ci95'][1]:.2f}]"
        L.append(f"| {rule} | {rep['free_dof'][rule]} | {cell(m)} | {cell(s)} |")
    if "relative_by_sign" in rep:
        L += ["", "| relative rule, stratified | k/n |", "|---|---|"]
        for k, x in rep["relative_by_sign"].items():
            L.append(f"| {k} | {x['k']}/{x['n']} |")
    e = rep["error"]
    if e:
        L += ["", f"Roll about the tool axis: mean {e['roll_deg']['mean']:+.1f}°, mean |roll| "
              f"{e['roll_deg']['mean_abs']:.1f}°, {100 * e['roll_deg']['share_positive']:.0f}% positive; swing median "
              f"{e['swing_deg']['median']:.1f}°; roll is {100 * rep['roll_share_of_rotation_error']:.0f}% of the "
              f"geodesic error (median)."]
    return "\n".join(L)


def rot_about(axis, deg: float) -> np.ndarray:
    """4x4 rotation about a unit axis through the origin (test / example helper)."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    t = math.radians(deg)
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K
    return T
