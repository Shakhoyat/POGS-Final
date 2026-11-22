"""
Plan rev. 2, E1 (CPU part): what makes the 10-24 deg pre-perturbation rotation floor and the +/- gap
in O4? Reads the schema-2 O4 runs (full SE(3), roll/swing about the drill-bit axis; E0) and writes
data_results/e1_floor/summary.md. Evaluation only (uses ground truth).

Tests, per method and setting:
  1. Decomposition of the pre-perturbation error: roll about the bit axis vs swing of the axis.
     Roll-dominated -> consistent with weak roll observability (plan R1/R9).
  2. Signed-bias artefact: stratify the post-hoc relative rule by the sign of the pre-perturbation
     signed roll relative to the perturbation direction. If 'same sign' vs 'opposite sign' explains the
     +/- gap, the gap is a metric artefact (plan rev. 1 Q3), not recovery.
  3. When does the floor appear: roll error at the first rest frame vs at the last pre-perturbation frame.
The GPU part of E1 (loss at estimate vs at ground truth, 1-D roll sweep) is in e2 (needs the tracker).

    .venv/bin/python -m experiments.e1_floor_analysis
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

from metrics.pose import roll_swing_error_deg
from metrics.stats import mean_std, wilson_interval

GRASP_AXIS_LOCAL = np.array([0.0, 0.0, 1.0])   # O4 perturbs about the vertical grasp axis (tool_perturbation)

RUNS = "data_results/o4_ingrasp/runs"
OUT = "data_results/e1_floor"
N_AFTER = 10


def load_rows() -> pd.DataFrame:
    rows = []
    for rp in glob.glob(os.path.join(RUNS, "*", "*", "row.json")):
        row = json.load(open(rp))
        if row.get("schema", 1) < 2:
            continue
        pf = json.load(open(os.path.join(os.path.dirname(rp), "per_frame.json")))
        ph = pf["phase"]
        p0 = ph.index("perturbed")
        k = min(p0 + N_AFTER, len(ph) - 1)
        roll = np.asarray(pf["roll_err_deg"])
        swing = np.asarray(pf["swing_err_deg"])
        rot = np.asarray(pf["rot_err_deg"])
        tip = np.asarray(pf["tip_err_cm"])
        pos = np.asarray(pf["pos_err_cm"])
        # signed pre-perturbation error about the perturbation axis (from the stored SE(3))
        yaw_pre = (roll_swing_error_deg(np.asarray(pf["pred_T"][p0 - 1]), np.asarray(pf["gt_T"][p0 - 1]),
                                        GRASP_AXIS_LOCAL)[0] if "pred_T" in pf else np.nan)
        last_approach = max(i for i, p in enumerate(ph) if p == "approach")   # drill still on the table
        last_lift = max(i for i, p in enumerate(ph) if p == "lift")
        rows.append({
            "case": row["case"], "angle": row["angle"], "seed": row["seed"], "method": row["method"],
            "roll_pre": roll[p0 - 1], "yaw_pre": yaw_pre, "swing_pre": swing[p0 - 1], "rot_pre": rot[p0 - 1],
            "roll_approach": roll[last_approach], "roll_lift": roll[last_lift], "roll_N": roll[k], "swing_N": swing[k], "rot_N": rot[k],
            "tip_N": tip[k], "tip_pre": tip[p0 - 1],
            # post-hoc relative rule of 2026-11-22 (labelled post hoc; reported for the artefact test)
            "rel_success": bool(rot[k] <= rot[p0 - 1] + 5.0 and tip[k] <= tip[p0 - 1] + 2.0
                                and np.nanmax(pos[p0:]) <= 10.0),
            "success": row["success"], "success_rollfree": row.get("success_rollfree"),
        })
    return pd.DataFrame(rows)


def _rate(s: pd.Series) -> str:
    k, n = int(s.sum()), len(s)
    if n == 0:
        return "–"
    lo, hi = wilson_interval(k, n)
    return f"{k}/{n} [{lo:.2f}, {hi:.2f}]"


def main() -> None:
    df = load_rows()
    os.makedirs(OUT, exist_ok=True)
    if df.empty:
        open(os.path.join(OUT, "summary.md"), "w").write("No schema-2 O4 rows yet (E0 not run).\n")
        print("[e1] no schema-2 rows")
        return
    df.to_csv(os.path.join(OUT, "rows.csv"), index=False)
    L = ["# E1: anatomy of the O4 rotation floor (evaluation only)", "",
         "## 1. Pre-perturbation error: roll about the bit axis vs swing of the axis", "",
         "| Setting | Method | n | |roll| pre (deg) | swing pre (deg) | geodesic pre (deg) | roll share | tip pre (cm) |",
         "|---|---|---|---|---|---|---|---|"]
    for (a, m), g in df.groupby(["angle", "method"], sort=False):
        r, rs = mean_std(g.roll_pre.abs())
        w, ws = mean_std(g.swing_pre)
        t, ts = mean_std(g.rot_pre)
        share = float(np.median(g.roll_pre.abs() / np.maximum(g.rot_pre, 1e-9)))
        tp, tps = mean_std(g.tip_pre)
        L.append(f"| {a:+g}° | {m} | {len(g)} | {r:.1f} ± {rs:.1f} | {w:.1f} ± {ws:.1f} | {t:.1f} ± {ts:.1f} | "
                 f"{share:.2f} | {tp:.2f} ± {tps:.2f} |")
    L += ["", "Roll share = median |roll| / geodesic error. Near 1: the floor is roll about the bit axis, which the "
          "paper's rule leaves free (POGS Sec. V-B).", "",
          "## 2. Signed-bias test of the post-hoc relative rule (POST HOC; artefact check only)", "",
          "Aligned = the pre-perturbation signed error about the *perturbation* (vertical grasp) axis has the "
          "same sign as the perturbation, so the jump moves a frozen estimate's error toward zero in that axis. "
          "If success concentrates in 'aligned', the +/- gap is a signed-bias artefact of the relative rule.", "",
          "| Setting | Method | Aligned: relative-rule success | Opposed: relative-rule success |",
          "|---|---|---|---|"]
    df["aligned"] = np.sign(df.yaw_pre) == np.sign(df.angle)
    for (a, m), g in df.groupby(["angle", "method"], sort=False):
        L.append(f"| {a:+g}° | {m} | {_rate(g[g.aligned].rel_success)} | {_rate(g[~g.aligned].rel_success)} |")
    L += ["", "## 3. When the floor appears", "",
          "| Method | |roll| end of approach, drill on table (deg) | end of lift (deg) | just before perturbation (deg) |",
          "|---|---|---|---|"]
    for m, g in df.groupby("method", sort=False):
        a, as_ = mean_std(g.roll_approach.abs())
        c, cs = mean_std(g.roll_lift.abs())
        b, bs = mean_std(g.roll_pre.abs())
        L.append(f"| {m} | {a:.1f} ± {as_:.1f} | {c:.1f} ± {cs:.1f} | {b:.1f} ± {bs:.1f} |")
    L += ["", "A floor already present on the table points to initialisation / observability; one that appears during "
          "the lift points to gripper occlusion or motion."]
    L += ["", "Static (no motion) is included in every table as the null model (CLAUDE.md rule 4)."]
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(L) + "\n")
    print(f"[e1] {len(df)} rows -> {OUT}/summary.md")


if __name__ == "__main__":
    main()
