"""POST-HOC (not pre-registered) O4 analysis: recovery relative to the pre-perturbation error.
The pre-registered rule (tip < 3 cm and rot < 5 deg at +10 frames) is dominated by a rotation error that is
already 10-24 deg BEFORE the perturbation, so it measures steady-state accuracy, not recovery.
Relative rule here: rot_err(k) <= pre + 5 deg, tip_err(k) <= pre + 2 cm, never lost after the perturbation."""
import glob, json, os
import numpy as np
from metrics.stats import wilson_interval

OUT = "data_results/o4_ingrasp"
METHODS = ["pogs", "pogs___kf___check", "pogs__release__no_per_frame_depth", "static__no_motion"]
lines = ["| Setting | Method | Relative recovery (k/n) | 95% CI | Pre rot (deg, median) | Post rot (deg, median) |", "|---|---|---|---|---|---|"]
for ang in ("+15", "-15", "+30", "-30"):
    for m in METHODS:
        ok = n = 0; pre = []; post = []
        for f in sorted(glob.glob(f"{OUT}/runs/grasp_{ang}_s*/{m}/per_frame.json")):
            d = json.load(open(f)); p0 = d["phase"].index("perturbed"); k = min(p0 + 10, len(d["phase"]) - 1)
            r, t, p = (np.asarray(d[x]) for x in ("rot_err_deg", "tip_err_cm", "pos_err_cm"))
            r0, t0 = np.median(r[max(0, p0 - 5):p0]), np.median(t[max(0, p0 - 5):p0])
            pre.append(r0); post.append(r[k]); n += 1
            ok += bool(r[k] <= r0 + 5 and t[k] <= t0 + 2 and not (p[p0:] > 10).any())
        lo, hi = wilson_interval(ok, n)
        lines.append(f"| {ang}° | {m} | {ok}/{n} | [{lo:.2f}, {hi:.2f}] | {np.median(pre):.1f} | {np.median(post):.1f} |")
open(f"{OUT}/posthoc_relative.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
