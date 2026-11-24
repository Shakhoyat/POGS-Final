"""OR6 scoring (main venv): official per-frame poses -> our O4 judge (both rules), vs E0 rows on the same trials.
Pre-registered in `docs/note/off_or6_recovery_preregistration.md`.

    .venv/bin/python -m experiments.off_repro.or6_score
"""
import json
import os

import numpy as np

from data.scene_generator.tool_perturbation import DRILL_ID, DRILL_TIP_LOCAL, build_tool_scene
from experiments.o3_perturbation import _tip_err_cm
from experiments.o4_ingrasp import judge
from metrics.pose import roll_swing_error_deg
from model.pogs.geometry import geodesic_rotation_error

OUT = "data_results/off_repro"
E0 = "data_results/o4_ingrasp/runs"
METHODS = {"Static (no motion)": "static__no_motion", "POGS (ours, release: no per-frame depth)":
           "pogs__release__no_per_frame_depth", "POGS (ours)": "pogs"}


def _geo_deg(R1, R2):
    return geodesic_rotation_error(R1, R2)             # degrees (numpy)


def main():
    d = json.load(open(os.path.join(OUT, "official_track_full.json")))
    c0 = np.asarray(d["setup"]["centroid"], float)
    drill = next(o for o in build_tool_scene() if o.obj_id == DRILL_ID)
    tip_local = np.asarray(drill.position) + DRILL_TIP_LOCAL - c0
    axis = np.r_[1.0, 0, 0]
    rows = []
    for c in d["cases"]:
        P = [np.asarray(r["pred_T"]) for r in c["rows"]]
        G = [np.asarray(r["gt_T"]) for r in c["rows"]]
        pf = {"phase": [r["phase"] for r in c["rows"]],
              "tip_err_cm": [_tip_err_cm(p, g, tip_local) for p, g in zip(P, G)],
              "rot_err_deg": [_geo_deg(p[:3, :3], g[:3, :3]) for p, g in zip(P, G)],
              "pos_err_cm": [float(np.linalg.norm(p[:3, 3] - g[:3, 3]) * 100) for p, g in zip(P, G)]}
        rs = [roll_swing_error_deg(p, g, axis) for p, g in zip(P, G)]
        pf["roll_err_deg"] = [r[0] for r in rs]
        pf["swing_err_deg"] = [r[1] for r in rs]
        j = judge(pf)
        e0 = {}
        for name, key in METHODS.items():
            f = os.path.join(E0, c["case"], key, "row.json")
            if os.path.exists(f):
                r = json.load(open(f))
                e0[name] = (bool(r.get("success")), bool(r.get("success_rollfree")))
        rows.append({"case": c["case"], "official": (j["success"], j["success_rollfree"]), "judge": j, "e0": e0})
    n = len(rows)
    def count(get):
        return sum(get(r)[0] for r in rows), sum(get(r)[1] for r in rows)
    off_full, off_rf = count(lambda r: r["official"])
    tab = {"POGS (official code)": (off_full, off_rf)}
    for name in METHODS:
        if all(name in r["e0"] for r in rows):
            tab[name] = count(lambda r, nm=name: r["e0"][nm])
    rel = tab.get("POGS (ours, release: no per-frame depth)", (np.nan, np.nan))[1]
    or6a = abs(off_rf - rel) <= 6
    or6b = off_rf < 32
    by_set = {}
    for r in rows:
        st = r["case"].split("_s")[0]
        by_set.setdefault(st, [0, 0, 0])
        by_set[st][0] += r["official"][0]
        by_set[st][1] += r["official"][1]
        by_set[st][2] += 1
    L = ["# OR6: official tracker on the in-grasp protocol (40 trials)", "",
         "Pre-registered in `docs/note/off_or6_recovery_preregistration.md`. Rules first.", "",
         "| Rule | Statistic | Result |", "|---|---|---|",
         f"| OR6a | official roll-free {off_rf}/{n} vs ours (release) {rel}/{n}; abs diff ≤ 6 | {'pass' if or6a else 'FAIL'} |",
         f"| OR6b | official roll-free {off_rf}/{n} < 32/40 | {'pass' if or6b else 'FAIL'} |", "",
         "| Method | Full rule | Roll-free (paper) |", "|---|---|---|"]
    L += [f"| {k} | {v[0]}/{n} | {v[1]}/{n} |" for k, v in tab.items()]
    L += ["", "| Setting | official full | official roll-free | n |", "|---|---|---|---|"]
    L += [f"| {k} | {v[0]} | {v[1]} | {v[2]} |" for k, v in sorted(by_set.items())]
    L += ["", "Exploratory: median official roll / swing / tip just before the perturbation: "
          f"{np.median([r['judge']['roll_pre_deg'] for r in rows]):+.1f}° / "
          f"{np.median([r['judge']['swing_pre_deg'] for r in rows]):.1f}° / "
          f"{np.median([r['judge']['tip_err_pre_cm'] for r in rows]):.2f} cm."]
    open(os.path.join(OUT, "summary_or6.md"), "w").write("\n".join(L) + "\n")
    json.dump(rows, open(os.path.join(OUT, "or6_rows.json"), "w"), indent=1, default=float)
    print("\n".join(L))
    print(f"FINDING: OR6a | official tracker roll-free recoveries {off_rf}/{n} vs ours (release) {rel}/{n} "
          f"({'pass' if or6a else 'FAIL'}) | {OUT}/summary_or6.md")
    print(f"FINDING: OR6b | official tracker roll-free {off_rf}/{n} below 32/40 ({'pass' if or6b else 'FAIL'}) | "
          f"{OUT}/summary_or6.md")


if __name__ == "__main__":
    main()
