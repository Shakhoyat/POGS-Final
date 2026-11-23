"""Figures for the supervisor progress report (docs/report/progress_report.tex). Reads only data_results/*;
writes docs/report/figs/*.pdf. Run from the repo root: .venv/bin/python docs/report/make_figures.py"""

from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = "docs/report/figs"
os.makedirs(OUT, exist_ok=True)

S1, S2, S3, GREY, CRIT, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#8a8d93", "#d03b3b", "#15171a", "#4e5157", "#e6e5e0"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "axes.edgecolor": "#b9b8b2", "axes.linewidth": 0.6, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})
W1, W2 = 3.4, 7.0   # single / double column width (in)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)


def bar_labels(ax, bars, fmt="{:g}", dy=0.6):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h), ha="center", va="bottom", fontsize=6.5, color=INK)


# --------------------------------------------------------------------------- O1
def fig_o1():
    d = json.load(open("data_results/o1_grouping/metrics.json"))
    rows = sorted(d["rows"], key=lambda r: -r["iou"])
    names = [r["object"].replace("_", "\n") for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    w = 0.26
    for j, (k, c, lab) in enumerate([("iou", S1, "IoU"), ("purity", S2, "Purity"), ("recall", S3, "Recall")]):
        ax.bar(x + (j - 1) * w, [r[k] for r in rows], w * 0.9, color=c, label=lab)
    ax.set_xticks(x, names, rotation=0, fontsize=7)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("score")
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    save(fig, "o1_grouping_scores.pdf")
    # downsample the qualitative figure for the PDF
    from PIL import Image
    im = Image.open("data_results/o1_grouping/figure.jpg")
    im.thumbnail((1600, 1600))
    im.save(os.path.join(OUT, "o1_renders.jpg"), quality=88)


# --------------------------------------------------------------------------- O3
def fig_o3():
    def stack(case, m, key):
        A = [json.load(open(p))[key] for p in sorted(glob.glob(f"data_results/o3_perturbation/runs/{case}_s*/{m}/per_frame.json"))]
        return np.array(A, float)

    pf = json.load(open(sorted(glob.glob("data_results/o3_perturbation/runs/step_p30_d5_z_s0/pogs/per_frame.json"))[0]))
    cmd = np.array(pf["commanded_deg"], float)
    ph = pf["phase"]
    f = np.arange(len(cmd))
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(W1, 2.9), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]})
    a0.step(f, cmd, where="mid", color=INK2, lw=1.2)
    a0.set_ylabel("commanded (°)")
    a0.set_yticks([0, 15, 30])
    ix = max(i for i, p in enumerate(ph) if p == "rest_x")
    iy = len(ph) - 1
    for i, lab in ((ix, "x"), (iy, "y")):
        a0.annotate(lab, (i, cmd[i]), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8, color=INK, weight="bold")
    ip = [i for i, p in enumerate(ph) if p == "peak_p"]
    a0.annotate("p", (ip[len(ip) // 2], 30), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8, color=INK, weight="bold")
    for m, c, lab in (("pogs", S1, "POGS"), ("static__no_motion", S2, "Static (no motion)")):
        R = stack("step_p30_d5_z", m, "rot_err_deg")
        mu = R.mean(0)
        a1.plot(f, mu, color=c, lw=1.6, label=lab)
        a1.fill_between(f, R.min(0), R.max(0), color=c, alpha=0.15, lw=0)
    a1.axhline(5, color=CRIT, lw=1, ls=(0, (4, 3)))
    a1.text(f[-1], 5.6, "5°", color=CRIT, ha="right", fontsize=7)
    a1.set_ylabel("rotation error (°)")
    a1.set_xlabel("frame (5 Hz)")
    a1.legend(loc="upper left", ncol=2)
    save(fig, "o3_step_trace.pdf")

    # jumps: recovered / 5
    conds = ["+15", "-15", "+30", "-30", "+45", "+60", "+90"]
    pog, sta = [], []
    for cnd in conds:
        for m, acc in (("pogs", pog), ("static__no_motion", sta)):
            R = [json.load(open(p)) for p in glob.glob(f"data_results/o3_perturbation/runs/jump_{cnd}_z_s*/{m}/row.json")]
            acc.append(sum(bool(r["recovered"]) for r in R))
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    x = np.arange(len(conds))
    b1 = ax.bar(x - 0.2, pog, 0.36, color=S1, label="POGS")
    b2 = ax.bar(x + 0.2, sta, 0.36, color=S2, label="Static (no motion)")
    bar_labels(ax, b1, dy=0.08)
    bar_labels(ax, b2, dy=0.08)
    ax.set_xticks(x, [c.replace("-", "−") + "°" for c in conds])
    ax.set_ylim(0, 5.9)
    ax.set_yticks(range(6))
    ax.set_ylabel("recovered / 5")
    ax.set_xlabel("single jump about the vertical axis")
    ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0))
    save(fig, "o3_jumps.pdf")


# --------------------------------------------------------------------------- O4
def fig_o4():
    cats = ["+15°", "−15°", "+30°", "−30°"]
    pre, pf, paper = [2, 14, 13, 5], [32, 25, 39, 17], [48, 48, 40, 40]
    fig, ax = plt.subplots(figsize=(W1, 2.1))
    x = np.arange(4)
    w = 0.26
    for j, (v, c, lab) in enumerate([(pre, S2, "Pre-registered rule"), (pf, S1, "Paper's rule (roll free)"),
                                     (paper, GREY, "Reported in the paper")]):
        b = ax.bar(x + (j - 1) * w, v, w * 0.9, color=c, label=lab)
        bar_labels(ax, b)
    ax.axhline(40, color=CRIT, lw=1, ls=(0, (4, 3)))
    ax.text(-0.45, 41.2, "80% target", color=CRIT, ha="left", fontsize=7)
    ax.set_xticks(x, cats)
    ax.set_ylim(0, 56)
    ax.set_ylabel("recovered / 50")
    ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=7)
    save(fig, "o4_recovery.pdf")

    # roll trace by frame (POGS, all 200 trials) with phase bands
    acc = {}
    for p in glob.glob("data_results/o4_ingrasp/runs/grasp_*/pogs/per_frame.json"):
        d = json.load(open(p))
        for i, (phs, r, s) in enumerate(zip(d["phase"], d["roll_err_deg"], d["swing_err_deg"])):
            acc.setdefault(i, {"ph": phs, "r": [], "s": []})
            acc[i]["r"].append(abs(r))
            acc[i]["s"].append(abs(s))
    fr = sorted(acc)[:20]
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    ax.plot(fr, [np.median(acc[i]["r"]) for i in fr], color=S1, lw=1.6, label="roll about the bit")
    ax.plot(fr, [np.median(acc[i]["s"]) for i in fr], color=S3, lw=1.6, label="swing of the bit axis")
    ax.axhline(5, color=CRIT, lw=1, ls=(0, (4, 3)))
    spans, cur = [], None
    for i in fr:
        if acc[i]["ph"] != cur:
            spans.append([acc[i]["ph"], i, i])
            cur = acc[i]["ph"]
        spans[-1][2] = i
    names = {"approach": "on table", "lift": "lift", "rest_x": "held", "perturbed": "after perturbation"}
    for k, (p, a, b) in enumerate(spans):
        if k % 2 == 0:
            ax.axvspan(a - 0.5, b + 0.5, color="#f1f0ec", lw=0, zorder=0)
        ax.text((a + b) / 2, 17.2, names.get(p, p), ha="center", fontsize=6.5, color=INK2)
    ax.set_ylim(0, 18.5)
    ax.set_xlim(-0.5, fr[-1] + 0.5)
    ax.set_ylabel("median error (°)")
    ax.set_xlabel("frame")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    save(fig, "o4_roll_trace.pdf")


# --------------------------------------------------------------------------- O5
def fig_o5():
    rows = [  # (role, model, median ms, note)
        ("Alignment", "CLIP ViT-B/16", 11.8, ""), ("Alignment", "MobileCLIPv2", 11.8, ""), ("Alignment", "SigLIP2 B/16", 12.5, ""),
        ("Features", "DINOv2 ViT-S/14", 15.7, ""), ("Features", "DINOv3 ViT-S/16", 19.0, ""), ("Features", "DINOv3 ViT-B/16", 27.1, ""),
        ("Features", "DINOv2 meets text", 170.8, "old stack"),
        ("Detection", "RF-DETR", 30.8, ""), ("Detection", "YOLO11x", 38.1, ""),
    ]
    fig, ax = plt.subplots(figsize=(W1, 2.5))
    y = np.arange(len(rows))[::-1]
    col = {"Alignment": S3, "Features": S1, "Detection": S2}
    for yi, (role, name, v, note) in zip(y, rows):
        ax.barh(yi, v, 0.66, color=col[role], alpha=0.55 if note else 1)
        ax.text(v + 2, yi, f"{v:.1f} ms" + (f" ({note})" if note else ""), va="center", fontsize=6.5, color=INK)
    ax.set_yticks(y, [r[1] for r in rows], fontsize=7)
    ax.set_xlim(0, 210)
    ax.set_xlabel("median latency per image (ms), fastest of fp32/fp16")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=col[k], label=k) for k in ("Detection", "Features", "Alignment")], ncol=3,
              loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.grid(axis="y", visible=False)
    save(fig, "o5_latency.pdf")


# --------------------------------------------------------------------------- O6
def fig_o6():
    pts = [(1.5, 100, "25 it, 500"), (2.2, 100, "15 it, 500"), (3.2, 100, "10 it, 500"), (3.4, 100, "10 it, 350"),
           (3.2, 100, "10 it, 250"), (5.6, 86, "5 it, 500"), (5.6, 68, "5 it, 350"), (5.7, 100, "5 it, 250")]
    fig, ax = plt.subplots(figsize=(W1, 2.0))
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=34, color=S1, edgecolor="white", linewidth=1.2, zorder=3)
    ax.axvline(5, color=CRIT, lw=1, ls=(0, (4, 3)))
    ax.text(4.93, 57.5, "5 Hz target", color=CRIT, ha="right", fontsize=7)
    for hz, acc, lab in pts:
        if lab in ("25 it, 500", "5 it, 350", "5 it, 250", "5 it, 500"):
            up = lab == "25 it, 500"
            ax.annotate(lab + " px", (hz, acc), xytext=(0, 7) if up else (6, -3), textcoords="offset points",
                        ha="center" if up else "left", fontsize=6.5, color=INK)
    ax.set_xlim(0, 8.2)
    ax.set_xticks(range(0, 9))
    ax.set_ylim(55, 106)
    ax.set_xlabel("tracking frequency (Hz)")
    ax.set_ylabel("ADD-S success (%)")
    save(fig, "o6_speed.pdf")


# --------------------------------------------------------------------------- finding
def fig_finding():
    g = json.load(open("data_results/b_bias/bgrender.json"))
    m = json.load(open("data_results/b_bias/masks.json"))
    def row(view):
        rb = next(r for r in g["rows"] if r["view"] == view)
        rm = next(r for r in m["rows"] if r["view"] == view)
        return [rb["argmin"]["object only (release)"], rb["argmin"]["object + rendered background"],
                rm["argmin"]["inside drill at base pose (fixed)"]]
    conds = [("tracking camera,\ngripper + shadow", row("track, gripper (frame 0)")),
             ("tracking camera,\nno shadows", row("track, gripper, no shadows")),
             ("tracking camera,\nno gripper", row("track, no gripper"))]
    hb = [r for r in g["rows"] if r["view"].startswith("heldout")]
    hm = [r for r in m["rows"] if r["view"].startswith("heldout")]
    conds.append(("held-out views,\nmean |argmin|", [np.mean([abs(r["argmin"]["object only (release)"]) for r in hb]),
                                                    np.mean([abs(r["argmin"]["object + rendered background"]) for r in hb]),
                                                    np.mean([abs(r["argmin"]["inside drill at base pose (fixed)"]) for r in hm])]))
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    y = np.arange(len(conds))[::-1]
    h = 0.24
    for j, (c, lab) in enumerate([(S2, "original loss"), (S1, "background rendered"), (S3, "drill pixels only (GT mask)")]):
        vals = [cd[1][j] for cd in conds]
        bars = ax.barh(y + (1 - j) * h, vals, h * 0.9, color=c, label=lab)
        for k, (b, v) in enumerate(zip(bars, vals)):
            lab_v = f"{v:.1f}°" if k == len(vals) - 1 else f"{v:+.0f}°"
            lab_v = lab_v.replace("-", "\u2212")
            ax.text(v + (0.4 if v >= 0 else -0.4), b.get_y() + b.get_height() / 2, lab_v,
                    va="center", ha="left" if v >= 0 else "right", fontsize=6.3, color=INK)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, [c[0] for c in conds], fontsize=6.8)
    ax.set_xlim(-13, 8)
    ax.set_xlabel("roll at the loss minimum (°); truth = 0")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=6.8)
    ax.grid(axis="y", visible=False)
    save(fig, "finding_argmin.pdf")

    fixes = [("POGS (reference)", [9, 12, 16, 23], S1), ("RGB term", [16, 13, 20, 18], GREY),
             ("DINO in drill mask", [1, 0, 4, 4], GREY), ("depth ×30", [2, 1, 0, 0], GREY),
             ("depth ×100", [0, 0, 0, 0], GREY), ("background rendered", [0, 0, 0, 0], GREY),
             ("depth only (no DINO)", [0, 0, 0, 0], GREY), ("Static (no motion)", [0, 0, 0, 0], GREY)]
    fig, ax = plt.subplots(figsize=(W1, 2.3))
    y = np.arange(len(fixes))[::-1]
    for yi, (lab, v, c) in zip(y, fixes):
        ax.barh(yi, sum(v), 0.62, color=c)
        ax.text(sum(v) + 1.5, yi, f"{sum(v)}   ({'/'.join(map(str, v))})", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(y, [f[0] for f in fixes], fontsize=7)
    ax.set_xlim(0, 100)
    ax.set_xlabel("recovered / 100 (paper's rule; per setting −30/−15/+15/+30°)")
    ax.grid(axis="y", visible=False)
    save(fig, "finding_fixes.pdf")


if __name__ == "__main__":
    fig_o1(); fig_o3(); fig_o4(); fig_o5(); fig_o6(); fig_finding()
    print("wrote", sorted(os.listdir(OUT)))
