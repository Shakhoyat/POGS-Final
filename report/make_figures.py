"""
Figures for the LaTeX report, built from the cached runs, the rendered dataset
and the trained checkpoint (CPU only, ~1 min).

    .venv/bin/python report/make_figures.py

Writes ``report/figures/*.pdf|png``.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene_generator.pick_place_benchmark import FACTORS, SCENE_VERSION  # noqa: E402

RUNS = "data_results/benchmark_runs"
DATA = "datasets/synthetic_scenes/pick_place_floor_table"
CKPT = "model_checkpoints/pick_place_floor_table"
OUT = "report/figures"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
COL = {
    "Static (no motion)": "#9e9e9e",
    "POGS": "#2b6cb0",
    "POGS (release: no per-frame depth)": "#90cdf4",
    "POGS + KF": "#dd8a2e",
    "POGS + KF + check": "#c0392b",
    "POGS, 10 iters": "#2b6cb0",
    "POGS + KF, 10 iters": "#dd8a2e",
    "POGS + KF + check, 10 iters": "#c0392b",
    "POGS, 5 iters": "#2b6cb0",
    "POGS + KF, 5 iters": "#dd8a2e",
    "POGS + KF + check, 5 iters": "#c0392b",
    "POGS + KF + check + latency comp.": "#8e44ad",
    "POGS + KF + check, 10 iters + latency comp.": "#8e44ad",
}
SHORT = {
    "Static (no motion)": "Static",
    "POGS (release: no per-frame depth)": "POGS (release)",
    "POGS + KF + check + latency comp.": "KF + check + LC",
    "POGS + KF + check, 10 iters + latency comp.": "KF + check + LC, 10 it.",
}
W1, W2 = 3.4, 7.0      # one / two column widths (inches)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("  ", name)


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #

def load_index():
    rows, dirs = [], {}
    for f in glob.glob(f"{RUNS}/*/*/row.json"):
        with open(f) as fh:
            r = json.load(fh)
        if r.get("scene_version") != SCENE_VERSION:
            continue
        rows.append(r)
        dirs[(r["case"], r["method"])] = os.path.dirname(f)
    df = pd.DataFrame(rows).drop_duplicates(["case", "method"])
    return df, dirs


def telemetry(dirs, case, method, target=None):
    with open(os.path.join(dirs[(case, method)], "telemetry.json")) as fh:
        tel = json.load(fh)
    out = {"frame": [], "pos": [], "rot": [], "occluded": []}
    for fr in tel:
        objs = fr["objects"]
        o = next((x for x in objs if target is None or x["name"] == target), objs[0])
        out["frame"].append(fr["frame"])
        out["pos"].append(o["trans_err_cm"])
        out["rot"].append(o["rot_err_deg"])
        out["occluded"].append(fr.get("occluded", False))
    return {k: np.asarray(v) for k, v in out.items()}


def meta(seq):
    with open(os.path.join(DATA, seq, "tracking_metadata.json")) as fh:
        return json.load(fh)


def offline(df):
    return df[~df["case"].str.contains("_30hz")]


def level_rows(df, cond, speed, depth):
    return df[(df["condition"] == cond) & (df["speed"] == speed) & (df["depth"] == depth)]


# --------------------------------------------------------------------------- #
# Dataset figures
# --------------------------------------------------------------------------- #

def fig_scan():
    """First row of the preview contact sheet: scan view, its depth, another view."""
    sheet = cv2.cvtColor(cv2.imread("data_results/scene_preview/contact_sheet.png"), cv2.COLOR_BGR2RGB)
    h = sheet.shape[0] // 3
    row = sheet[:h]
    cv2.imwrite(os.path.join(OUT, "scan_views.jpg"), cv2.cvtColor(row, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("   scan_views.jpg")


def fig_conditions():
    """One mid-carry frame per rendered condition (seed 0)."""
    conds = [("base", "base"), ("yaw15", "yaw 15"), ("yaw90", "yaw 90"), ("tilt30", "tilt 30"),
             ("tilt60", "tilt 60"), ("combined45", "yaw+tilt 45"), ("occ_none", "no occluder"),
             ("occ_hand", "hand (partial)"), ("occ_full", "hand (heavy)"), ("obj_mustard", "mustard bottle"),
             ("obj_cereal", "cereal box"), ("multi", "two objects")]
    fig, axes = plt.subplots(4, 3, figsize=(W2, W2 * 4 / 3 * 360 / 540 * 1.08))
    for ax, (c, lab) in zip(axes.flat, conds):
        m = meta(f"{c}_s0")
        if c.startswith("occ_"):
            f = int(np.argmax([fr["occlusion_frac"] for fr in m["frames"]])) if c != "occ_none" else 30
        elif c.startswith("tilt") or c == "combined45":
            f = int(np.argmax([abs(fr.get("tilt_deg", 0.0)) for fr in m["frames"]]))
        else:
            f = 30
        img = cv2.cvtColor(cv2.imread(os.path.join(DATA, f"{c}_s0", "rgb", f"frame_{f:04d}.png")), cv2.COLOR_BGR2RGB)
        ax.imshow(img[70:430, 230:770])
        fr = m["frames"][f]
        ax.set_title(f"{lab}: occl. {100 * fr['occlusion_frac']:.0f}%", pad=2)
        ax.axis("off")
    fig.subplots_adjust(wspace=0.02, hspace=0.12, left=0, right=1, top=0.97, bottom=0)
    save(fig, "conditions.jpg")


def fig_trajectories():
    """Ground-truth motion profiles (seed 0) and occlusion over time."""
    fig, axes = plt.subplots(1, 3, figsize=(W2, 1.9))
    cases = [("base", "base (yaw 30)"), ("yaw90", "yaw 90"), ("tilt60", "tilt 60"), ("combined45", "yaw+tilt 45")]
    for c, lab in cases:
        m = meta(f"{c}_s0")
        t = np.array([fr["timestamp_sec"] for fr in m["frames"]])
        axes[0].plot(t, [fr["translation_cm"] for fr in m["frames"]], lw=1.1, label=lab)
        axes[1].plot(t, [fr["rot_deg"] for fr in m["frames"]], lw=1.1, label=lab)
    m = meta("base_s0")
    phases = [fr["phase"] for fr in m["frames"]]
    t = np.array([fr["timestamp_sec"] for fr in m["frames"]])
    shade = {"lift": "#eef3fb", "carry": "#fdf2e6", "place": "#eef7ee"}
    for ax in axes[:2]:
        for ph, cc in shade.items():
            idx = [i for i, p in enumerate(phases) if p == ph]
            if idx:
                ax.axvspan(t[idx[0]], t[idx[-1]] + 0.2, color=cc, zorder=0, lw=0)
    for c, lab in [("occ_none", "none"), ("base", "gripper"), ("occ_hand", "+ hand (partial)"),
                   ("occ_full", "+ hand (heavy)")]:
        m = meta(f"{c}_s0")
        axes[2].plot([fr["timestamp_sec"] for fr in m["frames"]],
                     [100 * fr["occlusion_frac"] for fr in m["frames"]], lw=1.1, label=lab)
    axes[0].set(xlabel="time (s)", ylabel="displacement (cm)", title="(a) translation from start")
    axes[1].set(xlabel="time (s)", ylabel="rotation (deg)", title="(b) rotation from start")
    axes[2].set(xlabel="time (s)", ylabel="target occluded (%)", title="(c) occlusion of the mug")
    axes[1].legend(loc="upper left", frameon=False)
    axes[2].legend(loc="upper left", frameon=False)
    fig.tight_layout(w_pad=1.0)
    save(fig, "trajectories.pdf")


# --------------------------------------------------------------------------- #
# Reconstruction and grouping
# --------------------------------------------------------------------------- #

def fig_training():
    with open(os.path.join(CKPT, "train_summary.json")) as fh:
        s = json.load(fh)
    lg = s["log"]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.0))
    axes[0].plot(lg["step"], np.asarray(lg["n_gauss"]) / 1e3, color="#2b6cb0", lw=1.2)
    axes[0].set(xlabel="training step", ylabel="Gaussians (thousands)", title="(a) densification")
    axes[0].text(0.03, 0.95, f"final: {s['n_gaussians'] / 1e3:.0f}k Gaussians\n"
                 f"train-view PSNR {s['psnr_mean']:.1f} dB\ndepth MAE {100 * s['depth_mae_m']:.2f} cm\n"
                 f"{s['train_time_s'] / 60:.1f} min on one GPU",
                 transform=axes[0].transAxes, va="top", fontsize=6.5)
    for k, c in [("rgb", "#2b6cb0"), ("depth", "#27ae60"), ("dino", "#c0392b"), ("clip", "#8e44ad"),
                 ("group", "#dd8a2e")]:
        v = np.asarray(lg[k], dtype=float)
        v[v <= 0] = np.nan
        if np.isfinite(v).any():
            axes[1].plot(lg["step"], v, lw=1.0, label=k, color=c)
    axes[1].set_yscale("log")
    axes[1].set(xlabel="training step", ylabel="loss", title="(b) loss terms")
    for ax in axes:
        for x, lab in [(100, "DINO"), (2500, "CLIP + group")]:
            ax.axvline(x, color="#777", lw=0.6, ls=":")
    axes[1].legend(ncol=1, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=6.5)
    fig.tight_layout(w_pad=2.0)
    save(fig, "training.pdf")


def fig_grouping():
    import torch
    ck = torch.load(os.path.join(CKPT, "pogs_checkpoint.pt"), map_location="cpu", weights_only=False)
    gp = ck["gauss_params"]
    xyz = gp["means"].numpy()
    op = 1.0 / (1.0 + np.exp(-gp["opacities"].numpy()[:, 0]))
    rgb = np.clip(0.5 + 0.28209479 * gp["features_dc"].numpy(), 0, 1)
    labels = np.load(os.path.join(CKPT, "cluster_labels.npy"))
    with open(os.path.join(CKPT, "object_clusters.json")) as fh:
        oc = json.load(fh)
    keep = (op > 0.3) & (np.abs(xyz[:, 0]) < 0.45) & (np.abs(xyz[:, 1]) < 0.45) & (xyz[:, 2] < 0.35)
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(keep)
    idx = idx[rng.permutation(len(idx))[:90000]]
    pal_n = {"mustard bottle": "#e6b800", "red apple": "#c0392b", "cereal box": "#2b6cb0",
             "soup can": "#8e44ad", "ceramic mug": "#27ae60"}
    names = {g["cluster"]: g["name"].replace("_", " ") for g in oc["groups"]}
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.0))
    axes[0].scatter(xyz[idx, 0], xyz[idx, 1], c=rgb[idx], s=0.15, lw=0, rasterized=True)
    axes[0].set_title("(a) Gaussian centres, top view (own colour)")
    bg = idx[~np.isin(labels[idx], list(names))]
    axes[1].scatter(xyz[bg, 0], xyz[bg, 1], c="#dddddd", s=0.15, lw=0, rasterized=True)
    handles = [Patch(color="#dddddd", label="background (rejected)")]
    for i, (cl, nm) in enumerate(sorted(names.items())):
        sel = idx[labels[idx] == cl]
        sel = sel[xyz[sel, 2] > 0.015]
        axes[1].scatter(xyz[sel, 0], xyz[sel, 1], c=pal_n.get(nm, "#333"), s=0.3, lw=0, rasterized=True)
        g = next(g for g in oc["groups"] if g["cluster"] == cl)
        handles.append(Patch(color=pal_n.get(nm, "#333"), label=f"{nm} (IoU {g['iou_vs_gt']:.2f})"))
    axes[1].set_title("(b) HDBSCAN clusters selected as objects")
    axes[1].legend(handles=handles, loc="lower right", frameon=True, fontsize=6, markerscale=4)
    for ax in axes:
        ax.set_aspect("equal")
        ax.set(xlabel="x (m)", ylabel="y (m)", xlim=(-0.42, 0.45), ylim=(-0.42, 0.42))
        ax.grid(False)
    fig.tight_layout(w_pad=1.5)
    save(fig, "grouping.pdf")


# --------------------------------------------------------------------------- #
# Offline results
# --------------------------------------------------------------------------- #

def fig_factors(df):
    methods = ["Static (no motion)", "POGS", "POGS (release: no per-frame depth)", "POGS + KF", "POGS + KF + check"]
    d0 = offline(df)
    fig, axes = plt.subplots(2, 3, figsize=(W2, 4.6), gridspec_kw={"width_ratios": [6, 6, 4]})
    order = ["rotation", "speed", "occlusion", "depth", "object", "multi-object"]
    for ax, factor in zip(axes.flat, order):
        levels = FACTORS[factor]
        x = np.arange(len(levels))
        w = 0.8 / len(methods)
        for j, m in enumerate(methods):
            mu, sd = [], []
            for _, cond, speed, depth in levels:
                g = level_rows(d0, cond, speed, depth)
                v = 100 * g[g["method"] == m]["adds_success"]
                mu.append(v.mean())
                sd.append(v.std(ddof=1) if len(v) > 1 else 0)
            ax.bar(x + (j - (len(methods) - 1) / 2) * w, mu, w, yerr=sd, color=COL[m], label=SHORT.get(m, m),
                   error_kw={"lw": 0.5, "capsize": 0.8}, edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels([l[0].replace("gripper + hand ", "hand ").replace("noisy 10 mm + holes", "10 mm+holes")
                            .replace("(partial)", "part.").replace("(heavy)", "heavy").replace("mustard bottle", "mustard")
                            for l in levels], rotation=35, ha="right", rotation_mode="anchor")
        ax.set_ylim(0, 105)
        ax.set_title(factor, fontweight="bold")
        if ax in (axes[0, 0], axes[1, 0]):
            ax.set_ylabel("ADD-S success (%)")
        ax.grid(axis="x", visible=False)
    axes[0, 0].legend(ncol=5, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.1), fontsize=6.5)
    fig.tight_layout(w_pad=0.6)
    fig.subplots_adjust(hspace=0.62)
    save(fig, "factors.pdf")


def fig_budget(df):
    d0 = offline(df)
    fam = {"POGS": ["POGS", "POGS, 10 iters", "POGS, 5 iters"],
           "POGS + KF": ["POGS + KF", "POGS + KF, 10 iters", "POGS + KF, 5 iters"],
           "POGS + KF + check": ["POGS + KF + check", "POGS + KF + check, 10 iters", "POGS + KF + check, 5 iters"]}
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.2))
    for name, ms in fam.items():
        fps = [d0[d0["method"] == m]["fps"].mean() for m in ms]
        suc = [100 * d0[d0["method"] == m]["adds_success"].mean() for m in ms]
        lost = [100 * d0[d0["method"] == m]["lost_rate"].mean() for m in ms]
        axes[0].plot(fps, suc, "-o", color=COL[name], ms=3.5, lw=1.2, label=name)
        axes[1].plot(fps, lost, "-o", color=COL[name], ms=3.5, lw=1.2, label=name)
        for f, s, it in zip(fps, suc, (25, 10, 5)):
            axes[0].annotate(f"{it}", (f, s), textcoords="offset points", xytext=(3, 3), fontsize=6, color=COL[name])
    s = 100 * d0[d0["method"] == "Static (no motion)"]["adds_success"].mean()
    axes[0].axhline(s, color="#9e9e9e", ls="--", lw=0.8)
    axes[0].text(7.2, s + 1.5, "Static", fontsize=6, color="#777", ha="right")
    axes[0].set(xlabel="tracker rate (frames/s)", ylabel="ADD-S success (%)",
                title="(a) accuracy vs. speed, all 54 cases")
    axes[1].set(xlabel="tracker rate (frames/s)", ylabel="lost frames (%)", title="(b) lost-frame rate (>10 cm)")
    axes[0].legend(frameon=False, loc="lower left")
    fig.tight_layout(w_pad=2)
    save(fig, "budget.pdf")


def fig_speed(df):
    d0 = offline(df)
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.0), sharey=True)
    for ax, (suf, title) in zip(axes, [("", "25 iterations"), (", 10 iters", "10 iterations"),
                                       (", 5 iters", "5 iterations")]):
        for base in ["POGS", "POGS + KF", "POGS + KF + check"]:
            m = base + suf
            mu, sd = [], []
            for sp in (1, 2, 4):
                v = 100 * level_rows(d0, "base", sp, "stereo").query("method == @m")["adds_success"]
                mu.append(v.mean())
                sd.append(v.std(ddof=1))
            ax.errorbar([1, 2, 4], mu, yerr=sd, fmt="-o", color=COL[base], ms=3, lw=1.1, capsize=1.5,
                        elinewidth=0.6, label=base)
        st = 100 * level_rows(d0, "base", 1, "stereo").query("method == 'Static (no motion)'")["adds_success"].mean()
        ax.axhline(st, color="#9e9e9e", ls="--", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4])
        ax.set_xticklabels(["1x", "2x", "4x"])
        ax.set(title=title, xlabel="object speed per tracked frame")
    axes[0].set_ylabel("ADD-S success (%)")
    axes[0].legend(frameon=False, loc="lower left")
    fig.tight_layout(w_pad=0.8)
    save(fig, "speed.pdf")


def fig_cases(df, dirs):
    """Per-frame position error for four illustrative cases (seed 0)."""
    cases = [("base_s0__2x_stereo", "base, 2x speed", "ceramic_mug"),
             ("obj_cereal_s0__1x_stereo", "cereal box", "cereal_box"),
             ("tilt60_s0__1x_stereo", "tilt 60", "ceramic_mug"),
             ("occ_full_s0__1x_stereo", "hand occlusion (heavy)", "ceramic_mug")]
    methods = ["Static (no motion)", "POGS", "POGS + KF", "POGS + KF + check"]
    fig, axes = plt.subplots(2, 2, figsize=(W2, 3.4))
    for ax, (case, title, tgt) in zip(axes.flat, cases):
        fps = 5.0 / (2 if "2x" in case else 1)
        for m in methods:
            if (case, m) not in dirs:
                continue
            t = telemetry(dirs, case, m, tgt)
            ax.plot(t["frame"] / fps, t["pos"], color=COL[m], lw=1.1 if m != "Static (no motion)" else 0.8,
                    ls="--" if m == "Static (no motion)" else "-", label=SHORT.get(m, m))
            if m == "POGS":
                occ = t["occluded"]
        seq = case.split("__")[0]
        mm = meta(seq)
        stride = 2 if "2x" in case else 1
        of = np.array([fr["occlusion_frac"] for fr in mm["frames"]])[::stride]
        ax2 = ax.twinx()
        ax2.fill_between(np.arange(len(of)) / fps, 0, 100 * of, color="#bbb", alpha=0.25, lw=0, step="mid")
        ax2.set_ylim(0, 300)
        ax2.set_yticks([])
        ax2.grid(False)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.axhline(10, color="#555", lw=0.5, ls=":")
        ax.set(title=title, xlabel="time (s)", ylabel="position error (cm)")
        ax.set_ylim(0, None)
    axes[0, 0].legend(frameon=False, loc="upper left", ncol=2)
    fig.tight_layout(h_pad=1.0, w_pad=1.5)
    save(fig, "cases.pdf")


def fig_rotation(df, dirs):
    """Gripper-induced rotation error: occ_none vs base (POGS, seed 0)."""
    fig, ax = plt.subplots(figsize=(W1, 1.8))
    for case, lab, c in [("occ_none_s0__1x_stereo", "no occluder", "#27ae60"),
                         ("base_s0__1x_stereo", "gripper", "#2b6cb0"),
                         ("occ_hand_s0__1x_stereo", "gripper + hand", "#c0392b")]:
        t = telemetry(dirs, case, "POGS", "ceramic_mug")
        ax.plot(t["frame"] / 5.0, t["rot"], color=c, lw=1.1, label=lab)
    ax.axhline(5, color="#555", lw=0.5, ls=":")
    ax.set(xlabel="time (s)", ylabel="rotation error (deg)", title="POGS rotation error, mug, seed 0")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "rotation.pdf")


# --------------------------------------------------------------------------- #
# Real-time results
# --------------------------------------------------------------------------- #

RT = ["POGS", "POGS, 10 iters", "POGS + KF + check", "POGS + KF + check + latency comp.",
      "POGS + KF + check, 10 iters", "POGS + KF + check, 10 iters + latency comp."]


def _sched(dirs, case, m):
    with open(os.path.join(dirs[(case, m)], "schedule.json")) as fh:
        return {k: np.asarray(v) for k, v in json.load(fh).items()}


def fig_rt_schedule(dirs):
    case = "base_s0_30hz__rt1x_stereo"
    ms = ["POGS", "POGS, 10 iters", "POGS + KF + check, 10 iters"]
    fig, axes = plt.subplots(2, 1, figsize=(W2, 2.9), gridspec_kw={"height_ratios": [1.1, 1.4]}, sharex=True)
    ax = axes[0]
    for i, m in enumerate(ms):
        s = _sched(dirs, case, m)
        bars = [(a, b - a) for a, b in zip(s["proc_start"], s["proc_done"])]
        ax.broken_barh(bars, (i - 0.35, 0.7), facecolors=[COL[m] if k % 2 == 0 else matplotlib.colors.to_rgba(COL[m], 0.55)
                                                          for k in range(len(bars))], edgecolor="white", lw=0.3)
    ts = _sched(dirs, case, "POGS")["timestamps"]
    for x in ts[::3]:
        ax.axvline(x, color="#ccc", lw=0.2, zorder=0)
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels(["POGS (25 it.)", "POGS, 10 it.", "KF + check, 10 it."])
    ax.set_title("(a) when each tracker is busy (one bar = one tracked camera frame; grid = every 3rd camera frame)")
    ax.grid(False)
    ax.set_ylim(-0.6, len(ms) - 0.4)
    ax = axes[1]
    for m in ms:
        s = _sched(dirs, case, m)
        src = s["cam_src"]
        pf = s["proc_frames"]
        age = np.where(src >= 0, ts - ts[pf[np.clip(src, 0, None)]], np.nan)
        ax.plot(ts, 1000 * age, color=COL[m] if m != "POGS, 10 iters" else "#63b3ed", lw=0.9,
                label=SHORT.get(m, m))
    ax.set(xlabel="wall time (s)", ylabel="estimate age (ms)",
           title="(b) age of the camera frame behind the published pose", ylim=(0, 1600))
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout(h_pad=0.6)
    save(fig, "rt_schedule.pdf")


def fig_rt_error(dirs):
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.1), sharey=True)
    for ax, sp in zip(axes, (1, 2)):
        case = f"base_s0_30hz__rt{sp}x_stereo"
        for m in ["Static (no motion)"] + RT:
            if (case, m) not in dirs:
                continue
            t = telemetry(dirs, case, m, "ceramic_mug")
            col = COL[m] if m != "POGS, 10 iters" else "#63b3ed"
            ls = "--" if ("10 iters" in m and "KF" in m) or m == "Static (no motion)" else "-"
            ax.plot(t["frame"] / 30.0 / sp, t["pos"], color=col, lw=0.9, ls=ls, label=SHORT.get(m, m))
        ax.axhline(10, color="#555", lw=0.5, ls=":")
        ax.set(xlabel="wall time (s)", title=f"base, seed 0, {sp}x playback")
        ax.set_ylim(0, 40)
    axes[0].set_ylabel("position error now (cm)")
    axes[1].legend(frameon=False, fontsize=5.8, loc="upper left")
    fig.tight_layout(w_pad=0.8)
    save(fig, "rt_error.pdf")


def fig_rt_bars(df):
    d = df[df["case"].str.contains("_30hz")]
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.2), gridspec_kw={"width_ratios": [2.2, 1]})
    ms = ["Static (no motion)"] + RT
    x = np.arange(len(ms))
    for k, sp in enumerate((1, 2)):
        mu, sd = [], []
        for m in ms:
            v = 100 * d[(d["speed"] == sp) & (d["method"] == m)]["adds_success"]
            mu.append(v.mean())
            sd.append(v.std(ddof=1))
        axes[0].bar(x + (k - 0.5) * 0.38, mu, 0.38, yerr=sd, color=[COL.get(m, "#999") for m in ms],
                    alpha=1.0 if sp == 1 else 0.45, error_kw={"lw": 0.5, "capsize": 1}, edgecolor="none")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([SHORT.get(m, m).replace("POGS + ", "").replace(", 10 iters", ", 10 it.")
                             for m in ms], rotation=25, ha="right")
    axes[0].set(ylabel="ADD-S success, now (%)", title="(a) real-time accuracy (solid 1x, faded 2x)", ylim=(0, 105))
    axes[0].grid(axis="x", visible=False)
    ms2 = ["POGS", "POGS, 10 iters", "POGS + KF + check", "POGS + KF + check, 10 iters"]
    lat = [d[d["method"] == m]["latency_ms"].mean() for m in ms2]
    age = [d[d["method"] == m]["estimate_age_ms"].mean() for m in ms2]
    xx = np.arange(len(ms2))
    axes[1].bar(xx - 0.2, lat, 0.4, color=[COL[m] for m in ms2], label="latency")
    axes[1].bar(xx + 0.2, age, 0.4, color=[COL[m] for m in ms2], alpha=0.45, label="mean estimate age")
    axes[1].set_xticks(xx)
    axes[1].set_xticklabels(["POGS", "POGS 10", "KF+chk", "KF+chk 10"], rotation=25, ha="right")
    axes[1].set(ylabel="ms", title="(b) latency (solid) / estimate age")
    axes[1].grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.2)
    save(fig, "rt_bars.pdf")


def fig_dashboards():
    for src, dst in [
        ("data_results/realtime/videos/base_s0_30hz__rt1x_stereo__pogs_sample.png", "dash_rt_pogs.jpg"),
        ("data_results/realtime/videos/base_s0_30hz__rt1x_stereo__pogs___kf___check__10_iters___latency_comp_sample.png",
         "dash_rt_kf.jpg"),
        ("data_results/v1_base_pogs/videos/obj_cereal_s0__1x_stereo__pogs_sample.png", "dash_cereal_pogs.jpg"),
    ]:
        if os.path.exists(src):
            img = cv2.imread(src)
            cv2.imwrite(os.path.join(OUT, dst), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            print("  ", dst)


def main():
    os.makedirs(OUT, exist_ok=True)
    df, dirs = load_index()
    print(f"{len(df)} cached runs (scene v{SCENE_VERSION})")
    fig_scan()
    fig_conditions()
    fig_trajectories()
    fig_training()
    fig_grouping()
    fig_factors(df)
    fig_budget(df)
    fig_speed(df)
    fig_cases(df, dirs)
    fig_rotation(df, dirs)
    fig_rt_schedule(dirs)
    fig_rt_error(dirs)
    fig_rt_bars(df)
    fig_dashboards()


if __name__ == "__main__":
    main()
